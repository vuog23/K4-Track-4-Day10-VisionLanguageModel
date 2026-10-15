"""Student implementation helpers for the Day 10 Qwen3-VL lab.

The instructor-provided scripts/ and tests/ remain the source of truth for
splitting, scoring, and submission checks. Run this file from the repository
root (the Kaggle notebook does that after cloning the student's public fork).
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CLASSES = ("pothole", "crack", "manhole")
DATA_DIR = Path("data")
WORK_DIR = Path("work")
EVIDENCE_DIR = Path("report/evidence")

PROMPT_NORM1000 = (
    "Locate road damage in this image. Detect only objects of these classes: "
    "pothole, crack, manhole. Return only a JSON list of objects with keys "
    '"label" and "bbox_2d". bbox_2d is [x1, y1, x2, y2] as integers on a '
    "normalized 0-1000 scale relative to image width and height. "
    'Example: [{"label":"pothole","bbox_2d":[100,200,300,450]}]. '
    "If no damage is visible, return []."
)

PROMPT_PIXEL = (
    "Locate road damage in this image. Detect only objects of these classes: "
    "pothole, crack, manhole. Return only a JSON list of objects with keys "
    '"label" and "bbox_2d". bbox_2d is [x1, y1, x2, y2] in absolute image '
    "pixel coordinates, with x from 0 to image width and y from 0 to image "
    'height. Example: [{"label":"pothole","bbox_2d":[64,72,192,160]}]. '
    "If no damage is visible, return []."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


def load_split(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return read_jsonl(DATA_DIR / "splits" / f"{split}.jsonl")


def verify_prepared_data() -> dict[str, Any]:
    """Check split sizes, class counts, and session isolation after preparation."""
    meta_path = DATA_DIR / "splits" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run scripts/prepare_data.py before this check.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "train": (1422, {"pothole": 1074, "crack": 1750, "manhole": 691}),
        "val": (276, {"pothole": 66, "crack": 470, "manhole": 103}),
        "test": (311, {"pothole": 121, "crack": 299, "manhole": 163}),
    }
    seen_sessions: dict[str, str] = {}
    for split, (n_images, class_counts) in expected.items():
        records = load_split(split)
        if len(records) != n_images:
            raise ValueError(f"{split}: expected {n_images} images, found {len(records)}")
        counts = {label: 0 for label in CLASSES}
        for entry in records:
            for obj in entry["objects"]:
                counts[obj["label"]] += 1
            session = entry["session"]
            previous = seen_sessions.setdefault(session, split)
            if previous != split:
                raise ValueError(f"Session {session!r} appears in {previous} and {split}")
            image_path = DATA_DIR / entry["image_path"]
            if not image_path.exists():
                raise FileNotFoundError(image_path)
        if counts != class_counts:
            raise ValueError(f"{split}: class counts {counts} differ from expected {class_counts}")
        if meta["counts"][split]["images"] != n_images:
            raise ValueError(f"meta.json image count disagrees for {split}")
    version = meta.get("dataset_version", "")
    if not version.startswith("d10-"):
        raise ValueError(f"Unexpected dataset_version: {version!r}")
    print(f"dataset_version={version}; session isolation and all split counts verified")
    return meta


def yolo_to_box(
    label: str,
    yolo: list[float] | tuple[float, float, float, float],
    coord_scale: str = "norm1000",
    width: int = 640,
    height: int = 360,
) -> dict[str, Any]:
    """Convert normalized YOLO center boxes to the lab's output convention."""
    if label not in CLASSES:
        raise ValueError(f"Unknown class label: {label}")
    if coord_scale not in {"norm1000", "pixel"}:
        raise ValueError(f"Unsupported coordinate scale: {coord_scale}")
    xc, yc, box_w, box_h = (float(value) for value in yolo)
    if not all(math.isfinite(value) for value in (xc, yc, box_w, box_h)):
        raise ValueError(f"Non-finite YOLO box: {yolo}")
    scale_x, scale_y = (1000.0, 1000.0) if coord_scale == "norm1000" else (float(width), float(height))
    max_x, max_y = (1000, 1000) if coord_scale == "norm1000" else (width, height)
    x1 = int(round(max(0.0, min(float(max_x), (xc - box_w / 2) * scale_x))))
    y1 = int(round(max(0.0, min(float(max_y), (yc - box_h / 2) * scale_y))))
    x2 = int(round(max(0.0, min(float(max_x), (xc + box_w / 2) * scale_x))))
    y2 = int(round(max(0.0, min(float(max_y), (yc + box_h / 2) * scale_y))))
    # Keep boxes valid even if rounding collapses a very small source box.
    if x2 <= x1 and x1 < max_x:
        x2 = x1 + 1
    if y2 <= y1 and y1 < max_y:
        y2 = y1 + 1
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate box after conversion: {(x1, y1, x2, y2)}")
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(
    objects: list[dict[str, Any]],
    coord_scale: str = "norm1000",
    width: int = 640,
    height: int = 360,
) -> str:
    boxes = [
        yolo_to_box(obj["label"], obj["yolo"], coord_scale, width, height)
        for obj in objects
    ]
    boxes.sort(key=lambda item: (item["bbox_2d"][1], item["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def _image_path(entry: dict[str, Any]) -> Path:
    return DATA_DIR / entry["image_path"]


def show_examples(split: str = "train", count: int = 6, seed: int = 42) -> None:
    """Display a reproducible sample of ground-truth boxes for visual QA."""
    from IPython.display import display

    entries = load_split(split)
    rng = random.Random(seed)
    examples = rng.sample(entries, min(count, len(entries)))
    colors = {"pothole": "#ff453a", "crack": "#30d158", "manhole": "#0a84ff"}
    for entry in examples:
        image = Image.open(_image_path(entry)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in entry["objects"]:
            box = yolo_to_box(obj["label"], obj["yolo"], "pixel", entry["width"], entry["height"])["bbox_2d"]
            draw.rectangle(box, outline=colors[obj["label"]], width=3)
            draw.text((box[0] + 2, max(0, box[1] - 12)), obj["label"], fill=colors[obj["label"]])
        print(f"{entry['image_id']} — {len(entry['objects'])} objects — session {entry['session']}")
        display(image)


def _chat_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids", "labels"}
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key in allowed and value is not None
    }


def encode_example(
    image: Image.Image,
    prompt: str,
    answer: str,
    processor: Any,
) -> dict[str, torch.Tensor]:
    """Encode one image/chat example and mask prompt and image tokens from loss."""
    user_message = _chat_messages(image, prompt)
    prompt_enc = processor.apply_chat_template(
        user_message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        user_message + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_enc["input_ids"]
    input_ids = full_enc["input_ids"]
    prompt_length = prompt_ids.shape[1]
    if not torch.equal(input_ids[:, :prompt_length], prompt_ids):
        raise ValueError("Prompt token sequence is not a prefix of the full chat sequence")
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100

    out: dict[str, torch.Tensor] = {
        "input_ids": input_ids[0],
        "attention_mask": full_enc.get("attention_mask", torch.ones_like(input_ids))[0],
        "labels": labels[0],
    }
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        value = full_enc.get(key)
        if value is not None:
            out[key] = value[0] if key == "mm_token_type_ids" and value.ndim == 2 else value
    if "token_type_ids" in full_enc:
        # Qwen3-VL uses mm_token_type_ids; token_type_ids is not a model input.
        out.pop("token_type_ids", None)
    return out


def collate_train(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad token sequences, but concatenate variable-size image tensors."""
    result: dict[str, torch.Tensor] = {}
    for key, pad_value in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100), ("mm_token_type_ids", 0)):
        values = [row[key] for row in batch if key in row]
        if values:
            result[key] = pad_sequence(values, batch_first=True, padding_value=pad_value)
    for key in ("pixel_values", "image_grid_thw"):
        values = [row[key] for row in batch if key in row]
        if values:
            result[key] = torch.cat(values, dim=0)
    return result


class RoadDamageDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        processor: Any,
        coord_scale: str = "norm1000",
        prompt: str | None = None,
    ) -> None:
        self.entries = entries
        self.processor = processor
        self.coord_scale = coord_scale
        self.prompt = prompt or (PROMPT_PIXEL if coord_scale == "pixel" else PROMPT_NORM1000)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        with Image.open(_image_path(entry)) as image_file:
            image = image_file.convert("RGB")
        answer = objects_to_answer(entry["objects"], self.coord_scale, entry["width"], entry["height"])
        return encode_example(image, self.prompt, answer, self.processor)


def collate_generation(features: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Left-pad prompts for batched autoregressive generation."""
    result: dict[str, torch.Tensor] = {}
    max_len = max(feature["input_ids"].shape[0] for feature in features)
    for key, pad_value in (("input_ids", pad_token_id), ("attention_mask", 0), ("mm_token_type_ids", 0)):
        if not all(key in feature for feature in features):
            continue
        values = []
        for feature in features:
            value = feature[key]
            padded = torch.full((max_len,), pad_value, dtype=value.dtype)
            padded[max_len - value.shape[0]:] = value
            values.append(padded)
        result[key] = torch.stack(values)
    for key in ("pixel_values", "image_grid_thw"):
        values = [feature[key] for feature in features if key in feature]
        if values:
            result[key] = torch.cat(values, dim=0)
    return result


def load_model_and_processor(model_id: str = MODEL_ID, for_training: bool = False) -> tuple[Any, Any]:
    """Load Qwen3-VL with 4-bit NF4 LLM weights and fp16 vision/projector."""
    if not torch.cuda.is_available():
        raise RuntimeError("This lab requires a CUDA GPU; select the Kaggle GPU T4 x2 accelerator first.")
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right" if for_training else "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.use_cache = not for_training
    return model, processor


def run_inference(
    model: Any,
    processor: Any,
    split: str = "test",
    pred_path: str | Path = "work/predictions.jsonl",
    coord_scale: str = "norm1000",
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> Path:
    """Generate one raw JSON prediction per image and save incrementally."""
    if coord_scale not in {"norm1000", "pixel"}:
        raise ValueError(f"Unsupported coordinate scale: {coord_scale}")
    entries = load_split(split)
    prompt = PROMPT_PIXEL if coord_scale == "pixel" else PROMPT_NORM1000
    pred_path = Path(pred_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    with pred_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            features = []
            for entry in chunk:
                with Image.open(_image_path(entry)) as image_file:
                    image = image_file.convert("RGB")
                encoded = processor.apply_chat_template(
                    _chat_messages(image, prompt),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                feature = {key: value[0] if key in {"input_ids", "attention_mask", "mm_token_type_ids"} and value.ndim == 2 else value
                           for key, value in encoded.items()
                           if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}}
                features.append(feature)
            batch = collate_generation(features, int(pad_id))
            inputs = _model_inputs(batch, device)
            prompt_width = inputs["input_ids"].shape[1]
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for entry, raw in zip(chunk, decoded):
                stream.write(json.dumps({"image_id": entry["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved {len(entries)} predictions to {pred_path}")
    return pred_path


def _target_names(model: Any, setup: str) -> tuple[list[str], list[str]]:
    llm_re = re.compile(
        r"(?:^|\.)language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    vit_re = re.compile(
        r"(?:^|\.)visual\.blocks\.\d+\."
        r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
    )
    names = [name for name, _ in model.named_modules()]
    llm_names = [name for name in names if llm_re.search(name)]
    vit_names = [name for name in names if vit_re.search(name)] if setup == "E3" else []
    expected_llm = 252
    expected_vit = 96 if setup == "E3" else 0
    if len(llm_names) != expected_llm or len(vit_names) != expected_vit:
        raise ValueError(
            "Unexpected Qwen module layout: "
            f"found {len(llm_names)} LLM and {len(vit_names)} ViT targets; "
            f"expected {expected_llm} and {expected_vit}. Inspect model.named_modules()."
        )
    return llm_names, vit_names


def apply_setup(model: Any, setup: str, rank: int = 16) -> tuple[Any, dict[str, int]]:
    """Freeze/train the required blocks and attach the lab's LoRA adapters."""
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()
    llm_names, vit_names = _target_names(model, setup)
    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=llm_names + vit_names,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    projector_count = 0
    if setup in {"E2", "E3"}:
        for name, parameter in peft_model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                parameter.data = parameter.data.to(dtype=torch.float32)
                parameter.requires_grad = True
                projector_count += parameter.numel()
        if projector_count == 0:
            raise ValueError("Could not find the Qwen visual projector parameters to unfreeze")
    trainable_params = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    expected_floor = 100_000_000 if setup in {"E2", "E3"} else 1
    if trainable_params < expected_floor:
        raise ValueError(f"Only {trainable_params:,} parameters are trainable for {setup}; inspect target matching")
    info = {
        "llm_lora_modules": len(llm_names),
        "vit_lora_modules": len(vit_names),
        "projector_params": projector_count,
        "trainable_params": trainable_params,
    }
    print(f"{setup}: {info}")
    return peft_model, info


def _save_projector(model: Any, path: Path) -> None:
    state = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("visual.merger" in name or "visual.deepstack_merger_list" in name)
    }
    if not state:
        raise ValueError("Expected trainable projector weights, but none were found")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _save_training_snapshot(model: Any, run_dir: Path, backup_dir: str | Path | None) -> None:
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    if any("visual.merger" in name or "visual.deepstack_merger_list" in name
           for name, parameter in model.named_parameters() if parameter.requires_grad):
        _save_projector(model, run_dir / "extra_projector.pt")
    if backup_dir is not None:
        backup = Path(backup_dir) / run_dir.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, backup, dirs_exist_ok=True)


def train_setup(
    setup: str,
    run_name: str | None = None,
    rank: int = 16,
    train_fraction: float = 1.0,
    coord_scale: str = "norm1000",
    max_steps: int = 300,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    checkpoint_every: int = 100,
    backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train E1/E2/E3 or one controlled E4 variant and save real run metadata."""
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if coord_scale not in {"norm1000", "pixel"}:
        raise ValueError("coord_scale must be norm1000 or pixel")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; use Kaggle's GPU T4 x2 accelerator.")
    run_name = run_name or setup.lower()
    run_dir = WORK_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    entries = load_split("train")
    if train_fraction < 1.0:
        take = max(1, int(round(len(entries) * train_fraction)))
        entries = random.Random(seed).sample(entries, take)
    model, processor = load_model_and_processor(for_training=True)
    model, setup_info = apply_setup(model, setup, rank=rank)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = RoadDamageDataset(entries, processor, coord_scale=coord_scale)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=lambda batch: collate_train(batch, processor.tokenizer.pad_token_id),
    )
    iterator = iter(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(max_steps * 0.05)),
        num_training_steps=max_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    losses: list[float] = []

    for step in range(1, max_steps + 1):
        micro_losses = []
        for _ in range(grad_accum):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu_batch = next(iterator)
            batch = _model_inputs(cpu_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step}: {loss.item()}")
            micro_losses.append(float(loss.detach().float().item()))
            scaler.scale(loss / grad_accum).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_loss = sum(micro_losses) / len(micro_losses)
        losses.append(step_loss)
        if step == 1 or step % 10 == 0 or step == max_steps:
            print(f"{run_name} step {step}/{max_steps} loss={step_loss:.4f}", flush=True)

        if step % checkpoint_every == 0 or step == max_steps:
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            metadata = {
                **setup_info,
                "setup": setup,
                "completed_steps": step,
                "rank": rank,
                "train_fraction": train_fraction,
                "coord_scale": coord_scale,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "learning_rate": learning_rate,
                "seed": seed,
                "train_minutes": elapsed_minutes,
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "loss_last": step_loss,
                "loss_mean_last_20": sum(losses[-20:]) / len(losses[-20:]),
            }
            (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _save_training_snapshot(model, run_dir, backup_dir)

    final_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    del optimizer, scheduler, scaler, loader, iterator, dataset, processor, model
    gc.collect()
    torch.cuda.empty_cache()
    return final_metadata


def _restore_projector(model: Any, path: Path) -> None:
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise KeyError(f"Saved projector parameters not found in loaded model: {missing[:5]}")
    with torch.no_grad():
        for name, saved in state.items():
            parameter = parameters[name]
            parameter.data.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def load_experiment(run_name: str | None = None) -> tuple[Any, Any]:
    """Load base model plus a saved LoRA adapter and optional full projector."""
    from peft import PeftModel

    run_name = run_name or "e1"
    run_dir = WORK_DIR / run_name
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Train {run_name} first; adapter not found at {adapter_dir}")
    base_model, processor = load_model_and_processor(for_training=False)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    _restore_projector(model, run_dir / "extra_projector.pt")
    model.eval()
    return model, processor


def score_predictions(
    pred_path: str | Path,
    name: str,
    out_path: str | Path,
    coord_scale: str = "norm1000",
    metadata: dict[str, Any] | None = None,
    split: str = "test",
) -> Path:
    command = [
        sys.executable,
        "scripts/score.py",
        "--pred", str(pred_path),
        "--name", name,
        "--split", split,
        "--coord", coord_scale,
        "--out", str(out_path),
    ]
    for key, value in (metadata or {}).items():
        command.extend(["--meta", f"{key}={value}"])
    subprocess.run(command, check=True)
    return Path(out_path)


def evaluate_experiment(
    run_name: str,
    evidence_name: str,
    coord_scale: str = "norm1000",
    split: str = "test",
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run test inference, then call the untouched official scoring script."""
    model, processor = load_experiment(run_name)
    pred_path = WORK_DIR / f"{evidence_name}.jsonl"
    run_inference(model, processor, split, pred_path, coord_scale, batch_size)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    metadata: dict[str, Any] = {}
    run_meta = WORK_DIR / run_name / "run.json"
    if run_meta.exists():
        run = json.loads(run_meta.read_text(encoding="utf-8"))
        metadata = {key: run[key] for key in ("trainable_params", "peak_vram_gb", "train_minutes")}
    out_path = EVIDENCE_DIR / f"{evidence_name}.json"
    score_predictions(pred_path, evidence_name.upper(), out_path, coord_scale, metadata, split)
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_zero_shot(split: str = "test", batch_size: int = 4) -> dict[str, Any]:
    model, processor = load_model_and_processor(for_training=False)
    pred_path = WORK_DIR / "e0.jsonl"
    run_inference(model, processor, split, pred_path, "norm1000", batch_size)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    out_path = EVIDENCE_DIR / "e0.json"
    score_predictions(pred_path, "E0", out_path)
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_e4(
    axis: str,
    base_setup: str,
    backup_dir: str | Path | None = None,
    max_steps: int = 300,
) -> list[dict[str, Any]]:
    """Run the two values of one E4 variable, holding the other settings fixed."""
    if axis not in {"rank", "data_size", "coord_scale"}:
        raise ValueError("E4 axis must be rank, data_size, or coord_scale")
    if base_setup not in {"E1", "E2", "E3"}:
        raise ValueError("Choose the best measured base setup from E1-E3")
    variants: list[tuple[str, dict[str, Any], str]]
    if axis == "rank":
        variants = [("rank_4", {"rank": 4}, "norm1000"), ("rank_64", {"rank": 64}, "norm1000")]
    elif axis == "data_size":
        variants = [("data_0.1", {"train_fraction": 0.1}, "norm1000"), ("data_1.0", {"train_fraction": 1.0}, "norm1000")]
    else:
        variants = [("coord_norm1000", {}, "norm1000"), ("coord_pixel", {}, "pixel")]

    results = []
    for suffix, params, coord_scale in variants:
        run_name = f"e4_{suffix}"
        run_meta_path = WORK_DIR / run_name / "run.json"
        rank = params.get("rank", 16)
        train_fraction = params.get("train_fraction", 1.0)
        cached_run = None
        if run_meta_path.exists():
            cached_run = json.loads(run_meta_path.read_text(encoding="utf-8"))
        run_is_complete = bool(cached_run) and all((
            cached_run.get("completed_steps") == max_steps,
            cached_run.get("setup") == base_setup,
            cached_run.get("rank") == rank,
            cached_run.get("train_fraction") == train_fraction,
            cached_run.get("coord_scale") == coord_scale,
            cached_run.get("max_steps") == max_steps,
        ))
        if not run_is_complete:
            train_setup(
                base_setup,
                run_name=run_name,
                coord_scale=coord_scale,
                max_steps=max_steps,
                backup_dir=backup_dir,
                **params,
            )

        cached_evidence_path = EVIDENCE_DIR / f"{run_name}.json"
        cached_predictions_path = WORK_DIR / f"{run_name}.jsonl"
        current_meta_path = DATA_DIR / "splits" / "meta.json"
        cached_evidence = None
        if cached_evidence_path.exists() and cached_predictions_path.exists() and current_meta_path.exists():
            candidate = json.loads(cached_evidence_path.read_text(encoding="utf-8"))
            current_version = json.loads(current_meta_path.read_text(encoding="utf-8"))["dataset_version"]
            if (candidate.get("split") == "test" and candidate.get("coord") == coord_scale
                    and candidate.get("dataset_version") == current_version
                    and candidate.get("missing_predictions") == 0):
                cached_evidence = candidate
        if cached_evidence is not None:
            results.append(cached_evidence)
        else:
            results.append(evaluate_experiment(run_name, run_name, coord_scale=coord_scale))
    print(f"E4 complete: axis={axis}; compare the two CI95 intervals before drawing a conclusion.")
    return results


def _match_for_visuals(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    from scripts.score import iou_xyxy

    pairs = []
    for gi, ground_truth in enumerate(gt):
        for pi, prediction in enumerate(pred):
            if ground_truth["label"] == prediction["label"]:
                pairs.append((iou_xyxy(ground_truth["bbox"], prediction["bbox"]), gi, pi))
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for iou, gi, pi in pairs:
        if iou < 0.5:
            break
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
    return matched_gt, matched_pred


def render_prediction_examples(
    pred_path: str | Path = "work/e1.jsonl",
    split: str = "test",
    count: int = 6,
    coord_scale: str = "norm1000",
    output_dir: str | Path = "report/figures",
) -> list[Path]:
    """Save six GT/prediction overlays with TP/FP/FN labels for the report."""
    from scripts.score import parse_output

    entries = load_split(split)
    predictions = {row["image_id"]: row["raw"] for row in read_jsonl(pred_path)}
    candidates: dict[str, list[dict[str, Any]]] = {"TP": [], "FP": [], "FN": []}
    for entry in entries:
        raw = predictions.get(entry["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        pred = [] if parsed is None else parsed
        gt = []
        for obj in entry["objects"]:
            box = yolo_to_box(obj["label"], obj["yolo"], "norm1000", entry["width"], entry["height"])["bbox_2d"]
            gt.append({"label": obj["label"], "bbox": tuple(box)})
        if coord_scale == "pixel":
            pred = [{"label": p["label"], "bbox": (
                p["bbox"][0] / entry["width"] * 1000,
                p["bbox"][1] / entry["height"] * 1000,
                p["bbox"][2] / entry["width"] * 1000,
                p["bbox"][3] / entry["height"] * 1000,
            )} for p in pred]
        matched_gt, matched_pred = _match_for_visuals(gt, pred)
        tags = []
        if matched_gt:
            tags.append("TP")
        if len(pred) > len(matched_pred):
            tags.append("FP")
        if len(gt) > len(matched_gt):
            tags.append("FN")
        for tag in tags:
            candidates[tag].append({"entry": entry, "gt": gt, "pred": pred,
                                    "matched_gt": matched_gt, "matched_pred": matched_pred,
                                    "tags": tags})

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for tag in ("TP", "FP", "FN"):
        added = 0
        for candidate in candidates[tag]:
            image_id = candidate["entry"]["image_id"]
            if image_id in used_ids:
                continue
            selected.append(candidate)
            used_ids.add(image_id)
            added += 1
            if added == 2:
                break
    if len(selected) < count:
        for entry in entries:
            if entry["image_id"] in used_ids:
                continue
            raw = predictions.get(entry["image_id"])
            parsed = parse_output(raw) if raw is not None else None
            pred = [] if parsed is None else parsed
            gt = [{"label": obj["label"], "bbox": tuple(yolo_to_box(
                obj["label"], obj["yolo"], "norm1000", entry["width"], entry["height"])["bbox_2d"])}
                  for obj in entry["objects"]]
            if coord_scale == "pixel":
                pred = [{"label": p["label"], "bbox": (
                    p["bbox"][0] / entry["width"] * 1000,
                    p["bbox"][1] / entry["height"] * 1000,
                    p["bbox"][2] / entry["width"] * 1000,
                    p["bbox"][3] / entry["height"] * 1000,
                )} for p in pred]
            matched_gt, matched_pred = _match_for_visuals(gt, pred)
            selected.append({"entry": entry, "gt": gt, "pred": pred,
                             "matched_gt": matched_gt, "matched_pred": matched_pred,
                             "tags": []})
            used_ids.add(entry["image_id"])
            if len(selected) == count:
                break

    colors = {"pothole": (255, 69, 58), "crack": (48, 209, 88), "manhole": (10, 132, 255)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for index, item in enumerate(selected[:count], start=1):
        entry, gt, pred = item["entry"], item["gt"], item["pred"]
        image = Image.open(_image_path(entry)).convert("RGB")
        draw = ImageDraw.Draw(image)
        tp = len(item["matched_gt"])
        fp = len(pred) - len(item["matched_pred"])
        fn = len(gt) - len(item["matched_gt"])
        title = f"{entry['image_id']} | TP={tp} FP={fp} FN={fn} (green=GT, red=prediction)"
        draw.rectangle((0, 0, image.width, 18), fill=(20, 20, 20))
        draw.text((4, 3), title[:100], fill="white", font=font)
        for gi, box in enumerate(gt):
            x1, y1, x2, y2 = box["bbox"]
            coords = (x1 / 1000 * entry["width"], y1 / 1000 * entry["height"],
                      x2 / 1000 * entry["width"], y2 / 1000 * entry["height"])
            draw.rectangle(coords, outline=(48, 209, 88), width=3)
            label = f"GT {box['label']} {'TP' if gi in item['matched_gt'] else 'FN'}"
            draw.text((coords[0] + 2, max(19, coords[1] - 12)), label, fill=(48, 209, 88), font=font)
        for pi, box in enumerate(pred):
            x1, y1, x2, y2 = box["bbox"]
            coords = (x1 / 1000 * entry["width"], y1 / 1000 * entry["height"],
                      x2 / 1000 * entry["width"], y2 / 1000 * entry["height"])
            draw.rectangle(coords, outline=(255, 69, 58), width=3)
            label = f"PRED {box['label']} {'TP' if pi in item['matched_pred'] else 'FP'}"
            draw.text((coords[0] + 2, min(image.height - 12, coords[3] + 2)), label,
                      fill=(255, 69, 58), font=font)
        path = output_dir / f"case_{index:02d}_{entry['image_id']}.png"
        image.save(path)
        saved.append(path)
    print(f"Saved {len(saved)} annotated examples to {output_dir}")
    return saved

