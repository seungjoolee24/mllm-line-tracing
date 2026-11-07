"""Pinned Qwen3-VL activation collection; imports do not load Torch or a model.

Generation is measured first. A separate causal replay of the *generated* answer
captures tensors. Decoder query rows refer to predictor positions, not to the
token being predicted. Ground-truth teacher forcing is deliberately not mixed in.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time


SCHEMA_VERSION = "1.1"
TRANSFORMERS_VERSION = "4.57.1"
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_token_layout(prompt_ids, generated_ids, image_token_id, question_positions=None):
    """Pure indexing contract, using zero-based positions in prompt+answer.

    A token at generated_positions[j] is predicted by the hidden state at
    predictor_positions[j]. The latter starts at the last prompt position.
    """
    prompt_ids, generated_ids = list(prompt_ids), list(generated_ids)
    if not prompt_ids or not generated_ids:
        raise ValueError("both prompt and generated token sequences must be nonempty")
    images = [i for i, token in enumerate(prompt_ids) if token == image_token_id]
    if not images:
        raise ValueError("prompt contains no image token positions")
    if question_positions is None:
        question_positions = [i for i in range(len(prompt_ids)) if i not in set(images)]
    question_positions = list(question_positions)
    if len(set(question_positions)) != len(question_positions) or any(
        type(i) is not int or not 0 <= i < len(prompt_ids) or i in images
        for i in question_positions
    ):
        raise ValueError("question positions must be unique non-image prompt indices")
    predictors = list(range(len(prompt_ids) - 1, len(prompt_ids) + len(generated_ids) - 1))
    generated = list(range(len(prompt_ids), len(prompt_ids) + len(generated_ids)))
    queries = sorted(set(question_positions + predictors + generated))
    return {
        "prompt_length": len(prompt_ids), "generated_length": len(generated_ids),
        "full_length": len(prompt_ids) + len(generated_ids),
        "image_positions": images, "question_positions": question_positions,
        "predictor_positions": predictors, "generated_positions": generated,
        "attention_query_positions": queries,
        "predictor_attention_rows": [queries.index(i) for i in predictors],
        "question_attention_rows": [queries.index(i) for i in question_positions],
        "generated_attention_rows": [queries.index(i) for i in generated],
        "first_answer_predictor_position": len(prompt_ids) - 1,
    }


class TensorWriter:
    """Write one detached CPU snapshot at a time, retaining only its index."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def save(self, name, tensor, category):
        import torch
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: expected Tensor")
        if any(entry["name"] == name for entry in self.records):
            raise RuntimeError(f"duplicate tensor capture: {name}")
        # copy=True is essential: DeepStack mutates decoder outputs in-place.
        snapshot = tensor.detach().to(device="cpu", copy=True).contiguous()
        if snapshot.is_floating_point() and not bool(torch.isfinite(snapshot).all()):
            raise RuntimeError(f"nonfinite activation in {name}")
        relative = f"tensors/{name}.pt"
        target = self.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, target)
        entry = {
            "name": name, "file": relative, "shape": list(snapshot.shape),
            "dtype": str(snapshot.dtype), "category": category,
            "tensor_bytes": snapshot.numel() * snapshot.element_size(),
            "file_bytes": target.stat().st_size, "sha256": sha256_file(target),
        }
        self.records.append(entry)
        del snapshot
        return entry

    def size_summary(self):
        categories = {}
        for record in self.records:
            item = categories.setdefault(record["category"], {"files": 0, "tensor_bytes": 0, "file_bytes": 0})
            item["files"] += 1
            item["tensor_bytes"] += record["tensor_bytes"]
            item["file_bytes"] += record["file_bytes"]
        return {"categories": categories, "tensor_files": len(self.records),
                "tensor_bytes": sum(r["tensor_bytes"] for r in self.records),
                "file_bytes": sum(r["file_bytes"] for r in self.records)}


def expected_capture_names(vision_layers, decoder_layers, deepstack_mergers):
    names = ["vision/patch_embed/input", "vision/patch_embed/output"]
    for i in range(vision_layers):
        names.extend([f"vision/block_{i:03d}/input", f"vision/block_{i:03d}/output"])
    for label in ["final"] + [f"deepstack_{i:03d}" for i in range(deepstack_mergers)]:
        names.extend([f"merger/{label}/input", f"merger/{label}/output"])
    for i in range(decoder_layers):
        names.extend([f"decoder/layer_{i:03d}/input", f"decoder/layer_{i:03d}/output_before_deepstack",
                      f"decoder/layer_{i:03d}/attention_to_image",
                      f"decoder/layer_{i:03d}/attention_image_mass"])
    return names + ["decoder/final_norm"]


class CaptureSession:
    """Hooks for Transformers 4.57.1; full block states, selected attention rows.

    Decoder block inputs include any previous layer's DeepStack injection.
    Decoder block outputs are snapshotted BEFORE the outer DeepStack injection.
    Each selected attention row retains all heads and its original full-sequence
    softmax denominator; no re-normalization over image keys is performed.
    """

    def __init__(self, model, writer, image_positions, question_positions,
                 predictor_positions, generated_positions):
        self.model, self.writer = model, writer
        self.image_positions = list(image_positions)
        self.question_positions = list(question_positions)
        self.predictor_positions = list(predictor_positions)
        self.generated_positions = list(generated_positions)
        self.query_positions = sorted(set(self.question_positions + self.predictor_positions + self.generated_positions))
        self.handles = []
        self.vision = model.model.visual
        self.decoder = model.model.language_model

    def _activation(self, name, category):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            self.writer.save(name, value, category)
        return hook

    def _input(self, name, category):
        def hook(module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if value is None:
                raise RuntimeError(f"missing hidden_states for {name}")
            self.writer.save(name, value, category)
        return hook

    def _attention(self, layer):
        def hook(module, args, output):
            import torch
            if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
                raise RuntimeError("decoder attention weights unavailable: eager backend is required")
            activation, weights = output
            if weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError(f"unexpected attention shape {tuple(weights.shape)}")
            queries = torch.tensor(self.query_positions, dtype=torch.long, device=weights.device)
            images = torch.tensor(self.image_positions, dtype=torch.long, device=weights.device)
            # Crop on the GPU, transfer only the requested rectangular submatrix.
            crop = weights[0].index_select(1, queries).index_select(2, images)
            self.writer.save(f"decoder/layer_{layer:03d}/attention_to_image", crop, "decoder_attention")
            self.writer.save(f"decoder/layer_{layer:03d}/attention_image_mass", crop.float().sum(-1),
                             "decoder_attention_summary")
            del crop
            # The pinned Qwen3VLTextDecoderLayer ignores this second return value.
            # Returning None prevents accidental retention of full N x N weights.
            return activation, None
        return hook

    def __enter__(self):
        try:
            self.handles.append(self.vision.patch_embed.register_forward_pre_hook(
                self._input("vision/patch_embed/input", "vision_embedding"), with_kwargs=True))
            self.handles.append(self.vision.patch_embed.register_forward_hook(
                self._activation("vision/patch_embed/output", "vision_embedding")))
            for i, block in enumerate(self.vision.blocks):
                self.handles.append(block.register_forward_pre_hook(
                    self._input(f"vision/block_{i:03d}/input", "vision_hidden"), with_kwargs=True))
                self.handles.append(block.register_forward_hook(
                    self._activation(f"vision/block_{i:03d}/output", "vision_hidden")))
            mergers = [("final", self.vision.merger)] + [
                (f"deepstack_{i:03d}", module)
                for i, module in enumerate(self.vision.deepstack_merger_list)
            ]
            for name, merger in mergers:
                self.handles.append(merger.register_forward_pre_hook(
                    self._input(f"merger/{name}/input", "merger_hidden"), with_kwargs=True))
                self.handles.append(merger.register_forward_hook(
                    self._activation(f"merger/{name}/output", "merger_hidden")))
            for i, layer in enumerate(self.decoder.layers):
                self.handles.append(layer.register_forward_pre_hook(
                    self._input(f"decoder/layer_{i:03d}/input", "decoder_hidden"), with_kwargs=True))
                self.handles.append(layer.register_forward_hook(
                    self._activation(f"decoder/layer_{i:03d}/output_before_deepstack", "decoder_hidden")))
                self.handles.append(layer.self_attn.register_forward_hook(self._attention(i)))
            self.handles.append(self.decoder.norm.register_forward_hook(
                self._activation("decoder/final_norm", "decoder_hidden")))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def assert_complete(self):
        names = {record["name"] for record in self.writer.records}
        expected = expected_capture_names(len(self.vision.blocks), len(self.decoder.layers),
                                          len(self.vision.deepstack_merger_list))
        missing = set(expected) - names
        if missing:
            raise RuntimeError(f"incomplete capture: {sorted(missing)}")


def _find_subsequence(sequence, subsequence):
    return [start for start in range(len(sequence) - len(subsequence) + 1)
            if sequence[start:start + len(subsequence)] == subsequence]


def question_token_positions(tokenizer, prompt_ids, question, image_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    starts = _find_subsequence(prompt_ids, question_ids)
    if len(starts) == 1:
        return list(range(starts[0], starts[0] + len(question_ids))), "exact_question_subsequence"
    # Template fallback: the user message's text follows the image end marker.
    last_image = max(i for i, token in enumerate(prompt_ids) if token == image_token_id)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop = next((i for i in range(last_image + 1, len(prompt_ids)) if prompt_ids[i] == end_id), None)
    if stop is None:
        raise RuntimeError("cannot identify input question token span")
    special = set(tokenizer.all_special_ids)
    positions = [i for i in range(last_image + 1, stop) if prompt_ids[i] not in special]
    if not positions:
        raise RuntimeError("input question span is empty")
    return positions, "user_text_after_image_template_fallback"


def describe_generated_tokens(tokenizer, generated_ids, prompt_length):
    texts = [tokenizer.decode(generated_ids[:i], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False) for i in range(len(generated_ids) + 1)]
    records = []
    for i, token in enumerate(generated_ids):
        records.append({"generation_step": i, "token_id": int(token),
                        "token_piece": tokenizer.convert_ids_to_tokens(int(token)),
                        "position": prompt_length + i, "predictor_position": prompt_length + i - 1,
                        "decoded_char_start": len(texts[i]), "decoded_char_end": len(texts[i + 1])})
    spans = []
    pattern = r"\b(" + "|".join(COLORS) + r")\b"
    for match in re.finditer(pattern, texts[-1].lower()):
        indices = [i for i, row in enumerate(records)
                   if row["decoded_char_end"] > match.start() and row["decoded_char_start"] < match.end()]
        spans.append({"color": match.group(), "char_span": [match.start(), match.end()],
                      "generation_steps": indices,
                      "token_positions": [prompt_length + i for i in indices],
                      "predictor_positions": [prompt_length + i - 1 for i in indices]})
    return {"decoded_with_special_tokens": texts[-1], "tokens": records, "color_spans": spans}


def score_answer(raw_answer, expected_colors):
    tokens = raw_answer.strip().lower().split()
    expected = list(expected_colors) + ["end"]
    prefix = 0
    for actual, wanted in zip(tokens, expected):
        if actual != wanted:
            break
        prefix += 1
    return {"exact_match": tokens == expected,
            "format_valid": bool(tokens) and tokens[-1] == "end" and all(x in COLORS for x in tokens[:-1]),
            "correct_prefix_colors": min(prefix, len(expected_colors)),
            "expected_colors": list(expected_colors), "predicted_items": tokens}


def greedy_generation_config(model, max_new_tokens):
    """Neutral greedy decoding, retaining only the checkpoint's special IDs.

    Sampling defaults or repetition processors must not silently change the raw
    argmax when generation is compared with the causal replay.
    """
    from transformers import GenerationConfig
    original = model.generation_config
    return GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        bos_token_id=original.bos_token_id, eos_token_id=original.eos_token_id,
        pad_token_id=original.pad_token_id,
    )


def resolve_greedy_generation_config(model, max_new_tokens):
    """Resolve the exact configuration used at the generate entry point.

    Transformers 4.57.1 can replace a supplied GenerationConfig's globally
    default values with checkpoint defaults when use_model_defaults is None.
    Explicit kwargs and use_model_defaults=False prevent that fallback.
    """
    requested = greedy_generation_config(model, max_new_tokens)
    effective, _ = model._prepare_generation_config(
        requested, use_model_defaults=False, do_sample=False, num_beams=1,
        repetition_penalty=1.0, return_dict_in_generate=True, output_logits=True,
        output_attentions=False, output_hidden_states=False,
    )
    if effective.do_sample is not False or effective.num_beams != 1 or effective.repetition_penalty != 1.0:
        raise RuntimeError("effective generation settings are not unmodified greedy decoding")
    if effective.get_generation_mode().value != "greedy_search":
        raise RuntimeError(f"unexpected effective generation mode: {effective.get_generation_mode()}")
    return effective


def verify_completed_sample(directory, config_hash, check_shapes=True):
    """Validate resume artifacts, not just existence of a completion marker."""
    directory = Path(directory)
    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("run_config_sha256") != config_hash:
        raise ValueError("sample was not completed with the same collection configuration")
    records = metadata.get("tensors")
    spec = metadata.get("capture_spec")
    if not isinstance(records, list) or not records or not isinstance(spec, dict):
        raise ValueError("completed sample has an empty or missing tensor capture index")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported sample metadata schema")
    alignment = metadata.get("alignment", {})
    if alignment.get("replay_matches_generation_argmax") is not True or alignment.get("generation_raw_argmax_matches_emitted") is not True:
        raise ValueError("completed sample does not have verified generation/replay alignment")
    try:
        counts = [spec[key] for key in ("vision_layers", "decoder_layers", "deepstack_mergers")]
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("invalid capture layer counts")
        expected = set(expected_capture_names(*counts)) | {
            "generation/raw_logits", "generation/token_ids", "replay/predictor_logits"}
        names = [record["name"] for record in records]
        if len(set(names)) != len(names) or set(names) != expected:
            raise ValueError("completed sample tensor index does not cover every required block")
    except (KeyError, TypeError) as error:
        raise ValueError("malformed tensor capture specification") from error
    root = directory.resolve()
    for record in records:
        path = (directory / record["file"]).resolve()
        path.relative_to(root)
        if not path.is_file() or path.stat().st_size != record["file_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"tensor artifact missing or corrupted: {record['file']}")
        if check_shapes:
            import torch
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
                raise ValueError(f"tensor shape/dtype mismatch: {record['file']}")
            del tensor
    return metadata


def _read_dataset(dataset):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    annotations = dataset / manifest.get("annotations", "annotations.jsonl")
    annotations.resolve().relative_to(dataset)
    samples = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != manifest["count"] or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("dataset count or unique IDs disagree with manifest")
    for sample in samples:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample["id"]):
            raise ValueError("unsafe sample ID")
        image = (dataset / sample["files"]["image"]).resolve()
        image.relative_to(dataset)
        if sha256_file(image) != sample["sha256"]["image"]:
            raise ValueError(f"dataset image hash mismatch: {sample['id']}")
    return dataset, samples, sha256_file(annotations)


def preflight(output, require_t4=False):
    import torch
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"collector requires transformers=={TRANSFORMERS_VERSION}; got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; select a T4 Colab runtime")
    name = torch.cuda.get_device_name(0)
    if require_t4 and "T4" not in name:
        raise RuntimeError(f"T4 was requested but attached GPU is {name}")
    Path(output).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < 8 * 1024**3:
        raise RuntimeError("at least 8 GiB free output storage is required for complete activations")
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": name, "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "initial_free_disk_bytes": free}


def collect_one(model, processor, dataset, sample, directory, config_hash, environment,
                max_new_tokens=64):
    import torch
    from PIL import Image
    started = time.time()
    directory = Path(directory)
    writer = TensorWriter(directory)
    image_path = dataset / sample["files"]["image"]
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    if image.width % 32 or image.height % 32:
        raise ValueError("fixed-resolution image dimensions must be multiples of 32")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], do_resize=False, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    grid = inputs["image_grid_thw"].tolist()
    image_token_id = model.config.image_token_id
    q_positions, q_method = question_token_positions(processor.tokenizer, prompt_ids, sample["prompt"], image_token_id)
    inputs = inputs.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    model.model.rope_deltas = None
    effective_generation = resolve_greedy_generation_config(model, max_new_tokens)
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=effective_generation,
                                   use_model_defaults=False, do_sample=False, num_beams=1,
                                   repetition_penalty=1.0,
                                   return_dict_in_generate=True,
                                   output_logits=True, output_attentions=False, output_hidden_states=False)
    full_ids = generated.sequences.detach()
    generated_ids = full_ids[0, len(prompt_ids):].tolist()
    if not generated_ids:
        raise RuntimeError("model generated no tokens")
    generation_logits = torch.stack([value[0].detach().float().cpu() for value in generated.logits])
    if generation_logits.shape[0] != len(generated_ids):
        raise RuntimeError("generation logits and generated token count differ")
    raw_answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    layout = build_token_layout(prompt_ids, generated_ids, image_token_id, q_positions)
    merged_count = sum(t * h * w // (model.config.vision_config.spatial_merge_size**2) for t, h, w in grid)
    if len(layout["image_positions"]) != merged_count:
        raise RuntimeError("processor grid and inserted image tokens disagree")
    writer.save("generation/raw_logits", generation_logits, "logits")
    writer.save("generation/token_ids", full_ids[0].cpu(), "token_ids")
    del generated
    gc.collect()
    torch.cuda.empty_cache()
    replay_inputs = {key: value for key, value in inputs.items() if key not in {"input_ids", "attention_mask"}}
    replay_inputs["input_ids"] = full_ids
    replay_inputs["attention_mask"] = torch.ones_like(full_ids)
    predictors = torch.tensor(layout["predictor_positions"], dtype=torch.long, device=model.device)
    model.model.rope_deltas = None
    with CaptureSession(model, writer, layout["image_positions"], layout["question_positions"],
                        layout["predictor_positions"], layout["generated_positions"]) as capture:
        with torch.inference_mode():
            outputs = model(**replay_inputs, use_cache=False, output_hidden_states=False,
                            output_attentions=False, logits_to_keep=predictors, return_dict=True)
        capture.assert_complete()
    replay_logits = outputs.logits[0].detach().float().cpu()
    writer.save("replay/predictor_logits", replay_logits, "logits")
    actual = torch.tensor(generated_ids, dtype=torch.long)
    replay_argmax = replay_logits.argmax(-1)
    cached_argmax = generation_logits.argmax(-1)
    replay_mismatches = (replay_argmax != actual).nonzero().flatten().tolist()
    generation_mismatches = (cached_argmax != actual).nonzero().flatten().tolist()
    alignment = {
        "replay_matches_generation_argmax": not replay_mismatches,
        "generation_raw_argmax_matches_emitted": not generation_mismatches,
        "replay_mismatch_steps": replay_mismatches,
        "generation_mismatch_steps": generation_mismatches,
        "max_absolute_logit_difference": float((replay_logits - generation_logits).abs().max()),
        "generated_token_logprob": replay_logits.log_softmax(-1).gather(1, actual[:, None]).flatten().tolist(),
        "generated_token_logits": generation_logits.gather(1, actual[:, None]).flatten().tolist(),
        "replay_target_logits": replay_logits.gather(1, actual[:, None]).flatten().tolist(),
        "generation_top2_margins": (generation_logits.topk(2, dim=-1).values[:, 0]
                                    - generation_logits.topk(2, dim=-1).values[:, 1]).tolist(),
        "replay_top2_margins": (replay_logits.topk(2, dim=-1).values[:, 0]
                                - replay_logits.topk(2, dim=-1).values[:, 1]).tolist(),
        "collection_mode": "causal replay of actual greedy answer; not ground-truth teacher forcing",
    }
    metadata = {
        "schema_version": SCHEMA_VERSION, "sample_id": sample["id"], "status": "complete",
        "run_config_sha256": config_hash, "family": sample.get("family"),
        "input": {"question": sample["prompt"], "rendered_chat": chat_text,
                  "question_sha256": hashlib.sha256(sample["prompt"].encode()).hexdigest(),
                  "image": str(sample["files"]["image"]), "image_sha256": sha256_file(image_path),
                  "source_size": list(image.size), "image_grid_thw": grid,
                  "resize": False, "prompt_token_ids": prompt_ids,
                  "question_span_method": q_method},
        "layout": layout,
        "generation": {"raw_answer": raw_answer, "token_ids": generated_ids,
                       "token_details": describe_generated_tokens(processor.tokenizer, generated_ids, len(prompt_ids)),
                       "max_new_tokens": effective_generation.max_new_tokens,
                       "do_sample": effective_generation.do_sample,
                       "num_beams": effective_generation.num_beams,
                       "repetition_penalty": effective_generation.repetition_penalty,
                       "use_model_defaults": False,
                       "effective_generation_config": effective_generation.to_dict(),
                       "hit_token_limit": len(generated_ids) >= max_new_tokens},
        "score": score_answer(raw_answer, sample["answer_colors"]), "alignment": alignment,
        "capture_spec": {"vision_layers": len(model.model.visual.blocks),
                         "decoder_layers": len(model.model.language_model.layers),
                         "deepstack_mergers": len(model.model.visual.deepstack_merger_list)},
        "capture_semantics": {
            "vision": "block input and output, static image encoding during replay",
            "decoder_input": "before decoder block, includes previous DeepStack injection",
            "decoder_output": "after block MLP residual and before same-index DeepStack injection",
            "attention": "all heads, union(question, answer predictor, generated token positions) to decoder image key positions",
            "attention_normalization": "original full-sequence softmax; not image-only renormalized",
        },
        "environment": environment,
        "performance": {"elapsed_seconds": time.time() - started,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()},
        "storage": writer.size_summary(), "tensors": writer.records,
    }
    image_set, question_set = set(layout["image_positions"]), set(q_positions)
    metadata["input"]["token_table"] = [
        {"position": i, "token_id": int(token),
         "token_piece": processor.tokenizer.convert_ids_to_tokens(int(token)),
         "kind": "image" if i in image_set else "question" if i in question_set else "prompt_other"}
        for i, token in enumerate(prompt_ids)
    ]
    if replay_mismatches or generation_mismatches:
        metadata["status"] = "alignment_mismatch"
        write_json(directory / "meta.json", metadata)
        raise RuntimeError(f"generation/replay token alignment mismatch: {alignment}")
    write_json(directory / "meta.json", metadata)
    del outputs, replay_logits, generation_logits, inputs, replay_inputs, full_ids
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Only the first N fixed dataset cases; omit to resume all")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-revision", required=True, help="Pinned 40-character model commit SHA")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-t4", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        parser.error("--model-revision must be a full immutable commit SHA")
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("token limit and sample limit must be positive")
    dataset, samples, annotations_hash = _read_dataset(args.dataset)
    output = Path(args.output).resolve()
    environment = preflight(output, args.require_t4)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # Resolve with the actual loaded checkpoint BEFORE hashing/publishing a run
    # configuration. Schema 1.1 therefore cannot resume a 1.0 requested-only run.
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=torch.float16,
        attn_implementation="eager", device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    ).eval()
    if model.config.vision_config.depth != 24 or model.config.text_config.num_hidden_layers != 36:
        raise RuntimeError("unexpected architecture for the pinned 4B baseline")
    effective_generation = resolve_greedy_generation_config(model, args.max_new_tokens)
    run_config = {"schema_version": SCHEMA_VERSION, "model_id": args.model_id,
                  "model_revision": args.model_revision, "transformers": TRANSFORMERS_VERSION,
                  "dtype": "torch.float16", "attention_backend": "eager",
                  "do_sample": effective_generation.do_sample,
                  "num_beams": effective_generation.num_beams,
                  "repetition_penalty": effective_generation.repetition_penalty,
                  "use_model_defaults": False,
                  "effective_generation_config": effective_generation.to_dict(),
                  "max_new_tokens": args.max_new_tokens, "annotations_sha256": annotations_hash,
                  "collector_sha256": sha256_file(__file__), "intended_dataset_count": len(samples)}
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    config_path = output / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("run_config_sha256") != config_hash or not args.resume:
            raise RuntimeError("output already exists; use --resume only with an identical configuration")
    else:
        write_json(config_path, {**run_config, "run_config_sha256": config_hash})
    write_json(output / "environment.json", environment)
    selected = samples[:args.limit] if args.limit else samples
    to_run, completed = [], []
    for sample in selected:
        target = output / "samples" / sample["id"]
        if target.exists():
            if not args.resume:
                raise RuntimeError(f"existing sample output: {target}")
            completed.append(verify_completed_sample(target, config_hash))
        else:
            to_run.append(sample)
    failures = []
    write_json(output / "run_summary.json", {
        "status": "running", "intended_cases": len(samples),
        "selected_cases_this_invocation": len(selected), "run_config_sha256": config_hash,
        "completed_cases": len(completed),
    })
    if to_run:
        write_json(output / "model_config.json", model.config.to_dict())
        write_json(output / "generation_config.json", model.generation_config.to_dict())
        write_json(output / "greedy_generation_config.json", effective_generation.to_dict())
        for index, sample in enumerate(to_run):
            print(f"Collecting {sample['id']} ({index + 1}/{len(to_run)})", flush=True)
            parent = output / "samples"
            parent.mkdir(parents=True, exist_ok=True)
            partial = parent / ("." + sample["id"] + ".partial")
            if partial.exists():
                marker = partial / "attempt.json"
                if not args.resume or not marker.exists() or json.loads(marker.read_text()).get("run_config_sha256") != config_hash:
                    raise RuntimeError(f"unrecognized partial output: {partial}")
                shutil.rmtree(partial)
            partial.mkdir()
            write_json(partial / "attempt.json", {"run_config_sha256": config_hash, "sample_id": sample["id"]})
            try:
                metadata = collect_one(model, processor, dataset, sample, partial, config_hash,
                                       environment, args.max_new_tokens)
                partial.rename(parent / sample["id"])
                completed.append(metadata)
                print(json.dumps({"sample_id": sample["id"], "exact_match": metadata["score"]["exact_match"],
                                  "storage": metadata["storage"], "performance": metadata["performance"]}), flush=True)
            except Exception as error:
                failures.append({"sample_id": sample["id"], "error": repr(error)})
                write_json(partial / "error.json", failures[-1])
                # Stop after one failed case rather than repeating an OOM across all ten.
                break
    total_complete = []
    for sample in samples:
        path = output / "samples" / sample["id"] / "meta.json"
        if path.exists():
            meta = json.loads(path.read_text())
            if meta.get("status") == "complete" and meta.get("run_config_sha256") == config_hash:
                total_complete.append(meta)
    summary = {
        "schema_version": SCHEMA_VERSION, "status": "failed" if failures else (
            "complete" if len(total_complete) == len(samples) else "partial"),
        "intended_cases": len(samples), "selected_cases_this_invocation": len(selected),
        "completed_cases": len(total_complete), "failures": failures,
        "run_config_sha256": config_hash, "model_id": args.model_id,
        "model_revision": args.model_revision,
        "cases": [{"sample_id": row["sample_id"], "family": row["family"],
                   "score": row["score"], "storage": row["storage"], "performance": row["performance"]}
                  for row in total_complete],
        "total_tensor_bytes": sum(row["storage"]["tensor_bytes"] for row in total_complete),
        "total_tensor_file_bytes": sum(row["storage"]["file_bytes"] for row in total_complete),
        "max_peak_gpu_allocated_bytes": max((row["performance"]["peak_gpu_allocated_bytes"] for row in total_complete), default=0),
        "max_peak_gpu_reserved_bytes": max((row["performance"]["peak_gpu_reserved_bytes"] for row in total_complete), default=0),
    }
    write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
