"""CPU tests for real Qwen3-VL capture; no pretrained weights are downloaded."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


HERE = Path(__file__).resolve().parent


def load_collector():
    name = "trace_collect_test_target"
    spec = importlib.util.spec_from_file_location(name, HERE / "collect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tiny_model():
    """Same module types and DeepStack path as the 4B model, with tiny weights."""
    config = Qwen3VLConfig(
        vision_config={
            "depth": 2, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "patch_size": 4, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "out_hidden_size": 32,
            "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
        },
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 3, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128,
            "rope_scaling": {
                "rope_type": "default", "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        image_token_id=59, video_token_id=60,
        vision_start_token_id=61, vision_end_token_id=62,
    )
    for value in (config, config.text_config, config.vision_config):
        value._attn_implementation = "eager"
    torch.manual_seed(109)
    return Qwen3VLForConditionalGeneration(config).eval()


PROMPT = [1, 61, 59, 59, 59, 59, 62, 8, 9, 10, 11]
GENERATED = [12, 13]
IMAGE_POSITIONS = [2, 3, 4, 5]
QUESTION_POSITIONS = [7, 8, 9]
PREDICTOR_POSITIONS = [10, 11]
GENERATED_POSITIONS = [11, 12]


def tiny_inputs():
    torch.manual_seed(110)
    return {
        "input_ids": torch.tensor([PROMPT + GENERATED]),
        "attention_mask": torch.ones((1, len(PROMPT) + len(GENERATED)), dtype=torch.long),
        "pixel_values": torch.randn(16, 96),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "use_cache": False,
    }


class TokenLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collect = load_collector()

    def test_predictor_positions_are_shifted_before_emitted_tokens(self):
        layout = self.collect.build_token_layout(
            PROMPT, GENERATED, image_token_id=59,
            question_positions=QUESTION_POSITIONS,
        )
        self.assertEqual(layout["image_positions"], IMAGE_POSITIONS)
        self.assertEqual(layout["question_positions"], QUESTION_POSITIONS)
        self.assertEqual(layout["predictor_positions"], PREDICTOR_POSITIONS)
        self.assertEqual(layout["generated_positions"], GENERATED_POSITIONS)

    def test_greedy_config_removes_checkpoint_sampling_and_repetition_processors(self):
        model = tiny_model()
        model.generation_config.repetition_penalty = 1.2
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.do_sample = True
        config = self.collect.greedy_generation_config(model, max_new_tokens=3)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_beams, 1)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.eos_token_id, model.generation_config.eos_token_id)
        self.assertEqual(model.generation_config.repetition_penalty, 1.2)

    def test_cached_generation_and_causal_replay_predictor_alignment(self):
        model = tiny_model()
        inputs = tiny_inputs()
        inputs["input_ids"] = inputs["input_ids"][:, :len(PROMPT)]
        inputs["attention_mask"] = inputs["attention_mask"][:, :len(PROMPT)]
        inputs.pop("use_cache")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, generation_config=self.collect.greedy_generation_config(model, 3),
                return_dict_in_generate=True, output_scores=True,
            )
            replay_inputs = dict(inputs)
            replay_inputs["input_ids"] = generated.sequences
            replay_inputs["attention_mask"] = torch.ones_like(generated.sequences)
            replay = model(**replay_inputs, use_cache=False)
        answer_ids = generated.sequences[0, len(PROMPT):].tolist()
        layout = self.collect.build_token_layout(
            PROMPT, answer_ids, image_token_id=59,
            question_positions=QUESTION_POSITIONS,
        )
        self.assertEqual(len(layout["predictor_positions"]), len(generated.scores))
        for step, position in enumerate(layout["predictor_positions"]):
            with self.subTest(step=step):
                torch.testing.assert_close(
                    replay.logits[0, position], generated.scores[step][0],
                    atol=2e-5, rtol=2e-5,
                )
                self.assertEqual(
                    int(replay.logits[0, position].argmax()), answer_ids[step],
                )

    def test_invalid_token_layout_is_rejected(self):
        for prompt, answer, question in (
            ([], GENERATED, QUESTION_POSITIONS),
            (PROMPT, [], QUESTION_POSITIONS),
            ([1, 2, 3], GENERATED, [1]),
            (PROMPT, GENERATED, [2]),
            (PROMPT, GENERATED, [7, 7]),
            (PROMPT, GENERATED, [len(PROMPT)]),
        ):
            with self.subTest(prompt=prompt, answer=answer, question=question):
                with self.assertRaises(ValueError):
                    self.collect.build_token_layout(prompt, answer, 59, question)


class TensorWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collect = load_collector()

    def test_snapshot_survives_inplace_mutation_and_has_valid_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = self.collect.TensorWriter(temporary)
            source = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
            expected = source.detach().clone()
            record = writer.save("activation", source, "test")
            with torch.no_grad():
                source.fill_(900)
            path = Path(temporary) / record["file"]
            saved = torch.load(path, map_location="cpu", weights_only=True)
            torch.testing.assert_close(saved, expected)
            self.assertFalse(saved.requires_grad)
            self.assertEqual(record["shape"], [3, 4])
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["tensor_bytes"], expected.numel() * expected.element_size())
            with self.assertRaises(RuntimeError):
                writer.save("activation", source, "test")
            with self.assertRaises(RuntimeError):
                writer.save("nonfinite", torch.tensor([float("nan")]), "test")


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collect = load_collector()
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def hook_count(model):
        return sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                   for module in model.modules())

    def test_real_model_capture_preserves_logits_attention_and_deepstack_states(self):
        model = tiny_model()
        inputs = tiny_inputs()
        reference = {}
        def snapshot(name):
            def callback(module, args, output):
                reference[name] = output.detach().clone()
            return callback
        def attention_snapshot(module, args, output):
            reference["attention"] = output[1].detach().clone()
        def input_snapshot(module, args, kwargs):
            reference["layer1_input"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
        handles = [
            model.model.visual.patch_embed.register_forward_hook(snapshot("patch_embed")),
            model.model.visual.blocks[0].register_forward_hook(snapshot("vision0")),
            model.model.visual.deepstack_merger_list[0].register_forward_hook(snapshot("deepstack0")),
            model.model.language_model.layers[0].register_forward_hook(snapshot("layer0_before")),
            model.model.language_model.layers[1].register_forward_pre_hook(input_snapshot, with_kwargs=True),
            model.model.language_model.layers[0].self_attn.register_forward_hook(attention_snapshot),
        ]
        with torch.inference_mode():
            baseline = model(**inputs).logits.detach().clone()
        for handle in handles:
            handle.remove()
        baseline_hooks = self.hook_count(model)
        with tempfile.TemporaryDirectory() as temporary:
            writer = self.collect.TensorWriter(temporary)
            session = self.collect.CaptureSession(
                model, writer, IMAGE_POSITIONS, QUESTION_POSITIONS,
                PREDICTOR_POSITIONS, GENERATED_POSITIONS,
            )
            with session, torch.inference_mode():
                captured = model(**inputs).logits.detach().clone()
            session.assert_complete()
            torch.testing.assert_close(captured, baseline, atol=0, rtol=0)
            self.assertEqual(self.hook_count(model), baseline_hooks)
            self.assertEqual(len(writer.records), 23)
            records = {entry["name"]: entry for entry in writer.records}
            def load(name):
                record = records[name]
                path = Path(temporary) / record["file"]
                self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                value = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(list(value.shape), record["shape"])
                self.assertFalse(value.requires_grad)
                self.assertEqual(value.device.type, "cpu")
                return value
            torch.testing.assert_close(load("vision/patch_embed/input"), inputs["pixel_values"])
            torch.testing.assert_close(load("vision/patch_embed/output"), reference["patch_embed"])
            torch.testing.assert_close(load("vision/block_000/output"), reference["vision0"])
            for layer in range(2):
                self.assertEqual(tuple(load(f"vision/block_{layer:03d}/output").shape), (16, 16))
            for layer in range(3):
                self.assertEqual(tuple(load(f"decoder/layer_{layer:03d}/input").shape), (1, 13, 32))
                self.assertEqual(tuple(load(f"decoder/layer_{layer:03d}/output_before_deepstack").shape), (1, 13, 32))
            before = load("decoder/layer_000/output_before_deepstack")
            after = load("decoder/layer_001/input")
            torch.testing.assert_close(before, reference["layer0_before"])
            torch.testing.assert_close(after, reference["layer1_input"])
            injection = torch.zeros_like(before)
            injection[:, IMAGE_POSITIONS, :] = reference["deepstack0"]
            torch.testing.assert_close(after - before, injection, atol=1e-7, rtol=1e-5)
            # If the pre-injection snapshot shared storage, this difference would be zero.
            self.assertGreater(float((after - before).abs().max()), 0)
            query_positions = sorted(set(QUESTION_POSITIONS + PREDICTOR_POSITIONS + GENERATED_POSITIONS))
            expected_attention = reference["attention"][0][:, query_positions, :][:, :, IMAGE_POSITIONS]
            attention = load("decoder/layer_000/attention_to_image")
            torch.testing.assert_close(attention, expected_attention, atol=0, rtol=0)
            self.assertEqual(tuple(attention.shape), (4, 6, 4))
            mass = load("decoder/layer_000/attention_image_mass")
            torch.testing.assert_close(mass, attention.sum(-1))
            self.assertTrue(bool((mass > 0).all()))
            self.assertTrue(bool((mass < 1).all()))

    def test_context_always_removes_hooks_and_missing_blocks_fail(self):
        model = tiny_model()
        original = self.hook_count(model)
        with tempfile.TemporaryDirectory() as temporary:
            writer = self.collect.TensorWriter(temporary)
            session = self.collect.CaptureSession(
                model, writer, IMAGE_POSITIONS, QUESTION_POSITIONS,
                PREDICTOR_POSITIONS, GENERATED_POSITIONS,
            )
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with session:
                    self.assertGreater(self.hook_count(model), original)
                    raise RuntimeError("intentional")
            self.assertEqual(self.hook_count(model), original)
            with self.assertRaisesRegex(RuntimeError, "incomplete capture"):
                session.assert_complete()


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collect = load_collector()

    def test_resume_requires_complete_inventory_and_valid_saved_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self.collect.TensorWriter(root)
            names = list(self.collect.expected_capture_names(2, 3, 1)) + [
                "generation/raw_logits", "generation/token_ids", "replay/predictor_logits",
            ]
            self.assertEqual(len(names), 26)
            self.assertEqual(len(self.collect.expected_capture_names(24, 36, 3)), 203)
            for name in names:
                writer.save(name, torch.ones(2, 3), "fixture")
            metadata = {
                "schema_version": self.collect.SCHEMA_VERSION,
                "status": "complete", "run_config_sha256": "expected-config",
                "capture_spec": {"vision_layers": 2, "decoder_layers": 3, "deepstack_mergers": 1},
                "alignment": {
                    "replay_matches_generation_argmax": True,
                    "generation_raw_argmax_matches_emitted": True,
                },
                "tensors": writer.records,
            }
            self.collect.write_json(root / "meta.json", metadata)
            result = self.collect.verify_completed_sample(root, "expected-config")
            self.assertEqual(result["status"], "complete")
            with self.assertRaises(ValueError):
                self.collect.verify_completed_sample(root, "different-config")
            self.collect.write_json(root / "meta.json", dict(metadata, alignment={}))
            with self.assertRaisesRegex(ValueError, "alignment"):
                self.collect.verify_completed_sample(root, "expected-config")
            for records in ([], writer.records[:-1], writer.records + [writer.records[0]]):
                self.collect.write_json(root / "meta.json", dict(metadata, tensors=records))
                with self.assertRaises(ValueError):
                    self.collect.verify_completed_sample(root, "expected-config")
            self.collect.write_json(root / "meta.json", metadata)
            broken_path = root / writer.records[0]["file"]
            broken_path.write_bytes(broken_path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "missing or corrupted"):
                self.collect.verify_completed_sample(root, "expected-config")


class PipelineTests(unittest.TestCase):
    def test_collect_one_end_to_end_on_tiny_model_with_mocked_processor_and_gpu_stats(self):
        collect = load_collector()
        model = tiny_model()
        class Tokenizer:
            all_special_ids = [1, 2, 59, 60, 61, 62]
            def encode(self, text, add_special_tokens=False):
                return [8, 9, 10]
            def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                return " ".join(f"token_{value}" for value in ids
                                if not skip_special_tokens or value not in self.all_special_ids)
            def convert_ids_to_tokens(self, value):
                return f"token_{value}"
            def convert_tokens_to_ids(self, value):
                return 62
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "mock chat text"
            def __call__(self, text, images, do_resize=False, return_tensors="pt"):
                inputs = tiny_inputs()
                inputs.pop("use_cache")
                inputs["input_ids"] = inputs["input_ids"][:, :len(PROMPT)]
                inputs["attention_mask"] = inputs["attention_mask"][:, :len(PROMPT)]
                return BatchFeature(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            Image.new("RGB", (32, 32), "white").save(image)
            sample = {"id": "tiny", "family": "fixture", "prompt": "Follow the line.",
                      "files": {"image": "input.png"}, "answer_colors": ["red", "blue"]}
            with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.empty_cache"), \
                 patch("torch.cuda.max_memory_allocated", return_value=0), \
                 patch("torch.cuda.max_memory_reserved", return_value=0):
                result = collect.collect_one(
                    model, Processor(), root, sample, root / "result", "expected-config",
                    {"gpu": "mocked for CPU integration only"}, max_new_tokens=3,
                )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["alignment"]["replay_matches_generation_argmax"])
            self.assertTrue(result["alignment"]["generation_raw_argmax_matches_emitted"])
            self.assertEqual(result["storage"]["tensor_files"], 26)
            self.assertEqual(result["input"]["image_sha256"], collect.sha256_file(image))
            self.assertEqual(result["input"]["question_span_method"], "exact_question_subsequence")
            verified = collect.verify_completed_sample(root / "result", "expected-config")
            self.assertEqual(verified["capture_spec"], {"vision_layers": 2, "decoder_layers": 3,
                                                       "deepstack_mergers": 1})


if __name__ == "__main__":
    unittest.main()
