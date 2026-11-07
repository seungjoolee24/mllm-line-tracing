"""Descriptive analysis of existing run002 tensors; no model forward passes."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis import CaseAnalysis, overview


def inspect_case(run, dataset, sample_id, output):
    case = CaseAnalysis(run, sample_id, dataset)
    output.mkdir(parents=True, exist_ok=True)
    spans = case.meta["generation"]["token_details"]["color_spans"]
    steps = [span["generation_steps"][0] for span in spans]
    emitted = [span["color"] for span in spans]
    expected = [obj["color"] for obj in case.objects]
    lookup = {color: i for i, color in enumerate(expected)}
    predicted_indices = [lookup[color] for color in emitted]
    first_error = next((i for i, (a, b) in enumerate(zip(emitted, expected)) if a != b), None)
    selected_step = first_error if first_error is not None else 1
    coverage = np.asarray([case.geometry["object_fraction_of_merged_cell_area"][obj["id"]] for obj in case.objects])
    masses, image_masses, entropies, head_advantages = [], [], [], []
    for layer in range(36):
        attention = case.attention_rows(layer, "predictor")[steps]
        total = attention.sum(-1)
        probabilities = np.divide(attention, total[:, None], out=np.zeros_like(attention), where=total[:, None] > 1e-12)
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(-1) / np.log(attention.shape[1])
        entropy[total <= 1e-12] = np.nan
        masses.append(attention @ coverage.T)
        image_masses.append(total)
        entropies.append(entropy)
        weights = case.load(f"decoder/layer_{layer:03d}/attention_to_image").numpy().astype(np.float64)
        query_row = case.meta["layout"]["predictor_attention_rows"][steps[selected_step]]
        heads = weights[:, query_row, :]
        head_total = heads.sum(-1)
        head_object = heads @ coverage.T
        target = predicted_indices[selected_step]
        baseline = selected_step if first_error is not None else predicted_indices[selected_step - 1]
        head_advantages.append(np.divide(head_object[:, target] - head_object[:, baseline], head_total,
                                        out=np.full(32, np.nan), where=head_total > 1e-12))
        del weights, heads
    masses, image_masses, entropies = np.stack(masses), np.stack(image_masses), np.stack(entropies)
    conditional = np.divide(masses, image_masses[:, :, None], out=np.full_like(masses, np.nan), where=image_masses[:, :, None] > 1e-12)
    expected_mass = np.stack([conditional[:, i, i] for i in range(len(emitted))], axis=1)
    emitted_mass = np.stack([conditional[:, i, predicted_indices[i]] for i in range(len(emitted))], axis=1)
    previous_mass = np.full_like(emitted_mass, np.nan)
    for i in range(1, len(emitted)):
        previous_mass[:, i] = conditional[:, i, predicted_indices[i-1]]
    vision = case.activation_statistics("vision")
    decoder = case.activation_statistics("decoder")
    assert vision["norm"].shape == (24, 1600)
    assert decoder["norm"].shape == (36, case.meta["layout"]["full_length"])
    np.savez_compressed(output / "diagnostics.npz", object_mass_raw=masses, image_mass=image_masses,
                        image_conditional_object_mass=conditional, entropy=entropies,
                        expected_mass=expected_mass, emitted_mass=emitted_mass, previous_mass=previous_mass,
                        head_advantage=np.stack(head_advantages), vision_norm=vision["norm"],
                        decoder_norm=decoder["norm"], decoder_cosine=decoder["cosine_previous_layer"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, values, title in [(axes[0,0], np.log10(vision["norm"] + 1e-12), "ViT log10 hidden norm"),
                              (axes[0,1], vision["cosine_previous_layer"], "ViT cosine to previous layer"),
                              (axes[1,0], np.log10(decoder["norm"] + 1e-12), "Decoder log10 hidden norm"),
                              (axes[1,1], decoder["cosine_previous_layer"], "Decoder cosine to previous layer")]:
        limits = dict(vmin=-1, vmax=1) if "cosine" in title else {}
        image = ax.imshow(values, aspect="auto", origin="lower", **limits)
        ax.set(title=title, xlabel="Sequence position", ylabel="Layer (0 based)")
        fig.colorbar(image, ax=ax)
    fig.suptitle(sample_id + ": block outputs; decoder before DeepStack injection")
    fig.savefig(output / "activation_layers.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, values, title in [(axes[0,0], image_masses, "Original attention mass on image"),
                              (axes[0,1], entropies, "Conditional spatial entropy / log(400)"),
                              (axes[1,0], expected_mass, "Image-conditional expected-object mass"),
                              (axes[1,1], emitted_mass, "Image-conditional emitted-color-object mass")]:
        image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=0)
        ax.set(title=title, xlabel="Color occurrence (first BPE predictor)", ylabel="Layer (0 based)")
        fig.colorbar(image, ax=ax)
    fig.suptitle(sample_id)
    fig.savefig(output / "attention_layers.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for values, label in [(expected_mass, "expected next object"), (emitted_mass, "emitted-color object"), (previous_mass, "previous emitted object")]:
        axes[0].plot(np.arange(36), values[:, selected_step], label=label)
    axes[0].set(title=f"Color {selected_step}: expected {expected[selected_step]}, emitted {emitted[selected_step]}", xlabel="Layer (0 based)", ylabel="Image-conditional coverage-weighted mass")
    axes[0].legend()
    advantage = np.stack(head_advantages)
    vmax = max(float(np.nanmax(np.abs(advantage))), 1e-8)
    im = axes[1].imshow(advantage, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set(title="Emitted minus expected mass" if first_error is not None else "Emitted minus previous mass", xlabel="Head", ylabel="Layer (0 based)")
    fig.colorbar(im, ax=axes[1])
    fig.savefig(output / "selected_step_heads.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    for column, layer in enumerate([0, 17, 35]):
        for row, color_step in enumerate([0, selected_step]):
            values = case.attention_rows(layer, "predictor")[steps[color_step]]
            case.plot_overlay(values, f"Layer {layer}, predicts color {color_step}: {emitted[color_step]}", ax=axes[row,column])
    fig.savefig(output / "spatial_overlays.png", dpi=130)
    plt.close(fig)
    summary = {"sample_id": sample_id, "expected": expected, "emitted": emitted,
               "predicted_object_indices": predicted_indices, "first_error_color_index": first_error,
               "first_bpe_generation_steps": steps, "mapping": case.mapping_summary(),
               "verified_tensors_read": len(case.verified),
               "selected_step": selected_step,
               "selected_step_mean_expected_object_mass": float(np.nanmean(expected_mass[:,selected_step])),
               "selected_step_mean_emitted_object_mass": float(np.nanmean(emitted_mass[:,selected_step])),
               "selected_step_mean_previous_object_mass": float(np.nanmean(previous_mass[:,selected_step])),
               "selected_step_expected_larger_layers": np.flatnonzero(expected_mass[:,selected_step] > emitted_mass[:,selected_step]).tolist(),
               "selected_step_emitted_larger_layers": np.flatnonzero(emitted_mass[:,selected_step] > expected_mass[:,selected_step]).tolist(),
               "notes": ["Descriptive attention, not a causal explanation.", "Coverage projection assumes attention is uniform within each merged cell.", "case07 is a correct example, not a difficulty-matched control.", "Decoder transitions include DeepStack injection before the next block."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    exp = Path(args.experiment)
    run, dataset = exp / "results/run002", exp / "dataset"
    output = exp / "results/run002_exploration"
    output.mkdir(parents=True, exist_ok=True)
    rows = overview(run, dataset)
    (output / "behavior.json").write_text(json.dumps(rows, indent=2) + "\n")
    summaries = [inspect_case(run, dataset, identity, output / identity) for identity in ["case_03_swirl", "case_07_swirl"]]
    (output / "summary.json").write_text(json.dumps({"cases": summaries, "accuracy": sum(r["exact_match"] for r in rows)/len(rows), "source_run": "run002"}, indent=2) + "\n")
    print("ANALYSIS_COMPLETE", str(output), flush=True)


if __name__ == "__main__":
    main()
