"""Audit completed progressive training checkpoints and logs entirely on CPU."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch

from models.RSTT_progressive import ProgressiveRSTT
from models.RSTT_tiny import RSTTTiny


def read_json(path):
    return json.loads(path.read_text())


def load_checkpoint(path):
    # The runner stores NumPy RNG arrays as lists for weights-only loading.
    return torch.load(path, map_location="cpu", weights_only=True)


def same_scores(first, second):
    return first.keys() == second.keys() and all(
        math.isclose(first[key], second[key], rel_tol=0, abs_tol=1e-9) for key in first)


def audit_run(run_dir):
    run_dir = Path(run_dir).resolve()
    config = read_json(run_dir / "config.json")
    report = {"run_dir": str(run_dir), "pass": False, "errors": [], "stages": []}
    previous_best, completed_steps, observed_steps = None, 0, 0

    def check(condition, message):
        if not condition:
            report["errors"].append(message)

    for stage in config["stage_schedule"]:
        name, steps = stage["name"], stage["steps"]
        directory = run_dir / name
        errors_before = len(report["errors"])
        stage_report = {"stage": name, "expected_steps": steps}
        report["stages"].append(stage_report)
        try:
            records = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line]
            observed_steps += len(records)
            check([row["step"] for row in records] == list(range(1, steps + 1)), f"{name}: train steps are not contiguous 1..{steps}")
            check(all(row["cumulative_step"] == completed_steps + row["step"] for row in records), f"{name}: cumulative log steps mismatch")
            check(all(row["losses"] and all(math.isfinite(v) for v in row["losses"].values()) for row in records), f"{name}: nonfinite or missing losses")
            last = load_checkpoint(directory / "last.pth")
            best = load_checkpoint(directory / "best.pth")
            initial = load_checkpoint(directory / "initial.pth")
            summary = read_json(directory / "summary.json")
            before = read_json(directory / "before_validation.json")
            if previous_best is not None:
                check(same_scores(before, previous_best), f"{name}: before scores differ from previous stage best")
            check(same_scores(summary["before"], before), f"{name}: summary before scores mismatch")
            check(last["step"] == steps, f"{name}: last checkpoint step mismatch")
            check(last["cumulative_step"] == completed_steps + steps, f"{name}: last cumulative step mismatch")
            check(last["scheduler"]["last_epoch"] == steps, f"{name}: scheduler last_epoch mismatch")
            model = ProgressiveRSTT(**last["model_config"])
            model.load_state_dict(last["state_dict"], strict=True)
            named = list(model.named_parameters())
            optimizer = last["optimizer"]
            groups = optimizer["param_groups"]
            if len(groups) == 2:
                named = [pair for pair in named if pair[0].startswith("conv_last.")] + [pair for pair in named if not pair[0].startswith("conv_last.")]
                check(len(groups[0]["params"]) == 2, f"{name}: reconstruction-head optimizer group mismatch")
            ids = [index for group in groups for index in group["params"]]
            check(len(ids) == len(named) == len(set(ids)), f"{name}: optimizer parameter registration mismatch")
            expected = {index: (key, parameter, steps if key.startswith("conv_last.") else max(0, steps - stage.get("freeze_steps", 0)))
                        for index, (key, parameter) in zip(ids, named)}
            expected_ids = {index for index, (_, _, count) in expected.items() if count > 0}
            states = optimizer["state"]
            check(set(states) == expected_ids, f"{name}: optimizer state coverage {len(states)}/{len(expected_ids)}")
            wrong_steps, wrong_moments, step_histogram = [], [], Counter()
            for index, state in states.items():
                if index not in expected:
                    continue
                key, parameter, expected_step = expected[index]
                actual_step = float(state["step"])
                step_histogram[str(actual_step)] += 1
                if actual_step != expected_step:
                    wrong_steps.append(key)
                for moment in ("exp_avg", "exp_avg_sq"):
                    if moment not in state or state[moment].shape != parameter.shape or not torch.isfinite(state[moment]).all():
                        wrong_moments.append(f"{key}.{moment}")
            check(not wrong_steps, f"{name}: optimizer step mismatch in {len(wrong_steps)} parameters: {wrong_steps[:3]}")
            check(not wrong_moments, f"{name}: invalid optimizer moments: {wrong_moments[:3]}")
            candidates = [(0, initial["validation_psnr_y"])] + [(row["step"], row["validation_psnr_y"]) for row in records if "validation_psnr_y" in row]
            winning_step, winning_scores = max(candidates, key=lambda pair: pair[1]["balanced_mean"])
            check(best["step"] == summary["best_step"] == winning_step, f"{name}: best step differs from validation maximum")
            check(same_scores(best["validation_psnr_y"], winning_scores) and same_scores(summary["recovered_best"], winning_scores), f"{name}: best score differs from validation maximum")
            check(math.isclose(last["best_validation_psnr_y"], winning_scores["balanced_mean"], abs_tol=1e-9), f"{name}: last best-score tracking mismatch")
            check(summary["steps"] == steps and summary["cumulative_optimizer_steps"] == completed_steps + steps, f"{name}: summary steps mismatch")
            stage_report.update(observed_steps=len(records), parameter_tensors=len(named), optimizer_state_tensors=len(states),
                                optimizer_step_histogram=dict(step_histogram),
                                expected_optimizer_step_histogram=dict(Counter(str(float(count)) for _, _, count in expected.values() if count > 0)), best_step=winning_step,
                                best_scores=winning_scores, model_config=last["model_config"],
                                training_sample_visits=len(records) * config["batch_size"])
            previous_best = best["validation_psnr_y"]
            del model, last, best, initial
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            check(False, f"{name}: {type(error).__name__}: {error}")
        stage_report["pass"] = len(report["errors"]) == errors_before
        completed_steps += steps

    try:
        manifest = read_json(run_dir / "final_manifest.json")
        final = load_checkpoint(run_dir / "final.pth")
        tiny = RSTTTiny(**final["model_config"])
        tiny.load_state_dict(final["state_dict"], strict=True)
        chosen = max((row for row in report["stages"] if row.get("model_config", {}).get("embed_dim") == 32),
                     key=lambda row: row["best_scores"]["balanced_mean"])
        selected = load_checkpoint(run_dir / chosen["stage"] / "best.pth")
        check(final["selected_stage"] == manifest["selected_stage"] == chosen["stage"], "final: selected stage differs from best eligible validation score")
        check(final["state_dict"].keys() == selected["state_dict"].keys() and all(torch.equal(value, selected["state_dict"][key]) for key, value in final["state_dict"].items()), "final: exported state differs from selected stage best")
        check(same_scores(final["validation_psnr_y"], selected["validation_psnr_y"]) and same_scores(manifest["validation_psnr_y"], selected["validation_psnr_y"]), "final: validation metadata mismatch")
        digest = hashlib.sha256((run_dir / "final.pth").read_bytes()).hexdigest()
        check(digest == manifest["sha256"], "final: manifest SHA256 mismatch")
        check(read_json(run_dir / "training_summary.json")["optimizer_steps"] == completed_steps, "run: training summary total steps mismatch")
        report["final"] = {"selected_stage": chosen["stage"], "sha256": digest,
                           "parameters": sum(p.numel() for p in tiny.parameters()), "strict_tiny_load": True}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        check(False, f"final: {type(error).__name__}: {error}")
    report.update(expected_optimizer_steps=completed_steps, observed_optimizer_steps=observed_steps,
                  batch_size=config["batch_size"], expected_sample_visits=completed_steps * config["batch_size"],
                  observed_sample_visits=observed_steps * config["batch_size"])
    report["pass"] = not report["errors"]
    destination = run_dir / "training_audit.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)
