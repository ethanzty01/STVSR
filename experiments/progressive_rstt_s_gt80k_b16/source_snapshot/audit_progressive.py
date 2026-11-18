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


def audit_extension(run_dir, parent_run):
    """Audit supervised mixed-dataset recovery initialized by a progressive run."""
    run_dir, parent_run = Path(run_dir).resolve(), Path(parent_run).resolve()
    report = {"run_dir": str(run_dir), "parent_run": str(parent_run),
              "audit_type": "supervised_extension", "pass": False, "errors": []}

    def check(condition, message):
        if not condition:
            report["errors"].append(message)

    try:
        config, parent_config = read_json(run_dir / "config.json"), read_json(parent_run / "config.json")
        summary = read_json(run_dir / "training_summary.json")
        parent_summary = read_json(parent_run / "training_summary.json")
        steps, batch_size = config["steps"], config["batch_size"]
        records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines() if line]
        check([row["step"] for row in records] == list(range(1, steps + 1)), "extension: train steps are not contiguous 1..additional_steps")
        check(all(row["losses"] and all(math.isfinite(value) for value in row["losses"].values()) for row in records), "extension: nonfinite or missing losses")
        check(config["mode"] == "supervised", "extension: training mode must be supervised")
        check(all(row["losses"]["kd"] == 0 for row in records), "extension: KD losses must all equal zero")
        check(all(row["losses"]["total"] == row["losses"]["gt"] for row in records), "extension: total loss differs from GT loss")
        last, best = load_checkpoint(run_dir / "distilled_last.pth"), load_checkpoint(run_dir / "distilled_best.pth")
        initial, parent_final = load_checkpoint(run_dir / "initial.pth"), load_checkpoint(parent_run / "final.pth")
        parent_digest = hashlib.sha256((parent_run / "final.pth").read_bytes()).hexdigest()
        check(config["init_sha256"] == parent_digest, "extension: initializer SHA256 differs from parent final")
        initial_matches = initial["state_dict"].keys() == parent_final["state_dict"].keys() and all(torch.equal(value, parent_final["state_dict"][key]) for key, value in initial["state_dict"].items())
        check(initial_matches, "extension: initial weights differ from parent final")
        base_step = parent_final["cumulative_step"]
        check(initial["step"] == 0 and initial["base_step"] == config["base_step"] == base_step, "extension: initialization checkpoint ancestry mismatch")
        check(all(row["cumulative_step"] == base_step + row["step"] for row in records), "extension: cumulative log ancestry mismatch")
        check(last["step"] == steps and last["scheduler"]["last_epoch"] == steps, "extension: last checkpoint/scheduler step mismatch")
        check(last["base_step"] == base_step and last["cumulative_step"] == base_step + steps, "extension: last checkpoint ancestry mismatch")
        check(summary["additional_steps"] == steps and summary["base_step"] == base_step and summary["cumulative_step"] == base_step + steps, "extension: training summary step mismatch")
        check(summary["examples"] == steps * batch_size, "extension: training summary sample count mismatch")
        splits, parent_splits = read_json(run_dir / "split_ids.json"), read_json(parent_run / "split_ids.json")
        check(splits["train"] == parent_splits["train"], "extension: training clip IDs differ from parent")
        check(splits["val"] == parent_splits["validation"], "extension: validation clip IDs differ from parent")
        tiny = RSTTTiny(**best["model_config"])
        tiny.load_state_dict(best["state_dict"], strict=True)
        parameters = sum(parameter.numel() for parameter in tiny.parameters())
        check(parameters == 286140, "extension: Tiny parameter count differs from 286140")
        tiny.load_state_dict(last["state_dict"], strict=True)
        named = list(tiny.named_parameters())
        states = last["optimizer"]["state"]
        check(len(last["optimizer"]["param_groups"]) == 1, "extension: expected one optimizer group in model parameter order")
        ids = [index for group in last["optimizer"]["param_groups"] for index in group["params"]]
        check(len(named) == len(ids) == len(set(ids)) == len(states) == 344 and set(ids) == set(states), "extension: optimizer state coverage differs from all 344 parameter tensors")
        wrong_steps, wrong_moments, step_histogram = [], [], Counter()
        for index, (key, parameter) in zip(ids, named):
            if index not in states:
                continue
            state = states[index]
            actual_step = float(state["step"])
            step_histogram[str(actual_step)] += 1
            if actual_step != steps:
                wrong_steps.append(key)
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment not in state or state[moment].shape != parameter.shape or not torch.isfinite(state[moment]).all():
                    wrong_moments.append(f"{key}.{moment}")
        check(not wrong_steps, f"extension: optimizer step mismatch in {len(wrong_steps)} parameters: {wrong_steps[:3]}")
        check(not wrong_moments, f"extension: invalid optimizer moments: {wrong_moments[:3]}")
        candidates = [(0, read_json(run_dir / "initial_validation.json"))] + [(row["step"], row["validation_psnr_y"]) for row in records if "validation_psnr_y" in row]
        winning_step, winning_scores = max(candidates, key=lambda pair: pair[1]["balanced_mean"])
        check(best["step"] == winning_step and same_scores(best["validation_psnr_y"], winning_scores), "extension: best checkpoint differs from validation maximum")
        check(all(math.isclose(value, winning_scores["balanced_mean"], rel_tol=0, abs_tol=1e-9) for value in (best["best_validation_psnr_y"], last["best_validation_psnr_y"], summary["best_validation_psnr_y"])), "extension: best-score tracking mismatch")
        best_digest = hashlib.sha256((run_dir / "distilled_best.pth").read_bytes()).hexdigest()
        final_path = run_dir / "final.pth"
        if final_path.exists():
            check(hashlib.sha256(final_path.read_bytes()).hexdigest() == best_digest, "extension: final.pth byte hash differs from distilled_best.pth")
        parent_steps = parent_summary["optimizer_steps"]
        total_examples = parent_steps * parent_config["batch_size"] + steps * batch_size
        report.update(additional_steps=steps, observed_optimizer_steps=len(records), batch_size=batch_size,
                      parameter_tensors=len(named), optimizer_state_tensors=len(states),
                      optimizer_step_histogram=dict(step_histogram), strict_tiny_load=True, parameters=parameters,
                      initial_matches_parent=initial_matches, parent_final_sha256=parent_digest,
                      best_step=winning_step, best_scores=winning_scores, best_sha256=best_digest,
                      final_copy_present=final_path.exists(), checkpoint_ancestry_base_step=base_step,
                      parent_executed_optimizer_steps=parent_steps, total_executed_optimizer_steps=parent_steps + steps,
                      extension_sample_visits=steps * batch_size, total_executed_sample_visits=total_examples)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        check(False, f"extension: {type(error).__name__}: {error}")
    report["pass"] = not report["errors"]
    (run_dir / "training_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, help="Audit a supervised mixed continuation of this progressive run")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit_extension(args.run_dir, args.parent_run) if args.parent_run else audit_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)
