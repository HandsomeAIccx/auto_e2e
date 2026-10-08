"""Train joint heads from caller-supplied tensor batches, without a data SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_components.auto_e2e import AutoE2E
from training.scene_multitask import (
    SceneMultitaskObjective,
    run_scene_epoch,
    evaluate_scene,
    save_scene_checkpoint,
    load_scene_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train-batches",
        type=Path,
        required=True,
        help="torch.save(list_of_batch_dicts), tensors in documented scene contract",
    )
    parser.add_argument("--validation-batches", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or not 0 < args.learning_rate < float("inf"):
        parser.error("epochs and learning rate must be positive and finite")
    config = json.loads(args.config.read_text())
    device = torch.device(args.device)
    initialization = dict(config["model_kwargs"])
    if args.resume is not None:
        # The checkpoint restores every parameter; avoid downloading unused weights.
        initialization["is_pretrained"] = False
    model = AutoE2E(**initialization).to(device)
    head = model.Reactive_E2E.OccupancyFlowHead
    if head is None:
        parser.error("config must enable the occupancy head")
    objective_kwargs = {
        "num_classes": len(head.class_names),
        "dynamic_classes": head.dynamic_classes,
        **config.get("objective_kwargs", {}),
    }
    objective = SceneMultitaskObjective(**objective_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.resume is not None:
        load_scene_checkpoint(
            args.resume, model, optimizer=optimizer, objective=objective
        )
    batches = torch.load(args.train_batches, map_location="cpu", weights_only=True)
    validation = (
        None
        if args.validation_batches is None
        else torch.load(args.validation_batches, map_location="cpu", weights_only=True)
    )
    if isinstance(batches, dict):
        batches = [batches]
    if isinstance(validation, dict):
        validation = [validation]
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        report = {
            "epoch": epoch + 1,
            "training": run_scene_epoch(
                model, batches, objective, optimizer, device=device
            ),
        }
        if validation is not None:
            report["validation"] = evaluate_scene(
                model, validation, objective, device=device
            )
        history.append(report)
        save_scene_checkpoint(
            args.output / "checkpoint.pt",
            model,
            model_kwargs=config["model_kwargs"],
            optimizer=optimizer,
            objective=objective,
        )
        (args.output / "training.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    main()
