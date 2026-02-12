from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from deep_uncertainty.enums import DatasetType, OptimizerType
from deep_uncertainty.datamodules.tabular_datamodule import TabularDataModule
from deep_uncertainty.models.bayesian_uq.mc_dropout_nn import MCDropoutDoublePoissonNN
from deep_uncertainty.models.bayesian_uq.mc_dropout_nn import MLPDropoutBackbone
from deep_uncertainty.utils.configs import TrainingConfig


def _load_lightning_state_dict(ckpt_path: Path) -> dict[str, Any]:
    # NOTE:
    # Newer PyTorch versions may default to `weights_only=True`, which can refuse
    # to unpickle Lightning checkpoints that contain references to project classes.
    # Since this evaluation script is intended for *local, trusted* checkpoints,
    # we explicitly opt into full loading when supported.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have the `weights_only` argument.
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")


@torch.no_grad()
def _collect_mc_dropout_outputs(
    model: MCDropoutDoublePoissonNN, dataloader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    mu_means: list[np.ndarray] = []
    phi_means: list[np.ndarray] = []
    epistemic_vars: list[np.ndarray] = []
    aleatoric_vars: list[np.ndarray] = []
    total_stds: list[np.ndarray] = []

    for batch in dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        out = model.predict_mc(x)

        xs.append(x.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        mu_means.append(out["mu_mean"].detach().cpu().numpy())
        phi_means.append(out["phi_mean"].detach().cpu().numpy())
        epistemic_vars.append(out["epistemic_var"].detach().cpu().numpy())
        aleatoric_vars.append(out["aleatoric_var"].detach().cpu().numpy())
        total_stds.append(out["total_std"].detach().cpu().numpy())

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts, axis=0)

    return {
        "x": cat(xs),
        "y": cat(ys),
        "mu_mean": cat(mu_means),
        "phi_mean": cat(phi_means),
        "epistemic_var": cat(epistemic_vars),
        "aleatoric_var": cat(aleatoric_vars),
        "total_std": cat(total_stds),
    }


def _save_plots(log_dir: Path, outputs: dict[str, np.ndarray], title_prefix: str = "") -> None:
    x = outputs["x"]
    y = outputs["y"].reshape(-1)
    mu = outputs["mu_mean"].reshape(-1)
    epistemic = outputs["epistemic_var"].reshape(-1)
    aleatoric = outputs["aleatoric_var"].reshape(-1)
    total_std = outputs["total_std"].reshape(-1)

    # 1) Pred vs true.
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y, mu, s=10, alpha=0.4)
    lo = float(min(y.min(), mu.min()))
    hi = float(max(y.max(), mu.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1)
    ax.set_xlabel("y_true")
    ax.set_ylabel("mu_mean (prediction)")
    ax.set_title(f"{title_prefix}Prediction vs True")
    fig.tight_layout()
    fig.savefig(log_dir / "pred_vs_true.png", dpi=150)
    plt.close(fig)

    # 2) Uncertainty histograms.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(epistemic, bins=50, alpha=0.6, density=True, label="epistemic_var")
    ax.hist(aleatoric, bins=50, alpha=0.6, density=True, label="aleatoric_var")
    ax.set_xlabel("variance")
    ax.set_ylabel("density")
    ax.set_title(f"{title_prefix}Uncertainty decomposition (variance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(log_dir / "uncertainty_hist.png", dpi=150)
    plt.close(fig)

    # 3) 1D posterior predictive-style plot (only if x is 1-dimensional).
    if x.ndim == 2 and x.shape[1] == 1:
        x1 = x.reshape(-1)
        order = np.argsort(x1)
        upper = mu + 1.96 * total_std
        lower = np.maximum(0.0, mu - 1.96 * total_std)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(x1[order], y[order], facecolors="none", edgecolors="gray", alpha=0.35, label="Test data")
        ax.plot(x1[order], mu[order], color="black", linewidth=1, label="mu_mean")
        ax.fill_between(
            x1[order],
            lower[order],
            upper[order],
            color="tab:blue",
            alpha=0.2,
            label="approx 95% band (mu ± 1.96·total_std)",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}Posterior predictive (approx)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(log_dir / "posterior_predictive_1d.png", dpi=150)
        plt.close(fig)


def main(log_dir: Path, config_path: Path, chkp_path: Path, num_mc_samples: int, dropout_p: float):
    log_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig.from_yaml(config_path)

    if config.dataset_type != DatasetType.TABULAR:
        raise NotImplementedError(
            "eval_mc_dropout.py currently supports TABULAR datasets only (needs 2D tensor x)."
        )
    if config.head_type.value != "double_poisson":
        raise NotImplementedError("eval_mc_dropout.py currently supports head_type: double_poisson only.")

    datamodule = TabularDataModule(
        dataset_path=config.dataset_spec,
        batch_size=config.batch_size,
        num_workers=0,
        persistent_workers=False,
    )
    datamodule.setup("test")

    # Build an MC Dropout model with a checkpoint-compatible MLP backbone.
    model = MCDropoutDoublePoissonNN(
        num_mc_samples=num_mc_samples,
        backbone_type=MLPDropoutBackbone,
        backbone_kwargs={
            "input_dim": int(config.input_dim),
            "output_dim": int(config.hidden_dim),
            "p": float(dropout_p),
            "freeze_backbone": bool(config.freeze_backbone),
        },
        optim_type=OptimizerType(config.optim_type.value),
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        beta_scheduler_type=config.beta_scheduler_type,
        beta_scheduler_kwargs=config.beta_scheduler_kwargs,
    )

    # Load weights from checkpoint (trained without dropout is OK; architecture matches MLP).
    state_dict = _load_lightning_state_dict(chkp_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    with open(log_dir / "mc_dropout_load_report.yaml", "w") as f:
        yaml.dump(
            {"missing_keys": list(missing), "unexpected_keys": list(unexpected)},
            f,
        )

    # Run the standard Lightning test loop (so you still get rmse/mae/etc like eval_model.py).
    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        devices=1,
        num_nodes=1,
    )
    metrics = evaluator.test(model=model, datamodule=datamodule)[0]
    with open(log_dir / "test_metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    # Collect MC Dropout-specific outputs for plotting.
    device = torch.device("cuda" if (torch.cuda.is_available() and config.accelerator_type.value == "gpu") else "cpu")
    model.to(device)
    outputs = _collect_mc_dropout_outputs(model, datamodule.test_dataloader(), device=device)
    np.savez_compressed(log_dir / "mc_dropout_outputs_test.npz", **outputs)
    _save_plots(log_dir, outputs, title_prefix=f"{config.experiment_name} / ")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--log-dir", type=str, help="Directory to log eval metrics and plots in.")
    parser.add_argument("--config-path", type=str, help="Path to config.yaml used to train model.")
    parser.add_argument("--chkp-path", type=str, help="Path to .ckpt where model weights are saved.")
    parser.add_argument("--num-mc-samples", type=int, default=50, help="MC Dropout samples per input.")
    parser.add_argument("--dropout-p", type=float, default=0.2, help="Dropout probability used at inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=Path(args.log_dir),
        config_path=Path(args.config_path),
        chkp_path=Path(args.chkp_path),
        num_mc_samples=args.num_mc_samples,
        dropout_p=args.dropout_p,
    )

