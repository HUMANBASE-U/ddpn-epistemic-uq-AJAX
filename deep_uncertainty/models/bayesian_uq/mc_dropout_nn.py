from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.double_poisson_nn import DoublePoissonNN


# MLP backbone that is *checkpoint-compatible* with `deep_uncertainty.models.backbones.MLP`.
#
# Key idea:
# - Keep the exact same `self.layers = nn.Sequential(...)` module structure as the original MLP
#   so checkpoints trained with the original MLP can be loaded into this backbone.
# - Apply dropout *functionally* after each ReLU when the module is in train mode.
class MLPDropoutBackbone(Backbone):
    def __init__(self, input_dim: int = 1, output_dim: int = 64, p: float = 0.2, freeze_backbone: bool = False):
        super().__init__(output_dim=output_dim, freeze_backbone=freeze_backbone)
        self.p = float(p)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU):
                h = F.dropout(h, p=self.p, training=self.training)
        return h


class MCDropoutDoublePoissonNN(DoublePoissonNN):# automatic_optimization=True
    def __init__(
        self,
        num_mc_samples: int = 50,           #50 experts
        clamp_logmu: Tuple[float, float] = (-6.0, 5.0),
        clamp_logphi: Tuple[float, float] = (-3.0, 3.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mc_samples = num_mc_samples
        self.clamp_logmu = clamp_logmu
        self.clamp_logphi = clamp_logphi
        self.save_hyperparameters()


    #out_put here
    @torch.no_grad()
    def predict_mc(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        T = int(self.num_mc_samples)
        was_training = self.training
        self.train()

        samples = []
        for _ in range(T):
            y_hat_log = self._forward_impl(x)
            log_mu, log_phi = torch.split(y_hat_log, [1, 1], dim=-1)
            log_mu = torch.clamp(log_mu, *self.clamp_logmu)
            log_phi = torch.clamp(log_phi, *self.clamp_logphi)
            mu = torch.exp(log_mu)
            phi = torch.exp(log_phi)


            samples.append(torch.cat([mu, phi], dim=-1))

        stacked = torch.stack(samples, dim=0)
        mu_s, phi_s = torch.split(stacked, [1, 1], dim=-1)

        mu_mean = mu_s.mean(dim=0)
        phi_mean = phi_s.mean(dim=0)

        epistemic_var = mu_s.var(dim=0, unbiased=False)
        aleatoric_var = (mu_s / phi_s).mean(dim=0)
        total_var = epistemic_var + aleatoric_var
        total_std = torch.sqrt(total_var)

        
        if not was_training:
            self.eval()                     
        #dropout turned off


        return {
            "mu_mean": mu_mean,
            "phi_mean": phi_mean,
            "epistemic_var": epistemic_var,        #epistemic
            "aleatoric_var": aleatoric_var,
            "total_std": total_std,
        }




    def _predict_impl(self, x: torch.Tensor)  -> torch.Tensor:
        out = self.predict_mc(x)
        return torch.cat([out["mu_mean"], out["phi_mean"]], dim=-1)