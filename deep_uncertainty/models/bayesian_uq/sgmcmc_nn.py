from deep_uncertainty.models.bayesian_uq.bayesian_nn import DoublePoissonBayesianNN
from typing import Any, Dict, Optional
import torch
import posteriors
from deep_uncertainty.training.losses import double_poisson_nll as nll

class SGMCMCDoublePoissonNN(DoublePoissonBayesianNN):

    def __init__(self, burn_in, sample_interval, n_data, method, build_params, **kwargs):
        self.burn_in = burn_in
        self.sample_interval = sample_interval
        self.samples = []
        self.n_data = n_data
        self.method = method
        self.build_params = build_params
        self.global_step = 0
        super().__init__(**kwargs)

    def training_step(self, batch: Any):
        x_batch, y_batch = batch

        self.posterior_state, _ = self.posterior_transform.update(self.posterior_state, (x_batch, y_batch))
        if self.global_step > self.burn_in and self.global_step % self.sample_interval == 0:
            self.samples.append({k: v.detach().clone() for k, v in self.posterior_state.params.items()})
        
    def _sample_parameters(self, i) -> Optional[Dict[str, torch.Tensor]]:
        """
        Unlike Laplace and VI, SGMCMC collects samples from the Markov Chain process during training time.
        Laplace and VI fit a distribution during training then sample from it at inference time.
        """
        return self.samples[i]
    
    def _log_posterior(self, params, batch):
        x_batch, y_batch = batch
        output = self.functional((x_batch,))

        log_lik = - nll(output, y_batch) * self.n_data / len(x_batch)
        log_prior = posteriors.utils.diag_normal_log_prob(params, mean=0.0, sd_diag=0.5)
        return log_lik + log_prior, torch.tensor([])
    
    def init_posterior(self):
        """
        Subclasses must define the 'posteriors' transform (VI, Laplace, etc.)
        and initialize self.posterior_state here.
        """
        if self.method == 'sgld':
            self.posterior_transform = posteriors.sgmcmc.sgld.build(self._log_posterior, **self.build_params)
        elif self.method == 'sghmc':
            self.posterior_transform = posteriors.sgmcmc.sghmc.build(self._log_posterior, **self.build_params)
        else:
            raise NotImplementedError
