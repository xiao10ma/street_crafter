"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from vwm.modules.diffusionmodules.sampling_utils import to_d
from vwm.util import append_dims, default, instantiate_from_config

from einops import rearrange


class BaseDiffusionSampler:
    def __init__(
            self,
            discretization_config: Union[Dict, ListConfig, OmegaConf],
            num_steps: Union[int, None] = None,
            guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
            verbose: bool = False,
            device: str = "cuda"
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(guider_config)
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )  # type: ignore
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, cond_mask, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, cond_mask, uc))  # type: ignore
        denoised = self.guider(denoised, sigma)  # type: ignore
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling Setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps"
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EulerEDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)  # pred_original_sample

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
            self,
            denoiser,
            x,  # x is randn
            cond,
            uc=None,
            cond_frame=None,
            cond_mask=None,
            num_steps=None
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)
        replace_cond_frames = cond_mask is not None and cond_mask.any()
        for i in self.get_sigma_gen(num_sigmas):
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)  # type: ignore
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                cond_mask,
                uc,
                gamma
            )
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)  # type: ignore
        return x

class EulerEDMSamplerSDS(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, weight_clamp=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.weight_clamp = weight_clamp

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)  # pred_original_sample

        # add training free guided sampling
        # if 'sample_guidance' in cond and cond['sample_guidance']['masked_guidance']:
        #     # mask 1: for areas with high rendering acc, we use the render_latents
        #     render_latents = cond['sample_guidance']['input']  # (f, c, h, w)
        #     acc = cond['sample_guidance']['acc']  # (f, 1, h, w)
        #     mask = cond['sample_guidance']['mask']  # (f, 1, h, w) # we need training free guidance in this area
        #     top_mask = torch.zeros_like(acc)
        #     if cond['sample_guidance']['cond_masked_guidance']:
        #         top_mask = top_mask + mask  # use mask to control whether an area should be blended
        #     if cond['sample_guidance']['acc_masked_guidance']:
        #         top_mask = top_mask * acc  # use acc as blending weight for the areas with high acc, use stronger acc control
        #     top_mask[:, :, :int(top_mask.shape[-2] * 0.4)] = 1.0  # for the condition frame, use the original image
        #     top_mask[cond_mask == 1] = 0.0  # for the condition frame, use the original image
        #     top_mask = torch.repeat_interleave(top_mask, denoised.shape[1], dim=1)  # (f, c, h, w)
        #     denoised = denoised * (1 - top_mask) + render_latents * top_mask

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
            self,
            denoiser,
            x,  # x is rand
            cond,
            uc=None,
            cond_frame=None,
            cond_mask=None,
            num_steps=None,
            scale=0.3,
            ori_img_latent=None,
            ori_mask_latent=None
    ):

        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)  # type: ignore
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        uc = default(uc, cond)
        if 'sample_guidance' in cond:
            num_inference_steps = int(self.num_steps * scale)  # type: ignore
            start_step = self.num_steps - num_inference_steps  # type: ignore
            end_step = self.num_steps  # type: ignore
            render_latents = cond['sample_guidance']['input']  # (f, c, h, w)
            start_sigma = sigmas[start_step]
            x = render_latents + x * start_sigma  # noisy render frame
        else:
            start_step, end_step = 0, self.num_steps
            start_sigma = sigmas[start_step]
            x *= torch.sqrt(1.0 + start_sigma ** 2)

        replace_cond_frames = cond_mask is not None and cond_mask.any()

        for i in range(start_step, end_step):  # type: ignore
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)  # type: ignore
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            noise = torch.randn_like(ori_img_latent)
            sigma_t = sigmas[i]
            ori_img_noisy = ori_img_latent + noise * sigma_t
            x[1:, ...] = x[1:, ...] * ori_mask_latent + ori_img_noisy * (1 - ori_mask_latent)

            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                cond_mask,
                uc,
                gamma
            )
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(cond_mask, cond_frame.ndim)  # type: ignore
        return x
