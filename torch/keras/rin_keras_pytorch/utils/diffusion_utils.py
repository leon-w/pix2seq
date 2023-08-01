import math

import torch

from . import debug_utils


class Scheduler:
    """Time scheduling and add noise to data."""

    def __init__(self, train_schedule):
        self._time_transform = self.get_time_transform(train_schedule)
        # self.sample_ddim = functools.partial(self.generate, sampler_name="ddim")
        # self.sample_ddpm = functools.partial(self.generate, sampler_name="ddpm")

    def get_time_transform(self, schedule_name):
        """Returns time transformation function according to schedule name."""
        if schedule_name.startswith("log@"):
            start, end, reverse = schedule_name.split("@")[1].split(",")
            start, end = float(start), float(end)
            reverse = reverse.lower() in ["t", "true"]
            time_transform = lambda t: log_schedule(t, start, end, reverse)
        elif schedule_name.startswith("sigmoid@"):
            start, end, tau = schedule_name.split("@")[1].split(",")
            start, end, tau = float(start), float(end), float(tau)
            time_transform = lambda t: sigmoid_schedule(t, start, end, tau)
        elif schedule_name.startswith("cosine"):
            if "@" in schedule_name:
                start, end, tau = schedule_name.split("@")[1].split(",")
                start, end, tau = float(start), float(end), float(tau)
                time_transform = lambda t: cosine_schedule(t, start, end, tau)
            else:
                time_transform = cosine_schedule_simple
        elif schedule_name.startswith("simple_linear"):
            time_transform = simple_linear_schedule
        else:
            raise ValueError(f"Unknown train schedule {schedule_name}")
        return time_transform

    def time_transform(self, time_step):
        return self._time_transform(time_step).cuda()

    def sample_noise(self, shape):
        """Sample noises."""
        return torch.randn(shape)

    def add_noise(
        self,
        inputs: torch.Tensor,
        time_step: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_step_shape = [inputs.size(0)] + [1] * (inputs.ndim - 1)
        if time_step is None:
            time_step = torch.rand(time_step_shape)
        elif isinstance(time_step, float):
            time_step = torch.full(time_step_shape, time_step)
        else:
            time_step = time_step.reshape(time_step_shape)

        gamma = self.time_transform(time_step).cuda()
        noise = self.sample_noise(inputs.shape).cuda()
        inputs_noised = inputs * torch.sqrt(gamma) + noise * torch.sqrt(1 - gamma)

        return inputs_noised, noise, time_step.squeeze(), gamma

    def transition_step(self, samples, data_pred, noise_pred, gamma_now, gamma_prev, sampler_name):
        """Transition to states with a smaller time step."""
        ddpm_var_type = "large"
        if sampler_name.startswith("ddpm") and "@" in sampler_name:
            ddpm_var_type = sampler_name.split("@")[1]

        if sampler_name == "ddim":
            samples = data_pred * torch.sqrt(gamma_prev) + noise_pred * torch.sqrt(1 - gamma_prev)
        elif sampler_name.startswith("ddpm"):
            log_alpha_t = torch.log(gamma_now) - torch.log(gamma_prev)
            alpha_t = torch.clamp(torch.exp(log_alpha_t), 0.0, 1.0)
            x_mean = torch.rsqrt(alpha_t) * (samples - torch.rsqrt(1 - gamma_now) * (1 - alpha_t) * noise_pred)
            if ddpm_var_type == "large":
                var_t = 1.0 - alpha_t  # var = beta_t
            elif ddpm_var_type == "small":
                var_t = torch.exp(torch.log1p(-gamma_prev) - torch.log1p(-gamma_now)) * (1.0 - alpha_t)
            else:
                raise ValueError(f"Unknown ddpm_var_type {ddpm_var_type}")
            eps = self.sample_noise(data_pred.shape)
            samples = x_mean + torch.sqrt(var_t) * eps
        return samples

    # def generate(
    #     self,
    #     transition_f,
    #     iterations,
    #     samples_shape,
    #     hidden_shapes=None,
    #     pred_type="eps",
    #     schedule=None,
    #     td=0.0,
    #     x0_clip="",
    #     self_cond="none",
    #     self_cond_decay=0.0,
    #     guidance=0.0,
    #     sampler_name="ddim",
    # ):
    #     """A sampling function.

    #     Args:
    #     transition_f: `callable` function for producing transition variables.
    #     iterations: `int` number of iterations for generation.
    #     samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
    #     hidden_shapes: a `list` of shapes of hiddens from denoising network,
    #         excluding bsz dim. Set to None if only single tensor output.
    #     pred_type: `str`, the output type of `transition_f`.
    #     schedule: `str`, the sampling schedule. If None, use the one during train.
    #     td: `float` specifying an adjustment for next time step.
    #     x0_clip: `str` specifying the range of x0 clipping, e.g. '0,1'. Set to
    #         None or empty str for no clipping.
    #     self_cond: `str`.
    #     self_cond_decay: `float` decaying factor between 0 and 1.
    #     guidance: `float` for cf guidance strength.
    #     sampler_name: `str`.

    #     Returns:
    #     Generated samples in samples_shape.
    #     """
    #     num_samples = samples_shape[0]
    #     ts = torch.ones([num_samples] + [1] * (len(samples_shape) - 1))
    #     get_step = lambda t: ts * (1.0 - float32(t) / iterations)
    #     time_transform = self.time_transform if schedule is None else (self.get_time_transform(schedule))

    #     samples = self.sample_noise(samples_shape)
    #     noise_pred, data_pred = torch.zeros_like(samples), torch.zeros_like(samples)
    #     x0_clip_fn = get_x0_clipping_function(x0_clip)

    #     latent_prev = None
    #     tape_prev = None
    #     for t in torch.arange(iterations):
    #         t = float32(t)
    #         time_step = get_step(t)
    #         time_step_p = torch.max(get_step(t + 1 + td), float32(0.0))
    #         gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)
    #         ctx.update_context(
    #             {
    #                 "denoise_out": pred_out,
    #                 "data_pred": data_pred,
    #                 "noise_pred": noise_pred,
    #                 "pred_type": pred_type,
    #             }
    #         )

    #         pred_out_l = transition_f(ctx.contextualized_inputs(samples), gamma, training=False)
    #         pred_out = pred_out_l
    #         pred_out_l = pred_out_l[0] if isinstance(pred_out_l, tuple) else (pred_out_l)
    #         pred_out_ = pred_out_l
    #         x0_eps = get_x0_eps(samples, gamma, pred_out_, pred_type, x0_clip_fn, truncate_noise=True)
    #         noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
    #         samples = self.transition_step(
    #             samples=samples,
    #             data_pred=data_pred,
    #             noise_pred=noise_pred,
    #             gamma_now=gamma,
    #             gamma_prev=gamma_prev,
    #             sampler_name=sampler_name,
    #         )
    #     return data_pred


# class SelfCondEstimateContext:
#     """Context manager for self-conditioning estimate during the inference."""

#     def __init__(self, mode, samples_shape, momentum=0):
#         """Init function.

#         Args:
#           mode: self-conditioning option.
#           samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
#           momentum: `int` indiciating the momentum averaging for vars.
#         """
#         self._mode = mode
#         self._estimate = torch.zeros(samples_shape)
#         self._momentum = momentum

#     def init_denoise_out(self, samples_shape):
#         """Initial denoising network output for update_context."""
#         return torch.zeros(samples_shape)

#     def update_context(self, context):
#         """Update context / self-conditioning estimate."""
#         if self._mode == "none":
#             return

#         def ema(v_old, v_new, momentum):
#             return momentum * v_old + (1 - momentum) * v_new

#         estimate = get_self_cond_estimate(context["data_pred"], context["noise_pred"], self._mode, context["pred_type"])
#         self._estimate = ema(self._estimate, estimate, self._momentum)

#     def contextualized_inputs(self, samples):
#         """Instead of using samples as inputs, obtain inputs with self-cond vars."""
#         if self._mode == "none":
#             return samples
#         else:
#             return torch.cat([samples, self._estimate], dim=-1)


# class SelfCondHiddenContext:
#     """Context manager for self-conditioning variables during the inference."""

#     def __init__(self, mode, samples_shape, hidden_shapes, momentum=0):
#         """Init function.

#         Args:
#           mode: self-conditioning option.
#           samples_shape: `tuple` or `list` shape of samples, e.g., (bsz, h, w, 3).
#           hidden_shapes: a `list` of shapes of extra input/output elements to the
#             denoising network, excluding bsz dim. Set to None if not tracked.
#           momentum: `int` indiciating the momentum averaging for vars.
#         """
#         self._mode = mode
#         var_shapes = [[samples_shape[0]] + list(s) for s in hidden_shapes]
#         self._hiddens = [torch.zeros(shape) for shape in var_shapes]
#         self._momentum = momentum

#     def init_denoise_out(self, samples_shape):
#         """Initial denoising network output for update_context."""
#         return tuple([torch.zeros(samples_shape)] + self._hiddens)

#     def update_context(self, context):
#         """Update context / self-conditioning variables."""

#         def ema(v_old, v_new, momentum):
#             return momentum * v_old + (1 - momentum) * v_new

#         vars_update = []
#         hiddens = context["denoise_out"][1:]  # excluding 0-th which is estimate.
#         for var_old, var_new in zip(self._hiddens, hiddens):
#             vars_update.append(ema(var_old, var_new, self._momentum))
#         self._hiddens = vars_update

#     def contextualized_inputs(self, samples):
#         """Instead of using samples as inputs, obtain inputs with self-cond vars."""
#         return tuple([samples] + self._hiddens)


def float32(x, device=None):
    return torch.tensor(x, dtype=torch.float32, device=device)


def cosine_schedule_simple(t, ns=0.0002, ds=0.00025):
    """Cosine schedule.

    Args:
      t: `float` between 0 and 1.
      ns: `float` numerator constant shift.
      ds: `float` denominator constant shift.

    Returns:
      `float` of transformed time between 0 and 1.
    """
    return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2


def cosine_schedule(t, start=0.0, end=0.5, tau=1.0, clip_min=1e-9):
    """Cosine schedule.

    Args:
        t: `float` between 0 and 1.
        start: `float` starting point in x-axis of cosine function.
        end: `float` ending point in x-axis of cosine function.
        tau: `float` temperature.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    y_start = torch.cos(start * math.pi / 2) ** (2 * tau)
    y_end = torch.cos(end * math.pi / 2) ** (2 * tau)
    output = (torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau) - y_end) / (y_start - y_end)
    output.clamp_(clip_min, 1.0)
    return output


def sigmoid_schedule(t, start=-3.0, end=3.0, tau=1.0, clip_min=1e-9):
    """Sigmoid schedule.

    Args:
        t: `float` between 0 and 1.
        start: `float` starting point in x-axis of sigmoid function.
        end: `float` ending point in x-axis of sigmoid function.
        tau: `float` scaling temperature for sigmoid function.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    v_start = torch.sigmoid(start / tau)
    v_end = torch.sigmoid(end / tau)
    output = (-torch.sigmoid((t * (end - start) + start) / tau) + v_end) / (v_end - v_start)
    output.clamp_(clip_min, 1.0)
    return output


def log_schedule(t, start=1.0, end=100.0, reverse=False):
    """Log schedule.

    Args:
      t: `float` between 0 and 1.
      start: `float` starting point in x-axis of log function.
      end: `float` ending point in x-axis of log function.
      reverse: `boolean` whether to reverse the curving direction.

    Returns:
      `float` of transformed time between 0 and 1.
    """
    if reverse:
        start, end = end, start

    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    v_start = start.log()
    v_end = end.log()
    output = (-torch.log(t * (end - start) + start) + v_end) / (v_end - v_start)
    output.clamp_(0.0, 1.0)
    return output


def simple_linear_schedule(t, clip_min=1e-9):
    """Simple linear schedule.

    Args:
        t: `float` between 0 and 1.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    output = 1.0 - t
    output.clamp_(clip_min, 1.0)
    return output


def get_x0_clipping_function(x0_clip):
    """Get x0 clipping function."""
    if x0_clip is None or x0_clip == "":
        return lambda x: x
    else:
        x0_min, x0_max = x0_clip.split(",")
        x0_min, x0_max = float(x0_min), float(x0_max)
        return lambda x: torch.clamp(x, x0_min, x0_max)


def get_x0_from_eps(xt, gamma, noise_pred):
    data_pred = 1.0 / torch.sqrt(gamma) * (xt - torch.sqrt(1.0 - gamma) * noise_pred)
    return data_pred


def get_eps_from_x0(xt, gamma, data_pred):
    noise_pred = 1.0 / torch.sqrt(1 - gamma) * (xt - torch.sqrt(gamma) * data_pred)
    return noise_pred


def get_x0_from_v(xt, gamma, v_pred):
    return torch.sqrt(gamma) * xt - torch.sqrt(1 - gamma) * v_pred


def get_eps_from_v(xt, gamma, v_pred):
    return torch.sqrt(1 - gamma) * xt + torch.sqrt(gamma) * v_pred


def get_x0_eps(
    xt,
    gamma,
    denoise_out,
    pred_type,
    x0_clip_fn,
    truncate_noise=False,
):
    """Get x0 and eps from denoising output."""
    if pred_type == "eps":
        noise_pred = denoise_out
        data_pred = get_x0_from_eps(xt, gamma, noise_pred)
        data_pred = x0_clip_fn(data_pred)
        if truncate_noise:
            noise_pred = get_eps_from_x0(xt, gamma, data_pred)
    elif pred_type.startswith("x"):
        data_pred = denoise_out
        data_pred = x0_clip_fn(data_pred)
        noise_pred = get_eps_from_x0(xt, gamma, data_pred)
    elif pred_type.startswith("v"):
        v_pred = denoise_out
        data_pred = get_x0_from_v(xt, gamma, v_pred)
        data_pred = x0_clip_fn(data_pred)
        if truncate_noise:
            noise_pred = get_eps_from_x0(xt, gamma, data_pred)
        else:
            noise_pred = get_eps_from_v(xt, gamma, v_pred)
    else:
        raise ValueError(f"Unknown pred_type {pred_type}")
    return {"noise_pred": noise_pred, "data_pred": data_pred}


def get_self_cond_estimate(
    data_pred,
    noise_pred,
    self_cond,
    pred_type,
):
    """Returns self cond estimate given predicted data or noise."""
    assert self_cond in ["x", "eps", "auto"]
    if self_cond == "x":
        estimate = data_pred
    elif self_cond == "eps":
        estimate = noise_pred
    else:
        estimate = noise_pred if pred_type == "eps" else data_pred
    return estimate


# def add_self_cond_estimate(
#     x_noised,
#     gamma,
#     denoise_f,
#     pred_type,
#     self_cond,
#     x0_clip,
#     num_sc_examples,
#     drop_rate=0.0,
#     training=True,
# ):
#     """Returns x_noised with self cond estimate added for the first 1/2 batch."""
#     assert self_cond in ["x", "eps", "auto"]
#     if drop_rate > 0:
#         raise NotImplementedError("Self-Cond by masking is not implemented yet!")
#     x_noised_p = x_noised[:num_sc_examples]
#     gamma_p = gamma[:num_sc_examples]
#     placeholder = torch.zeros_like(x_noised_p)
#     pred_out = denoise_f(torch.cat([x_noised_p, placeholder], -1), gamma_p, training)
#     x0_clip_fn = get_x0_clipping_function(x0_clip)
#     x0_eps = get_x0_eps(x_noised_p, gamma_p, pred_out, pred_type, x0_clip_fn, truncate_noise=True)
#     estimate = get_self_cond_estimate(x0_eps["data_pred"], x0_eps["noise_pred"], self_cond, pred_type)
#     estimate = torch.cat([estimate, torch.zeros_like(x_noised[num_sc_examples:])], 0)
#     estimate = estimate.detach()
#     return torch.cat([x_noised, estimate], -1)


# def add_self_cond_hidden(
#     x_noised,
#     gamma,
#     denoise_f,
#     num_sc_examples,
#     hidden_shapes,
#     drop_rate=0.0,
#     training=True,
# ):
#     """Returns inputs (with self-cond hiddens) to denoising networks."""
#     bsz = x_noised.shape[0]  # assuming bsz > 1
#     x_noised_p = x_noised[:num_sc_examples]
#     gamma_p = gamma[:num_sc_examples]
#     placeholders1 = [torch.zeros([num_sc_examples] + s) for s in hidden_shapes]
#     placeholders2 = [torch.zeros([bsz - num_sc_examples] + s) for s in hidden_shapes]
#     pred_out = denoise_f(torch.cat([x_noised_p] + placeholders1, dim=1), gamma_p, training)
#     hiddens = [torch.cat([u, v], 0) for u, v in zip(pred_out[1:], placeholders2)]
#     if drop_rate > 0:  # The rate of masking out self-cond hiddens.
#         masks = torch.rand(bsz) > drop_rate
#         expand_dims = lambda x, h: x.view(bsz, *(1 for _ in range(h.dim() - 1)))
#         hiddens = [h * expand_dims(masks, h) for h in hiddens]
#     return [x_noised] + [h.detach() for h in hiddens]
