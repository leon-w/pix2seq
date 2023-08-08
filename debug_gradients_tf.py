import numpy as np
import tensorflow as tf
from einops import rearrange
from ml_collections import ConfigDict
from PIL import Image

from architectures.tape import ImageTapeDenoiser
from debug_utils import p, track
from models.image_diffusion_model import Model as RinDiffusionModel

config = dict(
    dataset=dict(
        image_size=32,
        num_classes=10,
    ),
    model=dict(
        arch_name="tape",
        b_scale=1.0,
        b_type="uint8",
        cond_decoupled_read=False,
        cond_dim=0,
        cond_dropout=0.0,
        cond_on_latent=True,
        cond_proj=True,
        cond_tape_writable=False,
        conditional="class",
        conv_drop_units=0.0,
        conv_kernel_size=0,
        drop_att=0.0,
        drop_path=0.1,  # 0.1
        drop_sc=0.0,
        drop_units=0.1,  # 0.1
        flip_rate=0.0,
        guidance=0.0,
        infer_iterations=100,
        infer_schedule="cosine",
        latent_dim=512,
        latent_mlp_ratio=4,
        latent_num_heads=16,
        latent_pos_encoding="learned",
        latent_slots=128,
        loss_type="eps",
        name="image_diffusion_model",
        normalize_noisy_input=False,
        num_layers="2,2,2",
        patch_size=2,
        pos_encoding="sin_cos",
        pred_type="eps",
        pretrained_ckpt="",
        rw_num_heads=8,
        sampler_name="ddpm",
        self_cond="latent",
        self_cond_by_masking=True,
        self_cond_rate=0.9,
        tape_dim=256,
        tape_mlp_ratio=2,
        tape_pos_encoding="learned",
        td=0.0,
        time_on_latent=True,
        time_scaling=1000,
        train_schedule="sigmoid@-3,3,0.9",
        use_cls_token=False,
        x0_clip="auto",
        xattn_enc_ln=False,
    ),
)
config = ConfigDict(config)

diffusion_model = RinDiffusionModel(config)


# pass some data to build the model
x = tf.zeros((1, 32, 32, 3))
t = tf.fill((1,), 0.0)
classes = tf.zeros((1, 10))
diffusion_model.denoiser(x, t, classes)


# load weights
weight_pre = np.load("rin_pytorch/rin_cifar10_pretrained_weights.npy", allow_pickle=True).item()
weights = list(weight_pre.values())

# iterate over weights
for i, w in enumerate(diffusion_model.denoiser.weights):
    w.assign(weights[i])


# pass some specific data
rng = np.random.default_rng(42)
x = tf.convert_to_tensor(rng.random((1, 3, 32, 32), dtype=np.float32).transpose(0, 2, 3, 1))
t = tf.convert_to_tensor(np.full((1,), 0.5, dtype=np.float32))
cond = tf.convert_to_tensor(rng.random((1, 10), dtype=np.float32))

# track("in", x=x, t=t, cond=cond)
# output, latent_prev, tape_prev = diffusion_model.denoiser(x, t, cond, training=False)
# track("out", output=output, latent_prev=latent_prev, tape_prev=tape_prev)

# output, latent_prev, tape_prev = diffusion_model.denoiser([x, latent_prev, tape_prev], t, cond, training=False)
# track("out-1", output=output, latent_prev=latent_prev, tape_prev=tape_prev)

with tf.GradientTape() as tape:
    tape.watch(diffusion_model.denoiser.trainable_weights)
    output, _, _ = diffusion_model.denoiser(x, t, cond, training=True)
    loss = tf.reduce_mean(tf.square(output - x))

    grads = tape.gradient(loss, diffusion_model.denoiser.trainable_weights)


p(loss=loss.numpy().item())

for g in grads:
    track("g", g=g)
