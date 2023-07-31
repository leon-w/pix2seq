import tensorflow as tf

from architectures.tape import ImageTapeDenoiser
from debug_utils import p, plot_dist

rin = ImageTapeDenoiser(
    num_layers="2,2,2",
    latent_slots=128,
    latent_dim=512,
    latent_mlp_ratio=4,
    latent_num_heads=16,
    tape_dim=256,
    tape_mlp_ratio=2,
    rw_num_heads=8,
    image_height=32,
    image_width=32,
    image_channels=3,
    patch_size=2,
    latent_pos_encoding="learned",
    tape_pos_encoding="learned",
    drop_path=0.1,
    drop_units=0.1,
    drop_att=0.0,
    time_scaling=1e4,
    self_cond="latent",
    time_on_latent=True,
    cond_on_latent_n=1,
    cond_tape_writable=False,
    cond_dim=0,
    cond_proj=True,
    cond_decoupled_read=False,
    xattn_enc_ln=False,
)

bs = 8
x = tf.random.normal((bs, 32, 32, 3))
t = tf.fill((bs,), 0.5)

classes = tf.one_hot(tf.random.uniform((bs,), maxval=10, dtype=tf.int32), depth=10)


# t_emb, _ = rin.initialize_cond(t, None, training=True)
# t_emb = t_emb.numpy()

# print(t_emb)


# pass some data to build the model
output, latent_prev, tape_prev = rin(x, t, classes)


# p(output_std=tf.math.reduce_std(output), output_mean=tf.math.reduce_mean(output))

p(x=x, t=t, classes=classes, output=output)
# plot_dist(x=x, output=output, latent_prev=latent_prev, tape_prev=tape_prev)

for w in rin.weights:
    p(w.name, w.shape)
