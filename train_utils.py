import tensorflow as tf

@tf.function
def train_step(model, x, optimizer, kl_factor):
    with tf.GradientTape() as model_tape:
        loss = compute_loss(model, x, kl_factor = kl_factor)
    gradients = model_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
def compute_loss(model, x, kl_factor = 1e-3):
    mean, logvar = model.encode(x)
    outputs = model.reparametrize(mean, logvar)
    outputs = model.decode(outputs)
    
    kl_loss = 0.5 * tf.reduce_sum(tf.exp(logvar) - 1 - logvar + tf.square(mean), axis = range(len(logvar.shape)))
    rc_loss = 0.5 * tf.reduce_mean(tf.losses.mean_squared_error(x, outputs), axis = range(len(x.shape)))
    return rc_loss + (kl_factor * kl_loss)

