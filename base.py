import tensorflow as tf
from tensorflow.keras import layers

class VAE(layers.Layer):
    def __init__(self, latent_dims = 20, input_shape = [28, 28, 1]):
        super().__init__()
        self.latent_dims = latent_dims
        self.input_shape = input_shape
        
        self.encoder = self.generate_encoder()
        self.decoder = self.generate_decoder()
        
    def generate_encoder(self):
        model = Sequential([layers.Input(shape = self.input_shape),
                            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
                            layers.Conv2D(filters = 3, kernel_size = 3, padding = 'same', strides = 2),
                            layers.LeakyReLU(),
                            layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', strides = 2),
                            layers.BatchNormalization(),
                            layers.LeakyReLU(),
                            layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', strides = 2),
                            layers.LeakyReLU(),
                            layers.Flatten(),
                            layers.Dense(units = self.latent_dims*2)
                           ])
        
        return model
    
    def generate_decoder(self):
        model = Sequential()
        
        model.add(layers.Input(shape = [self.latent_dims,]))
        model.add(layers.Dense(units = 7*7*16, activation = 'relu'))
        model.add(layers.Reshape(target_shape = [7, 7, 16]))
        model.add(layers.Conv2DTranspose(filters = 8, kernel_size = 3, strides = 2, activation = 'relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2DTranspose(filters = 3, kernel_size = 3, strides = 2, activation = 'relu'))
        model.add(layers.MaxPooling2D(pool_size = 3, strides = 2))
        model.add(layers.Conv2D(filters = 1, kernel_size = 2, strides = 1, padding = 'valid', activation = 'relu'))
        model.add(layers.UpSampling2D(size = (2, 2)))
        
        return model
    
    def encode(self, x, train = True):
        mean, logvar = tf.split(self.encoder(x, training = train), num_or_size_splits = 2, axis = 1)
        return mean, logvar
    
    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape = mean.shape)
        z = (eps * tf.exp(logvar * .5)) + mean
        
        return z
    
    def decode(self, z, training = True):
        logits = self.decoder(z, training = training)
        
        return logits
    
    def sample(self, eps = None, train = True):
        if eps is None:
            eps = tf.random.normal(shape = [100, self.latent_dims])
        
        return self.decode(eps, training = train)