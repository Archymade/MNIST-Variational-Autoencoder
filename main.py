import argparse
import os

import tensorflow as tf
from tensorflow.keras import optimizers

from data_ops import read_dataset
from base import VAE

from train_utils import train_step
from train_utils import compute_loss

from viz_utils import generate_and_save_images


def configure_args():
    parser = argparse.ArgumentParser(description="Provide relevant CLI arguments.")

    parser.add_argument('--epochs', type=int, default=20, help='Number of training rounds')

    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSprop'],
                        help='Optimizer algorithm for autoencoder training')

    parser.add_argument('--input_shape', default=[28, 28, 1], type=list,
                        help='Expected input image shape: [height, width, channels]')

    parser.add_argument('--latent_dims', type=int, default=100,
                        help='Dimensionality of latent vector space')

    parser.add_argument('--style', default='gruvboxd', type=str,
                        choices=['gruvboxd', 'solarizedd', 'solarizedl', 'onedork', 'oceans16', 'normal'],
                        help='Image visualization style')

    parser.add_argument('--save', default=True, choices=[True, False], type=bool,
                        help='Save sample images')

    parser.add_argument('--batch_size', default=32, type=int, help='Size for data batching')

    parser.add_argument('--data_dir', type=str, help='Local location of dataset',
                        default=os.path.join(os.getcwd().replace('variational autoencoder', ''), 'trainingSet',
                                             'trainingSet'))

    parser.add_argument('--lr', default=3e-4, type=float, help='Convergence rate')

    parser.add_argument('--beta1', default=0.5, type=float, help='First moment')

    parser.add_argument('--beta2', default=0.999, type=float, help='Second moment')

    parser.add_argument('--klf', type=int, default=10, help='KL-Loss factor')

    return parser


def check_args(args):
    ''' Validate CLI arguments. '''

    ### Epochs
    assert (args.epochs >= 1) & (type(args.epochs) == int), 'Epochs must be an object of type Int not less than 1.'

    ### Betas
    assert (type(args.beta1) == float) & (0 <= args.beta1 <= 1), 'Beta 1 must be in [0, 1).'

    assert (type(args.beta2) == float) & (0 <= args.beta2 <= 1), 'Beta 2 must be in [0, 1).'

    ### Learning rate
    assert (type(args.lr) == float), 'Learning rate must be in of type Float.'

    ### Input shape
    assert len(args.input_shape) == 3, 'Input shape must contain: [height, width, channels].'

    assert len(
        [*filter(lambda x: type(x) != int, args.input_shape)]) == 0, 'Input shape must comprise objects of type Int.'

    ### Latent space
    assert (type(args.latent_dims) == int) & (
            args.latent_dims >= 1), 'Latent size must be of type Int, and greater than or equal 1.'

    ### Batch size
    assert (type(args.batch_size) == int) & (
                args.batch_size >= 1), 'Batch size must be of type Int, and greater than or equal 1.'

    assert not (args.batch_size % 2), 'Batch size must be multiple of 2.'

    ### If all's well
    return args


def main():
    args = configure_args().parse_args()
    args = check_args(args)

    print('>>> Importing dataset...')
    train_images = read_dataset(args.data_dir)
    print('>>> Dataset imported!')
    print()
    cardinality = len(train_images)

    train_x = train_images.take(cardinality - 1)
    test_x = train_images.skip(1)

    sample = next(iter(test_x))

    print('>>> Invoking odel object...')
    model = VAE(latent_dims=args.latent_dims)
    print('>>> Model object obtained!')

    if args.optimizer == 'Adam':
        optimizer = optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2)

    elif args.optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(learning_rate=args.lr)

    else:
        optimizer = optimizers.SGD(learning_rate=args.lr)

    ### Training loop
    for epoch in range(args.epochs):
        for batch in train_x:
            train_step(model, batch, optimizer, args.klf)

        ### Image display
        choice = np.random.randint(low=0, high=2)
        if choice:
            generate_and_save_images(model, sample[0], style=args.style, epoch=epoch, save_image=args.save)
        else:
            z = tf.random.normal(shape=[args.batch_size, model.latent_dims])
            generate_and_save_images(model, z, style=args.style, epoch=epoch, save_image=args.save)
            
        if args.save:
            model.encoder.save_weights('artefacts/encoder_weights.h5')
            model.decoder.save_weights('artefacts/decoder_weights.h5')


if __name__ == '__main__':
    main()
