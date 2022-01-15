from tensorflow.keras import utils

def read_dataset(DIR, image_size = (28, 28), batch_size = 16):
    train_images = utils.image_dataset_from_directory(DIR, image_size = image_size, color_mode = 'grayscale',
                                                      batch_size = batch_size, shuffle = True, labels = None)
    return train_images