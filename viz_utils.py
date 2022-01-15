import tensorflow as tf
from matplotlib import pyplot as plt


def generate_and_save_images(model, sample, epoch, style = 'normal', title = None, save_image = False):
    
    if isinstance(sample, tf.Tensor):
        assert len(sample.shape) in [2, 4], f'Input tensor should be 2-D or 4-D. Shape: {sample.shape}'
        
        ### Generate images
        if len(sample.shape) == 4:
            m, l = model.encode(sample)
            z = model.reparametrize(m, l)
        else:
            z = sample

        z = model.decode(z)
        
    else:
        raise TypeError(f'Input tensor of type {type(sample)}. Should be of type tf.Tensor')
    
    if style != 'normal':
        jtplot.style(style)
    
    else:
        import seaborn as sns
        sns.set()
        
    ### Display and/or save generated images
    plt.figure(figsize = (15, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.squeeze(z[i]), cmap = 'gray')
        
    if save_image:
        if title:
            plt.savefig(f'images/{title}.png', dpi = 300)
        else:
            plt.savefig(f'images/Image generated at Epoch {epoch+1:03d}.png', dpi = 300)
    
    if title:
        plt.title(f'Images generated at Epoch {epoch+1:03d}', fontsize = 50, pad = 20)
    
    plt.show(); plt.close('all')
    
    return None