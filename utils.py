import numpy as np
import scipy.misc as misc
from PIL import Image
import os



def read_crop_data(path, batch_size, shape, factor):
    h = shape[0]
    w = shape[1]
    c = shape[2]
    filenames = os.listdir(path)
    rand_selects = np.random.randint(0, filenames.__len__(), [batch_size])
    batch = np.zeros([batch_size, h, w, c])
    downsampled = np.zeros([batch_size, h//factor, w//factor, c])
    for idx, select in enumerate(rand_selects):
        try:
            img = np.array(Image.open(path + filenames[select]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h // factor, w // factor])
        except:
            img = np.array(Image.open(path + filenames[0]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h//factor, w//factor])
    return batch, downsampled

def random_crop(img, size):
    h = img.shape[0]
    w = img.shape[1]
    start_x = np.random.randint(0, h - size + 1)
    start_y = np.random.randint(0, w - size + 1)
    return img[start_x:start_x + size, start_y:start_y + size, :]

