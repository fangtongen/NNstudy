import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_jaffe(img_rows=28, img_cols=28, fold=1):
    data_path = './jaffe'
    data = np.zeros((214, img_rows, img_cols, 1))
    index = 0
    for file_name in os.listdir(data_path):
        img = Image.open(data_path + '/' + file_name)

        resize_img = img.resize((img_rows, img_cols))
        # plt.imshow(resize_img)
        # plt.show()
        data[index, :, :, :] = np.expand_dims(resize_img, 2)

    print(data.shape)
    return data


if __name__ == '__main__':
    load_jaffe()