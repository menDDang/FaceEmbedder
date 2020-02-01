import tensorflow as tf
import numpy as np
import cv2
import os
import random
import itertools

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img /= 255
    return img

class DataGenerator:
    def __init__(self, hp, id_list):
        self.hp = hp
        self.id_list = id_list

    def gen(self):
        for n in itertools.count(1):
            x = list()
            _y = list()
            for i in range(self.hp.train.people_num):
                id = random.choice(self.id_list)
                for j in range(self.hp.train.img_num):
                    path = os.path.join(self.hp.base_dir, self.hp.data.data_path, id)
                    filename = os.path.join(path, random.choice(os.listdir(path)))
                    img = load_image(filename)
                    x.append(img)
                    _y.append([i])

            # shuffle
            x = np.array(x, dtype=np.float32)
            _y = np.array(_y, dtype=np.int32)
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)
            x = x[idx]
            _y = _y[idx]

            # create label
            y = np.zeros(shape=[x.shape[0], x.shape[0]], dtype=np.float32)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    if _y[i] == _y[j]:
                        y[i,j] = 1
                    else:
                        y[i,j] = 0

            yield x, y


def create_dataloader(hp, id_list):
    data_num = hp.train.people_num * hp.train.img_num
    x_shape = [data_num, hp.data.height, hp.data.width, 3]
    y_shape = [data_num, data_num]
    generator = DataGenerator(hp, id_list)
    dataloader = tf.data.Dataset.from_generator(
        generator=generator.gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(x_shape), tf.TensorShape(y_shape))
    )

    return dataloader.repeat().batch(1)
