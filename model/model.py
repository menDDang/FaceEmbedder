import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from model.loss import *
import os


class FaceEmbedder(tf.keras.Model):
    def __init__(self, hp, optimizer=tf.optimizers.Adam):
        super(FaceEmbedder, self).__init__()
        self.hp = hp
        self.model = tf.keras.Sequential([
            Conv2D(filters=8, kernel_size=[2,2], strides=[1,1], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2,2]),
            Conv2D(filters=16, kernel_size=[2,2], strides=[1,1], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2,2]),
            Conv2D(filters=32, kernel_size=[2, 2], strides=[1, 1], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2]),
            Conv2D(filters=64, kernel_size=[2, 2], strides=[1, 1], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2]),
            Flatten(),
            Dense(units=2048, activation='relu'),
            Dense(units=1024, activation='relu'),
            Dense(units=hp.model.emb_dims, activation='sigmoid')
        ])

        self.optimizer = optimizer

    def call(self, x):
        return self.model(x)

    def train_on_batch(self, batch_x):
        if self.optimizer is None:
            raise TypeError("Optimizer is Nonetype! Compile model first!")

        with tf.GradientTape() as tape:
            embeddings = self.model(batch_x)
            similarities = get_similarity_mat(embeddings, N=self.hp.train.people_num, M=self.hp.train.img_num, center=None)
            loss_value = calculate_loss(similarities)
            grads = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value

    def evaluate(self, x, center=None, N=7, M=3):
        # Embed images
        embeddings = self.model(x)

        # Get similarity matrix
        similarities = get_similarity_mat(embeddings, N=N, M=M, center=center)

        # Calculate loss
        loss_value = calculate_loss(similarities, N=N, M=M)

        # Calculate EER
        similarities = tf.reshape(similarities, [N, M, -1])
        diff = 1; EER = 0; # EER_threshold = 0; EER_FAR = 0; EER_FRR = 0;

        for threshold in [0.01 * i + 0.5 for i in range(50)]:
            # find points where score is higher than threshold
            S_thres = similarities > threshold

            # False Acception Ratio : false acceptance / mis-matched population ( enroll face != verification face )
            FAR = np.sum([np.sum(S_thres[i]) - np.sum(S_thres[i, :, i]) for i in range(N)]) / ((N-1) * N * M)

            # False Reject Ratio : false reject / matched population (enroll face == verification face)
            FRR = np.sum([M - np.sum(S_thres[i, :, i]) for i in range(N)]) / (M * N)


            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                EER = (FAR + FRR) / 2
                #EER_threshold = threshold
                #EER_FAR = FAR
                #EER_FAR = FRR

        return loss_value, EER
                