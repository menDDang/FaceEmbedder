import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class FaceEmbedder(tf.keras.Model):
    def __init__(self):
        super(FaceEmbedder, self).__init__()
        model = tf.keras.Sequential([
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
            Dense(units=512, activation='sigmoid')
        ])

    def call(self, x):
        return self.model(x)

    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            embeddings = self.model(x)
            similarities = get_similarity_matrix(embeddings)
            loss_value = loss(similarities, y)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value

    def fit(self, train_dataloader, test_dataloader, hp):
        train_epoch_num = hp.train.train_epoch_num
        learning_rate = hp.train.learning_rate

        for epoch in range(1, train_epoch_num+1):
            for x, y in train_dataloader.take(1):
                x, y = tf.squeeze(x, 0), tf.squeeze(y, 0)

                train_loss = self.train_on_batch(x, y)
                print(train_loss)

            if epoch % hp.log.summary_interval == 0:
                pass

            if epoch % hp.train.save_interbal == 0:
                self.save_weights()
                