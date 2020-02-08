import os
import argparse
import datetime
import tensorflow as tf
from model.model import FaceEmbedder
from model.loss import *
from utils.hparams import HParam
from utils.dataloader import create_dataloader

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config/config.yaml', help='configuration file')
args = parser.parse_args()

# Set hyper parameters
hp = HParam(args.config)

# Create summary writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join(hp.dir.log_dir, current_time)
writer = tf.summary.create_file_writer(log_dir)

# Create data loaders
train_data_loader = create_dataloader(hp, train=True)
test_data_loader = create_dataloader(hp, train=False)

# Build model
optimizer = tf.optimizers.Adam(learning_rate=hp.train.learning_rate)
model = FaceEmbedder(hp, optimizer=optimizer)

#
N = hp.train.people_num
M = hp.train.img_num
for epoch, batch_x in enumerate(train_data_loader.repeat().batch(N)):
    batch_x = tf.reshape(batch_x, [N * M, hp.data.size, hp.data.size, hp.data.channel_num])

    train_loss = model.train_on_batch(batch_x)

    print("Epoch : {}, Train Loss : {}".format(epoch, train_loss))
    #with writer.as_default():
    #    tf.summary.scalar('train loss', train_loss, step=epoch)

    if epoch % 10 == 0:
        test_loss_list = []
        test_eer_list = []
        for i, batch_x in enumerate(test_data_loader.batch(N)):
            if i == 10: break
            batch_x = tf.reshape(batch_x, [N * M, hp.data.size, hp.data.size, hp.data.channel_num])

            test_loss, test_eer = model.evaluate(batch_x, M=M, N=N)
            test_loss_list.append(test_loss)
            test_eer_list.append(test_eer)

        print("Epoch : {}, Test Loss : {}, Test EER : {}".format(epoch, np.mean(test_loss_list), np.mean(test_eer_list)))
