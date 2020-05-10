import datetime
import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

conv_layers = [
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=4, padding='same'),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=4, padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

]

fc_net = Sequential([
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])



def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x,y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000,) (10000, 32, 32, 3) (10000,)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
# sample: (128, 32, 32, 3) (128,) tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


def main():

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    conv_net = Sequential(conv_layers)

    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 128])
    optimizer = optimizers.Adam(lr=1e-4)

    # [1, 2] + [3, 4] => [1, 2, 3, 4] 需要同时对卷积和全连接求导 所以要两个
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):

        for step, (x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # print(out.shape)
                # flatten, => [b, 512]
                out = tf.reshape(out, [-1, 128])

                # 全连接
                # [b, 512] => [b, 100]
                logits = fc_net(out)

                # print(logits.shape)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=10)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

            with summary_writer.as_default():
                tf.summary.scalar("loss", float(loss), step=epoch)


        total_num = 0
        total_correct = 0
        for x,y in test_db:


            out = conv_net(x)
            out = tf.reshape(out, [-1, 128])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

        with summary_writer.as_default():
            tf.summary.scalar("acc", acc, step=epoch)


if __name__ == '__main__':
    main()
