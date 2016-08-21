# tensorboard --logdir=/Users/terry/PycharmProjects/untitled/train/
import re
import numpy as np
import tensorflow as tf

# Global variables.
batch_size = 100
num_epochs=10
lr = .00001


class Ladder(object):
    def __init__(self, trainData, trainLabels, valData, valLabels):
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.valData = valData
        self.valLabels = valLabels
        self.numTrain = trainData.shape[0]
        self.numVal = valData.shape[0]
        self.numClasses = trainLabels.shape[1]

    def get_classes(self):
        return self.numClasses

class DenseLadder(Ladder):
    def __init__(self, trainData, trainLabels, valData, valLabels, numLayers, numHidden):
        Ladder.__init__(self, trainData, trainLabels, valData, valLabels)
        self.numLayers = numLayers
        self.numHidden = numHidden
        if self.trainData.ndim > 2:
            self.trainData = np.reshape(self.trainData, [self.numTrain, -1])
        if self.valData.ndim > 2:
            self.valData = np.reshape(self.valData, [self.numVal, -1])
        self.inputSize = self.trainData.shape[1]
        self.z = []
        self.h = []
        self.z_noise = []
        self.h_noise = []

    def feed_forward(self, x, y):
        with tf.variable_scope("feedforward"):
            print "forward"
            for i in range(self.numLayers + 1):
                print i
                string = 'weight' + str(i)
                if i == 0:
                    W = tf.get_variable(name=string, shape=[self.inputSize, self.numHidden[i]], initializer=tf.contrib.layers.xavier_initializer())
                elif i == self.numLayers:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i-1], self.numClasses], initializer=tf.contrib.layers.xavier_initializer())
                else:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i-1], self.numHidden[i]], initializer=tf.contrib.layers.xavier_initializer())

                string = 'bias' + str(i)
                if i == self.numLayers:
                    B = tf.get_variable(name=string, shape=[self.numClasses], initializer=tf.constant_initializer(0.1))
                else:
                    B = tf.get_variable(name=string, shape=[self.numHidden[i]], initializer=tf.constant_initializer(0.1))

                if i == 0:
                    self.z.append(tf.nn.batch_normalization(tf.matmul(x, W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h.append(tf.nn.relu(tf.add(self.z[i], B)))
                elif i == self.numLayers:
                    self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h.append(tf.nn.sigmoid(tf.add(self.z[i], B)))
                else:
                    self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h.append(tf.nn.relu(tf.add(self.z[i], B)))

    def encoder_decoder(self, x, y, noiseMean, noiseStdDev):
        with tf.variable_scope("feedforward", reuse=True):
            print "encode"
            for i in range(self.numLayers + 1):
                print i
                string = 'weight' + str(i)
                if i == 0:
                    W = tf.get_variable(name=string, shape=[self.inputSize, self.numHidden[i]], initializer=tf.contrib.layers.xavier_initializer())
                elif i == self.numLayers:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i - 1], self.numClasses], initializer=tf.contrib.layers.xavier_initializer())
                else:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i - 1], self.numHidden[i]], initializer=tf.contrib.layers.xavier_initializer())

                string = 'bias' + str(i)
                if i == self.numLayers:
                    B = tf.get_variable(name=string, shape=[self.numClasses], initializer=tf.constant_initializer(0.1))
                else:
                    B = tf.get_variable(name=string, shape=[self.numHidden[i]], initializer=tf.constant_initializer(0.1))

                if i == 0:
                    n = tf.random_normal(shape=tf.shape(x), mean=noiseMean, stddev=noiseStdDev)
                    x = tf.add(x, n)
                    self.z_noise.append(tf.nn.batch_normalization(tf.matmul(x, W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    n = tf.random_normal(shape=tf.shape(B), mean=noiseMean, stddev=noiseStdDev)
                    self.h_noise.append(tf.nn.relu(tf.add(self.z[i], tf.add(B, n))))
                elif i == self.numLayers:
                    n = tf.random_normal(shape=tf.shape(B), mean=noiseMean, stddev=noiseStdDev)
                    self.z_noise.append(tf.nn.batch_normalization(tf.matmul(self.h[i - 1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h_noise.append(tf.nn.sigmoid(tf.add(self.z[i], tf.add(B, n))))
                else:
                    n = tf.random_normal(shape=tf.shape(B), mean=noiseMean, stddev=noiseStdDev)
                    self.z_noise.append(tf.nn.batch_normalization(tf.matmul(self.h[i - 1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h_noise.append(tf.nn.relu(tf.add(self.z[i], tf.add(B, n))))

        with tf.variable_scope("decoder"):
            print "decode"
            self.z_recon = [None]*len(self.z_noise)
            self.h_recon = [None]*len(self.h_noise)
            self.diff = tf.constant(0.0,dtype=np.float32)
            for i in range(self.numLayers, -1, -1):
                print i
                if i == 0:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numHidden[i])), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(tf.matmul(self.z_recon[i+1], V), mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numHidden[i], self.inputSize], initializer=tf.contrib.layers.xavier_initializer())
                elif i == self.numLayers:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numClasses)), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numClasses], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numClasses], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numClasses], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(self.h[i], mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numClasses, self.numHidden[i-1]], initializer=tf.contrib.layers.xavier_initializer())
                else:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numHidden[i])), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numHidden[i]], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(tf.matmul(self.z_recon[i+1], V), mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numHidden[i], self.numHidden[i-1]], initializer=tf.contrib.layers.xavier_initializer())


                e = tf.transpose(tf.pack([I, self.z_noise[i], mu, tf.mul(self.z_noise[i], mu)]), [1, 0, 2])
                part1 = tf.reduce_sum(tf.mul(a, e), 1)
                part2 = tf.mul(b, tf.nn.sigmoid(tf.reduce_sum(tf.mul(c, e), 1)))
                self.z_recon[i] = tf.add(part1, part2)
                string = 'lambda' + str(i)
                lam = tf.get_variable(string, shape=[1], initializer=tf.constant_initializer(0.5))
                self.diff = tf.add(self.diff,tf.div(tf.mul(lam, tf.reduce_sum(tf.squared_difference(self.z[i], self.z_recon[i]))), tf.to_float(tf.shape(self.z[i]))[1]))  # lambda * (sum||(z1-z1_n)||^2 ) / (layerWidth * numSamples)

                #print tf.Tensor.get_shape(tf.mul(self.z_recon[i], tf.ones(tf.shape(self.z_recon[i]))))

        lossSupervised = -tf.reduce_mean(y * tf.log(self.h[self.numLayers]) + (1 - y) * tf.log(1 - self.h[self.numLayers]))
        lossUnsupervised = tf.div(self.diff, tf.to_float(tf.shape(self.h[self.numLayers])[0]))
        self.loss = tf.add(lossSupervised, lossUnsupervised)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def get_loss(self):
        return self.loss

    def get_train_step(self):
        return self.train_step

    def get_size(self):
        return self.inputSize

def main(argv=None):
    path = "/Users/terry/Desktop/RESEARCH/FANTOM_DATA/"
    train_data_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_data_TRAIN_1.npy"
    train_labels_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_labels_TRAIN_1.npy"
    val_data_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_data_VAL_1.npy"
    val_labels_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_labels_VAL_1.npy"
    test_data_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_data_TEST.npy"
    test_labels_filename = path + "human_permissive_enhancers_phase_1_and_2_FIVE_FOLD_CV_1000bp_labels_TEST.npy"

    summary_path = path + "EVENTS/"
    model_path = path + "bestModel.txt"
    predictions_save_path = path + "predictions.txt"

    train_data = (np.load(train_data_filename)).astype(np.float32)
    train_labels = (np.load(train_labels_filename)).astype(np.int32)
    val_data = (np.load(val_data_filename)).astype(np.float32)
    val_labels = (np.load(val_labels_filename)).astype(np.int32)
    test_data = (np.load(test_data_filename)).astype(np.float32)
    test_labels = (np.load(test_labels_filename)).astype(np.int32)

    test_data = np.reshape(test_data, [test_data.shape[0], -1])

    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    nnTypes = ['dense','convolution','recurrent']
    nnType = nnTypes[0]
    if nnType.startswith('d'):
        numLayers = 4
        numHidden = [200, 250, 300, 350]

        L = DenseLadder(train_data, train_labels, val_data, val_labels, numLayers, numHidden)

        x = tf.placeholder(tf.float32, shape=[batch_size, L.get_size()])
        y = tf.placeholder(tf.float32, shape=[batch_size, L.get_classes()])
        L.feed_forward(x, y)
        L.encoder_decoder(x, y, noiseMean=0, noiseStdDev=0.2)
        loss = L.get_loss()
        train_step = L.get_train_step()

        best_val_loss = 10000
        with tf.Session() as s:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()

            for epoch in xrange(num_epochs):
                for batch in xrange(0, L.numTrain, batch_size):
                    if batch + batch_size > L.numTrain:
                        end = L.numTrain
                        break
                    else:
                        end = batch + batch_size
                    batch_data = L.trainData[batch:end, :]
                    batch_labels = L.trainLabels[batch:end]
                    _, cost = s.run([train_step, loss], feed_dict={x: batch_data, y: batch_labels, dropout_keep_prob: 1.0})
                    #print cost
                index = 0
                costT = 0
                for batch in xrange(0, L.numVal, batch_size):
                    if batch + batch_size > L.numVal:
                        end = L.numVal
                        break
                    else:
                        end = batch + batch_size
                    batch_data = L.valData[batch:end, :]
                    batch_labels = L.valLabels[batch:end]
                    cost = s.run(loss, feed_dict={x: batch_data, y: batch_labels, dropout_keep_prob: 1.0})
                    costT += cost[0]
                    index += 1
                costT /= index
                print
                print "val costs =", costT
                if costT < best_val_loss:
                    saver.save(s, model_path)
                    best_val_loss = costT

            print
            print "Testing..."
            costT = 0
            index = 0
            saver.restore(s, model_path)
            predictions = []
            costT = 0
            for batch in xrange(0, test_data.shape[0], batch_size):
                if batch + batch_size > test_data.shape[0]:
                    end = test_data.shape[0]
                    break
                else:
                    end = batch + batch_size
                batch_data = test_data[batch:end, :]
                batch_labels = test_labels[batch:end]
                cost = s.run([loss], feed_dict={x: batch_data, y: batch_labels, dropout_keep_prob: 1.0})
                pred = L.h[L.numLayers].eval(feed_dict={x: batch_data, y: batch_labels, dropout_keep_prob: 1.0})
                print pred[0]
                predictions.append(pred[0])
                costT += cost[0]
                index += 1
            costT /= index
            predictions = np.asarray(predictions)
            np.savetxt(predictions_save_path, predictions, fmt="%1.4f")
            print "test cost=", costT[0]


#############################################################################################################

if __name__ == '__main__':
    tf.app.run()

