# tensorboard --logdir=/Users/terry/PycharmProjects/untitled/train/
import re
import numpy as np
import tensorflow as tf

# Global variables.
batch_size = 500
num_epochs=10
lr = .0001
TOWER_NAME = 'tower'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations',x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _loss_summary(x):
    tf.scalar_summary(x.op.name, x)

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

class ConvLadder(Ladder):
    def __init__(self, trainData, trainLabels, valData, valLabels, numConv, numFilters, lambdas):
        Ladder.__init__(self, trainData, trainLabels, valData, valLabels)
        self.numConv = numConv
        self.numFilters = numFilters
        self.lambdas = lambdas
        if len(self.lambdas) != self.numLayers + 2:
            print "need lambda for input, output and every convolution layer"
            exit()
        self.inputSize = self.trainData.shape[1]
        self.z = []
        self.h = []
        self.z_noise = []
        self.h_noise = []

        # kernal_shape1 = [4, 17, 1, 300]  # height, width, num_input channels, num_output channels(kernals)
        # kernal_shape2 = [1, 7, kernal_shape1[3], 100]  # height, width, num_input channels, num_output channels(kernals)
        # kernal_stride = [1, 1, 1, 1]
        # pool_shape1 = [1, 1, 1, 1]
        # pool_shape2 = [1, 1, 1, 1]
    # with tf.variable_scope('conv1') as scope:
    #    # kernal = _variable_with_weight_decay('weights', shape = kernal_shape1, stddev = 1e-4, wd=0.0)
    #     kernal = _variable_on_cpu('weights', shape=kernal_shape1, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #     conv1 = tf.nn.conv2d(x,kernal,strides=kernal_stride, padding='Valid')
    #     biases = _variable_on_cpu('biases',kernal_shape1[3], tf.constant_initializer(0.1))
    #     conv1 = tf.nn.bias_add(conv1, biases)
    #     #conv1 = tf.nn.batch_normalization(conv1,mean=0,variance=1,offset=None,scale=None,variance_epsilon=1e-9)
    #     conv1 = tf.nn.relu(conv1, name=scope.name)
    #     _activation_summary(kernal)
    #     _activation_summary(biases)
    #     _activation_summary(conv1)

    # def feed_forward(self, x, y):
    #     with tf.variable_scope("feedforward"):
    #         print "forward"
    #         for i in range(self.numLayers + 2):
    #             print i
    #             if i == 0:
    #                 self.z.append(x)
    #                 self.h.append(x)
    #             elif i == self.numLayers + 1:
    #                 self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
    #                 self.h.append(tf.nn.sigmoid(tf.add(self.z[i], B)))
    #             else:
    #                 self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
    #                 self.h.append(tf.nn.relu(tf.add(self.z[i], B)))
    #
    #     self.loss_supervised = -tf.reduce_mean(y * tf.log(self.h[self.numLayers + 1]) + (1-y) * tf.log(1-self.h[self.numLayers + 1]))
    #     self.train_step_supervised = tf.train.AdamOptimizer(lr).minimize(self.loss_supervised)



class DenseLadder(Ladder):
    def __init__(self, trainData, trainLabels, valData, valLabels, numHidden, lambdas):
        Ladder.__init__(self, trainData, trainLabels, valData, valLabels)
        self.numLayers = len(numHidden)
        self.numHidden = numHidden
        self.lambdas = lambdas
        if len(self.lambdas) != self.numLayers + 2:
            print "need lambda for input, output and every hidden layer"
            exit()
        if self.trainData.ndim > 2:
            print "reshaping training data"
            self.trainData = np.reshape(self.trainData, [self.numTrain, -1])
        if self.valData.ndim > 2:
            print "reshaping validation data"
            self.valData = np.reshape(self.valData, [self.numVal, -1])
        self.inputSize = self.trainData.shape[1]
        self.z = []
        self.h = []
        self.z_noise = []
        self.h_noise = []

    def feed_forward(self, x, y):
        with tf.variable_scope("feedforward"):
            print "forward"
            for i in range(self.numLayers + 2):
                print i
                string = 'weight' + str(i)
                if i == 0:
                    pass
                elif i == 1:
                    W = tf.get_variable(name=string, shape=[self.inputSize, self.numHidden[i-1]], initializer=tf.contrib.layers.xavier_initializer())
                 #   _activation_summary(W)
                elif i == self.numLayers + 1:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i-2], self.numClasses], initializer=tf.contrib.layers.xavier_initializer())
                  #  _activation_summary(W)
                elif i < self.numLayers + 1:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i-2], self.numHidden[i-1]], initializer=tf.contrib.layers.xavier_initializer())
                 #   _activation_summary(W)

                string = 'bias' + str(i)
                if i == 0:
                    pass
                elif i == self.numLayers + 1:
                    B = tf.get_variable(name=string, shape=[self.numClasses], initializer=tf.constant_initializer(0.1))
                  #  _activation_summary(B)
                elif i < self.numLayers + 1:
                    B = tf.get_variable(name=string, shape=[self.numHidden[i-1]], initializer=tf.constant_initializer(0.1))
                  #  _activation_summary(B)

                if i == 0:
                    self.z.append(x)
                    self.h.append(x)
                elif i == self.numLayers + 1:
                    self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h.append(tf.nn.sigmoid(tf.add(self.z[i], B)))
                else:
                    self.z.append(tf.nn.batch_normalization(tf.matmul(self.h[i-1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None))
                    self.h.append(tf.nn.relu(tf.add(self.z[i], B)))

                #print "h", tf.Tensor.get_shape(tf.mul(self.h[i], tf.ones(tf.shape(self.h[i]))))
        self.loss_supervised = -tf.reduce_mean(y * tf.log(self.h[self.numLayers + 1]) + (1-y) * tf.log(1-self.h[self.numLayers + 1]))
        #_loss_summary(self.loss_supervised)
        self.train_step_supervised = tf.train.AdamOptimizer(lr).minimize(self.loss_supervised)

    def encoder(self, x, y, noiseMean, noiseStdDev):
        with tf.variable_scope("feedforward", reuse=True):
            print "encode"
            for i in range(self.numLayers + 2):
                print i
                string = 'weight' + str(i)
                if i == 0:
                    pass
                elif i == 1:
                    W = tf.get_variable(name=string, shape=[self.inputSize, self.numHidden[i - 1]], initializer=tf.contrib.layers.xavier_initializer())
                elif i == self.numLayers + 1:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i - 2], self.numClasses], initializer=tf.contrib.layers.xavier_initializer())
                elif i < self.numLayers + 1:
                    W = tf.get_variable(name=string, shape=[self.numHidden[i - 2], self.numHidden[i - 1]], initializer=tf.contrib.layers.xavier_initializer())

                string = 'bias' + str(i)
                if i == 0:
                    pass
                elif i == self.numLayers + 1:
                    B = tf.get_variable(name=string, shape=[self.numClasses], initializer=tf.constant_initializer(0.1))
                elif i < self.numLayers + 1:
                    B = tf.get_variable(name=string, shape=[self.numHidden[i - 1]], initializer=tf.constant_initializer(0.1))

                if i == 0:
                    n = tf.random_normal(shape=tf.shape(x), mean=noiseMean, stddev=noiseStdDev)
                    self.z_noise.append(tf.add(x, n))
                    self.h_noise.append(tf.add(x, n))
                elif i == self.numLayers + 1:
                    z_noise = tf.nn.batch_normalization(tf.matmul(self.h_noise[i - 1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None)
                    n = tf.random_normal(shape=tf.shape(z_noise), mean=noiseMean, stddev=noiseStdDev)
                    self.z_noise.append(tf.add(z_noise, n))
                    self.h_noise.append(tf.nn.sigmoid(tf.add(self.z_noise[i], B)))
                else:
                    z_noise = tf.nn.batch_normalization(tf.matmul(self.h_noise[i - 1], W), mean=0, variance=1, variance_epsilon=1e-9, offset=None, scale=None)
                    n = tf.random_normal(shape=tf.shape(z_noise), mean=noiseMean, stddev=noiseStdDev)
                    self.z_noise.append(tf.add(z_noise, n))
                    self.h_noise.append(tf.nn.relu(tf.add(self.z_noise[i], B)))

            lossSupervised_noise = -tf.reduce_mean(y * tf.log(self.h_noise[self.numLayers+1]) + (1 - y) * tf.log(1 - self.h_noise[self.numLayers+1]))
            self.loss_supervised_noise = lossSupervised_noise
            #_loss_summary(self.loss_supervised_noise)
            self.train_step_supervised_noise = tf.train.AdamOptimizer(lr).minimize(self.loss_supervised_noise)

    def decoder(self):
        with tf.variable_scope("decoder"):
            print "decode"
            self.z_recon = [None]*len(self.z_noise)
            self.diff = tf.constant(0.0,dtype=np.float32)
            for i in range(self.numLayers+1, -1, -1):
                print i

                if i == 0:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.inputSize)), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.inputSize], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.inputSize], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.inputSize], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(tf.matmul(self.z_recon[i+1], V), mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                elif i == 1:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numHidden[i-1])), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(tf.matmul(self.z_recon[i+1], V), mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numHidden[i-1], self.inputSize], initializer=tf.contrib.layers.xavier_initializer())
                elif i == self.numLayers+1:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numClasses)), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numClasses], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numClasses], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numClasses], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(self.h_noise[i], mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numClasses, self.numHidden[i-2]], initializer=tf.contrib.layers.xavier_initializer())
                else:
                    I = tf.Variable(initial_value=np.ones((batch_size, self.numHidden[i-1])), dtype=np.float32)
                    string = 'a' + str(i)
                    a = tf.get_variable(string, shape=[4, self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    string = 'b' + str(i)
                    b = tf.get_variable(string, shape=[self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    string = 'c' + str(i)
                    c = tf.get_variable(string, shape=[4, self.numHidden[i-1]], initializer=tf.constant_initializer(1.0))
                    mu = tf.nn.batch_normalization(tf.matmul(self.z_recon[i+1], V), mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-9)
                    string = 'weight' + str(i)
                    V = tf.get_variable(name=string, shape=[self.numHidden[i-1], self.numHidden[i-2]], initializer=tf.contrib.layers.xavier_initializer())

                #print "V", tf.Tensor.get_shape(tf.mul(V, tf.ones(tf.shape(V))))

                e = tf.transpose(tf.pack([I, self.z_noise[i], mu, tf.mul(self.z_noise[i], mu)]), [1, 0, 2])
                part1 = tf.reduce_sum(tf.mul(a, e), 1)
                part2 = tf.mul(b, tf.nn.sigmoid(tf.reduce_sum(tf.mul(c, e), 1)))
                self.z_recon[i] = tf.add(part1, part2)
                string = 'lambda' + str(i)
                lam = self.lambdas[i]
                self.diff = tf.add(self.diff,tf.div(tf.mul(lam, tf.reduce_sum(tf.squared_difference(self.z[i], self.z_recon[i]))), tf.to_float(tf.shape(self.z[i]))[1]))  # lambda * (sum||(z1-z1_n)||^2 ) / (layerWidth * numSamples)

        loss_unsupervised_noise = tf.div(self.diff, tf.to_float(tf.shape(self.h_noise[self.numLayers])[0]))
        #_loss_summary(loss_unsupervised_noise)
        self.loss = tf.add(self.loss_supervised_noise, loss_unsupervised_noise)
        #_loss_summary(self.loss)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def get_loss(self):
        return self.loss

    def get_loss_supervised(self):
        return self.loss_supervised

    def get_loss_supervised_noise(self):
        return self.loss_supervised_noise

    def get_train_step(self):
        return self.train_step

    def get_train_step_supervised(self):
        return self.train_step_supervised

    def get_train_step_supervised_noise(self):
        return self.train_step_supervised_noise

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


    # path = "/Users/terry/Desktop/RESEARCH/ENCODE/"
    # train_data_filename = path + "train.npy"
    # train_labels_filename = path + "train_labels.npy"
    # val_data_filename = path + "valid.npy"
    # val_labels_filename = path + "valid_labels.npy"
    # test_data_filename = path + "test.npy"
    # test_labels_filename = path + "test_labels.npy"


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

    #train_writer = tf.train.SummaryWriter(summary_path)

    nnTypes = ['dense','convolution','recurrent']
    nnType = nnTypes[0]
    if nnType.startswith('d'):
        numHidden = [4000, 2000, 2000, 2000, 2000, 1000] #the number of nodes in each hidden layer
        lambdas = [500.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #unsupervised loss coefficients:  first index is input layer, last index is output layer, the rest are the hidden layers

        noiseMean = 0.0
        noiseStdDev = 0.3
        L = DenseLadder(train_data, train_labels, val_data, val_labels, numHidden, lambdas)
        x = tf.placeholder(tf.float32, shape=[batch_size, L.get_size()])
        y = tf.placeholder(tf.float32, shape=[batch_size, L.get_classes()])
        L.feed_forward(x, y)
        L.encoder(x, y, noiseMean=noiseMean, noiseStdDev=noiseStdDev)
        L.decoder()
        loss = L.get_loss()                                 #ff, encoder, decoder
        #loss = L.get_loss_supervised()                     #ff
        #loss = L.get_loss_supervised_noise()               #encoder
        train_step = L.get_train_step()                     #ff, encoder, decoder
        #train_step = L.get_train_step_supervised()         #ff
        #train_step = L.get_train_step_supervised_noise()   #encoder
    if nnType.startswith('c'):
        numConv = [500,500,500]
        numFilters = [(15,4),(10,4),(7,4)]
        lambdas = [100.0, 10.0, 1.0, 1.0, 1.0] #unsupervised loss coefficients:  first index is input layer, last index is output layer, the rest are the convolution layers
        noiseMean = 0.0
        noiseStdDev = 0.3
        L = ConvLadder(train_data, train_labels, val_data, val_labels, numConv, numFilters, lambdas)
        x = tf.placeholder(tf.float32, shape=[batch_size, L.get_size()])
        y = tf.placeholder(tf.float32, shape=[batch_size, L.get_classes()])
        L.feed_forward(x, y)


    best_val_loss = 10000
    print
    with tf.Session() as s:
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        #merged = tf.merge_all_summaries()

        for epoch in xrange(num_epochs):
            for batch in xrange(0, L.numTrain, batch_size):
                if batch + batch_size > L.numTrain:
                    end = L.numTrain
                    break
                else:
                    end = batch + batch_size
                batch_data = L.trainData[batch:end, :]
                batch_labels = L.trainLabels[batch:end]
                _, cost = s.run([train_step, loss], feed_dict={x: batch_data, y: batch_labels})
                #train_writer.add_summary(summary, batch)
                print cost
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
                cost = s.run(loss, feed_dict={x: batch_data, y: batch_labels})
                costT += cost
                index += 1
            costT /= index
            print
            print "epoch = ", epoch, "\t", "val cost =", costT
            if costT < best_val_loss:
                saver.save(s, model_path)
                best_val_loss = costT

        print
        print "Testing..."
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
            cost = s.run([loss], feed_dict={x: batch_data, y: batch_labels})
            pred = L.h[L.numLayers+1].eval(feed_dict={x: batch_data, y: batch_labels})
            predictions.append(pred)
            costT += cost[0]
            index += 1
        costT /= index
        predictions = np.asarray(predictions)
        predictions = np.reshape(predictions,[predictions.shape[0]*predictions.shape[1], predictions.shape[2]])
        np.savetxt(predictions_save_path, predictions, fmt="%1.4f")
        print "test cost =", costT

#############################################################################################################

if __name__ == '__main__':
    tf.app.run()

