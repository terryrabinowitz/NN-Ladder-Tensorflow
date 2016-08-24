# tensorboard --logdir=/Users/terry/PycharmProjects/untitled/train/
import re
import numpy as np
import tensorflow as tf
from LadderClass import Ladder, ConvLadder, DenseLadder

TOWER_NAME = 'tower'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations',x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _loss_summary(x):
    tf.scalar_summary(x.op.name, x)

def main(argv=None):
    batch_size = 500
    num_epochs = 1
    lr = .0001
    lambdaSupervised = 10.0
    lambdas = [1.0, 1.0, 1.0]
    # unsupervised loss coefficients:  first index is input layer, last index is output layer, the rest are the hidden/convolution layers

    noiseMean = 0.0
    noiseStdDev = 0.3

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
        numHidden = [1000] #the number of nodes in each hidden layer

        L = DenseLadder(train_data, train_labels, val_data, val_labels, numHidden, lambdas, lr, batch_size, lambdaSupervised)
        x = tf.placeholder(tf.float32, shape=[batch_size, L.get_size()])
        y = tf.placeholder(tf.float32, shape=[batch_size, L.get_classes()])
        L.feed_forward(x, y)
        L.encoder(x, y, noiseMean=noiseMean, noiseStdDev=noiseStdDev)
        L.decoder()
        lossSN = L.loss_supervised_noise
        lossUN = L.loss_unsupervised_noise
        loss = L.get_loss()                                 #total
        train_step = L.get_train_step()                     #total
        #train_step = L.get_train_step_supervised()
        #train_step = L.get_train_step_supervised_noise()
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

    best_val_loss = np.inf
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
                _, cost, costSN, costUN = s.run([train_step, loss, lossSN, lossUN], feed_dict={x: batch_data, y: batch_labels})
                #train_writer.add_summary(summary, batch)
                print cost, costSN, costUN
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
            cost = s.run(loss, feed_dict={x: batch_data, y: batch_labels})
            pred = L.h[L.numLayers+1].eval(feed_dict={x: batch_data, y: batch_labels})
            predictions.append(pred)
            costT += cost
            index += 1
        costT /= index
        predictions = np.asarray(predictions)
        predictions = np.reshape(predictions,[predictions.shape[0]*predictions.shape[1], predictions.shape[2]])
        np.savetxt(predictions_save_path, predictions, fmt="%1.4f")
        print "test cost =", costT

#############################################################################################################

if __name__ == '__main__':
    tf.app.run()

