import tensorflow as tf
import first_basic_cnn as bcnn_1
import second_basic_cnn as bcnn_2
import third_basic_cnn as bcnn_3
import my_basic_ae as bae
from decorator2 import lazy_property


class AAEN:
    """
     The AAEN Model
    """

    def __init__(self, data, label_gt_1, label_gt_2, label_gt_3, p_keep, rerank):
        with tf.variable_scope('autoencoder'):
            self._autoencoder = bae.BasicAE(data)
        with tf.variable_scope('target1') as scope1:
            self._target_adv1 = bcnn_1.FirstBasicCnn(
                self._autoencoder.prediction, label_gt_1, p_keep, rerank
            )
            scope1.reuse_variables()
            self._target1 = bcnn_1.FirstBasicCnn(data, label_gt_1, p_keep, rerank)    
        with tf.variable_scope('target2') as scope2:
            self._target_adv2 = bcnn_2.SecondBasicCnn(
                self._autoencoder.prediction, label_gt_2, p_keep, rerank
            )
            scope2.reuse_variables()
            self._target2 = bcnn_2.SecondBasicCnn(data, label_gt_2, p_keep, rerank)
        with tf.variable_scope('target3') as scope3:
            self._target_adv3 = bcnn_3.ThirdBasicCnn(
                self._autoencoder.prediction, label_gt_3, p_keep, rerank
            )
            scope3.reuse_variables()
            self._target3 = bcnn_3.ThirdBasicCnn(data, label_gt_3, p_keep, rerank)

        self.data = data
        self.rerank = rerank
        self.label_gt_1 = label_gt_1
        self.label_gt_2 = label_gt_2
        self.label_gt_3 = label_gt_3
        self.prediction
        self.optimization

    @lazy_property
    def optimization(self):
        loss_beta = 0.01 
        learning_rate = 0.0001

        y_pred = self._autoencoder.prediction
        y_true = self.data

        loss_act = self._target_adv1.loss
        loss_per = self._target_adv2.loss
        loss_loc = self._target_adv3.loss
        
        loss = loss_act + loss_per - loss_beta*loss_loc
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "autoencoder"))
        return optimizer, loss, loss_act, loss_per, loss_loc, y_pred

    @lazy_property
    def prediction(self):
        return self._autoencoder.prediction

    def load(self, sess, path, prefix="AAEN"):
        self._autoencoder.load(sess, path+'/AAEN_CSI', name=prefix+'.ckpt')
        self._target1.load(sess, path+'/movement')
        self._target2.load(sess, path+'/person')
        self._target3.load(sess, path+'/loc')

    def save(self, sess, path, prefix="AAEN"):
        self._autoencoder.save(sess, path+'/AAEN_CSI', name=prefix+'.ckpt')
