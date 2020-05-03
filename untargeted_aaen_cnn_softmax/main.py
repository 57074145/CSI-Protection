import tensorflow as tf
import numpy as np
import aaen_model as aaen
import os
import matplotlib
matplotlib.use('Agg')  
import h5py 
from sklearn import preprocessing
import gc
import time
import scipy.io as sio

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', 0, 'Train and save the AAEN model.') # 0-testing, and 1-training 
protect_type_num = 7  # In this untargeted protection, the param is disabled

# Placeholder nodes.
images_holder = tf.placeholder(tf.float32, [None, 1800])
label_holder_1 = tf.placeholder(tf.float32, [None, 7])
label_holder_2 = tf.placeholder(tf.float32, [None, 10])
label_holder_3 = tf.placeholder(tf.float32, [None, 6])
p_keep_holder = tf.placeholder(tf.float32)
rerank_holder = tf.placeholder(tf.float32, [None, protect_type_num]) # In this untargeted protection, the param is disabled

"""
Load CSI Data

"""
data_path = './dataset/three_model_AAEN.mat'  

data = h5py.File(data_path,'r')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_x = np.transpose(data['train_x'][:])
test_x = np.transpose(data['test_x'][:])
protection_target = 2 # In this untargeted protection, the param is disabled



def norm(train_x, test_x): 
     
    train_x = train_x.reshape(-1, 20*90)
    train_x_max = np.max(train_x, axis=0)
    train_x_min = np.min(train_x, axis=0)
    norm_train_x = (train_x-train_x_min)/(train_x_max-train_x_min)
    test_x = test_x.reshape(-1, 20*90) 
    norm_test_x = (test_x-train_x_min)/(train_x_max-train_x_min)
    
    return norm_train_x, norm_test_x, train_x_max, train_x_min



norm_train_x, norm_test_x, train_x_max, train_x_min = norm(train_x,test_x)   #normalization

norm_train_x = norm_train_x.reshape(-1,20, 90,1)


norm_test_x = norm_test_x.reshape(-1, 20, 90,1)

target_1_train_y = np.transpose(data['train_act'][:])
target_1_test_y = np.transpose(data['test_act'][:])
target_2_train_y = np.transpose(data['train_per'][:])
target_2_test_y =np.transpose(data['test_per'][:])
target_3_train_y = np.transpose(data['train_loc'][:])
target_3_test_y =np.transpose(data['test_loc'][:])


del data
gc.collect()# Deallocating memory


def reverse_norm(norm_x, x_max, x_min):
    
    x = norm_x*(x_max-x_min)+x_min 
    x = x.reshape(-1, 20, 90)
    return x

    

def get_batch(faces,model_1_label,model_2_label, model_3_label, batch_size):
    array=np.arange(model_1_label.shape[0])
    np.random.shuffle(array)
    idx=array[:batch_size]
    batch_xs = faces[idx]
    model_1_batch_ys = model_1_label[idx]
    model_2_batch_ys = model_2_label[idx]
    model_3_batch_ys = model_3_label[idx]
    
    return batch_xs, model_1_batch_ys, model_2_batch_ys, model_3_batch_ys # generate a batch


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()

"""
Test AAEN

"""
def test():
    """
    """
    print("ok\n")
    batch_size = 2800
    temp_test_x = norm_test_x.reshape(-1, 1800) # transform each CSI image into a vector
    batch_xs, target_1_batch_ys, target_2_batch_ys, target_3_batch_ys = get_batch(temp_test_x, target_1_test_y, target_2_test_y, target_3_test_y, batch_size) #generate input CSI 
    
    starttime = time.time()
    
    model = aaen.AAEN(images_holder, label_holder_1, label_holder_2, label_holder_3, p_keep_holder, rerank_holder)# initialize AAEN model
	
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load(sess, './Models/AE_for_AAEN') # load AAEN
			
        adv_images = sess.run(
            model.prediction,

            feed_dict={images_holder: temp_test_x} # adversarial signal generation 
        )   
        
        endtime = time.time()
        
        dtime = endtime - starttime # time cost of adversarial signal generation 
        
        a_1, adv_ft_1 = sess.run(model._target1.prediction, feed_dict={
              images_holder: adv_images,
              label_holder_1: target_1_test_y,
              p_keep_holder: 1.0
          })
    
        b_1, ori_ft_1 = sess.run(model._target1.prediction, feed_dict={
              images_holder: temp_test_x,
              label_holder_1: target_1_test_y,
              p_keep_holder: 1.0
          })
        
        a_2, adv_ft_2 = sess.run(model._target2.prediction, feed_dict={
              images_holder: adv_images,
              label_holder_2: target_2_test_y,
              p_keep_holder: 1.0
          })
    
        b_2, ori_ft_2 = sess.run(model._target2.prediction, feed_dict={
               images_holder: temp_test_x,
               label_holder_2: target_2_test_y,
               p_keep_holder: 1.0
          })
        
        a_3, adv_ft_3 = sess.run(model._target3.prediction, feed_dict={
              images_holder: adv_images,
              label_holder_3: target_3_test_y,
              p_keep_holder: 1.0
          })
    
        b_3, ori_ft_3 = sess.run(model._target3.prediction, feed_dict={
              images_holder: temp_test_x,
              label_holder_3: target_3_test_y,
              p_keep_holder: 1.0
          })
    
        pre_res = np.argmax(a_1, 1)
        
        print('Adversarial action_accuracy: {0:0.5f}'.format(
            sess.run(model._target1.accuracy, feed_dict={
                images_holder: adv_images,
               
                label_holder_1: target_1_test_y,
                p_keep_holder: 1.0
            })))
        
        print('Original action_accuracy: {0:0.5f}'.format(
            sess.run(model._target1.accuracy, feed_dict={
             
                images_holder: temp_test_x,
                label_holder_1:  target_1_test_y,
                p_keep_holder: 1.0
            })))
    
        print('Adversarial person_accuracy: {0:0.5f}'.format(
            sess.run(model._target2.accuracy, feed_dict={
                images_holder: adv_images,
          
                label_holder_2: target_2_test_y,
                p_keep_holder: 1.0
            })))
        
        print('Original person_accuracy: {0:0.5f}'.format(
            sess.run(model._target2.accuracy, feed_dict={
       
                images_holder: temp_test_x,
                label_holder_2:  target_2_test_y,
                p_keep_holder: 1.0
            })))
    
        print('Adversarial location_accuracy: {0:0.5f}'.format(
            sess.run(model._target3.accuracy, feed_dict={
         
                images_holder: adv_images,
                label_holder_3:  target_3_test_y,
                p_keep_holder: 1.0
            })))
    
        print('Original location_accuracy: {0:0.5f}'.format(
            sess.run(model._target3.accuracy, feed_dict={
               
                images_holder: temp_test_x,
                label_holder_3:  target_3_test_y,
                p_keep_holder: 1.0
            })))
         
        
        print("adv time：%.8s s" % dtime)
        
        show_ori = reverse_norm(temp_test_x,  train_x_max, train_x_min)#denormalization 
        show_adv = reverse_norm(adv_images, train_x_max, train_x_min)


        #save original、adversarial CSI signals and the extrated features
        sio.savemat('saveddata.mat', {'pre_res': pre_res,'show_ori': show_ori,'show_adv': show_adv, 'ori_ft_1': ori_ft_1.reshape(-1, 6, 64), 'adv_ft_1': adv_ft_1.reshape(-1, 6, 64), 
                                      'ori_ft_2': ori_ft_2.reshape(-1, 6, 64), 'adv_ft_2': adv_ft_2.reshape(-1, 6, 64), 'ori_ft_3': ori_ft_3.reshape(-1, 6, 64), 'adv_ft_3': adv_ft_3.reshape(-1, 6, 64)}) 
"""
Train AAEN

"""
def train():
    """
    """
    
    training_epochs = 3000 # training iterations
    batch_size = 128 
    
    rerank_y = np.mat(np.ones((batch_size, protect_type_num)))
    rerank_y[:, :] = 0
    rerank_y[:, protection_target] = 1
    
    model = aaen.AAEN(images_holder, label_holder_1, label_holder_2, label_holder_3, p_keep_holder, rerank_holder) # AAEN model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        checkpoin_dir = "./Models/AE_for_AAEN/AAEN_CSI"
        ckpt = tf.train.get_checkpoint_state(checkpoin_dir)
        if ckpt and ckpt.model_checkpoint_path:
           model._autoencoder.load(sess, checkpoin_dir, name='AAEN.ckpt')
        model._target1.load(sess, './Models/AE_for_AAEN/movement') #load CSI action classifiers
        model._target2.load(sess, './Models/AE_for_AAEN/person') #load CSI person classifiers
        model._target3.load(sess, './Models/AE_for_AAEN/loc') #load CSI location classifiers
        
        for epoch in range(training_epochs):
            
                temp_train_x = norm_train_x.reshape(-1,1800) # training CSI data normalization
                batch_xs, target_1_batch_ys, target_2_batch_ys, target_3_batch_ys = get_batch(temp_train_x, target_1_train_y, target_2_train_y, target_3_train_y, batch_size) # generate training batch 

                 
                _, loss, Lx, Ly, Lz, prediction = sess.run(model.optimization, feed_dict={
                    images_holder: batch_xs,
                    label_holder_1: target_1_batch_ys,
                    label_holder_2: target_2_batch_ys,
                    label_holder_3: target_3_batch_ys,
                    p_keep_holder: 1,
                    rerank_holder: rerank_y
         
                })
                print('Eopch {0} completed. loss = {1}. Lx = {2}. Ly = {3}. Lz = {4}. pre = {5}'.format(epoch+1, loss, Lx, Ly, Lz, prediction))
                if epoch % 100 == 0: 
                 model.save(sess, './Models/AE_for_AAEN')
                 print("Trained params have been saved to './Models/AE_for_AAEN/AAEN_CSI'")
        print("Optimization Finished!")
      


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
