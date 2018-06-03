#!/usr/bin/python
import matplotlib.pyplot as plt
import os.path
import tensorflow as tf
import helper_carsonly
import warnings
from distutils.version import LooseVersion
import numpy as np
import cv2
from tensorflow.python.tools import inspect_checkpoint as chkp
#slpit data , to :training and validation

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    #load graph from file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    #grap the graph in variabel
    graph = tf.get_default_graph()
    #get desired layers by name
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    out_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    out_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    out_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, out_3, out_4, out_7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))#https://www.programcreek.com/python/example/90502/tensorflow.truncated_normal_initializer
    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    transpose_7 = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, strides=(2, 2),padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    add_7_4 = tf.add(transpose_7, conv_1x1_4)
    transpose_4 = tf.layers.conv2d_transpose(add_7_4, num_classes, 4, strides=(2, 2),padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    add_4_3 = tf.add(transpose_4 , conv_1x1_3 )
    transpose_3 = tf.layers.conv2d_transpose(add_4_3, num_classes, 16, strides=(8, 8),padding='same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2))
    return transpose_3

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes), name='labels')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss = cross_entropy_loss);
    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, num_classes, train_files):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    for epoch in range(epochs):
        print(" ----- starting epoch %d , batchsize=%d"%(epoch,batch_size))
        loss_list = []
        batch_num = 0
        batch_total_loss = 0
        for image, label in get_batches_fn(batch_size, train_files):
            if len(image) <= 0 or len(label) <= 0:
                continue
            #label = label.reshape([-1,288,416,num_classes])
            #first 8 epoches (batch_size=16) learning_rate = 0.000001
            train , batch_loss = sess.run([train_op,cross_entropy_loss], 
                     feed_dict={input_image:image,correct_label:label,keep_prob:0.3,learning_rate:1e-3})
            batch_total_loss = batch_total_loss + batch_loss
            print("    batch %d has loss %f"%(batch_num,batch_loss))
            batch_num +=1
            loss_list.append(batch_loss)
        epoch_loss=batch_total_loss/batch_num
        print ("  epoch %d finished with loss %f"%(epoch,epoch_loss))
    print(loss_list)
    #plt.plot(loss_list)
    #plt.ylabel('loss_list_001')
    #plt.savefig('loss_list_001',format='png')
    pass

def run(validate):
    num_classes = 2
    image_shape = (288,416)#(576, 800)
    data_dir = './data'
    train_dir = 'Train'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper_carsonly.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    epochs = 14#8#2#10
    batch_size = 4#10#6#8#
    #learning_rate = 10.0#from project_tests.py
    with tf.Session() as sess:
        #vars
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')#from project_tests.py
        #correct_label = np.reshape(correct_label, (-1, image_shape[0], image_shape[1], 2))
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')#from project_tests.py
            
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn, train_files, validation_files = helper_carsonly.gen_batch_function(os.path.join(data_dir, train_dir), image_shape)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        print("train_paths : %d"%len(train_files))
        print("validation_paths: %d"%len(validation_files))
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output,correct_label,learning_rate,num_classes)
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if validate :
            #################### load model if there are any ###############
            print(">>>>>>>>>> vars in saved model :")
            #chkp.print_tensors_in_checkpoint_file("./saved/segmentation_model_180529", tensor_name='', all_tensors=True)
            #saver = tf.train.import_meta_graph('./saved/segmentation_model_180531.meta')
            #saver.restore(sess,tf.train.latest_checkpoint('./saved/'))
            saver.restore(sess, "./saved/_180603_10_001/segmentation_model.ckpt")
            print("Model loaded!")
            #graph.get_tensor_by_name
            ################################################################
            # TODO: Train NN using the train_nn function
        else:
            #saver.restore(sess, "./saved/_180603_01_001/segmentation_model.ckpt")
            #print("Model loaded!")
            print("start train 180603_10_001 carsonly : epochs="+str(epochs)+" ,batch_size="+str(batch_size))
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,  correct_label, keep_prob, learning_rate, num_classes, train_files)
        
            #TODO: safe model : 
            saver.save(sess, './saved/_180603_10_001/segmentation_model.ckpt')
            print("Model Saved!")
            # TODO: Save inference data using helper.save_inference_samples
        helper_carsonly.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, validation_files)
        print("done")
        
def freeze():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./saved/_180603_10_001/segmentation_model.ckpt")
    print("Model loaded!")
    tf.train.write_graph(sess.graph, './saved/_180603_01_001/', 'train.pb', False)


import scipy.misc
import numpy as np
def test1():
    num_classes = 2
    image_shape = (288,416)#(576, 800)
    data_dir = './data'
    train_dir = 'Train'
    background_color = np.array([0, 0, 0])
    road_color = np.array([7, 0, 0])
    car_color = np.array([10, 0, 0])    
    data_folder = os.path.join(data_dir, train_dir)
    gt_image_file = os.path.join(data_folder, 'CameraSeg', '0.png')
    image_file = os.path.join(data_folder, 'CameraRGB', '0.png')
    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
    print("image.shape"+str(image.shape))
    gt_image_ = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
    print("gt_image_.shape"+str(gt_image_.shape))
    print("uniq="+str(np.unique(gt_image_,return_counts=True)))
    
    gt_bg_ = np.all(gt_image_ == background_color, axis=2)
    print("gt_bg_.shape"+str(gt_bg_.shape))
    print("uniq="+str(np.unique(gt_bg_,return_counts=True)))
    gt_bg = gt_bg_.reshape(*gt_bg_.shape, 1)
    print(" gt_bg.shape"+str(gt_bg.shape))
    print("  uniq="+str(np.unique(gt_bg,return_counts=True)))
    
    gt_road_ = np.all(gt_image_ == road_color, axis=2)
    print("gt_road_.shape"+str(gt_road_.shape))
    print("  uniq="+str(np.unique(gt_road_,return_counts=True)))
    gt_road = gt_road_.reshape(*gt_road_.shape, 1)
    print(" gt_road.shape"+str(gt_bg.shape))
    print("   uniq="+str(np.unique(gt_bg,return_counts=True)))
    
    gt_car_ = np.all(gt_image_ == car_color, axis=2)
    print("gt_car_.shape"+str(gt_car_.shape))
    print("  uniq="+str(np.unique(gt_car_,return_counts=True)))
    gt_car = gt_car_.reshape(*gt_car_.shape, 1)
    print(" gt_car.shape"+str(gt_car.shape))
    print("   uniq="+str(np.unique(gt_car,return_counts=True)))
    
    gt_image = np.concatenate((gt_bg, gt_road, gt_car), axis=2)
    print("gt_image.shape"+str(gt_image.shape))
    print("uniq="+str(np.unique(gt_image,return_counts=True)))

    print("===== res ====")
    print("uniq="+str(np.unique(gt_image[:,:,0],return_counts=True)))
    print("uniq="+str(np.unique(gt_image[:,:,1],return_counts=True)))
    print("uniq="+str(np.unique(gt_image[:,:,2],return_counts=True)))

if __name__ == '__main__':
    run(False)
    #test1()
    #freeze()
