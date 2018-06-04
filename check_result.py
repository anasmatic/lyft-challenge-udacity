import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc
import argparse  
import os
import scipy


file = sys.argv[-1]

image_shape = (288,416)

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")
'''
def load_graph(sess, frozen_graph_filename):
    tf.saved_model.loader.load(sess, [''], frozen_graph_filename)
    graph = tf.get_default_graph()
    return graph
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
'''
def load_graph(sess, graph_file):
    """Loads a frozen inference graph"""
    saver = tf.train.import_meta_graph(graph_file+'/segmentation_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(graph_file)) 
    graph = tf.get_default_graph()
    '''
    tf.global_variables_initializer()
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    '''
    return graph

def road_seg_function(sess,rgb_frame,road_image_input,road_keep_prob,road_softmax):
    image = scipy.misc.imresize(rgb_frame, image_shape)
    im_softmax = sess.run([road_softmax], {road_image_input: [image], road_keep_prob: 1.0})
    #sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])        
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    '''
    probs = sess.run([softmax], {image_input: [image], keep_prob: 1.0})
    im_softmax = probs[0].reshape(image_shape[0], image_shape[1], 3)
    road =  np.where(im_softmax[:,:,1] > 0.5,1,0)
    road = road * 255
    segmentation = np.zeros_like(image)
    segmentation[:,:,1] = road
    segmentation = scipy.misc.imresize(segmentation, (600,800))    
    '''
    return segmentation

def car_seg_function(sess,rgb_frame,car_image_input,car_keep_prob,car_softmax):    
    image = scipy.misc.imresize(rgb_frame, image_shape)
    im_softmax = sess.run([car_softmax], {car_image_input: [image], car_keep_prob: 1.0})
    #sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])        
    segmentation = (im_softmax > 0.3).reshape(image_shape[0], image_shape[1], 1)
    '''
    image = scipy.misc.imresize(rgb_frame, image_shape)
    probs = sess.run([car_softmax], {image_input: [image], keep_prob: 1.0})
    im_softmax = probs[0].reshape(image_shape[0], image_shape[1], 3)
    car = np.where(im_softmax[:,:,1] > 0.3,1,0)
    car = car * 255
    segmentation = np.zeros_like(image)
    segmentation[:,:,1] = car
    segmentation = scipy.misc.imresize(segmentation, (600,800))
    '''
    return segmentation

video = skvideo.io.vread(file)
'''
CAR_FILE = './lyft-challenge-udacity/saved/_180603_10_001/saved_model.pb'
car_graph = load_graph(sess,CAR_FILE)
car_image_input = car_graph.get_tensor_by_name('image_input:0')
car_keep_prob = car_graph.get_tensor_by_name('keep_prob:0')
car_softmax = car_graph.get_tensor_by_name('logits:0')

ROAD_FILE = './lyft-challenge-udacity/saved/_180603_7_001/train.pb'
road_graph = load_graph(sess,ROAD_FILE)
road_image_input = road_graph.get_tensor_by_name('image_input:0')
road_keep_prob = road_graph.get_tensor_by_name('keep_prob:0')
road_softmax = road_graph.get_tensor_by_name('logits:0')
'''

def seg_pipeline(rgb_frame):
    out_road = road_seg_function(rgb_frame)
    out_car = car_seg_function(rgb_frame)
    overlay = np.zeros_like(rgb_frame)
    overlay[:,:,1] = out_car[:,:,1] 
    overlay[:,:,2] = out_road[:,:,1] 
    overlay[496:,:,2] = 0
    final_frame = cv2.addWeighted(rgb_frame, 1, overlay, 0.3, 0, rgb_frame)
    return final_fram

answer_key = {}
car_answer_key = {}
road_answer_key = {}
tf.reset_default_graph()
tf.global_variables_initializer()
#with tf.Session(graph=car_graph) as sess:
with tf.Session() as sess:
    ROAD_FILE = './lyft-challenge-udacity/saved/_180603_7_001'#/train.pb'
    road_graph = load_graph(sess,ROAD_FILE)
    road_image_input = road_graph.get_tensor_by_name('image_input:0')
    road_keep_prob = road_graph.get_tensor_by_name('keep_prob:0')
    road_softmax = road_graph.get_tensor_by_name('logits:0')
    CAR_FILE = './lyft-challenge-udacity/saved/_180603_10_001'#/saved_model.pb'
    car_graph = load_graph(sess,CAR_FILE)
    car_image_input = car_graph.get_tensor_by_name('image_input:0')
    car_keep_prob = car_graph.get_tensor_by_name('keep_prob:0')
    car_softmax = car_graph.get_tensor_by_name('logits:0')

    #answer_key = {}
    # Frame numbering starts at 1
    frame = 1
    for rgb_frame in video:
        out_car = car_seg_function(sess,rgb_frame,car_image_input,car_keep_prob,car_softmax)
        out_road = road_seg_function(sess,rgb_frame,road_image_input,road_keep_prob,road_softmax)
        # Look for red cars :)
        binary_car_result = np.asarray(np.where(out_car[:,:,0]==True)).astype('uint8')
        # Look for road :)
        binary_road_result =  np.asarray(np.where(out_road[:,:,0]==True)).astype('uint8')
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        # Increment frame
        frame+=1
'''
with tf.Session(graph=road_graph) as sess:
    #answer_key = {}
    # Frame numbering starts at 1
    frame = 1
    for rgb_frame in video:
        out_road = road_seg_function(rgb_frame)
        # Look for red cars :)
        binary_car_result = np.where(out_car[:,:,1]==True).astype('uint8')
        # Look for road :)
        binary_road_result =  np.where(out_road[:,:,1]==True).astype('uint8')
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        # Increment frame
        frame+=1
    #
'''
# Print output in proper json format
print (json.dumps(answer_key))
#clip = clip1.fl_image(seg_pipeline)
#clip.write_videofile('./lyft-challenge-udacity/runs/segmentation_output_test.mp4', audio=False)