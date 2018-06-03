import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def preprocess_labels(label_image):#include labels : 0=gb, 7=road, 10=cars
    #exclude labels : 1,2,3,4,5,8,9,11,12
    other_pixels = (label_image[:,:,0] == 1).nonzero()#Buildings
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 2).nonzero()#Fences
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 3).nonzero()#Other
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 4).nonzero()#Pedestrians
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 5).nonzero()#Poles
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 8).nonzero()#Sidewalks
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 9).nonzero()#Vegetation
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 11).nonzero()#Walls
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 12).nonzero()#TrafficSigns
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 6).nonzero()#lanes
    label_image[other_pixels] = 0
    other_pixels = (label_image[:,:,0] == 7).nonzero()#roads
    label_image[other_pixels] = 0
    
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 235).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    label_image[hood_pixels] = 0
    # Return the preprocessed label image 
    #print("labels_new.shape: "+str(labels_new.shape))
    return label_image

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    train_paths, validation_paths = train_test_split(image_paths, test_size=0.15)
    def get_batches_fn(batch_size,train_paths):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        #label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))
        #background_color = np.array([255, 0, 0])
        random.shuffle(train_paths)
        for batch_i in range(0, len(train_paths), batch_size):
            images = []
            gt_images = []
            for image_file in train_paths[batch_i:batch_i+batch_size]:
                gt_image_file = image_file.replace("CameraRGB", "CameraSeg")
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_image = preprocess_labels(gt_image)
                background_color = np.array([0, 0, 0])
                car_color = np.array([10, 0, 0])    
                #gt_bg = np.all(gt_image == background_color, axis=2)
                #gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                #gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_car = np.all(gt_image == car_color, axis=2)
                gt_car = gt_car.reshape(*gt_car.shape, 1)
                gt_image = np.concatenate((gt_bg, gt_car), axis=2)
                images.append(image)
                gt_images.append(gt_image)
                '''
                items, car_count = np.unique(gt_car,return_counts=True)
                if len(car_count) > 1 and car_count[1] > 3000:
                    #augment ---------------------------
                    #  - grayscal with contrast
                    images.append(image)
                    gt_images.append(gt_image)
                    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    img = cv2.equalizeHist(img)#contrast http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html
                    cv2.normalize(img,img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)#, dtype=cv2.CV_32F)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    images.append(img)
                    gt_images.append(gt_image)
                    #########################
                    #  2-flip grayscale and original
                    img_h_flip = cv2.flip( img, 0 )
                    image_h_flip = cv2.flip( image, 0 )
                    try:
                        gt_image_h_flip = cv2.flip( gt_image, 0 )
                    except:
                        gt_image_h_flip = gt_image    
                    images.append(img_h_flip)
                    gt_images.append(gt_image_h_flip)
                    images.append(image_h_flip)
                    gt_images.append(gt_image_h_flip)

                    #augment end ----------------------------------------------------
                #end if
            c = list(zip(images, gt_images))
            random.shuffle(c)
            images, gt_images = zip(*c)
            '''
            yield np.array(images), np.array(gt_images)
    return get_batches_fn, train_paths, validation_paths


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    #image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    #label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))
    #for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):    
    is_printed_once = False
    for image_file in  data_folder:
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        im_softmax_none = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
        im_softmax_car = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        
        segmentation_none = (im_softmax_none > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_car = (im_softmax_car > 0.5).reshape(image_shape[0], image_shape[1], 1)
        if is_printed_once == False:
            print("a-im_softmax :"+str(im_softmax))
            print("a-len :"+str(len(im_softmax[0])))
            print("auniq="+str(np.unique(im_softmax,return_counts=True)))
            print(len(np.unique(im_softmax)))
            print("image_shape"+str(image_shape))
            print("c-im_softmax :"+str(im_softmax_car))
            print("c-len :"+str(len(im_softmax_car)))
            print("c-uniq="+str(np.unique(im_softmax_car,return_counts=True)))
            print(len(np.unique(im_softmax_car)))
            print("-segmentation_car :"+str(segmentation_car))
            print("-uniq="+str(np.unique(segmentation_car,return_counts=True)))
            print(len(np.unique(segmentation_car)))
            print("________")
            is_printed_once = True
        
        mask = np.dot(segmentation_car, np.array([[0, 0, 255, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
               
        mask = np.dot(segmentation_none, np.array([[255, 0, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(street_im)
        street_im.paste(mask, box=None, mask=mask)
        
        yield os.path.basename(image_file), np.array(street_im)
        #break#test only
        
        
def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, validation_files):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, validation_files, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
