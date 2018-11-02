"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import os
import glob
from random import shuffle, randint
import cv2

class ds_semseg_datareader:
    path = ""
    class_mappings = ""
    files = []
    images = []
    annotations = []
    test_images = []
    test_annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, deepscores_path, max_pages=None, crop=True, crop_size=[1000,1000], test_size=20, scale=0.5):
        """
        Initialize a file reader for the DeepScores classification data
        :param records_list: path to the dataset
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        """
        print("Initializing DeepScores segmentation Batch Dataset Reader...")
        self.path = deepscores_path
        self.max_pages = max_pages
        self.crop = crop
        self.crop_size = crop_size
        self.test_size = test_size
        self.scale = scale

        images_list = []
        images_glob = os.path.join(self.path, "images_png", '*.' + 'png')
        images_list.extend(glob.glob(images_glob))

        #shuffle image list
        shuffle(images_list)

        if max_pages is None:
            max_pages = len(images_list)
            print("max pages: ")

        if max_pages > len(images_list):
            print("Not enough data, only " + str(len(images_list)) + " available")
            print(" At "+ self.path)

        if test_size >= max_pages:
            print("Test set too big ("+str(test_size)+"), max_pages is: "+str(max_pages))
            print(" At " + self.path)
            import sys
            sys.exit(1)

        print("Splitting dataset, train: "+str(max_pages-test_size)+" images, test: "+str(test_size)+ " images")
        self.test_image_list = images_list[0:test_size]
        self.train_image_list = images_list[test_size:max_pages]

        # test_annotation_list = [image_file.replace("/images_png/", "/pix_annotations_png/") for image_file in test_image_list]
        # train_annotation_list = [image_file.replace("/images_png/", "/pix_annotations_png/") for image_file in train_image_list]

        #do lazy loading
        #self._read_images(test_image_list,train_image_list)

    def _read_images(self,test_image_list,train_image_list):

        dat_train = [self._transform(filename) for filename in train_image_list]
        for dat in dat_train:
            self.images.append(dat[0])
            self.annotations.append(dat[1])
        self.images = np.array(self.images)
        self.images = np.expand_dims(self.images, -1)

        self.annotations = np.array(self.annotations)
        self.annotations = np.expand_dims(self.annotations, -1)

        print("Training set done")
        dat_test = [self._transform(filename) for filename in test_image_list]
        for dat in dat_test:
            self.test_images.append(dat[0])
            self.test_annotations.append(dat[1])
        self.test_images = np.array(self.test_images)
        self.test_images = np.expand_dims(self.test_images, -1)

        self.test_annotations = np.array(self.test_annotations)
        self.test_annotations = np.expand_dims(self.test_annotations, -1)
        print("Test set done")


    def _transform(self, filename):
        image = misc.imread(filename)
        annotation = misc.imread(filename.replace("/images_png/", "/pix_annotations_png/"))
        if not image.shape[0:2] == annotation.shape[0:2]:
            print("input and annotation have different sizes!")
            import sys
            import pdb
            pdb.set_trace()
            sys.exit(1)

        if image.shape[-1] != 1:
            # take mean over color channels, image BW anyways --> fix in dataset creation
            image = np.mean(image, -1)

        if self.scale !=0:
            image = cv2.resize(image, None, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            annotation = cv2.resize(annotation, None, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

        if self.crop:
            coord_0 = randint(0, (image.shape[0] - self.crop_size[0]))
            coord_1 = randint(0, (image.shape[1] - self.crop_size[1]))

            image = image[coord_0:(coord_0+self.crop_size[0]),coord_1:(coord_1+self.crop_size[1])]
            annotation = annotation[coord_0:(coord_0 + self.crop_size[0]), coord_1:(coord_1 + self.crop_size[1])]

        return [image, annotation]

    # from PIL import Image
    # im = Image.fromarray(image)
    # im.show()
    # im = Image.fromarray(annotation)
    # im.show()


    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def get_test_records(self):
        images, annotations = self.load_batch(self.test_image_list)
        return images, annotations

    def next_batch_preloaded(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch_preloaded(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def load_batch(self, img_names):
        try:
            images = []
            annotations = []
            dat_train = [self._transform(filename) for filename in img_names]
            for dat in dat_train:
                images.append(dat[0])
                annotations.append(dat[1])
            images = np.array(images)
            images = np.expand_dims(images, -1)

            annotations = np.array(annotations)
            annotations = np.expand_dims(annotations, -1)
            return images, annotations
        except KeyboardInterrupt:
            raise
        except:
            print("there was a problem when loading this batch")
            print("Filename: "+ filename)
            print("try and return the next batch")
            return None, None


    def next_batch(self, batch_size):
        images = None
        annotations = None

        while images is None or annotations is None:
            start = self.batch_offset
            self.batch_offset += batch_size
            if self.batch_offset > len(self.train_image_list):
                # Finished epoch
                self.epochs_completed += 1
                print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
                # Shuffle the data
                shuffle(self.train_image_list)
                # Start next epoch
                start = 0
                self.batch_offset = batch_size
            end = self.batch_offset
            images, annotations = self.load_batch(self.train_image_list[start:end])

        return images, annotations

if __name__ == "__main__":
    data_reader = ds_semseg_datareader("/Users/tugg/datasets/DeepScores")
