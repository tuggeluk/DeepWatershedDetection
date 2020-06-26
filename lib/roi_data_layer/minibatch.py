# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen - Exteded by Lukas Tuggener and Ismail Elezi
# --------------------------------------------------------

# NB: Terms width, height etc have not been used consistently, but the results are correct (bounding boxes in the augmented images are visualized)

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import numpy.random as npr
from main.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from datasets.fcn_groundtruth import get_markers
import sys
from roi_data_layer.sample_images_for_augmentation import RandomImageSampler
from PIL import Image

counter = 0


def get_minibatch(roidb, args, assign, helper, ignore_symbols=0, visualize=0, augmentation_type='none'):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(args.scale_list),
                                    size=num_images)
    assert (args.batch_size % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, args.batch_size)

    # Get the input image blob
    im_blob, im_scales, crop_box = _get_image_blob(roidb, random_scale_inds, args)
    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != -1)[0]
    else:
        # For the COCO ground truth boxes, exclude the  ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)

    if args.crop == "True":
        # scale Coords
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]

        gt_boxes[:, 0:4] = gt_boxes[:, 0:4] - [crop_box[0][1], crop_box[0][0], crop_box[0][1], crop_box[0][0]]

    else:
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]

    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

    (batch_size, height, width, channels) = im_blob.shape

    # get the RandomImageSampler object to do augmentation
    if augmentation_type == 'up':
        im_s = RandomImageSampler(height, width)
        images, bboxes, horizontal, small_height, small_width = im_s.sample_image_up(ignore_symbols)
        new_blob = np.full((batch_size, height + small_height, width, channels), 255)
        new_blob[:, small_height:, :, :] = im_blob
    elif augmentation_type == 'full':
        im_s = RandomImageSampler(height, width)
        images, bboxes, horizontal, vertical, small_height, small_width = im_s.sample_image_full(ignore_symbols)
        new_blob = np.full((batch_size, height + small_height * vertical, width + small_width * horizontal, channels), 255)

    # remove nones, boxes which are outside of the images
    gt_boxes = [crop_boxes(blobs["data"].shape, box) for box in gt_boxes]
    gt_boxes = [x for x in gt_boxes if x is not None]

    new_boxes = []  # initialize a list of new_boxes where we put all the boxes of augmented images

    if augmentation_type != 'none':
        # here we shift bounding boxes of the real image
        for i in range(len(gt_boxes)):
            gt_boxes[i][1] += small_height
            gt_boxes[i][3] += small_height
        # here we should augment the image on the top of it
        for i in range(horizontal):
            if augmentation_type == 'up':
                im = np.expand_dims(images[i], 0)
                new_blob[:, 0:small_height, i * small_width:(i + 1) * small_width, :] = im * 255
                # new_blob[:, 0:small_height, i*small_width:(i+1) * small_width, 0] = im * 255
                # new_blob[:, 0:small_height, i*small_width:(i+1) * small_width, 1] = im * 255
                # new_blob[:, 0:small_height, i*small_width:(i+1) * small_width, 2] = im * 255  # workaround, delete this and the two rows above, uncomment the row before them
                # here we shift bounding boxes of the synthetic part of the image
                if not ignore_symbols:
                    for j in range(len(bboxes[i])):
                        bboxes[i][j][0] += (i * small_width)
                        bboxes[i][j][2] += (i * small_width)
                        new_boxes.append(bboxes[i][j])
                else:
                    bboxes[i][0] += (i * small_width)
                    bboxes[i][2] += (i * small_width)
                    new_boxes.append(bboxes[i])
            elif augmentation_type == 'full':
                for k in range(vertical):
                    im = np.expand_dims(images[i * vertical + k], 0)
                    new_blob[:, k * small_height:(k + 1) * small_height, i * small_width:(i + 1) * small_width, :] = im * 255
                    # here we shift bounding boxes of the synthetic part of the image
                    if not ignore_symbols:
                        for j in range(len(bboxes[i * vertical + k])):
                            bboxes[i * vertical + k][j][0] += (i * small_width)
                            bboxes[i * vertical + k][j][2] += (i * small_width)
                            bboxes[i * vertical + k][j][1] += (k * small_height)
                            bboxes[i * vertical + k][j][3] += (k * small_height)
                            new_boxes.append(bboxes[i * vertical + k][j])
                    else:
                        bboxes[i * vertical + k][0] += (i * small_width)
                        bboxes[i * vertical + k][2] += (i * small_width)
                        bboxes[i * vertical + k][1] += (k * small_height)
                        bboxes[i * vertical + k][3] += (k * small_height)
                        new_boxes.append(bboxes[i * vertical + k])
    else:
        new_blob = im_blob

    if not args.pad_to == 0:
        # pad to fit RefineNet #TODO fix refinenet padding problem
        y_mulity = int(np.ceil(new_blob.shape[1] / float(args.pad_to)))
        x_mulity = int(np.ceil(new_blob.shape[2] / float(args.pad_to)))
        canv = np.ones([args.batch_size, y_mulity * args.pad_to, x_mulity * args.pad_to, 3], dtype=np.uint8) * 255
        canv[:, 0:new_blob.shape[1], 0:new_blob.shape[2], :] = new_blob
        new_blob = canv

    blobs['data'] = new_blob

    for i1 in range(len(assign)):
        markers_list = get_markers(blobs['data'].shape, gt_boxes, args.nr_classes[0], assign[i1], 0, [])
        blobs["assign" + str(i1)] = dict()
        for i2 in range(len(assign[i1]["ds_factors"])):
            blobs["assign" + str(i1)]["gt_map" + str(i2)] = markers_list[i2]

    # Build loss masks
    # mask out background for class and bounding box predictions
    # also used for class/object weight balancing
    for i1 in range(len(assign)):
        for i2 in range(len(assign[i1]["ds_factors"])):
            if assign[i1]["balance_mask"] == "mask_bg":
                # background has weight zero
                if assign[i1]["stamp_args"]["loss"] == "softmax":
                    fg_map = np.argmax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                else:
                    fg_map = np.amax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                fg_map[fg_map != 0] = 1
                fg_map = fg_map / (np.sum(fg_map) + 1)

                blobs["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_map[0], -1)

            elif assign[i1]["balance_mask"] == "fg_bg_balanced":
                # foreground and background have the same weight
                if assign[i1]["stamp_args"]["loss"] == "softmax":
                    fg_map = np.argmax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                else:
                    fg_map = np.amax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                fg_map[fg_map != 0] = 1

                fg_copy = np.copy(fg_map).astype("float64")
                # weigh each position by the inverse of its size
                unique_counts = np.unique(fg_map, return_counts=1)

                for ele in range(len(unique_counts[0])):
                    fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                blobs["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)

            elif assign[i1]["balance_mask"] == "by_object":
                # each object has the same weight (background is one object)
                print("Unknown loss mask command")
                sys.exit(1)


            elif assign[i1]["balance_mask"] == "by_class":
                # each class has the same weight ( background is one class)
                if assign[i1]["stamp_args"]["loss"] == "softmax":
                    fg_map = np.argmax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                else:
                    fg_map = np.amax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)

                fg_copy = np.copy(fg_map).astype("float64")
                # weigh each position by the inverse of its size
                unique_counts = np.unique(fg_map, return_counts=1)

                for ele in range(len(unique_counts[0])):
                    fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                blobs["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)


            elif assign[i1]["balance_mask"] == "by_object_no_bg":
                # each object has the same weight (background has no weight)
                print("Unknown loss mask command")
                sys.exit(1)


            elif assign[i1]["balance_mask"] == "by_class_no_bg":
                # each class has the same weight ( background discarded)
                if assign[i1]["stamp_args"]["loss"] == "softmax":
                    fg_map = np.argmax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)
                else:
                    fg_map = np.amax(blobs["assign" + str(i1)]["gt_map" + str(i2)], -1)

                fg_copy = np.copy(fg_map).astype("float64")
                # weigh each position by the inverse of its size
                unique_counts = np.unique(fg_map, return_counts=1)

                for ele in range(1, len(unique_counts[0])):
                    fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                blobs["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)


            elif assign[i1]["balance_mask"] is None:
                # do nothing / multiply everything by 1
                blobs["assign" + str(i1)]["mask" + str(i2)] = np.ones(blobs["assign" + str(i1)]["gt_map" + str(i2)].shape[:-1] + (1,))[0]

            else:
                print("Unknown loss mask command")
                sys.exit(1)


    # set helper to None
    blobs["helper"] = None
    gt_boxes.extend(new_boxes)
    blobs['gt_boxes'] = np.expand_dims(gt_boxes, 0)
    blobs['im_info'] = np.array(
        [[new_blob.shape[1], new_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    # for deepscores average over last data dimension
    if "DeepScores" in args.dataset or "MUSICMA" in args.dataset:
        blobs["data"] = np.average(blobs["data"], -1)
        blobs["data"] = np.expand_dims(blobs["data"], -1)
    if visualize:
        global counter
        with open(os.path.join('/DeepWatershedDetection/visualization/pickle_files', str(counter) + '.pickle'), 'wb') as handle:
            pickle.dump(blobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        counter += 1
    return blobs


def crop_boxes(img_shape, coord):
    crop_coords = coord[0:4]
    crop_coords = np.maximum(crop_coords, 0)
    crop_coords[[0, 2]] = np.minimum(crop_coords[[0, 2]], img_shape[2])
    crop_coords[[1, 3]] = np.minimum(crop_coords[[1, 3]], img_shape[1])
    # if a dimension collapses kill element
    if crop_coords[0] == crop_coords[2] or crop_coords[1] == crop_coords[3]:
        return None

    coord[0:4] = crop_coords
    return coord


def _get_image_blob(roidb, scale_inds, args):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    crop_box = []

    for i in range(num_images):

        im = Image.open(roidb[i]['image'])
        im = np.array(im, dtype=np.float32)

        if "VOC2012" in roidb[i]['image']:
            # 1 2 0
            im = im[:, :, (0,1,2)]
            # substract mean
            if args.substract_mean == "True":
                mean = (122.67891434, 116.66876762, 104.00698793)
                im -= mean
            #im = im.transpose((2, 0, 1))

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        global_scale = args.scale_list[scale_inds[i]]
        im, im_scale, im_crop_box = prep_im_for_blob(im, global_scale, args)

        crop_box.append(im_crop_box)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, crop_box
