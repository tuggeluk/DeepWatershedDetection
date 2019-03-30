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
import cv2
from main.config import cfg
from utils.blob import im_list_to_blob, compute_scalings
from datasets.fcn_groundtruth import get_markers, stamp_class
import sys
from roi_data_layer.sample_images_for_augmentation import RandomImageSampler
from PIL import Image
import pickle

counter = 0



def get_minibatch(roidb, args, assign, helper, ignore_symbols=0, visualize=0, augmentation_type='none'):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)

    # Take care of all randomness whithin batch construction here!
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(args.scale_list),
                                    size=num_images)
    assert (args.batch_size % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, args.batch_size)

    #assert len(roidb) == 1, "Single batch only"
    # # flip names of macrophage data
    # path = "/share/DeepWatershedDetection/data/macrophages_2019/test/images/"
    # files = os.listdir(path)
    # for f in files:
    #     os.rename(path+"/"+f, path+"/"+f+"prern")
    #
    # files = os.listdir(path)
    # for f in files:
    #     if "mCherry" in f:
    #         os.rename(path + "/" + f, path + "/" + f.split("_")[0] + "_DAPI.tif")
    #     else:
    #         os.rename(path + "/" + f, path + "/" + f.split("_")[0] + "_mCherry.tif")
    #
    # # files = os.listdir(path)
    # # for f in files:
    # #     os.rename(path + "/" + f, path + "/" + f[:3]+"_"+f[3:])


    # iterate over batch elements
    minibatch = []
    for nr_ele, roidb_ele in enumerate(roidb):
        # iterate over sub elements (for paired data)
        scalings = None
        sub_batch = []
        for nr_subele, roidb_subele in enumerate(roidb_ele):
            #print( roidb_subele)

            if scalings is None:
                # figure out scaling factor and crop box for current paired images
                # scalings = [scaling_factor, crop_box]
                # assumes that all the images in minibatch have the same size!
                global_scale = args.scale_list[random_scale_inds[0]]
                scalings = compute_scalings(global_scale, roidb_subele, args)


            # Get the input image blob
            im_blob = _get_image_blob(roidb_subele, scalings, args)
            blob = {'data': im_blob}

            # gt boxes: (x1, y1, x2, y2, cls)
            if cfg.TRAIN.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(roidb_subele['gt_classes'] != 0)[0]
            else:
                # For the COCO ground truth boxes, exclude the  ones that are ''iscrowd''
                gt_inds = np.where(roidb_subele['gt_classes'] != 0 & np.all(roidb_subele['gt_overlaps'].toarray() > -1.0, axis=1))[0]

            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)

            if args.crop == "True":
                # scale Coords
                gt_boxes[:, 0:4] = roidb_subele['boxes'][gt_inds, :] * scalings[0]

                gt_boxes[:, 0:4] = gt_boxes[:, 0:4] - [scalings[1][1], scalings[1][0], scalings[1][1], scalings[1][0]]

            else:
                gt_boxes[:, 0:4] = roidb_subele['boxes'][gt_inds, :] * scalings[0]

            gt_boxes[:, 4] = roidb_subele['gt_classes'][gt_inds]

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
            gt_boxes = [crop_boxes(blob["data"].shape, box) for box in gt_boxes]
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

            blob['data'] = new_blob

            for i1 in range(len(assign)):
                if assign[i1]["stamp_func"][0] == "stamp_energy" and assign[i1]["use_obj_seg"] and roidb_subele["objseg_path"] is not None:
                    canvas = None

                    cache_path = roidb_subele["objseg_path"][0].replace("object_masks", "semseg_cache").split("/")
                    cache_path = "/"+os.path.join(*cache_path[:-1])+cache_path[-1][-8:-4]

                    if assign[i1]["use_obj_seg_cached"] and os.path.exists(cache_path+".npy"):
                        #im = Image.open(cache_path)
                        #canvas = np.array(im, dtype=np.float32)
                        canvas = np.load(cache_path+".npy")

                    else:
                        for objseg_img_path in roidb_subele["objseg_path"]:

                            im = Image.open(objseg_img_path)
                            im = np.array(im, dtype=np.float32)

                            if canvas is None:
                                # init canvas
                                canvas = np.zeros(im.shape, dtype=np.float32)

                            #print("build marker")
                            # build marker
                            im[im != 0] = 10000 # assume longest path is shorter than 10'000
                            dims = im.shape
                            not_done = True
                            save_val = 1
                            while not_done:
                                unlabeled_ind = np.where(im == 10000)
                                #print(len(unlabeled_ind[0]))
                                if len(unlabeled_ind[0]) == 0:
                                    not_done = False
                                    continue
                                for x1, x2 in zip(unlabeled_ind[0], unlabeled_ind[1]):
                                    #check neighborhood
                                    proposed_val = np.min(im[np.max((0,(x1-1))):np.min(((x1+2), dims[0])),np.max(((x2-1),0)):np.min(((x2+2),dims[1]))])+1
                                    if proposed_val != 10001 and proposed_val <= save_val:
                                        im[x1,x2] = proposed_val
                                save_val += 1

                            # add to canvas
                            im = im/np.max(im)*(cfg.TRAIN.MAX_ENERGY-1)
                            canvas += im

                        # cache
                        canvas = np.round(canvas)
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        np.save(cache_path,canvas)
                        #Image.fromarray(canvas.astype(np.uint8)).save(cache_path)

                    # crop and scale
                    # do scaling
                    canvas = cv2.resize(canvas, None, None, fx=scalings[0], fy=scalings[0],
                                    interpolation=cv2.INTER_NEAREST)
                    # do cropping
                    canvas = canvas[scalings[1][0]:scalings[1][2], scalings[1][1]:scalings[1][3]]


                    cavn_print = blob["data"]/np.max(blob["data"].shape)*255
                    cavn_print = np.squeeze(cavn_print[0], -1)
                    Image.fromarray(cavn_print.astype(np.uint8)).save("/share/DeepWatershedDetection/data/macrophages_2019/test_"+str(nr_subele)+".jpg")

                    cavn_print = canvas/np.max(canvas)*255
                    Image.fromarray(cavn_print.astype(np.uint8)).save("/share/DeepWatershedDetection/data/macrophages_2019/test_gt_"+str(nr_subele)+".jpg")

                    blob["assign" + str(i1)] = dict()
                    for i2 in range(len(assign[i1]["ds_factors"])):
                        # downsample
                        canv_downsamp = cv2.resize(canvas, None, None, fx=1/assign[i1]["ds_factors"][i2], fy=1/assign[i1]["ds_factors"][i2],
                                    interpolation=cv2.INTER_NEAREST)

                        # one-hot encode
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            canv_downsamp = np.round(canv_downsamp).astype(np.int32)
                            canv_downsamp = np.eye(cfg.TRAIN.MAX_ENERGY)[canv_downsamp[:, :]]
                        else:
                            canv_downsamp = np.expand_dims(canv_downsamp, -1)

                        canv_downsamp = np.expand_dims(canv_downsamp, 0)
                        blob["assign" + str(i1)]["gt_map" + str(i2)] = canv_downsamp


                elif assign[i1]["stamp_func"][0] == "stamp_class" and assign[i1]["use_sem_seg"] and roidb_subele["semseg_path"] is not None:

                    im = Image.open(roidb_subele["semseg_path"])
                    canvas = np.array(im, dtype=np.float32)


                    # apply semseg color -> class transform
                    for ind, val in enumerate(args.semseg_ind):
                        canvas[canvas ==  val] = ind

                    # crop and scale
                    # do scaling
                    canvas = cv2.resize(canvas, None, None, fx=scalings[0], fy=scalings[0],
                                    interpolation=cv2.INTER_NEAREST)
                    # do cropping
                    canvas = canvas[scalings[1][0]:scalings[1][2], scalings[1][1]:scalings[1][3]]

                    blob["assign" + str(i1)] = dict()
                    for i2 in range(len(assign[i1]["ds_factors"])):
                        # downsample
                        canv_downsamp = cv2.resize(canvas, None, None, fx=assign[i1]["ds_factors"][i2], fy=assign[i1]["ds_factors"][i2],
                                    interpolation=cv2.INTER_NEAREST)

                        # one-hot encode
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            canv_downsamp = np.round(canv_downsamp).astype(np.int32)
                            canv_downsamp = np.eye(args.nr_classes[0])[canv_downsamp[:, :]]
                        else:
                            canv_downsamp = np.expand_dims(canv_downsamp, -1)

                        canv_downsamp = np.expand_dims(canv_downsamp, 0)
                        blob["assign" + str(i1)]["gt_map" + str(i2)] = canv_downsamp

                else:
                    # bbox based assign
                    markers_list = get_markers(blob['data'].shape, gt_boxes, args.nr_classes[0], assign[i1], 0, [])
                    blob["assign" + str(i1)] = dict()
                    for i2 in range(len(assign[i1]["ds_factors"])):
                        blob["assign" + str(i1)]["gt_map" + str(i2)] = markers_list[i2]

            # ds_factors = set()
            # for i1 in range(len(assign)):
            #     ds_factors = ds_factors.union(set(assign[i1]["ds_factors"]))

            # #TODO add semseg GT if available
            # print("load, semseg gt")
            # if roidb_subele["semseg_path"] is not None:
            #     # load image
            #     im = Image.open(roidb_subele['semseg_path'])
            #     im = np.array(im, dtype=np.float32)
            #
            #     # do scaling
            #     im = cv2.resize(im, None, None, fx=scalings[0], fy=scalings[0],
            #                     interpolation=cv2.INTER_LINEAR)
            #     # do cropping
            #     im = im[scalings[1][0]:scalings[1][2], scalings[1][1]:scalings[1][3]]
            #     if len(im.shape) == 2:
            #         im = np.expand_dims(np.expand_dims(im, -1),0)
            #     # save downsampled versions
            #     for ds_factor in enumerate(ds_factors):
            #         blob["assign" + str(i1)]["gt_map" + str(i2)] = markers_list[i2]
            #
            # #TODO add obj-seg GT if available
            # print("load obj GT")
            #     # init canvas
            #     # load images
            #     # add marker according to energy assign
            #
            #     # apply scaling and cropping
            #
            #     # save downsampled versions according to energy task

            # Build loss masks
            # mask out background for class and bounding box predictions
            # also used for class/object weight balancing
            for i1 in range(len(assign)):
                for i2 in range(len(assign[i1]["ds_factors"])):
                    if assign[i1]["balance_mask"] == "mask_bg":
                        # background has weight zero
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            fg_map = np.argmax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        else:
                            fg_map = np.amax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        fg_map[fg_map != 0] = 1
                        fg_map = fg_map / (np.sum(fg_map) + 1)

                        blob["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_map[0], -1)

                    elif assign[i1]["balance_mask"] == "fg_bg_balanced":
                        # foreground and background have the same weight
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            fg_map = np.argmax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        else:
                            fg_map = np.amax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        fg_map[fg_map != 0] = 1

                        fg_copy = np.copy(fg_map).astype("float64")
                        # weigh each position by the inverse of its size
                        unique_counts = np.unique(fg_map, return_counts=1)

                        for ele in range(len(unique_counts[0])):
                            fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                        blob["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)

                    elif assign[i1]["balance_mask"] == "by_object":
                        # each object has the same weight (background is one object)
                        print("Unknown loss mask command")
                        sys.exit(1)


                    elif assign[i1]["balance_mask"] == "by_class":
                        # each class has the same weight ( background is one class)
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            fg_map = np.argmax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        else:
                            fg_map = np.amax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)

                        fg_copy = np.copy(fg_map).astype("float64")
                        # weigh each position by the inverse of its size
                        unique_counts = np.unique(fg_map, return_counts=1)

                        for ele in range(len(unique_counts[0])):
                            fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                        blob["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)


                    elif assign[i1]["balance_mask"] == "by_object_no_bg":
                        # each object has the same weight (background has no weight)
                        print("Unknown loss mask command")
                        sys.exit(1)


                    elif assign[i1]["balance_mask"] == "by_class_no_bg":
                        # each class has the same weight ( background discarded)
                        if assign[i1]["stamp_args"]["loss"] == "softmax":
                            fg_map = np.argmax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)
                        else:
                            fg_map = np.amax(blob["assign" + str(i1)]["gt_map" + str(i2)], -1)

                        fg_copy = np.copy(fg_map).astype("float64")
                        # weigh each position by the inverse of its size
                        unique_counts = np.unique(fg_map, return_counts=1)

                        for ele in range(1, len(unique_counts[0])):
                            fg_copy[fg_map == unique_counts[0][ele]] = sum(unique_counts[1])/unique_counts[1][ele]

                        blob["assign" + str(i1)]["mask" + str(i2)] = np.expand_dims(fg_copy[0], -1)


                    elif assign[i1]["balance_mask"] is None:
                        # do nothing / multiply everything by 1
                        blob["assign" + str(i1)]["mask" + str(i2)] = np.ones(blob["assign" + str(i1)]["gt_map" + str(i2)].shape[:-1] + (1,))[0]

                    else:
                        print("Unknown loss mask command")
                        sys.exit(1)


            # set helper to None
            blob["helper"] = None
            gt_boxes.extend(new_boxes)
            blob['gt_boxes'] = np.expand_dims(gt_boxes, 0)
            blob['im_info'] = np.array(
                [[new_blob.shape[1], new_blob.shape[2], scalings[0]]],
                dtype=np.float32)

            # for deepscores average over last data dimension
            if "DeepScores" in args.dataset or "MUSICMA" in args.dataset:
                blob["data"] = np.average(blob["data"], -1)
                blob["data"] = np.expand_dims(blob["data"], -1)
            if visualize:
                global counter
                with open(os.path.join('/DeepWatershedDetection/visualization/pickle_files', str(counter) + '.pickle'), 'wb') as handle:
                    pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
                counter += 1
            sub_batch.append(blob)
        # sub batch done stack inputs and outputs
        #TODO concatenate sub batches
        minibatch.append(sub_batch)
    return minibatch


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




def _get_image_blob(roidb, scalings, args):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """

    im = Image.open(roidb['image'])
    if im.mode ==  'I;16':
        # deal with wierd tif format
        #print("tif")
        im = np.array(im, dtype=np.float32)/256
        #Image.fromarray(im.astype(np.uint8)).save("/share/DeepWatershedDetection/tif_test.png")
    else:
        im = np.array(im, dtype=np.float32)


    # #fix last dimension 2
    # if im.shape[-1]==2:
    #     im = np.concatenate((im, np.zeros(im.shape[0:2]+(1,))), axis=-1)

    if "VOC2012" in roidb['image']:
        # 1 2 0
        im = im[:, :, (0,1,2)]
        # substract mean
        if args.substract_mean == "True":
            mean = (122.67891434, 116.66876762, 104.00698793)
            im -= mean
        #im = im.transpose((2, 0, 1))

    if roidb['flipped']:
        im = im[:, ::-1, :]

    # do scaling
    im = cv2.resize(im, None, None, fx=scalings[0], fy=scalings[0],
                    interpolation=cv2.INTER_LINEAR)
    # do cropping
    im = im[scalings[1][0]:scalings[1][2], scalings[1][1]:scalings[1][3]]

    # if not args.pad_to == 0:
    #   # pad to fit RefineNet #TODO fix refinenet padding problem
    #   y_mulity = int(np.ceil(im.shape[0] / float(args.pad_to)))
    #   x_mulity = int(np.ceil(im.shape[1] / float(args.pad_to)))
    #   canv = np.ones([y_mulity * args.pad_to, x_mulity * args.pad_to,3], dtype=np.uint8) * 255
    #   canv[0:im.shape[0], 0:im.shape[1]] = im
    #   im = canv

    if len(im.shape) == 2:
        im = np.expand_dims(im, -1)
    # Create a blob to hold the input images
    blob = im_list_to_blob([im])

    return blob
