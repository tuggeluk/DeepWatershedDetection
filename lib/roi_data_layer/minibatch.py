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
from PIL import Image, ImageDraw
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
    # path = "train/images/"
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
    # #
    # # files = os.listdir(path)
    # # for f in files:
    # #     os.rename(path + "/" + f, path + "/" + f[:3]+"_"+f[3:])


    # iterate over batch elements
    minibatch = []
    for nr_ele, roidb_ele in enumerate(roidb):
        # iterate over sub elements (for paired data)
        scalings = None
        sub_batch = []
        # put in list for paired 1
        if roidb_ele.__class__ == dict().__class__:
            roidb_ele = [roidb_ele]
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

            gt_boxes = np.empty((len(gt_inds), 9), dtype=np.float32)
            gt_boxes = [[[None],None] for i in range(len(gt_inds))]
            # translate all bboxes in 4 point path format
            if "boxes_full" in roidb_subele.keys():
                roidb_subele["boxes"] = roidb_subele["boxes_full"]
            else:
                for idx, elem in enumerate(roidb_subele["boxes"]):
                    x1, y1, x2, y2 = elem
                    roidb_subele["boxes"][idx] = [x1, y1, x1, y2, x2, y2, x1, y2]


            #gt_boxes[:, 8] = roidb_subele['gt_classes'][gt_inds]

            for ix, ele in enumerate(roidb_subele['gt_classes'][gt_inds]):
                gt_boxes[ix][1]= ele

            if args.crop == "True":
                debug_plots = False
                # scale Coords
                #gt_boxes[:, 0:8] = roidb_subele['boxes'][gt_inds, :] * scalings[0]

                for ix, ele in enumerate(roidb_subele['boxes'][gt_inds, :]):
                    gt_boxes[ix][0] = ele * scalings[0]

                left_side_map = {"[1 1]": [-1, 1], "[ 1 -1]": [1, 1], "[-1 -1]": [1, -1], "[-1  1]": [-1, -1],
                                 "[0 1]": [-1, 0], "[1 0]": [0, 1], "[ 0 -1]": [1, 0], "[-1  0]": [0, -1]}

                for ix, gt_box in enumerate(gt_boxes):
                    gt_box = gt_box[0]
                    #box corners are inside crop
                    new_path = []
                    b_inside = []
                    for i in range(int(len(gt_box)/2)):
                        p = gt_box[2*i:2*i+2]
                        if scalings[1][1] < p[0] and p[0] < scalings[1][3] and scalings[1][0] < p[1] and p[1] < scalings[1][2]:
                            b_inside.append(1)
                            new_path.append(p)
                        else:
                            b_inside.append(0)

                    #all inside --> done
                    if sum(b_inside) == len(b_inside):
                        y1, x1, y2, x2 = scalings[1]
                        crop_path = np.array([x1, y1, x1, y2, x2, y2, x2, y1])

                        if debug_plots:
                            y1, x1, y2, x2 = scalings[1]
                            crop_path = np.array([x1, y1, x1, y2, x2, y2, x2, y1])
                            import matplotlib.pyplot as plt
                            plt.plot(np.concatenate((gt_box[0::2], np.expand_dims(gt_box[0], -1))),
                                     np.concatenate((gt_box[1::2], np.expand_dims(gt_box[1], -1))))
                            plt.plot(np.concatenate((crop_path[0::2], np.expand_dims(crop_path[0], -1))),
                                     np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                            plt.show()
                            plt.plot(gt_box[0::2], gt_box[1::2])
                            plt.plot(np.concatenate((crop_path[0::2], np.expand_dims(crop_path[0], -1))),
                                     np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                            plt.show()

                    if not sum(b_inside) == len(b_inside):

                        y1,x1,y2,x2 = scalings[1]
                        crop_path = np.array([x1, y1, x1, y2, x2, y2, x2, y1])
                        xv, yv = np.array([x1, x1, x2, x2]), np.array([y1, y2, y2, y1])
                        # check for crop corners inside of box
                        c_inside = np.ones(xv.shape)
                        for point in range(int(len(gt_box)/2)):
                            dp = 2 * point
                            p = gt_box[dp:dp + 2].astype(np.int)

                            dp = 2 * (point + 1)
                            if point == int(len(gt_box)/2)-1:
                                dp = 0
                            p1 = gt_box[dp:dp + 2].astype(np.int)

                            u = p1 - p
                            u_len = np.dot(u, u)

                            v = [xv, yv] - np.expand_dims(p, -1)
                            # u dot v / u_len
                            gamma = (u[0] * v[0] + u[1] * v[1]) / u_len
                            proj = (v - [(gamma * u[0]), (gamma * u[1])])

                            signs = left_side_map[str(np.sign(u))]
                            c_inside = np.min((c_inside, np.sign(proj[0]) == signs[0], np.sign(proj[1]) == signs[1]), 0)

                        #disjoint boxes --> done
                        if sum(b_inside) == 0 and sum(c_inside) == 0:
                            crop_path = np.array([x1, y1, x1, y2, x2, y2, x2, y1])
                            if debug_plots:
                                import matplotlib.pyplot as plt
                                # plt.plot(np.concatenate((gt_box[0::2], np.expand_dims(gt_box[0], -1))),
                                #          np.concatenate((gt_box[1::2], np.expand_dims(gt_box[1], -1))))
                                # plt.plot(np.concatenate((crop_path[0::2], np.expand_dims(crop_path[0], -1))),
                                #          np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                                # plt.show()
                                plt.plot(gt_box[0::2], gt_box[1::2])
                                plt.plot(np.concatenate((crop_path[0::2], np.expand_dims(crop_path[0], -1))),
                                         np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                                plt.show()
                            gt_boxes[ix] = None
                            continue


                        if not (sum(b_inside) == 0 and sum(c_inside) == 0):

                            for i in range(len(c_inside)):
                                if c_inside[i] == 1:
                                    new_path.append([xv[i],yv[i]])

                                # find all intersections
                            # iterate over box paths
                            for i_box in range(int(len(gt_box) / 2)):
                                dp = 2 * i_box
                                p_box = gt_box[dp:dp + 2].astype(np.int)

                                dp = 2 * (i_box + 1)
                                if i_box == int(len(gt_box)/2)-1:
                                    dp = 0
                                p1_box = gt_box[dp:dp + 2].astype(np.int)

                                u_box = p1_box - p_box
                                # iterate over crop paths
                                for i_crop in range(int(len(crop_path) / 2)):
                                    dp = 2 * i_crop
                                    p_crop = crop_path[dp:dp + 2].astype(np.int)

                                    dp = 2 * (i_crop + 1)
                                    if i_crop == int(len(crop_path) / 2)-1:
                                        dp = 0
                                    p1_crop = crop_path[dp:dp + 2].astype(np.int)

                                    u_crop = p1_crop - p_crop
                                    # find intersection
                                    # solve p_box + alpha*u_box = p_crop+beta*u_crop
                                    A = np.transpose(np.stack((u_box, -u_crop)))
                                    b = p_crop - p_box
                                    try:
                                        x = np.linalg.solve(A,b)
                                    except:
                                        x = np.array([-1,-1])
                                    #p_box + x[0]*u_box
                                    # check if interesction is inside both edges
                                    try:
                                        if (0 <= x).all() and (x <= 1).all():
                                            new_path.append(p_crop+x[1]*u_crop)
                                    except:
                                        print("debug path")

                            #print("intersection pat hs")

                    new_path_mat = np.stack(new_path).astype(np.float)
                    for ix_delp, point in enumerate(new_path):
                        point_inds = np.where(np.sum(np.abs((new_path_mat - point)) < 1e-10, 1) == 2)
                        if len(point_inds[0]) > 1:
                            #remove element
                            new_path_mat = np.delete(new_path_mat, point_inds[0][1:], axis=0)

                    # reorder path
                    new_path_ordered = []

                    # start with point closest to origin
                    #new_path_ordered.append(new_path_mat[np.argmin(np.linalg.norm(new_path_mat,2,1))])

                    # start with path closest to y axis
                    new_path_ordered.append(new_path_mat[np.argmin(new_path_mat[: , 0])])
                    max_cor = np.array([0,-1]) # counter clockwise rotation

                    try:
                    # add paths that add the biggest scalar product with previous path ---> assume convexity
                        for i in range(new_path_mat.shape[0]-1):
                            conn_vectors = new_path_mat - new_path_ordered[i]
                            #remove dist=0 point
                            new_path_mat = new_path_mat[np.linalg.norm(conn_vectors, 2, 1) != 0]
                            conn_vectors = conn_vectors[np.linalg.norm(conn_vectors,2,1)!=0]

                            dot_prods = np.dot(conn_vectors,max_cor)/np.linalg.norm(conn_vectors,2,1)
                            new_path_ordered.append(new_path_mat[np.argmax(dot_prods)])
                            max_cor = new_path_ordered[-1]-new_path_ordered[-2]
                            max_cor = max_cor/np.linalg.norm(max_cor,2)
                    except Exception as e:
                        print("debug scala prod")
                        # reorder path
                        new_path_ordered = []
                        new_path_mat = np.stack(new_path).astype(np.float)

                        # start with point c losest to origin
                        # new_path_ordered.append(new_path_mat[np.argmin(np.linalg.norm(new_path_mat,2,1))])

                        # start with path closest to y axis
                        new_path_ordered.append(new_path_mat[np.argmin(new_path_mat[:, 0])])
                        max_cor = np.array([0, -1])  # counter clockwise rotation

                        # add paths that add the biggest scalar product with previous path ---> assume convexity
                        for i in range(len(new_path) - 1):
                            conn_vectors = new_path_mat - new_path_ordered[i]
                            # remove dist=0 point
                            new_path_mat = new_path_mat[np.linalg.norm(conn_vectors, 2, 1) != 0]
                            conn_vectors = conn_vectors[np.linalg.norm(conn_vectors, 2, 1) != 0]

                            dot_prods = np.dot(conn_vectors, max_cor) / np.linalg.norm(conn_vectors, 2, 1)
                            new_path_ordered.append(new_path_mat[np.argmax(dot_prods)])
                            max_cor = new_path_ordered[-1] - new_path_ordered[-2]
                            max_cor = max_cor / np.linalg.norm(max_cor, 2)
                    try:
                        gt_boxes[ix][0] = np.concatenate(new_path_ordered)
                    except Exception as e:
                        print("debug gt box assignement")
                        raise e

                    if debug_plots:
                        import matplotlib.pyplot as plt
                        # plt.plot(np.concatenate((gt_box[0::2],np.expand_dims(gt_box[0],-1))),
                        #          np.concatenate((gt_box[1::2], np.expand_dims(gt_box[1], -1))))
                        # plt.plot(np.concatenate((crop_path[0::2],np.expand_dims(crop_path[0],-1))),
                        #          np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                        # plt.plot(np.concatenate((np.concatenate(new_path_ordered)[0::2],np.expand_dims(np.concatenate(new_path_ordered)[0],-1))),
                        #          np.concatenate((np.concatenate(new_path_ordered)[1::2], np.expand_dims(np.concatenate(new_path_ordered)[1], -1))))
                        # plt.show()
                        # plt.plot(np.concatenate((np.concatenate(new_path_ordered)[0::2],np.expand_dims(np.concatenate(new_path_ordered)[0],-1))),
                        #          np.concatenate((np.concatenate(new_path_ordered)[1::2], np.expand_dims(np.concatenate(new_path_ordered)[1], -1))))
                        # plt.show()
                        plt.plot(np.concatenate(new_path_ordered)[0::2],np.concatenate(new_path_ordered)[1::2])
                        plt.show()
                        print("debug path ")
                    gt_boxes[ix][0][0::2] = gt_boxes[ix][0][0::2] - scalings[1][1]
                    gt_boxes[ix][0][1::2] = gt_boxes[ix][0][1::2] - scalings[1][0]

                    try:
                        stamp, coords = assign[0]["stamp_func"][1](gt_boxes[ix], assign[0]["stamp_args"], args.nr_classes[0])
                    except Exception as e:
                        print("debug stamp")
                        import matplotlib.pyplot as plt
                        plt.plot(np.concatenate((gt_box[0::2],np.expand_dims(gt_box[0],-1))),
                                 np.concatenate((gt_box[1::2], np.expand_dims(gt_box[1], -1))))
                        plt.plot(np.concatenate((crop_path[0::2],np.expand_dims(crop_path[0],-1))),
                                 np.concatenate((crop_path[1::2], np.expand_dims(crop_path[1], -1))))
                        plt.plot(np.concatenate((np.concatenate(new_path_ordered)[0::2],np.expand_dims(np.concatenate(new_path_ordered)[0],-1))),
                                 np.concatenate((np.concatenate(new_path_ordered)[1::2], np.expand_dims(np.concatenate(new_path_ordered)[1], -1))))
                        plt.show()
                        plt.plot(np.concatenate((np.concatenate(new_path_ordered)[0::2],np.expand_dims(np.concatenate(new_path_ordered)[0],-1))),
                                 np.concatenate((np.concatenate(new_path_ordered)[1::2], np.expand_dims(np.concatenate(new_path_ordered)[1], -1))))
                        plt.show()
                        stamp, coords = assign[0]["stamp_func"][1](gt_boxes[ix], assign[0]["stamp_args"], args.nr_classes[0])

                # # subtract crop distance
                # for ix, ele in enumerate(gt_boxes):
                #     if ele is not None:
                #         gt_boxes[ix][0][0::2] = ele[0][0::2] - scalings[1][1]
                #         gt_boxes[ix][0][1::2] = ele[0][1::2] - scalings[1][0]
            else:
                for ix, ele in enumerate(roidb_subele['boxes'][gt_inds, :]):
                    gt_boxes[ix][0] = ele * scalings[0]



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
            #gt_boxes = [crop_boxes(blob["data"].shape, box) for box in gt_boxes]
            #gt_boxes = [x for x in gt_boxes if x is not None]

            new_boxes = []  # initialize a list of new_boxes where we put all the boxes of augmented images

            if augmentation_type != 'none':
                # here we shift bounding boxes of the real image
                for i in range(len(gt_boxes)):
                    gt_boxes[i][1] += small_height
                    gt_boxes[i][3] += small_height
                    gt_boxes[i][5] += small_height
                    gt_boxes[i][7] += small_height
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
                                bboxes[i][j][4] += (i * small_width)
                                bboxes[i][j][6] += (i * small_width)
                                new_boxes.append(bboxes[i][j])
                        else:
                            bboxes[i][0] += (i * small_width)
                            bboxes[i][2] += (i * small_width)
                            bboxes[i][4] += (i * small_width)
                            bboxes[i][6] += (i * small_width)
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
                                    bboxes[i * vertical + k][j][4] += (i * small_width)
                                    bboxes[i * vertical + k][j][6] += (i * small_width)

                                    bboxes[i * vertical + k][j][1] += (k * small_height)
                                    bboxes[i * vertical + k][j][3] += (k * small_height)
                                    bboxes[i * vertical + k][j][5] += (k * small_height)
                                    bboxes[i * vertical + k][j][7] += (k * small_height)

                                    new_boxes.append(bboxes[i * vertical + k][j])
                            else:
                                bboxes[i * vertical + k][0] += (i * small_width)
                                bboxes[i * vertical + k][2] += (i * small_width)
                                bboxes[i * vertical + k][4] += (i * small_width)
                                bboxes[i * vertical + k][6] += (i * small_width)

                                bboxes[i * vertical + k][1] += (k * small_height)
                                bboxes[i * vertical + k][3] += (k * small_height)
                                bboxes[i * vertical + k][5] += (k * small_height)
                                bboxes[i * vertical + k][7] += (k * small_height)

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
                if assign[i1]["stamp_func"][0] == "stamp_energy" and assign[i1]["use_obj_seg"] and "objseg_path" in roidb_subele.keys():
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


                elif assign[i1]["stamp_func"][0] == "stamp_class" and assign[i1]["use_sem_seg"] and "semseg_path" in roidb_subele.keys():

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
                    markers_list = get_markers(blob['data'].shape, gt_boxes, args.nr_classes[0], assign[i1], 0, [],args.model)
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


                    elif assign[i1]["balance_mask"] is None or assign[i1]["balance_mask"] == "None":
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
            if "Dota" in args.dataset:
                #expand dims if its just one
                if blob["data"].shape[-1] == 1:
                    blob["data"] = np.concatenate([blob["data"] for i in range(3)], -1)
                    #print("expand_dims")
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
    crop_coords = coord[0:8]
    crop_coords = np.maximum(crop_coords, 0)
    crop_coords[[0, 2, 4, 6]] = np.minimum(crop_coords[[0, 2, 4, 6]], img_shape[2])
    crop_coords[[1, 3, 5, 7]] = np.minimum(crop_coords[[1, 3, 5, 7]], img_shape[1])
    # one of the diagonals is zero? --> kill element
    if  (crop_coords[0] * crop_coords[3] - crop_coords[1] * crop_coords[2]) + \
        (crop_coords[2] * crop_coords[5] - crop_coords[3] * crop_coords[4]) + \
        (crop_coords[4] * crop_coords[7] - crop_coords[5] * crop_coords[6]) + \
        (crop_coords[6] * crop_coords[1] - crop_coords[7] * crop_coords[0]) == 0:
        return None

    coord[0:8] = crop_coords
    return coord




def _get_image_blob(roidb, scalings, args):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """

    im = Image.open(roidb['image'])

    # draw = ImageDraw.Draw(im)
    # for ix, bbox in enumerate(roidb["boxes_full"]):
    #     draw.polygon([(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])], fill="green", outline="red")
    # im.save("/share/DeepWatershedDetection/bbox8.png")

    if im.mode == 'I;16':
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
