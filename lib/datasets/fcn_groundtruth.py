# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Tuggener
# --------------------------------------------------------


from PIL import Image, ImageDraw
import numpy as np
import random
from main.config import cfg
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import sys

marker_size = [4,4]

def objectness_energy(data, gt_boxes):
    objectness = np.zeros(data[0].shape[0:2], dtype=np.int16)-10
    mark = objectness_marker(marker_size[0], marker_size[1], func_nothing)

    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        part_marker, part_coords = get_partial_marker(objectness.shape, coords, mark)
        if part_marker is not None:
            objectness[part_coords[0]:part_coords[1],part_coords[2]:part_coords[3]] = part_marker

    #debug_energy_maps(data,gt_boxes,objectness)
    return np.expand_dims(np.expand_dims(objectness,-1),0)


def fcn_class_labels(data, gt_boxes):
    fcn_class = np.zeros(data[0].shape[0:2], dtype=np.int16)
    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        mark = np.ones((marker_size[0]*2+1, marker_size[1]*2+1))*row[4]

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        part_marker, part_coords = get_partial_marker(fcn_class.shape, coords, mark)
        if part_marker is not None:
            fcn_class[part_coords[0]:part_coords[1],part_coords[2]:part_coords[3]] = part_marker

    return np.expand_dims(np.expand_dims(fcn_class,-1),0)


def fcn_foreground(data, gt_boxes):
    fcn_foreground = np.zeros(data[0].shape[0:2], dtype=np.int16)
    for row in gt_boxes:
        # crop to fit data
        x_1 = int(max(0,row[0]))
        x_2 = int(min(data[0].shape[0], row[2]))

        y_1 = int(max(0,row[1]))
        y_2 = int(min(data[0].shape[1], row[3]))

        # set foreground to 1
        fcn_foreground[y_1:y_2, x_1:x_2] = 1

    #show_image(data,gt_boxes,True)

    return np.expand_dims(np.expand_dims(fcn_foreground,-1),0)


def fcn_bbox_labels(data, gt_boxes):
    fcn_bbox = np.zeros(data[0].shape[0:2]+(2,), dtype=np.int16)
    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        mark = np.ones((marker_size[0]*2+1, marker_size[1]*2+1,2))
        mark[:, :, 0] = mark[:, :, 0]*(row[2]-row[0])
        mark[:, :, 1] = mark[:, :, 1]*(row[3] - row[1])

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        part_marker, part_coords = get_partial_marker(fcn_bbox.shape, coords, mark)
        if part_marker is not None:
            fcn_bbox[part_coords[0]:part_coords[1],part_coords[2]:part_coords[3]] = part_marker

    return np.expand_dims(fcn_bbox,0)


def sanatize_coords(canvas_shape, coords):

    if coords[0] < 0 or coords[1] > canvas_shape[0] or coords[2] < 0  or coords[3] > canvas_shape[1]:
        print("skipping marker, coords: " + str(coords) + " img_shape: "+ str(canvas_shape))
        return False
    else:
        # print("ok marker, coords: " + str(coords) + " img_shape: " + str(canvas_shape))
        return True


def get_partial_marker(canvas_shape, coords, mark):


    # if coords are outside of bbox
    crop_coords = np.asarray(coords)
    crop_coords = np.maximum(crop_coords, 0)
    crop_coords[[1,3]] = np.minimum(crop_coords[[1,3]], canvas_shape[0]-1)
    crop_coords[[0,2]] = np.minimum(crop_coords[[0,2]], canvas_shape[1]-1)

    # if a dimension collapses kill element
    if crop_coords[0] == crop_coords[1] or crop_coords[2]==crop_coords[3]:
        return None, None

    # if marker has bigger size than coords
    size_offsets = np.asarray(mark.shape[0:2])-[crop_coords[3]-crop_coords[1],crop_coords[2]-crop_coords[0]]
    # if sum(size_offsets == [0,0])<2:
    #     1==1

    # shrink mark by size_offsets
    mark = mark[int(0+np.ceil(size_offsets[0]/2.0)):int(mark.shape[0]-np.floor(size_offsets[0]/2.0)),
           int(0+np.ceil(size_offsets[1]/2.0)):int(mark.shape[1]-np.floor(size_offsets[1]/2.0))]

    # check size-offsets again
    size_offsets = np.asarray(mark.shape[0:2]) - [crop_coords[3] - crop_coords[1], crop_coords[2] - crop_coords[0]]
    if sum(size_offsets == [0,0])<2:
        1==1


    return mark, crop_coords


def objectness_energy_high_dym(data, gt_boxes, num_classes):
    objectness = np.zeros([data[0].shape[0],data[0].shape[1],num_classes], dtype=np.int16) - 10
    mark = objectness_marker(4, 4, func_nothing)

    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        objectness[int(center_0-mark.shape[0]/2):int(center_0+mark.shape[0]/2+1),
        int(center_1 - mark.shape[1] / 2):int(center_1 + mark.shape[1] / 2)+1, int(row[4])] = mark

    return objectness


def func_nothing(grid):
    return grid


def func_square(grid):
    return np.square(grid)


def debug_energy_maps(data, gt_boxes, orig_object):
    orig_object = orig_object+10
    im = Image.fromarray(orig_object.astype("uint8"))
    im.save("/Users/tugg/cclabel/test_orig.png")

    objectness = np.zeros(data[0].shape[0:2], dtype=np.int16)-10
    mark = objectness_marker(marker_size[0], marker_size[1], func_nothing)
    mark1 = objectness_marker(marker_size[0]+2, marker_size[1]+2, func_nothing)
    mark2 = objectness_marker(marker_size[0]+4, marker_size[1]+4, func_nothing)

    marks = [mark, mark1, mark2]

    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        mark = random.choice(marks)

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        if sanatize_coords(objectness.shape, coords):
            objectness[coords[0]:coords[1],coords[2]:coords[3]] = mark

    objectness = objectness+10
    im = Image.fromarray(objectness.astype("uint8"))
    im.save("/Users/tugg/cclabel/test_diff.png")

    obj_binary = (objectness < 1)*255
    im = Image.fromarray(obj_binary.astype("uint8"))
    im = im.convert("1")
    im.save("/Users/tugg/cclabel/test_diff_bin.png")
    return None


def objectness_marker(sx=3,sy=3,fnc=func_nothing):
    grid_x = np.concatenate((range(sx),range(sx,-1,-1)))
    grid_y = np.concatenate((range(sy),range(sy,-1,-1)))
    exec_grid = np.min(np.stack(np.meshgrid(grid_x, grid_y)),0)
    return fnc(exec_grid)


def get_markers(size, gt, nr_classes, objectness_settings, downsample_ind = 0, maps_list = []):

    #   ds_factors, downsample_marker, overlap_solution, samp_func, samp_args
    #
    #   build objectness gt
    #   size:               size of the largest feature map the gt is built for
    #   ds_factors:         factors of downsampling for which de gt is built
    #   gt:                 list of the ground-truth bounding boxes
    #   downsample_marker:  if True the marker will be downsampled accordingly otherwise the marker has the same size
    #                       for each feature map
    #   overlap_solution:   what to do with overlaps
    #                       "no": the values just get overwritten
    #                       "max": the maximum of both values persists (per pixel)
    #                       "closest": each pixel gets assigned according to its closet center
    #                       --> prints to console if conflicts exist
    #   stamp_func:         function that defines individual markers
    #   stamp_args:         dict with  additional arguments passed on to stamp_func


    # downsample size and bounding boxes
    samp_factor = 1.0/objectness_settings["ds_factors"][downsample_ind]
    sampled_size = (int(size[1]*samp_factor),int(size[2]*samp_factor))

    sampled_gt = [x*[samp_factor,samp_factor,samp_factor,samp_factor,1] for x in gt]


    #init canvas
    last_dim = objectness_settings["stamp_func"][1](None,objectness_settings["stamp_args"],nr_classes)


    if objectness_settings["stamp_args"]["loss"] == "softmax":
        canvas = np.eye(last_dim)[np.zeros(sampled_size, dtype=np.int)]

    else:
        canvas = np.zeros(sampled_size + (last_dim,), dtype=np.float)

    if objectness_settings["stamp_func"][0] == "stamp_energy" and objectness_settings["stamp_args"]["loss"] == "reg":
        canvas = canvas - 10

    used_coords = []
    for bbox in sampled_gt:
        stamp, coords = objectness_settings["stamp_func"][1](bbox, objectness_settings["stamp_args"], nr_classes)
        stamp, coords = get_partial_marker(canvas.shape,coords,stamp)
        if stamp is None:
            continue #skip this element

        if objectness_settings["overlap_solution"] == "max":
            if objectness_settings["stamp_args"]["loss"]=="softmax":
                maskout_map = np.argmax(canvas[coords[1]:coords[3],coords[0]:(coords[2])], -1) < np.argmax(stamp,-1)
                canvas[coords[1]:coords[3], coords[0]:coords[2]][maskout_map] = stamp[maskout_map]
            else:
                canvas[coords[1]:coords[3], coords[0]:coords[2]] = np.expand_dims(np.max(np.concatenate([canvas[coords[1]:coords[3], coords[0]:coords[2]], stamp],-1),-1),-1)
        elif objectness_settings["overlap_solution"] == "no":
            canvas[coords[1]:coords[3], coords[0]:coords[2]] = stamp
        elif objectness_settings["overlap_solution"] == "nearest":
            closest_mask = get_closest_mask(coords, used_coords)
            if objectness_settings["stamp_args"]["loss"] == "softmax":
                closest_mask = np.expand_dims(closest_mask,-1)
                canvas[coords[1]:coords[3],coords[0]:coords[2]] = \
                    (1 - closest_mask) * canvas[coords[1]:coords[3], coords[0]:coords[2]] + closest_mask * stamp
            else:
                if len(closest_mask.shape) < len(stamp.shape):
                    closest_mask = np.expand_dims(closest_mask,-1)

                canvas[coords[1]:coords[3], coords[0]:coords[2]] = (1-closest_mask)*canvas[coords[1]:coords[3], coords[0]:coords[2]] + closest_mask*stamp

            used_coords.append(coords)
            # get overlapping bboxes
            # shave off pixels that are closer to another center
        else:
            raise NotImplementedError("overlap solution unkown")
    #color_map(canvas, objectness_settings, True)
    # set base energ y to minus 10




    # if downsample marker --> use cv2 to downsample gt
    if objectness_settings["downsample_marker"]:
        if objectness_settings["stamp_args"]["loss"] == "softmax":
            sm_dim = canvas.shape[-1]
            canvas_un_oh = np.argmax(canvas, -1).astype("uint8")

            for x in objectness_settings["ds_factors"][1:]:
                shape = (int(np.ceil(float(canvas_un_oh.shape[1]) / x)), int(np.ceil(float(canvas_un_oh.shape[0]) / x)))
                resized = np.round(cv2.resize(canvas_un_oh, shape, cv2.INTER_NEAREST))
                resized_oh = np.eye(sm_dim)[resized[:,:]]
                maps_list.append(np.expand_dims(resized_oh,0))
        else:
            for x in objectness_settings["ds_factors"][1:]:
                # if has 3rd dimension downsample dim3 one by one
                # if len(canvas.shape) == 3:
                #     for dim in range(3):
                #         rez_l = []
                #         rez_l.append()
                # else:
                shape = (int(np.ceil(float(canvas.shape[1]) / x)), int(np.ceil(float(canvas.shape[0]) / x)))
                resized = cv2.resize(canvas, shape, cv2.INTER_NEAREST)
                if len(resized.shape) < len(canvas.shape):
                    maps_list.append(np.expand_dims(np.expand_dims(resized, -1),0))
                else:
                    maps_list.append(np.expand_dims(resized, 0))

        # one-hot encoding
        maps_list.insert(0,np.expand_dims(canvas, 0))
    # if do not downsample marker, recursively rebuild for each ds level
    else:
        if (downsample_ind+1) == len(objectness_settings["ds_factors"]):
            return maps_list
        else:
            return get_markers(size, gt, nr_classes, objectness_settings,downsample_ind+1, maps_list)

    return maps_list


def get_closest_mask(coords, used_coords):
    # coords format x1,y1,x2,y2
    mask = np.ones((int(coords[3]-coords[1]), int(coords[2]-coords[0]))).astype(np.int)
    center = [int((coords[3]+coords[1])*0.5),int((coords[2]+coords[0])*0.5)]

    x_coords = np.array(range(coords[0], coords[2]))
    y_coords = np.array(range(coords[1], coords[3]))
    coords_grid = np.stack(np.meshgrid(x_coords, y_coords))

    for used_coord in used_coords:
        used_center = [int((used_coord[3]+used_coord[1])*0.5), int((used_coord[2]+used_coord[0])*0.5)]
        closer_map = obj_closer(coords_grid,center,used_center)
        mask = np.min(np.stack([mask, closer_map],-1),-1)
    return mask

def obj_closer(grid, pos1, pos2):
    dist1 = np.sqrt(np.square(grid[0] - pos1[1]) + np.square(grid[1] - pos1[0]))
    dist2 = np.sqrt(np.square(grid[0] - pos2[1]) + np.square(grid[1] - pos2[0]))

    return (dist1 < dist2)*1


def stamp_directions(bbox,args,nr_classes):

    #   for bbox == -1 return dim
    #   Builds gt for objectness energy gradients has the shape of an oval
    #
    #   must be contained by args:
    #   marker_dim:         if it is not None every object will have the same size marker
    #   size_percentage:    percentage of the oval axes w.r.t. to the corresponding bounding boxes, only applied if
    #                       marker_dim is None
    #   hole:               percentage of the oval on which we ignore gt (rounded down), from the center point
    #                       --> make it a doughnut
    #
    #   return patch, and coords
    if bbox is None:
        return 2

    if args["marker_dim"] is None:
        # use bbox size
        # determine marker size
        marker_size = (int(args["size_percentage"]*(bbox[3]-bbox[1])),int(args["size_percentage"]*(bbox[2]-bbox[0])))

    else:
        marker_size = args["marker_dim"]

    # transpose marker size bc of wierd bbox definition
    marker_size = np.asarray(marker_size)[[1,0]]

    coords_offset = np.round(([bbox[2]-bbox[0], bbox[3]-bbox[1]] - np.asarray(marker_size))*0.5)
    topleft = np.round([bbox[0]+coords_offset[0],bbox[1]+coords_offset[1]]).astype(np.int)
    coords = [topleft[0],topleft[1],
              topleft[0]+marker_size[0],topleft[1]+marker_size[1]]

    marker = get_direction_marker(marker_size, args["shape"],args["hole"])

    return marker, coords


def get_direction_marker(size, shape, hole):

    # get directions map
    grid_y = range(size[0])
    grid_x = range(size[1])

    # push center slightly off otherwise the norms would not be defined exactly at the center
    center = np.asanyarray(size) * 0.5 + (np.array([0.00001, 0.00001])*random.choice([-1.0, 1.0]))

    marker = np.stack(np.meshgrid(grid_y, grid_x),-1)-center
    # norm last dimension
    marker = marker / np.expand_dims(np.linalg.norm(marker,ord=2,axis=-1),-1)

    center = np.asanyarray(size) * 0.5

    # if smaller than one pixel return
    if np.min(center) < 1:
        return marker

    if shape == "oval":
        # mask out non-oval parts
        y_coords = np.array(range(size[0]))+0.5
        x_coords = np.array(range(size[1]))+0.5
        coords_grid = np.stack(np.meshgrid(y_coords, x_coords))

        oval = np.sqrt(np.square((coords_grid[0] - center[0])/(size[0])) + np.square((coords_grid[1] - center[1])/(size[1])))
        largest = max(oval[int(center[1]),0],oval[0,int(center[0])])
        oval = 1-(oval / float(np.max(largest)))
        marker[oval < 0] = 0


    if hole is not None:
        # punch hole in the middle
        hole_size = np.round(np.asanyarray(size)*hole).astype(np.int)
        hole_center = hole_size * 0.5

        y_coords = np.array(range(hole_size[0]))+0.5
        x_coords = np.array(range(hole_size[1]))+0.5
        coords_grid = np.stack(np.meshgrid(y_coords, x_coords))

        oval = np.sqrt(np.square((coords_grid[0] - hole_center[0])/(hole_size[0])) + np.square((coords_grid[1] - hole_center[1])/(hole_size[1])))
        largest = max(oval[int(hole_center[1]),0],oval[0, int(hole_center[0])])
        oval = 1-(oval / float(np.max(largest)))

        hole_map = np.ones(marker.shape[0:2])*-1
        # place oval at the center
        hole_map[int(center[1]-hole_center[1]):(int(center[1]-hole_center[1])+oval.shape[0]),
        int(center[0]-hole_center[0]):(int(center[0]-hole_center[0])+oval.shape[1])] = oval
        marker[hole_map > 0] = 0

    return marker


def stamp_energy(bbox,args,nr_classes):

    #   for bbox == -1 return dim
    #
    #   Builds gt for objectness energy
    #
    #   must be contained by args:
    #   marker_dim:         if it is not None every object will have the same size marker
    #   size_percentage:    percentage of the oval axes w.r.t. to the corresponding bounding boxes, only applied if
    #                       marker_dim is None
    #   shape:              "oval" or "square" detemines the shape of the energy marker
    #   loss:               softmax, regression
    #   energy_shape:       function that maps from position in patch to objectnes energy i.e.
    #                       (x-x_0,y-y_0)--> R, rounded for loss softmax
    #
    #   return patch, and coords

    if bbox is None:
        if args["loss"]== "softmax":
            return cfg.TRAIN.MAX_ENERGY
        else:
            return 1


    if args["marker_dim"] is None:
        # use bbox size

        # determine marker size
        marker_size = (int(args["size_percentage"]*(bbox[3]-bbox[1])),int(args["size_percentage"]*(bbox[2]-bbox[0])))

    else:
        marker_size = args["marker_dim"]

    # transpose marker size bc of wierd bbox definition
    marker_size = np.asarray(marker_size)[[1,0]]

    coords_offset = np.round(([bbox[2]-bbox[0], bbox[3]-bbox[1]] - np.asarray(marker_size))*0.5)
    topleft = np.round([bbox[0]+coords_offset[0],bbox[1]+coords_offset[1]]).astype(np.int)
    coords = [topleft[0],topleft[1],
              topleft[0]+marker_size[0],topleft[1]+marker_size[1]]

    marker = get_energy_marker(marker_size, args["shape"])

    # apply shape function
    if args["energy_shape"] == "linear":
        1==1 # do nothing
    elif args["energy_shape"] == "root":
        marker = np.sqrt(marker)
        marker = marker/np.max(marker)* (cfg.TRAIN.MAX_ENERGY-1)
    elif args["energy_shape"] == "quadratic":
        marker = np.square(marker)
        marker = marker / np.max(marker) * (cfg.TRAIN.MAX_ENERGY-1)

    if args["loss"]== "softmax":
        marker = np.round(marker).astype(np.int32)
        marker = np.eye(cfg.TRAIN.MAX_ENERGY)[marker[:, :]]
        # turn into one-hot softmax targets
    else:
        # expand last dim
        marker = np.expand_dims(marker,-1)


    return marker, coords


def get_energy_marker(size, shape):
    if shape == "square":
        sy = size[0] / 2
        sx = size[1] / 2
        if size[0]%2 == 1:
            grid_y = np.concatenate((range(sy),range(sy,-1,-1)))
        else:
            grid_y = np.concatenate((range(sy), range(sy-1, -1, -1)))

        if size[1]%2 == 1:
            grid_x = np.concatenate((range(sx), range(sx, -1, -1)))
        else:
            grid_x = np.concatenate((range(sx), range(sx-1, -1, -1)))


        marker = np.min(np.stack(np.meshgrid(grid_x, grid_y)),0)

        if min(marker.shape) > 0:
            marker = marker/float(np.max(marker))*(cfg.TRAIN.MAX_ENERGY-1)
        return marker
    if shape == "oval":
        center = np.asanyarray(size) *0.5

        y_coords = np.array(range(size[0]))+0.5
        x_coords = np.array(range(size[1]))+0.5
        coords_grid = np.stack(np.meshgrid(y_coords, x_coords))

        marker = np.sqrt(np.square((coords_grid[0] - center[0])/(size[0])) + np.square((coords_grid[1] - center[1])/(size[1])))
        largest = max(marker[int(center[1]),0],marker[0,int(center[0])])
        marker = 1-(marker / float(np.max(largest)))
        marker[marker < 0] = 0
        marker = marker * (cfg.TRAIN.MAX_ENERGY-1)
        return marker
    return None


def stamp_class(bbox, args, nr_classes):

    #   for bbox == -1 return dim
    #
    #   Builds gt for class prediction
    #
    #   must be contained by args:
    #   marker_dim:         if it is not None every object will have the same size marker
    #   size_percentage:    percentage of the oval axes w.r.t. to the corresponding bounding boxes, only applied if
    #                       marker_dim is None
    #   shape:              square or oval
    #   class_resolution:   "binary" for background/foreground or "class"
    #
    #   return patch, and coords
    if bbox is None:
        if args["class_resolution"]== "binary":
            return 2
        else:
            return nr_classes

    if args["marker_dim"] is None:
        # use bbox size
        # determine marker size
        marker_size = (int(args["size_percentage"] * (bbox[3] - bbox[1])), int(args["size_percentage"] * (bbox[2] - bbox[0])))
    else:
        marker_size = args["marker_dim"]


    coords_offset = np.round(([bbox[3]-bbox[1],bbox[2]-bbox[0]] - np.asarray(marker_size))*0.5)
    topleft = np.round([bbox[0]+coords_offset[1],bbox[1]+coords_offset[0]]).astype(np.int)
    coords = [topleft[0],topleft[1],
              topleft[0]+marker_size[1],topleft[1]+marker_size[0]]


    # transpose bc of wierd bbox definition
    marker_size = np.asarray(marker_size)[[1, 0]]

    # piggy back on energy marker
    marker = get_energy_marker(marker_size, args["shape"])

    if args["class_resolution"] == "class":
        marker[marker != 0] = int(bbox[4])
        marker = marker.astype(np.int)
        # turn into one-hot softmax targets
        marker = np.eye(nr_classes)[marker[:, :]]
    elif args["class_resolution"] == "binary":
        marker[marker != 0] = 1
        marker = marker.astype(np.int)
        # turn into one-hot softmax targets
        marker = np.eye(2)[marker[:, :]]

    else:
        raise NotImplementedError("unknown class resolution:"+args["class_resolution"])

    return marker, coords


def stamp_bbox(bbox, args, nr_classes):

    #   for bbox == -1 return dim
    #
    #   Builds gt for class prediction
    #
    #   must be contained by args:
    #   marker_dim:         if it is not None every object will have the same size marker
    #   size_percentage:    percentage of the oval axes w.r.t. to the corresponding bounding boxes, only applied if
    #                       marker_dim is None
    #   shape:              square or oval
    #   class_resolution:   "binary" for background/foreground or "class"
    #
    #   return patch, and coords
    if bbox is None:
        return 2

    if args["marker_dim"] is None:
        # use bbox size
        # determine marker size
        marker_size = (int(args["size_percentage"] * (bbox[3] - bbox[1])), int(args["size_percentage"] * (bbox[2] - bbox[0])))
    else:
        marker_size = args["marker_dim"]


    coords_offset = np.round(([bbox[3]-bbox[1],bbox[2]-bbox[0]] - np.asarray(marker_size))*0.5)
    topleft = np.round([bbox[0]+coords_offset[1],bbox[1]+coords_offset[0]]).astype(np.int)
    coords = [topleft[0],topleft[1],
              topleft[0]+marker_size[1],topleft[1]+marker_size[0]]

    # transpose bc of wierd bbox definition
    marker_size = np.asarray(marker_size)[[1, 0]]

    # piggy back on energy marker
    marker = get_energy_marker(marker_size, args["shape"])

    # expand marker and multiply by bbox size
    marker = np.expand_dims(marker,-1)
    bbox_marker = np.concatenate([(bbox[3] - bbox[1])*(marker>0), (bbox[2] - bbox[0])*(marker>0)], -1)
    return bbox_marker, coords


def try_all_assign(data_layer, args, nr_tries = 1):
    for assign in args.training_assignements:
        for i in range(nr_tries):
            print(i)
            batch_not_loaded = True
            while batch_not_loaded:
                data = data_layer.forward(args, assign,None)
                batch_not_loaded = len(data["gt_boxes"].shape) != 3

            fetched_maps =[]
            for key in data.keys():
                if "map" in key:
                    fetched_maps.append(data[key])

            get_gt_visuals(data, assign, data["gt_boxes"][0], show=True)

    return None



def color_map(img_map, assign,show=False):
    if assign["stamp_func"][0] == "stamp_energy":
        if assign["stamp_args"]["loss"] == "softmax":
            img_map = np.argmax(img_map, -1)
        else:
            img_map = np.squeeze(img_map,-1)
            # normalize to 0 to 19
            img_map = img_map-np.min(img_map)
            img_map = img_map/np.max(img_map)*19

            img_map = img_map.astype(np.int)


        colors = np.asarray(cm.rainbow(np.linspace(0, 1, 20)))[:, 0:3]

        colored_map = (colors[img_map, :] * 255).astype(np.uint8)

    if assign["stamp_func"][0] == "stamp_class":

        nr_classes = img_map.shape[-1]
        img_map = np.argmax(img_map, -1)

        colors = np.asarray(cm.nipy_spectral(np.linspace(0, 1, nr_classes)))[:, 0:3]
        colored_map = (colors[img_map, :] * 255).astype(np.uint8)

    if assign["stamp_func"][0] == "stamp_directions":
        nonzeros = img_map != 0

        # Normalize
        img_map = img_map / np.expand_dims(np.linalg.norm(img_map, axis=2), -1)

        #apply arccos along last dimension
        img_map[nonzeros] = np.arccos(img_map[nonzeros])
        img_map = np.round(img_map/math.pi*255)
        # add zero B channel
        img_map = np.concatenate([img_map, np.zeros(img_map.shape[:-1]+(1,))],-1)
        colored_map = img_map.astype(np.uint8)
    if assign["stamp_func"][0] == "stamp_bbox":
        size_map = np.sqrt(np.maximum(img_map[:,:,0]*img_map[:,:,1],0))
        size_map = size_map/np.max(size_map)*255
        colored_map = size_map.astype(np.uint8)
        # make colored map have last dim 3
        colored_map = np.stack([colored_map,colored_map,colored_map],-1)



    if show:
        im = Image.fromarray(colored_map)
        im.save(sys.argv[0][:-24]+"gt.png")
        im.show()
    return colored_map


def overlayed_image(image,gt_boxes,pred_boxes,fill=False,show=False):
    # add mean
    #image += cfg.PIXEL_MEANS[0][0][[0,1,2]]
    if len(image.shape)==3 and  image.shape[2]>=3:
        # image = image[:,:,[2,1,0]] # Switch to rgb
        im = Image.fromarray(image.astype("uint8"))
    else:
        im = Image.fromarray(np.squeeze(image.astype("uint8"),-1))

    if fill:
        outline = [None,None]
        fill = ["red","green"]
    else:
        outline = ["red","green"]
        fill = [None,None]

    draw = ImageDraw.Draw(im)
    if gt_boxes is not None:
        for row in gt_boxes:
            draw.rectangle(((row[0], row[1]), (row[2], row[3])), fill=fill[0], outline=outline[0])

    if pred_boxes is not None:
        for row in pred_boxes:
            draw.rectangle(((row[0], row[1]), (row[2], row[3])), fill=fill[1], outline=outline[1])
    if show:
        im.save(sys.argv[0][:-17] + "inp.png")
        im.show()
    return np.asarray(im).astype("uint8")


def get_gt_visuals(data,assign,assign_nr,pred_boxes=None,show=False):
    vis = []
    for i in range(len(assign["ds_factors"])):
        img_map = data["assign"+str(assign_nr)]["gt_map" + str(i)]
        colored_map = color_map(img_map[0],assign,show)
        vis.append(colored_map)

    colored_gt = overlayed_image(image=data["data"][0],gt_boxes=data["gt_boxes"][0],pred_boxes=pred_boxes,fill=False,show=show)
    vis.append(colored_gt)
    return vis


def get_map_visuals(fetched_maps,assign,show=False):
    vis = []
    for fetched_map in fetched_maps:
        colored_map = color_map(fetched_map[0],assign,show)
        vis.append(colored_map)

    return vis


# pil_im = Image.fromarray(im)
def show_image(data, gt_boxes=None):
    from PIL import Image, ImageDraw
    if gt_boxes is None:
        im = Image.fromarray(data[0].astype("uint8"))
        im.show()
    else:
        im = Image.fromarray(data[0].astype("uint8"))
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes[0]:
            draw.rectangle(((row[0],row[1]),(row[2],row[3])), fill="red")
        im.show()
    return