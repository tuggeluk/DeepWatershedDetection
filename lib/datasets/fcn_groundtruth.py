# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Tuggener
# --------------------------------------------------------


from PIL import Image, ImageDraw
import numpy as np
import random
from main.config import cfg
import cv2

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
    orig_coords = np.asarray(coords)
    crop_coords = np.asarray(coords)
    crop_coords = np.maximum(crop_coords, 0)
    crop_coords[0:2] = np.minimum(crop_coords[0:2],canvas_shape[0])
    crop_coords[2:4] = np.minimum(crop_coords[2:4],canvas_shape[1])

    # if a dimension collapses kill element
    if crop_coords[0] == crop_coords[1] or crop_coords[2]==crop_coords[3]:
        return None, None

    reorder_idx = [0,2,1,3]
    mark_subs = np.asarray((0,0)+mark.shape)[reorder_idx] - (orig_coords - crop_coords)
    mark = mark[mark_subs[0]:mark_subs[1],mark_subs[2]:mark_subs[3]]

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
    #   stamp_args:         dict with additional arguments passed on to stamp_func

    # downsample size and bounding boxes
    samp_factor = 1/objectness_settings["ds_factors"][downsample_ind]
    sampled_size = (int(size[1]*samp_factor),int(size[2]*samp_factor))

    sampled_gt = [x*[samp_factor,samp_factor,samp_factor,samp_factor,1] for x in gt]


    print("init canvas")
    last_dim = objectness_settings["stamp_func"][1](-1,objectness_settings["stamp_args"],nr_classes)
    canvas = np.zeros(sampled_size+(last_dim,), dtype=np.int16)

    used_coords = []
    for bbox in sampled_gt:
        stamp, coords = objectness_settings["stamp_func"][1](bbox, objectness_settings["stamp_args"], nr_classes)
        if objectness_settings["overlap_solution"] == "max":
            canvas[coords] = np.max(canvas[coords], stamp)
        elif objectness_settings["overlap_solution"] == "no":
            canvas[coords] = stamp
        elif objectness_settings["overlap_solution"] == "nearest":
            closest_mask = get_closest_mask(coords, used_coords)
            canvas[coords] =  (1-closest_mask)*canvas[coords]+closest_mask*stamp
            used_coords.append(coords)
            # get overlapping bboxes
            # shave off pixels that are closer to another center
        else:
            raise NotImplementedError("overlap solution unkown")

    maps_list.append(canvas)
    # if downsample marker --> use cv2 to downsample gt
    if objectness_settings["downsample_marker"]:
        for x in range(1,len(objectness_settings["ds_factors"])):
            maps_list.append(cv2.resize(canvas,fx=1/x,fy=1/x,interpolation=cv2.INTER_NEAREST))

    # if do not downsample marker, recursively rebuild for each ds level
    else:
        if (downsample_ind+1) == len(objectness_settings["ds_factors"]):
            return maps_list
        else:
            return get_markers(size, gt, nr_classes, objectness_settings, downsample_ind+1, maps_list)

    return maps_list




def get_closest_mask(coords, used_coords):
    # coords format x1,y1,x2,y2
    mask = np.ones((int(coords[3]-coords[1]), int(coords[2]-coords[0])))
    center = [int((coords[3]+coords[1])*0.5),int((coords[2]+coords[0])*0.5)]

    x_coords = np.array(range(coords[0], coords[2]))
    y_coords = np.array(range(coords[1], coords[3]))
    coords_grid = np.stack(np.meshgrid(y_coords, x_coords))

    for used_coord in used_coords:
        used_center = [int((used_coord[3]+used_coord[1])*0.5), int((used_coord[2]+used_coord[0])*0.5)]
        closer_map = obj_closer(coords_grid,center,used_center)
        mask = np.min(mask,closer_map)
    return mask

def obj_closer(grid, pos1, pos2):
    dist1 = np.sqrt(np.square(grid[0] - pos1[0]) + np.square(grid[1] - pos1[1]))
    dist2 = np.sqrt(np.square(grid[0] - pos2[0]) + np.square(grid[1] - pos2[1]))

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
    if bbox == -1:
        return 2

    return None


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

    if args["marker_dim"] is None:
        # use bbox size
        print("use percentage bbox size")
        # determine marker size
        marker_size = (int(args["size_percentage"]*(bbox[3]-bbox[1])),int(args["size_percentage"]*(bbox[2]-bbox[0])))
    else:
        marker_size = args["marker_dim"]

    marker = get_energy_marker(marker_size, args["shape"])

    # apply shape function
    shape_fnc = None
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

    return marker

    if bbox == -1:
        if args["loss"]== "softmax":
            return cfg.TRAIN.MAX_ENERGY
        else:
            return 1
    return None


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
    if bbox == -1:
        if args["class_resolution"]== "binary":
            return 2
        else:
            return nr_classes

    return None


# pil_im = Image.fromarray(im)
def show_image(data, gt_boxes=None, gt=False):
    im = Image.fromarray(data[0].astype("uint8"))
    im.show()

    if gt:
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes:
            draw.rectangle(((row[0],row[1]),(row[2],row[3])), fill="red")
        im.show()
    return