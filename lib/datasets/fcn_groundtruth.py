# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Tuggener
# --------------------------------------------------------


from PIL import Image, ImageDraw
import numpy as np
import random

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

# TODO marker and objectness energy grad
def get_markers():
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

    return None


def stamp_directions(bbox,args):

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

    return None


def stamp_energy(bbox,args):

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

    return None


def stamp_class(bbox, args):
    # for bbox == -1 return dim
    #   Builds gt for objectness energy gradients has the shape of an oval
    #
    #   must be contained by args:
    #   marker_dim:         if it is not None every object will have the same size marker
    #   size_percentage:    percentage of the oval axes w.r.t. to the corresponding bounding boxes, only applied if
    #                       marker_dim is None
    #   hole:               percentage of the oval on which we ignore gt from the center point --> make it a doughnut
    #
    #   return patch, and coords

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