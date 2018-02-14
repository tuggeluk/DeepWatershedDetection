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

        if sanatize_coords(objectness.shape, coords):
            objectness[coords[0]:coords[1],coords[2]:coords[3]] = mark

    #debug_energy_maps(data,gt_boxes,objectness)
    return np.expand_dims(np.expand_dims(objectness,-1),0)


def fcn_class_labels(data, gt_boxes):
    fcn_class = np.zeros(data[0].shape[0:2], dtype=np.int16)
    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        mark = np.ones((marker_size[0]*2+1, marker_size[1]*2+1))*row[4]

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        if sanatize_coords(fcn_class.shape, coords):
            fcn_class[coords[0]:coords[1],coords[2]:coords[3]] = mark

    return np.expand_dims(np.expand_dims(fcn_class,-1),0)



def fcn_bbox_labels(data, gt_boxes):
    fcn_bbox = np.zeros(data[0].shape[0:2]+(2,), dtype=np.int16)
    for row in gt_boxes:
        center_1 = np.round((row[2]-row[0])/2 + row[0])
        center_0 = np.round((row[3] - row[1]) / 2 + row[1])

        mark = np.ones((marker_size[0]*2+1, marker_size[1]*2+1,2))
        mark[:, :, 0] = mark[:, :, 0]*(row[2]-row[0])
        mark[:, :, 1] = mark[:, :, 1]*(row[3] - row[1])

        coords = [int(center_0-mark.shape[0]/2), int(center_0+mark.shape[0]/2+1), int(center_1 - mark.shape[1] / 2), int(center_1 + mark.shape[1] / 2)+1]

        if sanatize_coords(fcn_bbox.shape, coords):
            fcn_bbox[coords[0]:coords[1],coords[2]:coords[3]] = mark
    return np.expand_dims(fcn_bbox,0)


def sanatize_coords(canvas_shape, coords):

    if coords[0] < 0 or coords[1] > canvas_shape[0] or coords[2] < 0  or coords[3] > canvas_shape[1]:
        print("skipping marker, coords: " + str(coords) + " img_shape: "+ str(canvas_shape))
        return False
    else:
        # print("ok marker, coords: " + str(coords) + " img_shape: " + str(canvas_shape))
        return True




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