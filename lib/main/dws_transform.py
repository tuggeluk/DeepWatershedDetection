from PIL import Image
import cv2
import random
from itertools import product
from utils.ufarray import *
import numpy as np

def perform_dws(dws_energy, class_map, bbox_map,cutoff=[7,1],min_ccoponent_size=0, return_ccomp_img = False, store_ccomp_img=False,cfg=None,counter=0):
    bbox_list = []

    dws_energy = np.squeeze(dws_energy)
    class_map = np.squeeze(class_map)
    bbox_map = np.squeeze(bbox_map)


    def merge_markers(base_marker, markers):
        # merges markers form new (lower-cutoff) transform onto base marker if they have no overlap with any base marker
        new_index = np.max(base_marker)+1
        markers_new = markers.copy() # store copy for later
        # offset new markers by 10K (assumes that each page has less than 10K symbols)
        markers = markers * 1E5
        # overlay with base marker
        markers = markers + base_marker
        #check for pure markers > 10k
        tainted_ind = np.unique(np.floor(np.unique(markers)[np.unique(markers) % 1E5 > 0] / 1E5))
        merge_inds = [x for x in np.unique(markers_new) if x not in tainted_ind and not x == 0]
        print("merge down")
        merge_pos = np.in1d(markers_new, merge_inds).reshape(markers_new.shape)
        # merge down
        base_marker[merge_pos] = markers_new[merge_pos]+new_index

        return base_marker

    base_marker = None
    for cut in cutoff:
        # Treshhold and binarize dws energy
        binar_energy = dws_energy.copy()
        binar_energy[binar_energy <= cut] = 0
        binar_energy[binar_energy > cut] = 255

        # asdf = cv2.watershed(binar_energy.astype(np.uint8))
        ret, markers = cv2.connectedComponents(binar_energy.astype(np.uint8), connectivity=8)
        # Image.fromarray(markers.astype(np.uint8)).show()
        # cv2_img = Image.fromarray(markers.astype(np.uint8))
        # cv2_img.save(cfg.ROOT_DIR + "/output_images/inference/"+"cv2out.png")

        # get connected components
        # labels, out_img = find_connected_comp(np.transpose(binar_energy)) # works with inverted indices
        if base_marker is None:
            base_marker = markers
        else:
            base_marker = merge_markers(base_marker, markers)

    if store_ccomp_img:
        out_img = Image.fromarray(base_marker.astype(np.uint8))
        out_img.save(cfg.ROOT_DIR + "/output_images/inference/" + 'ccomp' + str(counter) + '.png')

    # invert labels dict
    # labels_inv = {}
    # for k, v in labels.items():
    #     labels_inv[v] = labels_inv.get(v, [])
    #     labels_inv[v].append(k)

    labels = np.unique(base_marker, return_counts=True)
    filtered_labels = []
    for ind, key in enumerate(labels[0]):
        if key == 0:
            continue
        if labels[1][ind] > min_ccoponent_size:
            filtered_labels.append(key)


    for key in filtered_labels:
        key_coords = np.where(base_marker == key)
        # average off all coordinates as center
        center = np.average(key_coords,1).astype(int)

        # mayority vote for class
        class_label = np.bincount(class_map[key_coords[0], key_coords[1]]).argmax()

        # average for box size
        #labels_inv[key]["bbox_size"] = np.average(bbox_map[labels_inv[key]["pixel_coords"][:, 1], labels_inv[key]["pixel_coords"][:, 0]],0).astype(int)
        bbox_size = np.amax(bbox_map[key_coords[0],key_coords[1]], 0).astype(int)

        # produce bbox element, append to list
        bbox = []
        bbox.append(int(np.round(center[1] - (bbox_size[1]/2.0), 0))) # xmin
        bbox.append(int(np.round(center[0] - (bbox_size[0]/2.0), 0))) # ymin
        bbox.append(int(np.round(center[1] + (bbox_size[1]/2.0), 0))) # xmax
        bbox.append(int(np.round(center[0] + (bbox_size[0]/2.0), 0))) # ymax
        bbox.append(int(class_label))
        bbox_list.append(bbox)

    if return_ccomp_img:
        return bbox_list, out_img
    return bbox_list



def get_class(component,class_map):
    return None

def get_bbox(component,):
    return None

#
# Implements 8-connectivity connected component labeling
#
# Algorithm obtained from "Optimizing Two-Pass Connected-Component Labeling
# by Kesheng Wu, Ekow Otoo, and Kenji Suzuki
#
def find_connected_comp(input):
    data = input
    width, height = input.shape

    # Union find data structure
    uf = UFarray()

    #
    # First pass
    #

    # Dictionary of point:label pairs
    labels = {}

    for y, x in product(range(height), range(width)):

        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #

        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 255:
            pass

        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            labels[x, y] = uf.makeLabel()

    #
    # Second pass
    #

    uf.flatten()

    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:

        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component

        # Associate a random color with this component
        if component not in colors:
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Colorize the image
        outdata[x, y] = colors[component]

    return (labels, output_img)