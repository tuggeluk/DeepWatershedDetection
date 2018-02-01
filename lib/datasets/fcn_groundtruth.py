# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Tuggener
# --------------------------------------------------------


from PIL import Image


def objectness_energy(data, gt_boxes):

    im = Image.fromarray(data[0].astype("uint8"))
    return data