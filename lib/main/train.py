import os
import sys
sys.path.append(os.path.abspath('../'))
import train_dwd as dwd
import argparse
import numpy.random as ran
import random
import pdb
import os.path as osp


def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--random_seed", type=int, default=317,
                        help="init randon number generator")

    root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    parser.add_argument("--root_dir", type=str, default=root_dir,
                        help="directory of the project")

    parser.add_argument("--data_dir", type=str, default=osp.abspath(osp.join(root_dir, 'data')),
                        help="directory of the project")

    parser.add_argument("--pretrained_dir", type=str, default=osp.abspath(osp.join(root_dir, 'pretrained')),
                        help="directory of the project")

    parser.add_argument("--exp_dir", type=str, default=osp.abspath(osp.join(root_dir, 'experiments')),
                        help="directory of the project")

    parser.add_argument("--out_dir", type=str, default=osp.abspath(osp.join(root_dir, 'output')),
                        help="directory of the project")


    parser.add_argument("--scale_list", type=list, default=[1],
                        help="global scaling factor randomly chosen from this list")
    parser.add_argument("--crop", type=str, default="True", help="should images be cropped")
    parser.add_argument("--crop_top_left_bias", type=float, default=0.3,
                        help="fixed probability that the crop will be from the top left corner")
    augmentation_type = 'none'  # none - no augmentation, up - augmentation on the upper side of the image, full - only augmentation, full synthetic image
    if augmentation_type == 'full':
        parser.add_argument("--augmentation_type", type=str, default=augmentation_type,
                            help="Type of augmentation, none = train on real data; full = train on synthetic data")
        parser.add_argument("--max_edge", type=int, default=0,
                            help="if there is no cropping - scale such that the longest edge has this size / if there is cropping crop to max_edge * max_edge")
    elif augmentation_type == 'up' or augmentation_type == 'none':
        parser.add_argument("--augmentation_type", type=str, default=augmentation_type,
                            help="Augment synthetic data at the top of  the image")
        parser.add_argument("--max_edge", type=int, default=256*3,
                            help="if there is no cropping - scale such that  the longest edge has this size / if there is cropping crop to max_edge * max_edge")
    parser.add_argument("--use_flipped", type=str, default="False",
                        help="wether or not to append Horizontally flipped images")
    parser.add_argument("--substract_mean", type=str, default="False",
                        help="wether or not to substract the mean of the VOC images")
    parser.add_argument("--pad_to", type=int, default=0,
                        help="pad the final image to have edge lengths that are a multiple of this - use 0 to do nothing")
    parser.add_argument("--pad_with", type=int, default=0, help="use this number to pad images")

    parser.add_argument("--prefetch", type=str, default="False", help="use additional process to fetch batches")
    parser.add_argument("--prefetch_len", type=int, default=20, help="prefetch queue len")
    parser.add_argument("--prefetch_proc", type=int, default=2, help="how many processes should be spawned to prefetch batches")
    parser.add_argument("--prefetch_cache_dir", type=str, default="../../data/cache/prefetch_chunks", help="where to store the used cache chunks for next execution")
    parser.add_argument("--prefetch_size", type=int, default=40, help="number of batches stored in one chunk")


    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size for training")

    parser.add_argument("--continue_training", type=str, default="True", help="load checkpoint")
    parser.add_argument("--pretrain_lvl", type=str, default="class",
                        help="What kind of pretraining to use: no,class,semseg, DeepScores_to_300dpi")
    learning_rate = 1e-4  # rnd(3, 5) # gets a numb er (log uniformly) on interval 10^(-3) to 10^(-5)
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate for the Optimizer")
    optimizer = 'rmsprop'  # at the moment it supports only 'adam', 'rmsprop' and 'momentum'
    parser.add_argument("--optim", type=str, default=optimizer, help="type of the optimizer")
    regularization_coefficient = 6.1e-4  # rnd(3, 6) # gets a number (log uniformly) on interval 10^(-3) to 10^(-6)
    parser.add_argument("--regularization_coefficient", type=float, default=regularization_coefficient,
                        help="Value for regularization parameter")

    dataset = "macrophages_2019_train"
    if dataset == "DeepScores_2017_debug":
        parser.add_argument("--dataset", type=str, default="DeepScores_2017_debug", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug",
                            help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="DeepScores_2017_test", help="dataset to perform inference on")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "DeepScores_300dpi_2017_train":
        parser.add_argument("--dataset", type=str, default="DeepScores_300dpi_2017_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="DeepScores_300dpi_2017_val", help="dataset to perform inference on, we use val for evaluation, test can be used only visually")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "DeepScores_ipad_2017_train":
        parser.add_argument("--dataset", type=str, default="DeepScores_ipad_2017_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "MUSICMA++_2017_train":
        parser.add_argument("--dataset", type=str, default="MUSICMA++_2017_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_test", help="dataset to perform inference on")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "Dota_2018_train":
        parser.add_argument("--dataset", type=str, default="Dota_2018_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="Dota_2018_valid", help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="Dota_2018_valid", help="--> test can only used visually")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "voc_2012_train":
        parser.add_argument("--dataset", type=str, default="voc_2012_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="voc_2012_val", help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="voc_2012_val", help="dataset to perform inference on, voc_2012_val/voc_2012_train")

        parser.add_argument("--paired_data", type=int, default=1, help="is data paired? use 1 for unpaired")
    elif dataset == "macrophages_2019_train":
        parser.add_argument("--dataset", type=str, default="macrophages_2019_train", help="DeepScores, voc or coco")
        parser.add_argument("--dataset_validation", type=str, default="macrophages_2019_test", help="DeepScores, voc, coco or no - validation set")
        parser.add_argument("--test_set", type=str, default="macrophages_2019_test", help="dataset to perform inference on")

        parser.add_argument("--paired_data", type=int, default=2, help="is data paired? use 1 for unpaired")
    else:
        raise ValueError("This dataset is not supported, the only supported datasets are DeepScores_2017_train, DeepScores_300dpi_2017_train, DeepScores_ipad_2017_train, MUSICMA++_2017_train, "
                         "Dota_2018_train and voc_2012_train. Are you sure that you are using the correct dataset?")

    parser.add_argument("--use_all_gt", type=str, default="True",
                        help="also include iscrowd tagged gt from COCO")

    parser.add_argument("--print_interval", type=int, default=20,
                        help="after how many iterations the loss is printed to console")

    parser.add_argument("--tensorboard_interval", type=int, default=100,
                        help="after how many iterations is tensorboard updated")

    parser.add_argument("--validation_loss_task", type=int, default=100,
                        help="Compute validation loss on current task")

    parser.add_argument("--validation_loss_task_nr_batch", type=int, default=2,
                        help="batch size for validation loss estimation")

    parser.add_argument("--validation_loss_final", type=int, default=1000000,
                        help="Compute validation loss on bounding box level")

    parser.add_argument("--save_interval", type=int, default=500,
                        help="after how many iterations are the weights saved")


    parser.add_argument("--nr_classes", type=list, default=[], help="ignore, will be overwritten by program")
    parser.add_argument("--semseg_ind", type=list, default=[], help="ignore, will be overwritten by program")


    parser.add_argument('--model', type=str, default="RefineNet-Res101",
                        help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152"
                             "                                  UNet")

    parser.add_argument('--training_help',  type=list, default=[None], help="sample gt into imput / currently unused")

    parser.add_argument('--individual_upsamp', type=str, default="sub_task", help="use individual refine-Net heads per task (No,task,sub_task)")
    parser.add_argument('--sparse_heads', type=str, default="True", help="only initialize used heads (True/False)")


    parser.add_argument('--training_assignements', type=list,
                        default=[

                            # energy markers
                            {'ds_factors': [1], 'downsample_marker': True, 'overlap_solution': 'max',
                             'stamp_func': 'stamp_energy', 'layer_loss_aggregate': 'avg',
                             'stamp_args': {'marker_dim': None, 'size_percentage': 1, "shape": "oval",
                                            "loss": "softmax", "energy_shape": "linear"},
                             'balance_mask': 'fg_bg_balanced', # by_class, by_object, fg_bg_balanced, mask_bg, None
                             'balance_coef': 0.25,  # % of loss given to background
                             'use_obj_seg': True, # use object segmentation if available
                             'use_obj_seg_cached': True
                             },

                            # # class markers
                            {'ds_factors': [1],  'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_class', 'layer_loss_aggregate': 'avg',
                             'stamp_args': {'marker_dim': None, 'size_percentage': 1, "shape": "oval",
                                            "class_resolution": "class", "loss": "softmax"},
                             'balance_mask': 'mask_bg',
                             'use_sem_seg': True # use semantic segmentation if avialable
                             },

                            # bbox markers
                            {'ds_factors': [1],  'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_bbox', 'layer_loss_aggregate': 'avg',
                             'stamp_args': {'marker_dim': None, 'size_percentage': 1, "shape": "oval", "loss": "reg"},
                             'balance_mask': 'mask_bg'
                             }

                        ], help="configure how groundtruth is built, see datasets.fcn_groundtruth")

    parser.add_argument('--max_energy', type=int, default=20, help="energy level for center")

    parser.add_argument('--bbox_estimation', type=str, default="energy_shape", help="compute bbox by estimating from energy or by bbox layer, (bbox_head/energy_shape)")
    parser.add_argument('--bbox_angle', type=str, default="aligned", help="does the bbox have to be parallel to the image (aligned/estimated)")
    parser.add_argument('--class_estimation', type=str, default="energy_foreground", help="compute class by estimating from energy (only works for one class) or by bbox layer, (class_head/energy_foreground)")



    Itrs0, Itrs1, Itrs2, Itrs0_1, Itrs_combined = 5, 5, 5, 5, 1000000
    parser.add_argument('--do_assign', type=list,
                        default=[
                            {"assign": 0, "help": 0, "Itrs": Itrs0},
                            {"assign": 1, "help": 0, "Itrs": Itrs1},
                            #{"assign": 2, "help": 0, "Itrs": Itrs2},
			                {"assign": 0, "help": 0, "Itrs": Itrs0_1}
                         ], help="configure how assignements get repeated")

    # current prefetcher only works with TRUE
    parser.add_argument('--train_only_combined', type=str, default="True", help="only initialze opt for combined task (save memory space)")

    # when assigned in both overrides stuff defined in assign
    parser.add_argument('--combined_assignements', type=list,
                        default=[{"assigns": [0,2], "loss_factors": [3,0], "pair_balancing":[[3,1],[1,1]],   "Running_Mean_Length": 5, "Itrs": Itrs_combined},
                                 {"assigns": [0,2], "loss_factors": [2,1],   "Running_Mean_Length": 5, "Itrs": Itrs_combined},
                                 ],help="configure how groundtruth is built, see datasets.fcn_groundtruth")
    
    dict_info = {'augmentation': augmentation_type, 'learning_rate': learning_rate, 'Itrs_energy': Itrs0, 'Itrs_class': Itrs1, 'Itrs_bb': Itrs2, 'Itrs_energy2': Itrs0_1, 'Itrs_combined': Itrs_combined,
		 'optimizer': optimizer, 'regularization_coefficient': regularization_coefficient}
    parser.add_argument("--dict_info", type=dict, default=dict_info, help="a dictionary containing information about some of the hyperparameters")
    parsed = parser.parse_known_args()
    dwd.main(parsed)


if __name__ == '__main__':
    main()
