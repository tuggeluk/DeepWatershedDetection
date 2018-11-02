from main.train_dwd import main
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Holds arguments to run DWD on pascal voc

    parser.add_argument("--scale_list", type=list, default=[0.9,1,1.1], help="global scaling factor randomly chosen from this list")
    parser.add_argument("--crop", type=str, default="False", help="should images be cropped")
    parser.add_argument("--crop_size", type=bytearray, default=[160,160], help="size of the image to be cropped to")
    parser.add_argument("--crop_top_left_bias", type=float, default=0.3, help="fixed probability that the crop will be from the top left corner")
    parser.add_argument("--max_edge", type=int, default=1280, help="if there is no cropping - scale such that the longest edge has this size")
    parser.add_argument("--use_flipped", type=str, default="False", help="wether or not to append Horizontally flipped images")
    parser.add_argument("--substract_mean", type=str, default="True", help="wether or not to substract the mean of the VOC images")
    parser.add_argument("--pad_to", type=int, default=160, help="pad the final image to have edge lengths that are a multiple of this - use 0 to do nothing")
    parser.add_argument("--pad_with", type=int, default=0,help="use this number to pad images")

    parser.add_argument("--prefetch", type=str, default="False", help="use additional process to pre-fetch batches")
    parser.add_argument("--prefetch_len", type=int, default=2, help="prefetch queue len")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training") # code only works with batchsize 1!
    parser.add_argument("--continue_training", type=str, default="False", help="continue from latest checkpoint")
    parser.add_argument("--pretrain_lvl", type=str, default="class", help="What kind of pretraining to use: no,class,semseg")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
    parser.add_argument("--dataset", type=str, default="voc_2012_train", help="Dataset to be used in training")
    parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")
    parser.add_argument("--print_interval", type=int, default=100, help="after how many iterations is tensorboard updated")
    parser.add_argument("--tensorboard_interval", type=int, default=50, help="after how many iterations is tensorboard updated")
    parser.add_argument("--save_interval", type=int, default=1000, help="after how many iterations are the weights saved")
    parser.add_argument("--nr_classes", type=list, default=[],help="ignore, will be overwritten by program")

    parser.add_argument('--model', type=str, default="RefineNet-Res101", help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152")


    parser.add_argument('--training_help', type=list, default=[None,
                                                               {"level": "bbox", "samp_prob": 1.0},
                                                               {"level": "bbox", "samp_prob": 0.75},
                                                               {"level": "bbox", "samp_prob": 0.5},
                                                               {"level": "bbox", "samp_prob": 0.25}], help="Randomly add ground truth to the input")

    parser.add_argument('--training_assignements', type=list,
                        default=[
    # direction markers 0.3 to 0.7 percent, downsample
    #                         {'ds_factors': [1,8,16,32], 'downsample_marker': True, 'overlap_solution': 'nearest',
    #                          'stamp_func': 'stamp_directions', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
    #                          'stamp_args': {'marker_dim': None, 'size_percentage': 0.7,"shape": "oval", 'hole': None, 'loss': "reg"}},
    # energy markers
                            {'ds_factors': [1,8,16,32], 'downsample_marker': False, 'overlap_solution': 'max',
                                 'stamp_func': 'stamp_energy', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
                                 'stamp_args':{'marker_dim': (8,8),'size_percentage': 0.8, "shape": "oval", "loss": "softmax", "energy_shape": "linear"}}
    # class markers 0.8% - size-downsample
    #                         {'ds_factors': [1,8,16,32], 'downsample_marker': True, 'overlap_solution': 'nearest',
    #                          'stamp_func': 'stamp_class', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
    #                          'stamp_args': {'marker_dim': None, 'size_percentage': 0.8, "shape": "square", "class_resolution": "class", "loss": "softmax"}}

                        ],help="a list of training_assignements, which are modes in which the network is trained")

    parser.add_argument('--do_assign', type=list,
                        default=[
                            {"assign": 0, "help": 0, "Itrs": 1000},
                            {"assign": 0, "help": 2, "Itrs": 1000},
                            {"assign": 0, "help": 3, "Itrs": 1000},
                            {"assign": 0, "help": 4, "Itrs": 1000},
                            {"assign": 1, "help": 0, "Itrs": 1000},


                        ], help="configure order and iterations of above defined assignements")

    parser.add_argument('--combined_assignements', type=list,
                        default=[],help="tuples of assignements to be simultaniousely optimized")

    parsed = parser.parse_known_args()

    main(parsed)
