# --------------------------------------------------------
# Test which which datasets are found
# --------------------------------------------------------


from datasets.factory import get_imdb, list_imdbs
from roi_data_layer.layer import RoIDataLayer
import roi_data_layer.roidb as rdl_roidb
from main.config import cfg


def get_training_roidb(imdb):
  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


if __name__ == '__main__':
    print("looking for available datasets")
    #print(list_imdbs())
    #print("looking for coco 2017")
    #get_imdb("coco_2017_train")
    #print("looking for voc 2012")
    #get_imdb("voc_2012_train")
    ds_imdb = get_imdb("deep_scores_2017_train")
    print('Loaded dataset `{:s}` for training'.format(ds_imdb.name))

    roidb = get_training_roidb(ds_imdb)

    data_layer = RoIDataLayer(roidb, ds_imdb.num_classes)

    blobs = data_layer.forward()

