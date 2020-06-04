# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/DeepWatershedDetection/lib')

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.deep_scores import deep_scores
from datasets.deep_scoresV2 import deep_scoresV2
from datasets.deep_scores_300dpi import deep_scores_300dpi
from datasets.deep_scores_ipad import deep_scores_ipad
from datasets.musicma import musicma
from datasets.dota import dota

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up DeepScores dataset
for year in ['2017']:
  for split in ['train', 'val', 'test', 'debug','train100','train10000', 'test100']:
    name = 'DeepScores_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: deep_scores(split, year))

# Set up DeepScores dataset
for year in ['2020']:
  for split in ['train', 'val']:
    name = 'DeepScoresV2_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: deep_scoresV2(split, year))

# Set up DeepScores_300dpi dataset
for year in ['2017']:
  for split in ['train', 'val', 'test', 'debug']:
    name = 'DeepScores_300dpi_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: deep_scores_300dpi(split, year))

# Set up DeepScores_300dpi dataset
for year in ['2017']:
  for split in ['train', 'val', 'test', 'debug']:
    name = 'DeepScores_ipad_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: deep_scores_ipad(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up for Musicma++
for year in ['2017']:
  for split in ['train', 'test', 'val']:
    name = 'MUSICMA++_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: musicma(split, year))

# Set up coco_2014_<split>
for year in ['2017']:
  for split in ['train', 'val']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up for Dota_2018_<split>
for year in ['2018']:
  for split in ['train', 'val', 'test', 'debug']:
    name = 'Dota_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: dota(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  print(name)
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  print(__sets[name]())
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
