# --------------------------------------------------------
# Test which which datasets are found
# --------------------------------------------------------


from datasets.factory import get_imdb, list_imdbs


if __name__ == '__main__':
    print("looking for available datasets")
    #print(list_imdbs())
    #print("looking for coco 2017")
    #get_imdb("coco_2017_train")
    #print("looking for voc 2012")
    #get_imdb("voc_2012_train")
    ds_imdb = get_imdb("deep_scores_2017_train")

    for db in list_imdbs():
        imdb = get_imdb(db)


    print("asdf")
