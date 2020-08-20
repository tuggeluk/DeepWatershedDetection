import pickle
import pandas as pd
import cv2
import os
import re

def load_results(eval_pickle):
    with open(eval_pickle, 'rb') as input_file:
        results_dict = pickle.load(input_file)
    return results_dict

def choose_tresh(results_dict, tresh):
    for cls_key, tresh_dict in results_dict.items():
        results_dict[cls_key] = results_dict[cls_key][tresh]
    return results_dict

def average_overlaps(results_dict):
    for cls_key, tresh_dict in results_dict.items():
        averaged_dict = {'accuracy': 0,
                         'precision': 0,
                         'recall': 0}
        for _, eval_dict in tresh_dict.items():
            for key in averaged_dict.keys():
                averaged_dict[key] += eval_dict[key]
        for key in averaged_dict.keys():
            averaged_dict[key] = averaged_dict[key] / len(tresh_dict)
        results_dict[cls_key] = averaged_dict


def print_latex(print_dict):
    cols = ('Nr','class', 'No. Occurences', 'mAP', 'AP at 0.5')
    print_df = pd.DataFrame(columns=cols)
    for id, (cls_key, tresh_dict) in enumerate(print_dict.items()):
        averaged_dict = {'ap': 0,
                         'precision': 0,
                         'recall': 0}
        no_occ = tresh_dict['no_occurences']
        del tresh_dict['no_occurences']
        for _, eval_dict in tresh_dict.items():
            for key in averaged_dict.keys():
                averaged_dict[key] += eval_dict[key]
        for key in averaged_dict.keys():
            averaged_dict[key] = averaged_dict[key] / len(tresh_dict)

        print_df = print_df.append(pd.Series([id, cls_key,no_occ, round(averaged_dict['ap'], 3), round(tresh_dict[0.5]['ap'], 3)], index=cols), ignore_index=True)
    print(print_df.to_latex(index=False))
    return print_df

def merge_dfs(to_merge):

    combined_classes = set()
    for df in to_merge:
        combined_classes.update(df['class'])
    combined_classes = list(combined_classes)
    combined_classes.sort()

    cols = ('class', 'No. Occurences', 'mAP', 'AP at 0.5','mAP_2','AP at 0.5_2')
    merged = pd.DataFrame(columns=cols)
    for cla in combined_classes:
        row = [cla, 0]
        for df in to_merge:
            selected = df[df['class'] == cla]
            if len(selected) > 0:
                row[1] = int(selected['No. Occurences'])
                row.extend([float(selected['mAP']), float(selected['AP at 0.5'])])
            else:
                row.extend([0.0, 0.0])
        merged = merged.append(pd.Series(row, index=cols), ignore_index=True)

    # sort by occurences
    print("asdf")
    merged = merged.sort_values(['No. Occurences'], ascending=False)
    merged = merged.reset_index()

    del merged['index']

    return merged

def add_global_averages(df):
    mean_row = ['mean', '-']
    mean_row.extend(list(df.iloc[:, 2:].mean().round(3)))

    w_mean_row = ['weighted mean', '-']
    w_mean_row.extend(list(df.iloc[:, 2:].multiply(df['No. Occurences'] / df['No. Occurences'].sum(), axis="rows").sum().round(3)))

    df = df.append(pd.Series(mean_row, index=df.columns), ignore_index=True)
    df = df.append(pd.Series(w_mean_row, index=df.columns), ignore_index=True)

    print("asd")
    return df

if __name__ == '__main__':
    eval_pickle_dwd = 'evaluation_renamed.pickle'
    results_dict_dwd = load_results(eval_pickle_dwd)
    dwd_df = print_latex(results_dict_dwd)

    eval_pickle_rcnn = 'evaluation_renamed_rcnn.pickle'
    results_dict_rcnn = load_results(eval_pickle_rcnn)
    rcnn_df = print_latex(results_dict_rcnn)

    merged = merge_dfs([dwd_df, rcnn_df])
    merged = add_global_averages(merged)

    print(merged.to_latex())
    halved = pd.concat([merged[:54].reset_index(drop=True), merged[54:].reset_index(drop=True)], axis=1)
    print(halved.to_latex(index=False))