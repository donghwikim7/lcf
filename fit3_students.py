import scipy as sp
import scipy.optimize
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import models


def get_multiple_extrapolations_mean_curve_robust(df):
    model_n =10
    extrap_n = 1
    # model_names = ['pow2', 'pow3', 'log2', 'exp3']
    #model_names = ['pow4', 'pow3', 'pow2', 'log2', 'exp2', 'exp3', 'lin2', 'last1', 'vap3', 'mmf4', 'wbl4', 'exp4',
    # 'expp3', 'ilog2', 'expd3', 'logpower3']
    model_names = ['pow3']
    rows = []
    pbar = tqdm(total=len(pd.unique(df["openmlid"])) * len(pd.unique(df["learner"])) * len(model_names), smoothing=0,
                miniters=1)
    number_of_data_to_process = 0
    for openmlid, df_dataset in df.groupby("openmlid"):

        for learner, df_learner in df_dataset.groupby("learner"):
            if number_of_data_to_process < model_n:
                sizes = sorted(pd.unique(df_learner["size_train"]))
                scores = []
                for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                    sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                    scores.append(
                        [scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
                scores = np.array(scores)
                mean_scores =  np.nanmean(scores, axis=0)
                # sizes, scores = df_seeded["size_train"].values, df_seeded["score_valid"].values
                for i in range(0, len(model_names)):
                    if len(sizes) <= 4+extrap_n:
                        extrap_n = len(sizes)-4
                    for offset in range(4, 4+extrap_n):
                        # for offset in range(4, 6):
                        m = models.Models(model_names[i], np.array(sizes[:offset]), np.array(mean_scores[:offset]),
                                          np.array(sizes), "least_square_lm")
                        m.optimise()
                        rows.append(
                            [openmlid, learner, np.array(sizes)[offset - 1], m.predict(), model_names[i], m.best_beta,
                             mean_scores[offset] - m.predict()[offset],
                             0])
                    pbar.update(1)

                number_of_data_to_process += 1

    pbar.close()
    return pd.DataFrame(rows, columns=["openmlid", "learner", "max_anchor_seen", "prediction", "curve_model", "beta",
                                       "fails_init", "fails_fit"])


def get_anchors_and_scores_mean_curve(df):
    rows = []
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = sorted(pd.unique(df_learner["size_train"]))
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                scores.append([scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
            scores = np.array(scores)
            mean_scores = np.nanmean(scores, axis=0)
            rows.append([openmlid, learner, sizes, mean_scores])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "anchor_prediction", "score"])


def metrics_per_row(row, score, anchor_prediction):
    max_anchor_seen = row.max_anchor_seen
    prediction = row.prediction
    max_anchor = np.max(anchor_prediction)
    percentage_train = max_anchor_seen / max_anchor

    trn_ind = np.argwhere(max_anchor_seen == anchor_prediction)[0][0]  # recover offset
    trn_indices = range(0, (trn_ind + 1))
    tst_indices = range(trn_ind + 1, len(anchor_prediction))
    n_trn = len(trn_indices)

    y_trn_hat = prediction[trn_indices]
    y_trn = score[trn_indices]
    y_tst_hat = prediction[tst_indices]
    y_tst = score[tst_indices]

    bad_score_trn = np.isnan(y_trn)
    bad_score_tst = np.isnan(y_tst)

    y_trn = y_trn[bad_score_trn == False]
    y_trn_hat = y_trn_hat[bad_score_trn == False]

    y_tst = y_tst[bad_score_tst == False]
    y_tst_hat = y_tst_hat[bad_score_tst == False]
    # y_tst_hat = np.where(y_tst_hat > 1, 1.0, y_tst_hat)
    # print(y_tst)
    # print(y_tst_hat)
    MSE_trn = np.mean((y_trn - y_trn_hat) ** 2)
    MSE_tst = np.mean((y_tst - y_tst_hat) ** 2)
    MSE_tst_last = (y_tst[-1] - y_tst_hat[-1]) ** 2
    L1_trn = np.mean((y_trn - y_trn_hat) ** 2)
    L1_tst = np.mean((y_tst - y_tst_hat) ** 2)
    L1_tst_last = (y_tst[-1] - y_tst_hat[-1]) ** 2

    return [MSE_trn, MSE_tst, MSE_tst_last, L1_trn, L1_tst, L1_tst_last, max_anchor_seen, percentage_train, n_trn,
            row.curve_model]


def get_info_mean_curve(df_info, openmlid, learner):
    q = df_info.query('openmlid==@openmlid and learner==@learner')
    q = q.iloc[0, :]
    return [q.anchor_prediction, q.score]


def df_compute_metrics_mean_curve(df, df_info):
    pbar = tqdm(total=len(df))
    rows_metrics = []
    for i in range(0, len(df)):
        row = df.iloc[i, :]
        anchor_prediction, score = get_info_mean_curve(df_info, row.openmlid, row.learner)
        rows_metrics.append(metrics_per_row(row, score, anchor_prediction))
        pbar.update(1)
    pbar.close()
    df_metrics = pd.DataFrame(rows_metrics,
                              columns=['MSE trn', 'MSE tst', 'MSE tst last', 'L1 trn', 'L1 tst', 'L1 tst last',
                                       'max anchor seen', 'percentage', 'n', 'curve_model'])
    return df_metrics


def select_part(part, df_all, datasets):
    num = 20
    indices = range(part * num, part * num + num)
    if part == 9:
        indices = range(part * num, len(datasets))
    datasets_todo = []
    for i in indices:
        datasets_todo.append(datasets[i])
    df_selected = df_all.loc[df_all['openmlid'].isin(datasets_todo)]
    return df_selected


def do_job(part):
    print('starting part %d' % part)

    df_all = pd.read_csv("lcdb_new.csv")
    np.random.seed(42)
    datasets = df_all['openmlid'].unique()
    np.random.shuffle(datasets)

    df_selected = select_part(part, df_all, datasets)

    print('computing extrapolations...')
    df_extrapolations = get_multiple_extrapolations_mean_curve_robust(df_selected)
    df_extrapolations.to_csv('extrapolations%d.csv' % part)
    df_extrapolations.to_pickle('extrapolations%d.p' % part)

    print('computing anchors and scores...')
    df_anchors_and_scores = get_anchors_and_scores_mean_curve(df_selected)
    df_anchors_and_scores.to_csv('anchors_scores%d.csv' % part)
    df_anchors_and_scores.to_csv('anchors_scores%d.p' % part)

    print('computing metrics....')
    df_metrics = df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
    df_metrics.to_csv('metrics%d.csv' % part)
    df_metrics.to_pickle('metrics%d.p' % part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", type=int)
    args = parser.parse_args()
    part = args.part
    do_job(part)


if __name__ == "__main__":
    df_performances = pd.read_csv("lcdb_new.csv")
    df_performances
    df_performances_one_dataset = df_performances.query('openmlid == 6')
    df_performances_one_dataset

    print('computing extrapolations...')
    # compute the fits
    df_extrapolations = get_multiple_extrapolations_mean_curve_robust(df_performances_one_dataset)
    df_extrapolations.to_pickle('extrapolations_example.gz', protocol=3)

    print('computing anchors and scores...')
    # compute the X, Y values and store them for later use
    df_anchors_and_scores = get_anchors_and_scores_mean_curve(df_performances_one_dataset)
    df_anchors_and_scores.to_pickle('anchors_scores_example.gz', protocol=3)

    print('computing metrics....')
    # compute the metrics and other information (L2 losses, etc.)
    df_metrics = df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
    df_metrics.to_pickle('metrics_example.gz', protocol=3)
