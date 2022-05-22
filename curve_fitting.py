import scipy as sp
import scipy.optimize
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import models
from mpi4py import MPI
from helpers import get_anchors_and_scores_mean_curve, convert_table2, remove_rows_with_nan_or_inf, load_from_parts, remove_bad_fits, failed_fits_statistics,  prepare_total_dataframe, plot_data2, plot_trn_data2, plot_prediction2, plot_prediction_smooth2, get_fun_model_id, get_XY2, load_from_parts_example
from helpers import load_from_parts_example, get_anchors_and_scores_mean_curve, remove_fails, remove_nan_and_inf, remove_performace_too_bad, get_info_mean_curve, convert_table2, remove_rows_with_nan_or_inf, wilcoxon_holm, determine_plotting, graph_ranks, convert_to_cd_tables, prepare_data_for_cd, get_ranks_from_tables, build_rank_table, print_pretty_rank_table_transpose, filter_table, from_tables_print_pretty_rank_table, make_all_cd_plots, plot_data, plot_trn_data, plot_prediction_smooth, plot_prediction, get_curve_models, get_curve_model, load_from_parts_example_gathered_result
from scipy.cluster.vq import vq, kmeans2, whiten, kmeans
import pickle

def flatten(arr):
    arrnew = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arrnew.append(arr[i][j])
    return np.array(arrnew)

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


class CurveFitting:
    def __init__(self, df, model_ids=None, extrapolation_n=3, training_model_n=5, only_full_size= False, initailparams=None):
        self.df = df
        if model_ids is None:
            model_ids = ['pow4', 'pow3', 'pow2', 'log2', 'exp2', 'exp3', 'lin2', 'last1', 'vap3', 'mmf4', 'wbl4',
                         'exp4',
                         'expp3', 'ilog2', 'expd3', 'logpower3']
        self.model_ids = model_ids
        self.extrapolation_n = extrapolation_n
        self.training_model_n = training_model_n
        self.only_full_size = only_full_size
        self.initailparams = initailparams

    def k_mean_centroids(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        arr = []
        for m in self.model_ids:
            cf = None
            if rank == 0:
                df_performances = pd.read_csv("lcdb_new.csv")
                df_performances_one_dataset = df_performances.query(
                    '(openmlid ==  6| openmlid ==  1468) ==False')  # df_performances
                # df_performances_one_dataset = df_performances
                cf = CurveFitting(df_performances_one_dataset, model_ids=[m],
                                  extrapolation_n=300, training_model_n=12, only_full_size=True)
                print('computing extrapolations...')
                # compute the fits
            cf = comm.bcast(cf, root=0)
            df_extrapolations = cf.get_multiple_extrapolations_mean_curve_robust()
            if rank == 0:
                df_extrapolations.to_pickle('extrapolations_example.gz', protocol=3)

                print('computing anchors and scores...')
                # compute the X, Y values and store them for later use
                df_anchors_and_scores = cf.get_anchors_and_scores_mean_curve()
                df_anchors_and_scores.to_pickle('anchors_scores_example.gz', protocol=3)

                print('computing metrics....')
                # compute the metrics and other information (L2 losses, etc.)
                df_metrics = cf.df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
                df_metrics.to_pickle('metrics_example.gz', protocol=3)
            if rank ==0:
                [df_anchors_and_scores, df_metrics, df_extrapolations] = load_from_parts_example()
                df_total = prepare_total_dataframe(df_anchors_and_scores, df_metrics, df_extrapolations)
                dff = df_total[df_total['curve_model'] == m]
                if len(dff[dff['MSE_trn'] <= 0.0025])>15:
                    dff = dff[dff['MSE_trn'] <= 0.0025]
                djj = np.array(dff[['beta']]).flatten()
                djj = (np.stack(djj).astype(None))
                whitened = whiten(djj)
                a = whitened[0] / djj[0]
                # print(a)
                z = kmeans(whitened, 15)
                arr.append({'model': m, 'centroids' : ((z[0]*( djj[0]/whitened[0])))})
        if rank == 0:
            with open('james.p', 'wb') as file:
                pickle.dump(arr, file)
        return True


    def get_multiple_extrapolations_mean_curve_robust(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        rank = comm.Get_rank()
        model_n = self.training_model_n
        extrap_n = self.extrapolation_n
        df = self.df
        rows = []
        if rank == 0:
            pbar = tqdm(total=len(pd.unique(df["openmlid"])) * len(pd.unique(df["learner"])) * len(self.model_ids),
                        smoothing=0,
                        miniters=1)
        number_of_data_to_process = 0
        for openmlid, df_dataset in df.groupby("openmlid"):
            arr1 = np.array([])
            for learner, df_learner in df_dataset.groupby("learner"):
                arr1 = np.append(arr1, learner)
            arr1 = np.array_split(arr1[:model_n], size)
            # print(arr1)
            for learner, df_learner in df_dataset.groupby("learner"):
                if learner in arr1[rank]:
                    sizes = sorted(pd.unique(df_learner["size_train"]))
                    scores = []
                    for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                        sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                        scores.append(
                            [scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
                    scores = np.array(scores)
                    mean_scores = np.nanmean(scores, axis=0)
                    # sizes, scores = df_seeded["size_train"].values, df_seeded["score_valid"].values
                    for i in range(0, len(self.model_ids)):
                        if len(sizes) <= 4 + extrap_n:
                            extrap_n = len(sizes) - 4
                        if self.only_full_size:
                            offset = len(sizes)-1
                            m = models.Models(self.model_ids[i], np.array(sizes[:offset]),
                                              np.array(mean_scores[:offset]),
                                              np.array(sizes), 'newton_cg')
                            m.optimise()
                            rows.append(
                                [openmlid, learner, np.array(sizes)[offset - 1], m.predict(), self.model_ids[i],
                                 m.best_beta,
                                 m.fitness_function(m.best_beta),
                                 0])
                        else:
                            for offset in range(4, 4 + extrap_n):
                            #for offset in range(len(sizes)-1, len(sizes)):
                                # for offset in range(4, 6):
                                m = models.Models(self.model_ids[i], np.array(sizes[:offset]), np.array(mean_scores[:offset]),
                                                  np.array(sizes), 'newton_cg', initailparams = self.initailparams)
                                m.optimise()
                                rows.append(
                                    [openmlid, learner, np.array(sizes)[offset - 1], m.predict(), self.model_ids[i],
                                     m.best_beta,
                                     m.fitness_function(m.best_beta),
                                     0])
                        if rank == 0:
                            pbar.update(1)

                number_of_data_to_process += 1
        if rank == 0:
            pbar.close()

        rows = comm.gather(rows, root=0)
        if rank == 0:
            rows = flatten(rows)
            return pd.DataFrame(rows,
                                columns=["openmlid", "learner", "max_anchor_seen", "prediction", "curve_model", "beta",
                                         "fails_init", "fails_fit"])
        else:
            return None

    def get_anchors_and_scores_mean_curve(self):
        df = self.df
        rows = []
        for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
            for learner, df_learner in df_dataset.groupby("learner"):
                sizes = sorted(pd.unique(df_learner["size_train"]))
                scores = []
                for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                    sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                    scores.append(
                        [scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
                scores = np.array(scores)
                mean_scores = np.nanmean(scores, axis=0)
                rows.append([openmlid, learner, sizes, mean_scores])
        return pd.DataFrame(rows, columns=["openmlid", "learner", "anchor_prediction", "score"])

    def df_compute_metrics_mean_curve(self, df, df_info):

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

    def convert_table2(self, df,  performance_measure, logscale=False):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        rank = comm.Get_rank()
        dff = self.df

        curve_models = df['curve_model'].unique()

        rows = []
        info_rows = []
        for (openmlid, df_dataset) in tqdm(df.groupby('openmlid')):
            arr1 = np.array([])
            for learner, df_learner in df_dataset.groupby("learner"):
                arr1 = np.append(arr1, learner)
            arr1 = np.array_split(arr1, size)
            for (learner, df_learner) in df_dataset.groupby('learner'):
                if learner in arr1[rank]:
                    for (n, df_n) in df_learner.groupby('n'):
                        new_row = []
                        bucket = df_n.iloc[0, :].percentage_bucket

                        percentage = df_n.iloc[0, :].percentage

                        # print(percentage)
                        # print(type(percentage))

                        info_rows.append([openmlid, learner, n, bucket, float(percentage)])
                        for curve_model in curve_models:
                            row = df_n.query('curve_model == @curve_model')
                            score = np.nan
                            if len(row) > 0:
                                row = row.iloc[0, :]
                                score = row[performance_measure]
                            new_row.append(score)
                        rows.append(new_row)
        rows = comm.gather(rows, root=0)
        info_rows = comm.gather(info_rows, root=0)
        if rank == 0:
            rows = flatten(rows)
            info_rows = flatten(info_rows)
            a = np.array(rows)
            if logscale == True:
                a = np.log(a)
            a = pd.DataFrame(a, columns=curve_models)
            a_info = pd.DataFrame(info_rows, columns=['openmlid', 'learner', 'n', 'bucket', 'percentage'])
            b = pd.concat([a_info, a], axis=1)
            return b
        else:
            return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    cf = None
    if True:
        if rank == 0:
            df_performances = pd.read_csv("lcdb_new.csv")
            df_performances_one_dataset = df_performances.query('(openmlid ==  6| openmlid ==  1468) ')#df_performances
            #df_performances_one_dataset = df_performances
            cf = CurveFitting(df_performances_one_dataset, model_ids=None, extrapolation_n=300, training_model_n=12)
            print('computing extrapolations...')
            # compute the fits
        cf = comm.bcast(cf, root=0)
        cf.k_mean_centroids()

    if True:
        cf = None
        if rank == 0:
            with open('james.p', 'rb') as file:
                arr = pickle.load(file)
                print(arr)
            df_performances = pd.read_csv("lcdb_new.csv")
            df_performances_one_dataset = df_performances.query('(openmlid ==  6| openmlid ==  1468) ')
            #df_performances_one_dataset = df_performances
            cf = CurveFitting(df_performances_one_dataset, model_ids=None, extrapolation_n=300, training_model_n=12, initailparams=arr)
            print('computing extrapolations...')
            # compute the fits
        cf = comm.bcast(cf, root=0)
        df_extrapolations = cf.get_multiple_extrapolations_mean_curve_robust()
        if rank == 0:
            df_extrapolations.to_pickle('extrapolations_example.gz', protocol=3)

            print('computing anchors and scores...')
            # compute the X, Y values and store them for later use
            df_anchors_and_scores = cf.get_anchors_and_scores_mean_curve()
            df_anchors_and_scores.to_pickle('anchors_scores_example.gz', protocol=3)

            print('computing metrics....')
            # compute the metrics and other information (L2 losses, etc.)
            df_metrics = cf.df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
            df_metrics.to_pickle('metrics_example.gz', protocol=3)
    if False:
        df_clean_buckets =None
        if rank ==0:
            [df_anchors_and_scores, df_metrics, df_extrapolations] = load_from_parts_example()
            df_total = prepare_total_dataframe(df_anchors_and_scores, df_metrics, df_extrapolations)

            df_extrapolations_no_curve_model = df_extrapolations.loc[:, df_extrapolations.columns != 'curve_model']
            df_total = pd.concat([df_extrapolations_no_curve_model, df_metrics], axis=1)
            df_total = df_total.rename(
                columns={'MSE trn': 'MSE_trn', 'MSE tst': 'MSE_tst', 'MSE tst last': 'MSE_tst_last', 'L1 trn': 'L1_trn',
                         'L1 tst': 'L1_tst', 'L1 tst last': 'L1_tst_last'})

            [df_total_no_fail, df_fail] = remove_fails(df_total)
            df_no_fail_no_nan_or_inf, df_nan_or_inf = remove_nan_and_inf(df_total_no_fail)
            numeric = df_no_fail_no_nan_or_inf.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
            numeric = numeric.isin([np.inf, -np.inf, np.nan]).any(axis=1)

            df_no_fail_no_nan_or_inf_no_too_bad, df_too_bad = remove_performace_too_bad(df_no_fail_no_nan_or_inf)

            numeric = df_no_fail_no_nan_or_inf_no_too_bad.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
            threshold = 100
            df_clean = df_no_fail_no_nan_or_inf_no_too_bad


            list_temp = []
            for (openmlid, df_dataset) in tqdm(df_clean.groupby('openmlid')):
                for (learner, df_learner) in df_dataset.groupby('learner'):
                    anchor_prediction, score = get_info_mean_curve(df_anchors_and_scores, openmlid, learner)
                    max_anchor = np.max(anchor_prediction)
                    new_percentage = df_learner['max_anchor_seen'].values / max_anchor
                    df_learner['percentage'] = new_percentage
                    list_temp.append(df_learner)

            df_clean_new_percentage = pd.concat(list_temp, axis=0)

            df_n_to_anchor = df_clean.copy()
            df_n_to_anchor = df_n_to_anchor[['n', 'max_anchor_seen']]
            df_n_to_anchor = df_n_to_anchor.drop_duplicates()
            df_n_to_anchor

            percentage_buckets = [1, 0.8, 0.4, 0.2, 0.1, 0.05]
            percentage_buckets = np.array(percentage_buckets)

            bucket_list = [np.nan] * (len(df_clean))
            bucket_list = np.array(bucket_list)

            for i in range(0, len(percentage_buckets)):
                p = percentage_buckets[i]
                inbucket = df_clean_new_percentage['percentage'] < p
                bucket_list = np.where(inbucket.values, p, bucket_list)

            df_clean_buckets = df_clean_new_percentage.copy()
            df_clean_buckets.insert(0, 'percentage_bucket', bucket_list)

        df_clean_buckets = comm.bcast(df_clean_buckets, root=0)
        cf = CurveFitting(df_clean_buckets)
        table_MSE_tst_last = cf.convert_table2(df_clean_buckets, 'MSE_tst_last')
        table_MSE_tst = cf.convert_table2(df_clean_buckets, 'MSE_tst')

        table_MSE_trn = cf.convert_table2(df_clean_buckets, 'MSE_trn')
        if rank == 0:
            table_MSE_tst_last.to_pickle('table_MSE_tst_last3.gz', protocol=3)
            table_MSE_tst.to_pickle('table_MSE_tst3.gz', protocol=3)

            table_MSE_trn.to_pickle('table_MSE_trn3.gz', protocol=3)


