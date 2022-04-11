# Utility functions

# import mysql.connector
import logging
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def save_to_csv(df, file_name):
    if os.path.isfile(file_name):
        print(file_name + "already exists")
        raise RuntimeError(file_name + " already exists, check first!")
    else:
        df.to_csv(file_name, index=False)


def remove_mistaken(df):
    """
    Delete mistaken data.
    Weight_kg: (0, 300]
    BMI_kg/m2: [5, 150]
    SBP_mmHg: [50, 300], > DBP_mmHg
    DBP_mmHg: [30, 300], > SBP_mmHg
    HbA1c_%: [3, 20]
    FPG_(mmol/l: [1, 40]
    OGTT2hr_(mmol/l): [1, 40]
    LDLc_(mmol/l): [0.1, 10]
    TC_(mmol/l): -
    TG_(mmol/l): [0.1, 30]
    HDLc_(mmol/l): -

    :param df: dataframe
    :return:
    """
    df.drop(df[(df.Weight_kg <= 0) | (df.Weight_kg > 300)].index, inplace=True)
    df.drop(df[(df['BMI_kg/m2'] < 5) | (df['BMI_kg/m2'] > 150)].index, inplace=True)
    df.drop(df[(df.SBP_mmHg < 50) | (df.SBP_mmHg > 300)].index, inplace=True)
    df.drop(df[(df.DBP_mmHg < 30) | (df.DBP_mmHg > 300)].index, inplace=True)
    df.drop(df[df.SBP_mmHg < df.DBP_mmHg].index, inplace=True)
    df.drop(df[(df['HbA1c_%'] < 3) | (df['HbA1c_%'] > 20)].index, inplace=True)
    df.drop(df[(df['FPG_(mmol/l'] < 1) | (df['FPG_(mmol/l'] > 40)].index, inplace=True)
    df.drop(df[(df['OGTT2hr_(mmol/l)'] < 1) | (df['OGTT2hr_(mmol/l)'] > 40)].index, inplace=True)
    df.drop(df[(df['LDLc_(mmol/l)'] < 0.1) | (df['LDLc_(mmol/l)'] > 10)].index, inplace=True)
    df.drop(df[(df['TG_(mmol/l)'] < 0.1) | (df['TG_(mmol/l)'] > 30)].index, inplace=True)


def fill_data_knn(df, col, knn, knn_metric_list):
    """
    Insert mean value into missing field based on knn.
    - Mean value of male/female, m1
    - Mean value of n neighbours based on 'knn_metric_list', if nan then fill m1

    :param df: data frame to fill
    :param col: column to fill
    :param knn: k nearest neighbors
    :param knn_metric_list: nearest neighbors metrics
    :return:
    """
    mean_m = np.around(df[(df['Gender_Male'] == 1)][col].mean(), decimals=2)
    mean_f = np.around(df[(df['Gender_Female'] == 1)][col].mean(), decimals=2)

    df_to_fill = df.loc[df[col].isna()]
    if df_to_fill.shape[0] == 0:
        # if no missing data to fill, return
        return
    neigh_index = knn.kneighbors(df_to_fill.loc[:, knn_metric_list], return_distance=False)
    # to_fill_df = to_fill_df.reset_index()
    neigh_index_count = 0
    for index, row in df_to_fill.iterrows():
        m = np.around(df.loc[neigh_index[neigh_index_count], col].mean(), decimals=2)
        neigh_index_count += 1
        if np.isnan(m):
            df.loc[index, col] = mean_m if df.loc[index, "Gender_Male"] == 1 else mean_f
        else:
            df.loc[index, col] = m
    # df.loc[(df[col].isna()) & (df['Gender_Male'] == 1), col] = mean_m
    # df.loc[(df[col].isna()) & (df['Gender_Female'] == 1), col] = mean_f


def fill_data_gender_age_race(df, col, metric_list):
    """
    Insert mean value into missing field using mean of samples with same gender, race or similar age.

    :param df: data frame to fill
    :param col: column to fill
    :param metric_list: metrics
    :return:
    """
    mean_m = np.around(df[(df['Gender'] == 'Male')][col].mean(), decimals=2)
    mean_f = np.around(df[(df['Gender'] == 'Female')][col].mean(), decimals=2)

    df_to_fill = df.loc[df[col].isna()]
    if df_to_fill.shape[0] == 0:
        # if no missing data to fill, return
        return
    df_temp = df[df[col].notna()]
    for index, row in df_to_fill.iterrows():
        for metric in metric_list:
            if metric == "Age":
                df_temp = df_temp[(abs(df[metric] - row[metric]) <= 5)]
            else:
                df_temp = df_temp[(df[metric] == row[metric])]
            if df_temp[col].shape[0] > 0:
                m = np.around(df_temp[col].mean(), decimals=2)
        if np.isnan(m):
            df.loc[index, col] = mean_m if row['Gender'] == 'Male' else mean_f
        else:
            df.loc[index, col] = m
    # df.loc[(df[col].isna()) & (df['Gender_Male'] == 1), col] = mean_m
    # df.loc[(df[col].isna()) & (df['Gender_Female'] == 1), col] = mean_f


def fill_data_mean(df):
    """
    Insert mean values into missing fields

    :param df: data frame to fill
    :return:
    """
    df.fillna(df.mean(), inplace=True)


def fill_data_zero(df):
    """
    Insert 0 into missing fields

    :param df: data frame to fill
    :return:
    """
    df.fillna(0, inplace=True)


def get_logger(logger_name, file_name):
    log = logging.getLogger(logger_name)
    log.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler for logger.
    fh = logging.FileHandler(file_name)
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log

# sql functions
# def if_db_exists(cursor, db_name):
#     pass
#
# def if_tbl_exists(cursor, tbl_name):
#     pass
#
# def create_table(cursor, tbl_name):
#     sql_cmd = "create table if exists {}"
#     cursor.execute(sql_cmd.format(tbl_name))
#
# # drop a table if exists
# def drop_tbl(cursor, tbl_name):
#     try:
#         print("Dropping table {}: ".format(tbl_name), end='')
#         cursor.execute("drop table {}".format(tbl_name))
#     except mysql.connector.Error as err:
#         print(err.msg)
#     else:
#         print("OK")
#
# def query_from_tbl(cursor, tbl_name, cols, condtions='1'):
#     try:
#         cursor.execute("select {} from {} where {}".format(cols, tbl_name, condtions))
#     except mysql.connector.Error as err:
#         print(err.msg)
#
#     return cursor
