import copy
from utils import get_logger, save_to_csv
from utils import remove_mistaken, fill_data_knn, fill_data_gender_age_race, fill_data_zero, fill_data_mean
# import mysql.connector
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import date
import os
# import datawig

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def get_from_raw_datafile():
    """
    get data from multiple raw data files and save into one data file

    :return:
    """
    # log
    logger = get_logger("preprocessing1", "preprocessing1.log")

    # read data from csv files
    raw_dir = 'csv_file/raw/'
    diabetes_diagnosis = pd.read_csv(raw_dir + 'diabetes_diagnosis.csv')
    pid_preDiabetesOnly = pd.read_csv(raw_dir + 'pid_preDiabetesOnly.csv')
    pid_preDiabetesToDiabetes = pd.read_csv(raw_dir + 'pid_preDiabetesToDiabetes.csv')
    all_complication = pd.read_csv(raw_dir + 'all_complication.csv')
    bmi_readings = pd.read_csv(raw_dir + 'bmi_readings.csv')
    bp_readings = pd.read_csv(raw_dir + 'bp_readings.csv')
    glucosetest_readings = pd.read_csv(raw_dir + 'glucosetest_readings.csv')
    hba1c_readings = pd.read_csv(raw_dir + 'hba1c_readings.csv')
    lipidspanel_readings = pd.read_csv(raw_dir + 'lipidspanel_readings.csv')
    poi_demography = pd.read_csv(raw_dir + 'poi_demography.csv')
    smk_asssesment = pd.read_csv(raw_dir + 'smk_asssesment.csv')
    surgical_procedures_all = pd.read_csv(raw_dir + 'surgical_procedures_all.csv')

    data_dict = {
        'PID': [],
        'Gender': [],
        'Age': [],
        'Race': [],
        'Weight_kg': [],
        'Diagnosis_Date': [],
        'CDMS_Disease_Grp': [],
        'OGTT2hr_(mmol/l)': [],
        'FPG_(mmol/l': [],
        'HbA1c_%': [],
        'LDLc_(mmol/l)': [],
        'TC_(mmol/l)': [],
        'TG_(mmol/l)': [],
        'HDLc_(mmol/l)': [],
        'Smoking_Status': [],
        'BMI_kg/m2': [],
        'SBP_mmHg': [],
        'DBP_mmHg': [],
        'Hypertension (Y/N)': [],
        'Dyslipidaemia (Y/N)': [],
        'Surgical_procedure (Y/N)': [],
        'No_of_complications': [],
        'Counts_of_visits': [],
    }
    df = pd.DataFrame(data_dict)
    results_file_name = "diagnosis_records.csv"
    # save head to csv file first
    save_to_csv(df, results_file_name)

    # get all diagnosis records from several tables based on table 7 (nhgp_diagnosis_2010_2017)
    for index, row in diabetes_diagnosis.iterrows():
        start_time = time.time()

        timedelta_30days = pd.Timedelta("30 days")
        timedelta_180days = pd.Timedelta("180 days")
        timedelta_360days = pd.Timedelta("360 days")
        dict_onerow = {}
        diagnosis_date = pd.to_datetime(row.Diagnosis_Date)

        dict_onerow['PID'] = row.PID
        dict_onerow['Diagnosis_Date'] = row.Diagnosis_Date
        dict_onerow['CDMS_Disease_Grp'] = row.CDMS_Disease_Grp

        # ------------  table 1 (all_complication)  ------------------
        # column 21: No_of_complications
        # Remarks: Active complications on date of diagnosis

        # pandas convert to_datetime('9999-12-31') out of bounds error,
        # so convert 'Complication_End_Date' to another big date first
        all_complication.loc[
            all_complication['Complication_End_Date'] > "2222-01-01", 'Complication_End_Date'] = "2222-01-01"

        temp_df = all_complication[(all_complication['PID'] == row.PID) &
                                   (pd.to_datetime(all_complication['Complication_Onset_Date']) < diagnosis_date) &
                                   (pd.to_datetime(all_complication['Complication_End_Date']) > diagnosis_date)]
        dict_onerow['No_of_complications'] = temp_df.shape[0]

        # ------------  tbl 5 (nhgp_bmi_readings)  ------------------
        # 4: Weight_kg; 15: BMI_kg/m2
        # Remarks: Examination happened 30 days before or after date of diagnosis
        temp_df = bmi_readings[(bmi_readings['PID'] == row.PID) &
                               (pd.to_datetime(
                                   bmi_readings['Date_of_Physical_Examination']) - diagnosis_date < timedelta_30days) &
                               (diagnosis_date - pd.to_datetime(
                                   bmi_readings['Date_of_Physical_Examination']) < timedelta_30days)]
        if temp_df.shape[0] > 0:
            dict_onerow['Weight_kg'] = temp_df.iloc[0]['Weight_kg']
            dict_onerow['BMI_kg/m2'] = temp_df.iloc[0]['BMI_kg/m2']

        # ------------  tbl 6 (nhgp_bp_readings)  ------------------
        # 16: SBP_mmHg; 17: DBP_mmHg
        # Remarks: Examination happened 30 days before or after date of diagnosis
        temp_df = bp_readings[(bp_readings['PID'] == row.PID) &
                              (pd.to_datetime(
                                  bp_readings['Date_of_Physical_Examination']) - diagnosis_date < timedelta_30days) &
                              (diagnosis_date - pd.to_datetime(
                                  bp_readings['Date_of_Physical_Examination']) < timedelta_30days)]
        if temp_df.shape[0] > 0:
            dict_onerow['SBP_mmHg'] = temp_df.iloc[0]['SBP_mmHg']
            dict_onerow['DBP_mmHg'] = temp_df.iloc[0]['DBP_mmHg']

        # ------------  tbl 9 (nhgp_glucosetest_readings)  ------------------
        # 7: OGTT2hr_(mmol/l); 8: FPG_(mmol/l
        # Remarks: Examination happened 30 days before or after date of diagnosis
        temp_df = glucosetest_readings[(glucosetest_readings['PID'] == row.PID) &
                                       (pd.to_datetime(glucosetest_readings[
                                                           'Date_of_Lab_Test_Result']) - diagnosis_date < timedelta_30days) &
                                       (diagnosis_date - pd.to_datetime(
                                           glucosetest_readings['Date_of_Lab_Test_Result']) < timedelta_30days)]
        if temp_df.shape[0] > 0:
            dict_onerow['OGTT2hr_(mmol/l)'] = temp_df.iloc[0]['OGTT2hr_(mmol/l)']
            dict_onerow['FPG_(mmol/l'] = temp_df.iloc[0]['FPG_(mmol/l']

        # ------------  tbl 10 (nhgp_hba1c_readings)  ------------------
        # 9: HbA1c_%
        # Remarks: Examination happened 30 days before or after date of diagnosis
        temp_df = hba1c_readings[(hba1c_readings['PID'] == row.PID) &
                                 (pd.to_datetime(
                                     hba1c_readings['Date_of_Lab_Test_Result']) - diagnosis_date < timedelta_30days) &
                                 (diagnosis_date - pd.to_datetime(
                                     hba1c_readings['Date_of_Lab_Test_Result']) < timedelta_30days)]
        if temp_df.shape[0] > 0:
            dict_onerow['HbA1c_%'] = temp_df.iloc[0]['HbA1c_%']

        # ------------  tbl 11 (nhgp_lipidspanel_readings)  ------------------
        # 10: LDLc_(mmol/l); 11:TC_(mmol/l); 12:TG_(mmol/l); 13:HDLc_(mmol/l);
        # Remarks: Examination happened 30 days before or after date of diagnosis
        temp_df = lipidspanel_readings[(lipidspanel_readings['PID'] == row.PID) &
                                       (pd.to_datetime(lipidspanel_readings[
                                                           'Date_of_Lab_Test_Result']) - diagnosis_date < timedelta_30days) &
                                       (diagnosis_date - pd.to_datetime(
                                           lipidspanel_readings['Date_of_Lab_Test_Result']) < timedelta_30days)]
        if temp_df.shape[0] > 0:
            dict_onerow['LDLc_(mmol/l)'] = temp_df.iloc[0]['LDLc_(mmol/l)']
            dict_onerow['TC_(mmol/l)'] = temp_df.iloc[0]['TC_(mmol/l)']
            dict_onerow['TG_(mmol/l)'] = temp_df.iloc[0]['TG_(mmol/l)']
            dict_onerow['HDLc_(mmol/l)'] = temp_df.iloc[0]['HDLc_(mmol/l)']

        # ------------  tbl 13 (poi_demography)  ------------------
        # 1: Gender; 2: Age (in MONTH); 3: Race; 18: Hypertension (Y/N); 19: Dyslipidaemia (Y/N); 22: Counts_of_visits
        # Remarks: -
        temp_df = poi_demography[(poi_demography['PID'] == row.PID)]
        if temp_df.shape[0] > 0:
            dict_onerow['Gender'] = temp_df.iloc[0]['Gender']
            dict_onerow['Age'] = diagnosis_date.year - pd.to_datetime(temp_df.iloc[0]['Date_of_Birth']).year
            dict_onerow['Race'] = temp_df.iloc[0]['Race']
            dict_onerow['Hypertension (Y/N)'] = 1 if pd.to_datetime(
                temp_df.iloc[0]['Hypertension']) < diagnosis_date else 0
            dict_onerow['Dyslipidaemia (Y/N)'] = 1 if pd.to_datetime(
                temp_df.iloc[0]['Dyslipidaemia']) < diagnosis_date else 0
            dict_onerow['Counts_of_visits'] = temp_df.iloc[0][str(diagnosis_date.year)]

        # ------------  tbl 14 (smk_asssesment)  ------------------
        # 14: Smoking_Status
        # Remarks: Examination happened 6 months before or after date of diagnosis
        temp_df = smk_asssesment[(smk_asssesment['PID'] == row.PID) &
                                 (pd.to_datetime(
                                     smk_asssesment['Smoking_Assessment_Date']) - diagnosis_date < timedelta_180days) &
                                 (diagnosis_date - pd.to_datetime(
                                     smk_asssesment['Smoking_Assessment_Date']) < timedelta_180days)]
        if temp_df.shape[0] > 0:
            dict_onerow['Smoking_Status'] = temp_df.iloc[0]['Smoking_Status']

        # ------------  tbl 15 (surgical_procedures_all)  ------------------
        # 20: Surgical_procedure (Y/N)
        # Remarks: 1 if there are surgical procedures records 1 year before diagnosis date, otherwise 0
        temp_df = surgical_procedures_all[(surgical_procedures_all['PID'] == row.PID) &
                                          (pd.to_datetime(surgical_procedures_all[
                                                              'Surgical_Procedure_Date']) - diagnosis_date < timedelta_180days) &
                                          (diagnosis_date - pd.to_datetime(
                                              surgical_procedures_all['Surgical_Procedure_Date']) < timedelta_180days)]
        dict_onerow['Surgical_procedure (Y/N)'] = 1 if temp_df.shape[0] > 0 else 0

        print(dict_onerow)
        df.loc[df.shape[0]] = dict_onerow
        print("{} / {} completed, time taken: {:.2f}.".format(index + 1, diabetes_diagnosis.shape[0],
                                                              time.time() - start_time))

        # append data to csv file every 500 records, and reset dataframe
        if (index + 1) % 500 == 0 or index == diabetes_diagnosis.shape[0] - 1:
            logger.info("{} / {} finished and saved".format(index + 1, diabetes_diagnosis.shape[0]))
            df.to_csv(results_file_name, mode='a', header=False, index=False)
            df = pd.DataFrame(data_dict)

    logger.info("pre-process finished \n")

    # f = open('diagnosis_records.csv', 'a')
    # writer = csv.writer(f)
    # f.close()
    return 0


def remove_mistaken_data():
    """
    remove mistaken data and unimportant data

    :return:
    """
    # read data from csv file
    # results_file_name = "diagnosis_records_3000.csv"
    results_file_name = "diagnosis_records.csv"
    df = pd.read_csv(results_file_name)

    # delete mistaken data
    remove_mistaken(df)

    # add label/outcome  based on column CDMS_Disease_Grp
    # CDMS_Disease_Grp
    # ['Diabetes Mellitus', 'Pre-Diabetes']
    df.loc[df.CDMS_Disease_Grp == 'Diabetes Mellitus', 'outcome'] = 1
    df.loc[df.CDMS_Disease_Grp == 'Pre-Diabetes', 'outcome'] = 0

    # Gender
    # ['Male', 'Female', 'Unknown'] => [1, 0, remove]
    gender_list = ['Male', 'Female']
    df.drop(df[df.Gender == 'Unknown'].index, inplace=True)
    # df['Gender'].replace(gender_list, [1, 0], inplace=True)
    # df.loc[df['Gender'] == 'Male', 'Gender'] = 1
    # df.loc[df['Gender'] == 'Female', 'Gender'] = 0

    # Age

    # Race
    # ['Malay', 'Chinese', 'Indian', 'Others', 'Eurasian', 'Caucasian', 'Unknown']
    # => [0, 1, 2, 3, 4, 5, remove]
    race_list = ['Malay', 'Chinese', 'Indian', 'Others', 'Eurasian', 'Caucasian']
    df.drop(df[df.Race == 'Unknown'].index, inplace=True)
    # df['Race'].replace(race_list, [0, 1, 2, 3, 4, 5], inplace=True)

    # # data filling
    # # missing value: mean of all male/female with pre-diabetes/diabetes mellitus
    # to_fill_cols = ['Weight_kg', 'OGTT2hr_(mmol/l)', 'FPG_(mmol/l', 'HbA1c_%',
    #                     'LDLc_(mmol/l)', 'TC_(mmol/l)', 'TG_(mmol/l)', 'HDLc_(mmol/l)',
    #                     'SBP_mmHg', 'DBP_mmHg', 'BMI_kg/m2']
    # for col in to_complete_cols:
    #     fill_data(df, col)

    # Smoking_Status
    # ['nan', 'Non-Smoker', 'Ex-Smoker', 'Smoker', 'Not Applicable']
    # => [0, 0, 1, 2, remove]
    # *** patients with no smoking assessment records within 12 month are set to 'Non-Smoker'
    smoking_status_list = ['Non-Smoker', 'Ex-Smoker', 'Smoker']
    df.drop(df[df['Smoking_Status'] == 'Not Applicable'].index, inplace=True)
    # df['Smoking_Status'].replace(smoking_status_list, [0, 1, 2], inplace=True)
    df.loc[df['Smoking_Status'].isna(), 'Smoking_Status'] = 'Non-Smoker'

    save_to_csv(df, "no_filling_" + results_file_name)
    pass


def split_and_filling(fill_method='zero'):
    """
    split into train, test
    one-hot encoding
    filling data with nearest neighbors based on Gender and Age

    :param fill_method:
        "knn":
        "gender_age_race":
        "zero":
    :return:
    """
    # data_file = "no_filling_diagnosis_records_3000.csv"
    data_file = "no_filling_diagnosis_records.csv"
    df = pd.read_csv(data_file)
    df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])
    df = df.loc[df.groupby(['PID', 'outcome']).Diagnosis_Date.idxmin()]
    df = df.drop(['PID', 'Diagnosis_Date', 'CDMS_Disease_Grp'], axis=1)

    # train/test split
    df_train, df_test = train_test_split(df, random_state=123, test_size=0.20)
    # reset index
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # split dataset
    x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    # one-hot encoding
    categorical_cols = ['Gender', 'Race', 'Smoking_Status']
    x_train = pd.get_dummies(x_train, columns=categorical_cols)
    x_test = pd.get_dummies(x_test, columns=categorical_cols)

    x_list = [x_test, x_train]

    # data filling
    print("filling data...")
    time_start = time.time()
    cols_to_fill = ['Weight_kg', 'OGTT2hr_(mmol/l)', 'FPG_(mmol/l', 'HbA1c_%',
                    'LDLc_(mmol/l)', 'TC_(mmol/l)', 'TG_(mmol/l)', 'HDLc_(mmol/l)',
                    'SBP_mmHg', 'DBP_mmHg', 'BMI_kg/m2']
    for x in x_list:
        if fill_method == "knn1000":
            # kNN
            knn_metric_list = ['Gender_Male', 'Gender_Female', 'Age', 'Race_Caucasian', 'Race_Chinese',
                               'Race_Eurasian', 'Race_Indian', 'Race_Malay', 'Race_Others']
            df_temp = x[knn_metric_list]
            knn = NearestNeighbors(n_neighbors=1000)
            knn.fit(df_temp)
            for col in cols_to_fill:
                fill_data_knn(x, col, knn, knn_metric_list)
                print("Filled column {}".format(col))
        elif fill_method == "knn_all50":
            # kNN
            knn_metric_list = ['Age', 'Hypertension (Y/N)', 'Dyslipidaemia (Y/N)',
                               'Surgical_procedure (Y/N)', 'No_of_complications', 'Counts_of_visits',
                               'Gender_Female', 'Gender_Male', 'Race_Caucasian', 'Race_Chinese',
                               'Race_Eurasian', 'Race_Indian', 'Race_Malay', 'Race_Others',
                               'Smoking_Status_Ex-Smoker', 'Smoking_Status_Non-Smoker',
                               'Smoking_Status_Smoker']
            df_temp = x[knn_metric_list]
            knn = NearestNeighbors(n_neighbors=50)
            knn.fit(df_temp)
            for col in cols_to_fill:
                fill_data_knn(x, col, knn, knn_metric_list)
                print("Filled column {}".format(col))
        elif fill_method == "gender_age_race":
            # ['Gender_Female', 'Gender_Male', 'Age', 'Race_Caucasian',
            # 'Race_Chinese', 'Race_Eurasian', 'Race_Indian', 'Race_Malay', 'Race_Others']
            metric_list = ['Gender', 'Age', 'Race']
            for col in cols_to_fill:
                fill_data_gender_age_race(x, col, metric_list)
                print("Filled column {}".format(col))
        elif fill_method == "zero":
            fill_data_zero(x)
        elif fill_method == "5":
            x.fillna(5, inplace=True)
        elif fill_method == "10":
            x.fillna(10, inplace=True)
        elif fill_method == "15":
            x.fillna(15, inplace=True)
        elif fill_method == "20":
            x.fillna(20, inplace=True)
        elif fill_method == "mean":
            fill_data_mean(x)
        elif fill_method == "mice":
            # fill_data_mice(x)
            lr = LinearRegression()
            imp = IterativeImputer(estimator=lr, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')

            imp.fit(x)
            arr_temp = imp.transform(x)
            df_temp = pd.DataFrame(arr_temp)
            df_temp.columns = x.columns
            if df_temp.shape == x_train.shape:
                x_train = df_temp
            else:
                x_test = df_temp
        elif fill_method == "datawig":
            df_temp = datawig.SimpleImputer.complete(x, num_epochs=30)
            if x.shape == x_train.shape:
                x_train = df_temp
            else:
                x_test = df_temp
            # input_cols = ['Age', 'Hypertension (Y/N)', 'Dyslipidaemia (Y/N)',
            #               'Surgical_procedure (Y/N)', 'No_of_complications', 'Counts_of_visits',
            #               'Gender_Female', 'Gender_Male', 'Race_Caucasian', 'Race_Chinese',
            #               'Race_Eurasian', 'Race_Indian', 'Race_Malay', 'Race_Others',
            #               'Smoking_Status_Ex-Smoker', 'Smoking_Status_Non-Smoker',
            #               'Smoking_Status_Smoker']
            # for col in cols_to_fill:
            #     fill_data_datawig(x, input_cols, col)

            pass
        else:
            raise Exception("fill method not found")
    print("filling data done. Time consumed(s): {:.0f}".format(time.time() - time_start))

    # save to csv file
    save_dir = "csv_file/filled/" + fill_method + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_to_csv(x_train, save_dir + "x_train.csv")
    save_to_csv(y_train, save_dir + "y_train.csv")
    save_to_csv(x_test, save_dir + "x_test.csv")
    save_to_csv(y_test, save_dir + "y_test.csv")


def fill_data_datawig(df, input_cols, output_col):
    df_train, df_test = datawig.utils.random_split(df)
    imputer = datawig.SimpleImputer(
        input_columns=input_cols,  # column(s) containing information about the column we want to impute
        output_column=output_col,  # the column we'd like to impute values for
        output_path='imputer_model'  # stores model data and metrics
    )
    imputer.fit(train_df=df_train, num_epochs=5)
    # a = imputer.complete(df)
    imputed = imputer.predict(df)
    pass


def test():
    pass


if __name__ == '__main__':
    # Step 1: from original csv files to one csv file; lots of missing fields; haven't assigned label
    # get_from_raw_datafile()
    # Step 2: remove mistaken value
    # remove_mistaken_data()
    # Step 3: train/test split; one-hot encoding; data filling
    # fill_method: zero, gender_age_race, knn, mice, datawig
    split_and_filling(fill_method="knn_all50")

    # test()
    pass
