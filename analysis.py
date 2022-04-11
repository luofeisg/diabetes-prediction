"""
To understand raw data comprehensively
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# raw data
raw_dir = 'csv_file/raw/'
diabetes_diagnosis = pd.read_csv(raw_dir + 'diabetes_diagnosis.csv')
pid_preDiabetesOnly = pd.read_csv(raw_dir + 'pid_preDiabetesOnly.csv')
pid_preDiabetesToDiabetes = pd.read_csv(raw_dir + 'pid_preDiabetesToDiabetes.csv')
pid_DiabetesOnly = pd.read_csv(raw_dir + 'pid_DiabetesOnly.csv')
preDiabetesOnly = pd.read_csv(raw_dir + 'preDiabetesOnly.csv')
preDiabetesToDiabetes = pd.read_csv(raw_dir + 'preDiabetesToDiabetes.csv')
DiabetesOnly = pd.read_csv(raw_dir + 'DiabetesOnly.csv')
NonDiabetes = pd.read_csv(raw_dir + 'NonDiabetes.csv')
all_complication = pd.read_csv(raw_dir + 'all_complication.csv')
bmi_readings = pd.read_csv(raw_dir + 'bmi_readings.csv')
bp_readings = pd.read_csv(raw_dir + 'bp_readings.csv')
glucosetest_readings = pd.read_csv(raw_dir + 'glucosetest_readings.csv')
hba1c_readings = pd.read_csv(raw_dir + 'hba1c_readings.csv')
lipidspanel_readings = pd.read_csv(raw_dir + 'lipidspanel_readings.csv')
diabetes_demography = pd.read_csv(raw_dir + 'diabetes_demography.csv')
diabetes_demography_withlabel = pd.read_csv(raw_dir + 'diabetes_demography_withlabel.csv')
all_demography = pd.read_csv(raw_dir + 'all_demography.csv')
smk_asssesment = pd.read_csv(raw_dir + 'smk_asssesment.csv')
surgical_procedures_all = pd.read_csv(raw_dir + 'surgical_procedures_all.csv')

# import data from csv file
no_filling_diagnosis_records = pd.read_csv("no_filling_diagnosis_records.csv")
# df_all = pd.read_csv(data_file)


def development_time():
    df = no_filling_diagnosis_records
    df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])

    timedelta_0days = pd.Timedelta("0 days")
    timedelta_30days = pd.Timedelta("30 days")
    timedelta_180days = pd.Timedelta("180 days")
    timedelta_360days = pd.Timedelta("360 days")

    df_min_pre_date = df.loc[df[df['outcome'] == 0].groupby('PID').Diagnosis_Date.idxmin()]
    df_max_pre_date = df.loc[df[df['outcome'] == 0].groupby('PID').Diagnosis_Date.idxmax()]
    df_min_dia_date = df.loc[df[df['outcome'] == 1].groupby('PID').Diagnosis_Date.idxmin()]

    pre_pid_list = df_min_pre_date['PID'].to_list()
    dia_pid_list = df_min_dia_date['PID'].to_list()
    df_pre_min = df_min_pre_date[df_min_pre_date['PID'].isin(dia_pid_list)].sort_values('PID')
    df_pre_max = df_max_pre_date[df_max_pre_date['PID'].isin(dia_pid_list)].sort_values('PID')
    df_dia_min = df_min_dia_date[df_min_dia_date['PID'].isin(pre_pid_list)].sort_values('PID')

    row_count = 0
    count = 0
    result_df = diabetes_demography[diabetes_demography['PID'].isin(df_pre_min['PID'].to_list())]
    result_df = result_df[['PID', 'Gender', 'Race']]
    result_df['Development_time'] = pd.Timedelta("0 days")
    for index, row_pre in df_pre_min.iterrows():
        print("{} / {}".format(row_count+1, df_pre_min.shape[0]))
        pre_min = row_pre
        pre_max = df_pre_max.iloc[row_count]
        dia_min = df_dia_min.iloc[row_count]

        result_df.iat[row_count, 3] = dia_min['Diagnosis_Date'] - pre_min['Diagnosis_Date']
        if dia_min['Diagnosis_Date'] < pre_max['Diagnosis_Date']:
            count += 1

        row_count += 1

    print(result_df[(result_df['Development_time'] >= pd.Timedelta("0 days"))].shape[0])
    print(result_df[(result_df['Development_time'] >= pd.Timedelta("1440 days")) & (result_df['Development_time'] < pd.Timedelta("1800 days"))].shape[0])
    pass


def missing_data_analysis():
    df_all = no_filling_diagnosis_records
    df = df_all
    cols = ['OGTT2hr_(mmol/l)', 'FPG_(mmol/l', 'LDLc_(mmol/l)', 'HDLc_(mmol/l)',
            'HbA1c_%', 'TC_(mmol/l)', 'TG_(mmol/l)', 'Weight_kg', 'BMI_kg/m2', 'SBP_mmHg', 'DBP_mmHg']
    outcome_list = list(df_all['outcome'].unique())
    for label in outcome_list:
        break
        df = df_all[df_all['outcome'] == label]
        print("\n-----label: {}\n".format(label))
        no_all_records = df.shape[0]
        for col in cols:
            df_na = df[df[col].isna()]
            print("{}   {:.4f}".format(col, df_na.shape[0] / no_all_records))

        # earliest records
        df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])
        df_earliest = df.loc[df.groupby(['PID']).Diagnosis_Date.idxmin()]
        no_all_records = df_earliest.shape[0]
        for col in cols:
            df_na = df_earliest[df[col].isna()]
            print("{}   {:.4f}".format(col, df_na.shape[0] / no_all_records))

    df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])
    df_earliest = df.loc[df.groupby(['PID', 'outcome']).Diagnosis_Date.idxmin()]
    no_all_records = df_earliest.shape[0]
    for col in cols:
        df_na = df_earliest[df[col].isna()]
        print("{}   {:.4f}".format(col, df_na.shape[0]/no_all_records))

    df_max_pre_date = df.loc[df[df['outcome'] == 0].groupby('PID').Diagnosis_Date.idxmax()]
    df_missing345 = df_earliest[df_earliest[cols].isnull().all(1)]
    df_missing345_pid = df_missing345['PID']
    timedelta_360days = pd.Timedelta("360 days")
    min_date = df_missing345['Diagnosis_Date'].min()
    date_counts = df_missing345[(df_missing345['outcome'] == 1) & ((df_missing345['Diagnosis_Date'] - min_date)<timedelta_360days)]

    for col in cols:
        df = df.loc[df[col].notna()]
    pid_prediabetes_to_diabetes_set = set(diabetes_demography_withlabel[diabetes_demography_withlabel['label'] == 'prediabetes_to_diabetes']['PID'])
    pass


def missing_distribution():
    df = no_filling_diagnosis_records
    cols = ['OGTT2hr_(mmol/l)', 'FPG_(mmol/l', 'HbA1c_%']
    for col in cols:
        df_temp = df.loc[df[col].isna(), ['outcome']]
        print(col)
        results = df_temp.squeeze().value_counts()
        print(results)
        print(results[0]/(results[0]+results[1]))


def race_gender_plot_distribution():
    # df_list = [pid_preDiabetesOnly, pid_preDiabetesToDiabetes, pid_preDiabetesToDiabetes]
    # cols = ["Race", "Gender"]
    # for df in df_list:
    #     # arr = df['PID'].unique()
    #     df[cols] = None
    #     for index in range(df.shape[0]):
    #         pid = df.iloc[index]['PID']
    #         records = all_demography[all_demography['PID'] == pid]
    #         if records.shape[0] > 0:
    #             df.loc[index][cols] = records.iloc[0][cols]

    df_dict = {'preDiabetesOnly': preDiabetesOnly,
               'preDiabetesToDiabetes': preDiabetesToDiabetes,
               'DiabetesOnly': DiabetesOnly,
               'NonDiabetes': NonDiabetes
               }
    cols = ['Race', 'Gender']
    gender_remove_list = ['Unknown']
    race_remove_list = ['Unknown']

    fig, big_axes = plt.subplots(4, 1, figsize=(10, 10))
    for title, big_ax in zip(list(df_dict.keys()), big_axes):
        big_ax.set_title(title)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    plot_counter = 1
    for index_df, (name, df) in enumerate(df_dict.items()):
        for v in gender_remove_list:
            df.drop(df[df['Gender'] == v].index, inplace=True)
        for v in race_remove_list:
            df.drop(df[df['Race'] == v].index, inplace=True)
        for index_col, col in enumerate(cols):
            # df[col].value_counts().sort_index().hist()
            plot_data = df[col].value_counts().sort_index()
            ax = fig.add_subplot(4, 2, plot_counter)
            ax.bar(plot_data.index, plot_data.values)
            # ax.set_title('Plot title ' + str(plot_counter))
            plot_counter += 1
            # big_axs[index_df-1, index_col-1].bar(plot_data.index, plot_data.values)

            # plt.xlabel(name+'_'+col)
            # plt.show()
            # fig.savefig("images/distribution_analysis/" + name + '_' + col + '_distribution_uniquePID.png')
            # plt.clf()
    # plt.show()
    fig.tight_layout()
    fig.savefig("images/distribution_analysis/" + 'all_distribution_uniquePID.png')


def race_gender_distribution():
    # df_list = [pid_preDiabetesOnly, pid_preDiabetesToDiabetes, pid_preDiabetesToDiabetes]
    # cols = ["Race", "Gender"]
    # for df in df_list:
    #     # arr = df['PID'].unique()
    #     df[cols] = None
    #     for index in range(df.shape[0]):
    #         pid = df.iloc[index]['PID']
    #         records = all_demography[all_demography['PID'] == pid]
    #         if records.shape[0] > 0:
    #             df.loc[index][cols] = records.iloc[0][cols]

    df_dict = {'preDiabetesOnly': preDiabetesOnly,
               'preDiabetesToDiabetes': preDiabetesToDiabetes,
               'DiabetesOnly': DiabetesOnly,
               'NonDiabetes': NonDiabetes
               }
    cols = ['Race', 'Gender']
    gender_remove_list = ['Unknown']
    race_remove_list = ['Unknown']

    for index_df, (name, df) in enumerate(df_dict.items()):
        for v in gender_remove_list:
            df.drop(df[df['Gender'] == v].index, inplace=True)
        for v in race_remove_list:
            df.drop(df[df['Race'] == v].index, inplace=True)
        for index_col, col in enumerate(cols):
            # df[col].value_counts().sort_index().hist()
            data = df[col].value_counts().sort_index()
            print(name)
            print(col)
            print(data)
            pass
        pass


def data_plot_distribution_within_race():
    df = diabetes_demography_withlabel
    gender_remove_list = ['Unknown']
    race_list = ['Indian', 'Chinese', 'Malay']
    plot_cols = ['Gender', 'label']

    for v in gender_remove_list:
        df.drop(df[df['Gender'] == v].index, inplace=True)

    fig, big_axes = plt.subplots(3, 1, figsize=(12, 10))
    for title, big_ax in zip(race_list, big_axes):
        big_ax.set_title(title)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    plot_counter = 1
    for race in race_list:

        df_race = df[df['Race'] == race]
        for col in plot_cols:
            plot_data = df_race[col].value_counts().sort_index()
            ax = fig.add_subplot(3, 2, plot_counter)
            ax.bar(plot_data.index, plot_data.values)
            plot_counter += 1

    fig.tight_layout()
    fig.savefig("images/distribution_analysis/" + 'data_distribution_within_race.png')


def data_distribution_within_race():
    df = diabetes_demography_withlabel
    gender_remove_list = ['Unknown']
    race_list = ['Indian', 'Chinese', 'Malay']
    plot_cols = ['Gender', 'label']

    for v in gender_remove_list:
        df.drop(df[df['Gender'] == v].index, inplace=True)

    for race in race_list:
        df_race = df[df['Race'] == race]
        for col in plot_cols:
            plot_data = df_race[col].value_counts().sort_index()
            print(race)
            print(col)
            print(plot_data)
        pass


def age_distribution():
    df = no_filling_diagnosis_records
    df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])
    df = df.loc[df.groupby(['PID', 'outcome']).Diagnosis_Date.idxmin()]
    fig = plt.figure()
    ax = df['Age'].plot.hist()
    aa = df['Age'].sort_values()
    plt.xlabel("Age", fontsize=17)
    plt.ylabel("Number of patients", fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()
    fig.savefig("images/distribution_analysis/" + 'age_distribution.png')
    pass


if __name__ == '__main__':
    # missing_data_analysis()
    # race_gender_distribution()
    # development_time()
    # data_distribution_within_race()
    age_distribution()
    pass

