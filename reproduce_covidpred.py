import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

def dfByDate(df, start_date, end_date, date_str='test_date'):
    start_date = np.datetime64(start_date) 
    end_date = np.datetime64(end_date)
    mask = (df[date_str] >= start_date) & (df[date_str] <= end_date)
    return df.loc[mask]

def replaceValuesDF(df):
    df = df.replace("None", np.nan)
    df["test_indication"] = df["test_indication"].replace('Other', 0)
    df["test_indication"] = df["test_indication"].replace('Abroad', 0)
    df["test_indication"] = df["test_indication"].replace('Contact with confirmed', 1)
    df.rename(columns={'test_indication': 'Contact with confirmed', 'head_ache': 'Headache', 'sore_throat': 'Sore throat', 'shortness_of_breath': 'Shortness of breath', 'cough': 'Cough', 'fever': 'Fever'}, inplace=True)
    falsestr = 'negative'
    truestr = 'positive'
    nonestr = 'other'
    df["corona_result"] = df["corona_result"].replace(falsestr, 0)
    df["corona_result"] = df["corona_result"].replace(truestr, 1)
    df["corona_result"] = df["corona_result"].replace(nonestr, np.nan)
    corona_val = [0, 1]
    df = df[df["corona_result"].isin(corona_val)]
    gender_val = ['male', 'female']
    df = df[df["gender"].isin(gender_val)]
    df["gender"] = df["gender"].replace('male', 1)
    df["gender"] = df["gender"].replace('female', 0)
    df.rename(columns={'gender': 'Male'}, inplace=True)
    df["age_60_and_above"] = df["age_60_and_above"].replace('Yes', 1)
    df["age_60_and_above"] = df["age_60_and_above"].replace('No', 0)
    df.rename(columns={'age_60_and_above': 'Age 60+'}, inplace=True)
    return df


def prepareDF(df):
    label = "corona_result"
    drop_columns = ["test_date"]
    df.drop(labels=drop_columns, axis=1, inplace=True)
    y_train = df[label]
    final_lables = [label]
    df.drop(labels=final_lables, axis=1, inplace=True)
    return df.to_numpy(), y_train


testdf = pd.read_csv('corona_tested_individuals_ver_006.english.csv.zip', error_bad_lines=False, index_col=False, encoding='utf-8', low_memory=False)
gbm = lgb.Booster(model_file='lgbm_model_all_features.txt')  # load model
testdf = replaceValuesDF(testdf)

filename = 'finalized_model.sav'
pickle.dump(gbm, open(filename, 'wb'))

testdf['test_date'] = pd.to_datetime(testdf['test_date'])

test_dates = ['2020-04-01', '2020-04-07']
testdf = dfByDate(testdf, test_dates[0], test_dates[1])

testdf, y_test = prepareDF(testdf)

y_test = y_test.astype(float)
y_pred = gbm.predict(testdf, num_iteration=gbm.best_iteration)

print('The AUC of prediction is:', roc_auc_score(y_test, y_pred))

# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])
# print("Result: ", result)
