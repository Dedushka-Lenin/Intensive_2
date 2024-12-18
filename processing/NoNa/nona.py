import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def nona(data, algreg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1)), algclass = RandomForestClassifier(max_depth=2, random_state=0)):

    for i in tqdm(data.columns):
        if i == data.columns[0] and data[i].isna().sum() != 0:
            data[i] = data[i].fillna(data[i].median())
            continue

        if data[i].isna().sum() != 0:
            indexna = data[data[i].isna()].index
            datanona = data.loc[:, data.isna().any()==False]
            X_train_nona = datanona.loc[datanona.index.isin(indexna) == False]
            X_test_nona = datanona.loc[indexna]
            y_train_nona = data[i].loc[datanona.index.isin(indexna) == False]

            if data[i].nunique() < 20 and float(data[i].unique()[~np.isnan (data[i].unique())][0]).is_integer():
                try: 
                    class_nona = algclass
                    class_nona.fit(X_train_nona, y_train_nona)
                    data.loc[data[i].isna(), i] = class_nona.predict(X_test_nona)

                except: pass

            else:
                reg_nona = algreg
                reg_nona.fit(X_train_nona, y_train_nona)
                data.loc[data[i].isna(), i] = reg_nona.predict(X_test_nona)