import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def nona(data, algreg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1)), algclass = RandomForestClassifier(max_depth=2, random_state=0)):

    """
        Функция прогнозирования пропущенных значений. Проходим по всем столбцам, выявляя столбец с пропущенными значениями. Делим выборку на обучающую и тестовую. Прогнозируем пропущенные значения с помощью машинного обучения.
            Параметры:
                Data: подготовленный набор данных
                algreg: Алгоритм регрессии для прогнозирования пропущенных значений в столбцах
                algclss: Алгоритм классификации для прогнозирования пропущенных значений в столбцах
    """

    for i in tqdm(data.columns): # пройтись по всем столбцам от первого до последнего
        # если в первом столбце есть пробелы, заполните пропущенные значения медианой
        if i == data.columns[0] and data[i].isna().sum() != 0:
            data[i] = data[i].fillna(data[i].median())
            continue

        # проверьте есть ли пробелы в столбце, если есть - алгоритм работает
        if data[i].isna().sum() != 0:
            indexna = data[data[i].isna()].index # отображать индексы строк в столбце с пробелами
            datanona = data.loc[:, data.isna().any()==False] # выводить столбцы без пробелов
            X_train_nona = datanona.loc[datanona.index.isin(indexna) == False] # создаем обучающую выборку для обучения, включающую только столбцы без пробелов, выполняем волокнистый анализ индекса тестовой выборки, оставляем только те столбцы, в которых знаем прогнозируемое значение
            X_test_nona = datanona.loc[indexna] # создайте тестовую выборку, оставьте столбцы, в которых мы не знаем прогнозируемое значение
            y_train_nona = data[i].loc[datanona.index.isin(indexna) == False] # оставьте значения в прогнозируемом столбце без пробелов

            # если предсказанное число является целым числом и число предсказанных значений меньше 20, мы решаем классификацию
            if data[i].nunique() < 20 and float(data[i].unique()[~np.isnan (data[i].unique())][0]).is_integer():
                class_nona = algclass
                class_nona.fit(X_train_nona, y_train_nona)
                data.loc[data[i].isna(), i] = class_nona.predict(X_test_nona) # предсказать значения и вставить их вместо пропущенных значений в столбце

            else:
                reg_nona = algreg
                reg_nona.fit(X_train_nona, y_train_nona)
                data.loc[data[i].isna(), i] = reg_nona.predict(X_test_nona) # предсказать значения и вставить их вместо пропущенных значений в столбце