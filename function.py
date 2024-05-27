import numpy as np
import pandas as pd


# 读取数据
def read_data(path):
    data = pd.read_csv(path)
    data = data.set_index("date")
    data.index = pd.to_datetime(data.index)
    return data


# 对齐数据
def alignment_data(data, i, c):
    data = data.loc[i, c]
    data = data.sort_index(ascending=True)
    data = data.sort_index(axis=1, ascending=True)
    return data


# 月频转日频
def month_to_day(factor):
    ret = pd.read_csv("path to ret", index_col=0)
    ret.index = pd.to_datetime(ret.index)
    day_factor = ret.copy()[ret.index >= factor.index[0]]
    day_factor.loc[:, :] = np.nan
    j = 0
    for i in day_factor.index:
        if j + 1 < len(factor.index):
            if i < factor.index[j + 1]:
                day_factor.loc[i, :] = factor.iloc[j, :]
            else:
                j += 1
                day_factor.loc[i, :] = factor.iloc[j, :]
        elif i >= factor.index[j]:
            day_factor.loc[i, :] = factor.iloc[j, :]
    return day_factor
