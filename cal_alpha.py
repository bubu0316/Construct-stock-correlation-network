import pandas as pd
from function import *
N = 20

# 读取数据
adj_factor = read_data("path to adj_factor.csv")
close = read_data("path to close.csv")

# 对齐数据
i_l = list(set(adj_factor.index).intersection(set(close.index)))
c_l = list(set(adj_factor.columns).intersection(set(close.columns)))
adj_factor = alignment_data(adj_factor, i_l, c_l)
close = alignment_data(close, i_l, c_l)

adj_close = adj_factor * close
N_rise = adj_close.iloc[N:, :].sub(adj_close.shift(N).iloc[N:, :]).div(adj_close.shift(N).iloc[N:, :])
N_rise = N_rise.loc['start_date':]
med = N_rise.median(axis=1)
alpha = N_rise.sub(med, axis=0)
alpha.to_csv("path to alpha")
