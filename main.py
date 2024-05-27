import pandas as pd
import numpy as np
from function import *
import statsmodels.api as sm
from tqdm import tqdm
import os
from statsmodels.formula.api import ols


pd.set_option('future.no_silent_downcasting', True)

# 读取数据
alpha = pd.read_csv("path to alpha").set_index("date")

# 计算原始因子
date_list = alpha.index
exp_ave = pd.DataFrame(index=date_list, columns=alpha.columns)

# 基金网络文件名列表
path = "path to fund_holding_net_files"
files = []
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        files.append(file)

# 读取并计算
for date in tqdm(date_list):
    if date + ".h5" in files:
        K = pd.read_hdf("path to fund_holding_net" + date + ".h5")

        # 将对角线上元素置为0
        for i in range(len(K)):
            K.iat[i, i] = 0

        vector = alpha.loc[date]
        K.reindex(index=vector.index, columns=vector.index)
        res = vector * K
        res = res.replace(0, np.nan)
        res = res[res.notnull().any(axis=1)].mean(axis=1)
        res.reindex(index=exp_ave.columns)
        exp_ave.loc[date] = res
    else:
        continue

exp_ave = exp_ave.dropna(axis=0, how="all").fillna(0).infer_objects(copy=False).mask(alpha.loc[date_list].isnull()).rename_axis('date')
exp_ave.to_csv("path to fund_exp_ave")

# 中性化处理
exp_ave = pd.read_csv("path to fund_exp_ave").set_index("date")
industry = pd.read_csv("path to industry", index_col=0, low_memory=False)
industry.index.name = "date"

i_l = list(set(alpha.index).intersection(set(exp_ave.index)).intersection(set(industry.index)))
c_l = list(set(alpha.columns).intersection(set(exp_ave.columns)).intersection(set(industry.columns)))
alpha = alignment_data(alpha, i_l, c_l)
exp_ave = alignment_data(exp_ave, i_l, c_l)
industry = alignment_data(industry, i_l, c_l)

date = industry.index
industry = industry.stack(dropna=False)
ind = pd.get_dummies(industry).replace({True: 1, False: 0})
x = pd.concat([alpha.stack(dropna=False), ind], axis=1).rename(columns={0: 'alpha'})

exp_ave = exp_ave.stack(dropna=False)
notnull_list = list(set(x[x.notnull().all(axis=1)].index).intersection(set(exp_ave[exp_ave.notnull()].index)))
exp_ave = exp_ave.loc[notnull_list].sort_index(ascending=True)
x = x.loc[notnull_list, :].sort_index(ascending=True)
o = pd.concat([exp_ave, x], axis=1).astype(float).dropna()

o['resid'] = np.nan
for i in tqdm(range(len(date))):
    y = o.loc[date[i]].iloc[:, 0]
    ols_model = sm.OLS(np.asarray(y), sm.add_constant(np.asarray(o.loc[date[i]].iloc[:, 1:-1])))
    ols_results = ols_model.fit().resid
    o.loc[date[i], 'resid'] = ols_results

Traction20d = (o.copy().iloc[:, -1].unstack().dropna(how='all').dropna(axis=1, how='all')
               .reindex(columns=alpha.columns).sort_index(ascending=True).sort_index(axis=1, ascending=True))
Traction20d = Traction20d.loc[(Traction20d.fillna(0) != 0).any(axis=1)]
Traction20d.to_csv("path to Traction20d")
