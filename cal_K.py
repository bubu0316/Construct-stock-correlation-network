import pandas as pd
import numpy as np
from function import *
from tqdm import tqdm
import ray


# 初始化ray
ray.init(num_cpus=48)


# 并行任务
@ray.remote
def task(date, s_l, a_d_m):
    print(date)
    key = max([x for x in s_l if x <= date])
    holding = pd.read_hdf('path to top_ten_per_fund.h5', key="date_" + key.replace('-', '_'))
    I = (holding / a_d_m.loc[date]).fillna(0)
    res = pd.DataFrame(index=I.columns, columns=I.columns).infer_objects(copy=False).fillna(0)
    for i in I.index:
        data = np.minimum.outer(I.loc[i].values, I.loc[i].values)
        J = pd.DataFrame(data, columns=I.columns, index=I.columns)
        res = res + J
    res.to_hdf("path to fund_holding_net" + str(date) + ".h5", key="net")


# 读取成交额数据
amount = pd.read_csv("path to amount").set_index("date")
amount_day20_mean = amount.rolling(20).mean().loc['start_date':].replace(0, 1)

# 读取基金数据并计算每天基金的所持股票
fund = pd.read_csv("path to fund_date", low_memory=False)
fund = fund.loc[:, ['FUND_ID', 'REPORT_DATE', 'TICKER_SYMBOL', 'MARKET_VALUE']]

# 只要季度末的数据
fund = fund[fund['REPORT_DATE'].str.endswith(('03-31', '06-30', '09-30', '12-31'))]

fund = fund.set_index('REPORT_DATE').sort_values(by=['REPORT_DATE', 'FUND_ID'], ascending=[True, True])
fund = fund[fund['TICKER_SYMBOL'].str.startswith(('00', '3', '6'))]
fund['TICKER_SYMBOL'] = np.where(fund['TICKER_SYMBOL'].astype(str).str.startswith('6'),
                                 fund['TICKER_SYMBOL'].astype(str) + '.SH', fund['TICKER_SYMBOL'].astype(str) + '.SZ')
fund_date = np.unique(fund.index)

# 保存数据
for i in tqdm(fund_date):
    temp = fund.loc[i].set_index('TICKER_SYMBOL')
    top_ten_per_fund = temp.groupby('FUND_ID')['MARKET_VALUE'].nlargest(10).unstack().reindex(
        columns=amount_day20_mean.columns)
    top_ten_per_fund.to_hdf("path to top_ten_per_fund.h5", key="date_" + i.replace('-', '_'), mode='a')

# 计算K值
date_list = amount_day20_mean.index

alpha = pd.read_csv("path to alpha").set_index("date")

fundCode_dict = pd.HDFStore('path to top_ten_per_fund.h5', mode='r')
season_list = fundCode_dict.keys()
season_list = [s[6:].replace('_', '-') for s in season_list]

# for date in tqdm(date_list):
#     key = max([x for x in season_list if x <= date])
#     holding = fundCode_dict["date_" + key.replace('-', '_')]
#     I = (holding / amount_day20_mean.loc[date]).fillna(0)
#     res = I.apply(min_i, axis=1).sum().unstack()

# 将season_list和amount_day20_mean放入Ray对象存储中
season_list_id = ray.put(season_list)
amount_day20_mean_id = ray.put(amount_day20_mean)

# 并行计算
ray.get([task.remote(date, season_list_id, amount_day20_mean_id) for date in date_list])

fundCode_dict.close()
ray.shutdown()
