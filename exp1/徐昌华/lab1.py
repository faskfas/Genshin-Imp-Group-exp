# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 1. 库的导入与数据的读入
print("1. 导入必要的库并读取数据...")
try:
    # 尝试不同的编码格式
    primitive_data = pd.read_csv("http://storage.amesholland.xyz/data.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        primitive_data = pd.read_csv("http://storage.amesholland.xyz/data.csv", encoding='gbk')
    except UnicodeDecodeError:
        try:
            primitive_data = pd.read_csv("http://storage.amesholland.xyz/data.csv", encoding='gb2312')
        except UnicodeDecodeError:
            try:
                primitive_data = pd.read_csv("http://storage.amesholland.xyz/data.csv", encoding='latin1')
            except:
                print("无法用常见编码格式读取文件，请检查文件编码")
                exit()
# 获取原始数据的列名
columns = primitive_data.columns
# 创建30行空数据
empty_rows = pd.DataFrame([[np.nan] * len(columns)] * 30, columns=columns)
# 将空数据添加到原始数据末尾
primitive_data_with_empty = pd.concat([primitive_data, empty_rows], ignore_index=True)
print("原始数据读取完成，前5行和后5行如下：")
print("前5行：")
print(primitive_data.head())
print("\n后5行：")
print(primitive_data.tail())
print("数据形状：", primitive_data_with_empty.shape)
print()

# 2. 删除空行
print("2. 删除所有包含空值的行...")
primitive_data_1 = primitive_data.dropna(how='any')
print("删除空行后的数据形状：", primitive_data_1.shape)
print("前5行和后5行如下：")
print("前5行：")
print(primitive_data_1.head())
print("\n后5行：")
print(primitive_data_1.tail())
print()

# 3. 数据过滤：traffic != 0 且 from_level == '一般节点'
print("3. 过滤数据：traffic != 0 且 from_level == '一般节点'...")
data_after_filter_1 = primitive_data_1.loc[primitive_data_1["traffic"] != 0]
data_after_filter_2 = data_after_filter_1.loc[data_after_filter_1["from_level"] == '一般节点']
print("过滤后的数据形状：", data_after_filter_2.shape)
print("前5行和后5行如下：")
print("前5行：")
print(data_after_filter_2.head())
print("\n后5行：")
print(data_after_filter_2.tail())
print()

# 4. 加权采样：to_level为"一般节点"和"网络核心"的权重比为1:5
print("4. 进行加权采样（权重比：一般节点:网络核心 = 1:5）...")
data_before_sample = data_after_filter_2.copy()
data_before_sample['weight'] = data_before_sample['to_level'].apply(
    lambda x: 1 if x == '一般节点' else 5
)
weight_sample_finish = data_before_sample.sample(n=50, weights='weight', random_state=42)
weight_sample_finish = weight_sample_finish.drop(columns=['weight'])  # 移除权重列
print("加权采样后的50个样本（前5行和后5行）：")
print("前5行：")
print(weight_sample_finish.head())
print("\n后5行：")
print(weight_sample_finish.tail())
print("采样数据形状：", weight_sample_finish.shape)
print()

# 5. 随机抽样
print("5. 进行随机抽样...")
random_sample_finish = data_before_sample.sample(n=50, random_state=42)
random_sample_finish = random_sample_finish.drop(columns=['weight'])  # 移除权重列
print("随机抽样后的50个样本（前5行和后5行）：")
print("前5行：")
print(random_sample_finish.head())
print("\n后5行：")
print(random_sample_finish.tail())
print("采样数据形状：", random_sample_finish.shape)
print()

# 6. 分层抽样：按 to_level 分层，一般节点抽17个，网络核心抽33个
print("6. 进行分层抽样：一般节点17个，网络核心33个...")
ybjd = data_before_sample[data_before_sample['to_level'] == '一般节点']
wlhx = data_before_sample[data_before_sample['to_level'] == '网络核心']
ybjd_sample = ybjd.sample(n=17, random_state=42)
wlhx_sample = wlhx.sample(n=33, random_state=42)
after_sample = pd.concat([ybjd_sample, wlhx_sample])
after_sample = after_sample.drop(columns=['weight'])  # 移除权重列
print("分层抽样后的50个样本（前5行和后5行）：")
print("前5行：")
print(after_sample.head())
print("\n后5行：")
print(after_sample.tail())
print("采样数据形状：", after_sample.shape)
print("各类别数量：")
print(after_sample['to_level'].value_counts())