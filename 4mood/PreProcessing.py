import json
import pandas as pd
import re
import opencc
import jieba
import matplotlib.pyplot as plt
import csv
import random
import time

converter = opencc.OpenCC('t2s.json')  # 't2s.json' 表示从繁体转换为简体，'s2t.json' 表示从简体转换为繁体

#正则化
re_obj = re.compile(
    r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—！，。？、￥…（）：【】《》‘’“”\s]+")
re_obj2=re.compile(
    r'[a-zA-Z0-9’!"#$%&\'()*+,-.<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+')

def preprocess_text(text):
    res = re_obj.sub('', text)
    # 去除字符串两端的空格和换行符
    text = text.strip()
    # 去除非中文、数字、符号、空格
    text = re.sub('[^\u4e00-\u9fa5]+','',text)
    # 繁简转化
    text = converter.convert(text)
    return text



filename = 'simplifyweibo_4_moods.csv'  # CSV文件名
data = pd.read_csv(filename)
data_filtered = pd.DataFrame()
for label in data['label'].unique():
    # 获取当前类别的数据
    subset_df = data[data['label'] == label].head(30000)
    # 将当前类别的数据添加到过滤后的DataFrame
    data_filtered = pd.concat([data_filtered, subset_df])

print(len(data_filtered))
label_categories = data_filtered['label'].unique()
print(label_categories)

start_time = time.time()

data_filtered['review'] = data_filtered['review'].apply(preprocess_text)
sample_df = data_filtered.sample(n=50)

# 将DataFrame转换为字符串并打印
print(sample_df.to_string(index=False))

end_time = time.time()
execution_time = end_time - start_time
# 打印执行时间
print(f"遍历函数执行时间：{execution_time}秒")


df= data_filtered
# 去除空数据
df.dropna(inplace=True)
# 去除重复数据
df.drop_duplicates(inplace=True)
# 去除review长度小于15的数据
df = df[df['review'].str.len() >= 15]
# 重置索引
df.reset_index(drop=True, inplace=True)
print(len(df))

label_counts = df['label'].value_counts()
print(label_counts)

plt.bar(label_counts.index, label_counts.values)
# 添加标题和标签
plt.title('Label Counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


stopwords = []
with open('stopword.txt', 'r', encoding='utf-8') as file:
    stopwords = [line.strip() for line in file.readlines()]

# 分词函数
def tokenize(text):
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords]
    return words

# 每个元素进行分词并去除停用词
df['review'] = df['review'].apply(tokenize)
df.sample(10)

# 存储为新的CSV文件
df.to_csv('processed_4mood_data.csv', index=False)

# data_list = df.to_dict('records')
# # 将数据列表转换为JSON字符串
# json_data = json.dumps(data_list, ensure_ascii=False, indent=4)
# # 将JSON数据写入txt文件
# with open('processed_4mood_data.txt', 'w', encoding='utf-8') as file:
#     file.write(json_data)
