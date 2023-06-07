import json
import pandas as pd
import re
import opencc
import jieba
import matplotlib.pyplot as plt


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



# 读取文件
with open('dataset/train/usual_train.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)

print(len(data))
df = pd.DataFrame(columns=('content', 'tag'))
label_mapping = {'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'surprise': 4, 'neutral': 5}

# 遍历数据
for item in data:
    content = item['content']
    # 正则匹配
    processed_content = preprocess_text(content)
    label = item['label']
    print("Content:", processed_content)
    print("Label:", label)
    df = pd.concat([df, pd.DataFrame({'content': [processed_content], 'tag': [label]})], ignore_index=True)

# 去除重复的content
df = df.drop_duplicates(subset=['content'])
# 去除空的content
df = df.dropna(subset=['content'])
# 重置索引
df = df.reset_index(drop=True)
print(len(df))
df = df[df['content'].apply(lambda x: len(x) >= 15)]
print(len(df))

label_counts = df['tag'].value_counts()
print(label_counts)

plt.bar(label_counts.index, label_counts.values)
# 添加标题和标签
plt.title('Tag Counts')
plt.xlabel('Tag')
plt.ylabel('Count')
plt.show()


df['tag'] = df['tag'].map(label_mapping)
label_counts = df['tag'].value_counts()
print(label_counts)



stopwords = []
with open('stopword.txt', 'r', encoding='utf-8') as file:
    stopwords = [line.strip() for line in file.readlines()]

# 对content列的每个元素进行分词并去除停用词
df['content'] = df['content'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))
df.sample(10)

data_list = df.to_dict('records')
# 将数据列表转换为JSON字符串
json_data = json.dumps(data_list, ensure_ascii=False, indent=4)
# 将JSON数据写入txt文件
with open('processed_usual_data.txt', 'w', encoding='utf-8') as file:
    file.write(json_data)

