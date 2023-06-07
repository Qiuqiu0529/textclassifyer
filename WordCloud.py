from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# 假设处理后的数据已经存储在名为'processed_data.txt'的文件中
data = pd.read_json('processed_data.txt')

# 合并所有文本内容为一个字符串，用于构建总的词云图
all_content = ' '.join(data['content'].tolist())

font_path="C:\\Windows\\Fonts\\STFANGSO.ttf"

# 创建总的词云图对象
all_wordcloud = WordCloud(font_path=font_path,background_color='white').generate(all_content)

# 绘制总的词云图
plt.figure()
plt.imshow(all_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Total Word Cloud')

# 显示总的词云图
plt.show()

# 根据标签分组
grouped = data.groupby('tag')

# 构建每个标签的词云图
for tag, group in grouped:
    # 提取对应标签组的内容并合并为一个字符串
    content = ' '.join(group['content'].tolist())

    # 创建词云对象
    wordcloud = WordCloud(font_path=font_path,background_color='white').generate(content)

    # 绘制词云图
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tag: {}'.format(tag))

# 显示所有词云图
plt.show()