from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

font_path="C:\\Windows\\Fonts\\STFANGSO.ttf"


# 读取新的CSV文件
df = pd.read_csv('processed_4mood_data.csv')
# df_sampled = df.sample(frac=0.2, random_state=42)
# df=df_sampled

# 合并所有review文本
all_reviews = ' '.join(df['review'].sum())

# 生成总的词云图
wordcloud = WordCloud(width=800, height=400,font_path=font_path, background_color='white').generate(all_reviews)


# 绘制总的词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('total')
plt.show()

# 按tag分组，合并每个tag的review文本
grouped_reviews = df.groupby('label')['review'].sum()

# 生成每个tag的词云图
for label, reviews in grouped_reviews.items():
    tag_wordcloud = WordCloud(width=800, height=400,font_path=font_path, background_color='white').generate(' '.join(reviews))

    # 绘制每个tag的词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(tag_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{label}')
    plt.show()