import json
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm


# with open('processed_usual_data.txt', 'r', encoding='utf-8') as file:
#     train_data = json.load(file)
#
# with open('processed_usual_test_data.txt', 'r', encoding='utf-8') as file:
#     test_data = json.load(file)
#
# train_texts, train_tags = [], []
# test_texts, test_tags = [], []
#
# for data in train_data:
#     train_texts.append(data['content'])
#     train_tags.append(data['tag'])
#
# for data in test_data:
#     test_texts.append(data['content'])
#     test_tags.append(data['tag'])

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(train_texts, train_tags, test_size=0.2, random_state=42)


df = pd.read_csv('processed_4mood_data.csv')

# 提取特征列和标签列
features = df['review']  # 假设特征列为 'content'
labels = df['label']  # 假设标签列为 'tag'

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)


# 输出训练集和测试集的样本数量
print("训练集样本数量:", len(X_train))
print("测试集样本数量:", len(X_test))

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
# 在训练集上进行文本向量化
X_train_tf = vectorizer.fit_transform(X_train)
# 在测试集上进行文本向量化
X_test_tf = vectorizer.transform(X_test)
# 向量化后产生过多特征，用方差分析进行特征选择
f_classif(X_train_tf,y_train)
# 使用方差作为评估函数，选择20000个最佳特征
selector = SelectKBest(f_classif, k=min(20000,X_train_tf.shape[1]))
# 对训练数据进行特征选择和降维
selector.fit(X_train_tf, y_train)
selected_features = selector.get_support(indices=True)
print("Selected Features:", selected_features)
X_train_tf = selector.transform(X_train_tf)
X_test_tf = selector.transform(X_test_tf)


# 创建词袋向量化器
vectorizer_bag=CountVectorizer()
# 词频
X_train_bag = vectorizer_bag.fit_transform(X_train)
X_test_bag = vectorizer_bag.transform(X_test)
print(X_train_bag)
f_classif(X_train_bag,y_train)
# 使用方差作为评估函数，选择20000个最佳特征
selector1 = SelectKBest(f_classif, k=min(20000,X_train_bag.shape[1]))
selector1.fit(X_train_bag, y_train)
selected_features_bag = selector1.get_support(indices=True)
print("Selected Features:", selected_features_bag)
X_train_bag = selector1.transform(X_train_bag)
X_test_bag = selector1.transform(X_test_bag)

def draw_accuracy_comparison(accuracy_tf, accuracy_bag):
    labels = ['TF-IDF', 'Bag-of-Words']
    accuracies = [accuracy_tf, accuracy_bag]
    plt.bar(labels, accuracies)
    plt.xlabel('Feature Representation')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.show()


def softmax():
    # softmax
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=30)
    # 使用 TF-IDF 特征训练模型
    softmax.fit(X_train_tf, y_train)
    y_pred_tf = softmax.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(accuracy_tf)
    print(classification_report(y_test, y_pred_tf))

    # 使用词袋模型特征训练模型
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=30)
    softmax.fit(X_train_bag, y_train)
    y_pred_bag = softmax.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(accuracy_bag)
    print(classification_report(y_test, y_pred_bag))

    draw_accuracy_comparison(accuracy_tf, accuracy_bag)


softmax()

def KNN():
    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train_tf, y_train)
    y_pred_tf = knn.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))

    knn.fit(X_train_bag, y_train)
    y_pred_bag = knn.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))

    draw_accuracy_comparison(accuracy_tf, accuracy_bag)

def decision_tree():
    # decision_tree
    dt = DecisionTreeClassifier()

    dt.fit(X_train_tf, y_train)
    y_pred_tf = dt.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))

    dt.fit(X_train_bag, y_train)
    y_pred_bag = dt.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))

    draw_accuracy_comparison(accuracy_tf, accuracy_bag)

def native_bayes():
    nb = MultinomialNB()
    nb.fit(X_train_tf, y_train)
    y_pred_tf = nb.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))

    nb.fit(X_train_bag, y_train)
    y_pred_bag = nb.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))

    draw_accuracy_comparison(accuracy_tf, accuracy_bag)


def SVM():
    clf = svm.SVC(kernel='linear')

    clf.fit(X_train_tf, y_train)
    y_pred_tf = clf.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))

    clf.fit(X_train_bag, y_train)
    y_pred_bag = clf.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))

    draw_accuracy_comparison(accuracy_tf, accuracy_bag)
