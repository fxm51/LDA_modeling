import math
import os
import glob
import jieba
import json

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import matplotlib.pyplot as plt

folder_path = ".\\data"  # 文件夹路径
file_extension = "*.txt"  # 文件扩展名


# 数据采样，获得以词、字为单元的段落
def paragraph_sampling():
    # 获取path和label
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    # print(file_paths)
    labels = []
    words_list = []
    for path in file_paths:
        path = path.replace('.\\data\\', '', 1)
        labels.append(path.replace('.txt', '', 1))
    # print(labels)

    # 分割词
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stop_words = [line.strip() for line in lines]

    # 循环遍历所有文件路径，并读取文件内容
    # 数据处理，清除无用数据
    english = u'[a-zA-Z0123456789’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str1 = u'[①②③④⑤⑥⑦⑧⑨⑩_“”、。《》！，：；？‘’」「…『』（）<>【】．·.—*-~﹏]'
    remove_set = set(english + str1)
    for path in file_paths:
        # print(path)
        with open(path, "r", encoding='ANSI') as file:
            temp = file.read()
            result = filter(lambda x: x not in remove_set, temp)
            new_temp = ''.join(result)
            new_temp = new_temp.replace("\n", '')
            new_temp = new_temp.replace("\u3000", '')
            new_temp = new_temp.replace(" ", '')
            words_list.append(new_temp)

    count = 0
    f = open('data_final.json', 'w', encoding='utf-8', newline='\n')
    json_data = {}
    # 对于整理后的文本进行均匀采样
    for text in words_list:
        new_text = []
        new_words = []
        label = labels[count]
        split_words = list(jieba.cut(text))
        for word in split_words:
            if word not in stop_words:
                new_text.append(word)
                new_words.append(word)
                new_words.append(' ')
        cut = int(len(new_text)/13)
        paragraph_begin = 0
        sub_data = {}
        for num in range(13):
            paragraph = ''.join(new_text[paragraph_begin:paragraph_begin + 500])
            paragraph_word = ''.join(new_words[paragraph_begin:paragraph_begin + 1000])
            paragraph_char = ''
            for char in paragraph:
                paragraph_char += char
                paragraph_char += ' '
            paragraph_begin += cut
            sub_data[str(num)] = {
                "paragraph": paragraph,
                "paragraph of word": paragraph_word,
                "paragraph of char": paragraph_char
            }
        json_data[label] = sub_data
        count += 1
    f.write(json.dumps(json_data, ensure_ascii=False, indent=1))
    f.close()


def lda_modeling(paragraph_word, topics_num, mode):
    # 将段落数据转换为文本表示
    paragraphs_text = [''.join(paragraph_word[p]) for p in range(len(paragraph_word))]
    if mode == "char":
        # 构建字频矩阵
        vectorizer = CountVectorizer(strip_accents='unicode', analyzer='char_wb', max_df=0.5, min_df=10)
        frequency_matrix = vectorizer.fit_transform(paragraph_word)
    elif mode == "word":
        # 构建词频矩阵
        vectorizer = CountVectorizer(strip_accents='unicode', analyzer='word', max_df=0.5, min_df=10)
        frequency_matrix = vectorizer.fit_transform(paragraphs_text)
    # 训练LDA模型
    lda_model = LatentDirichletAllocation(n_components=topics_num, max_iter=50, learning_method='online',
                                        learning_offset=50, doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0)
    lda_model.fit(frequency_matrix)
    topics = lda_model.transform(frequency_matrix)

    # 计算困惑度
    perplexity = lda_model.perplexity(frequency_matrix)
    # print("Perplexity: {:.2f}".format(perplexity))

    # 输出每个主题的关键词
    for topic_idx, topic in enumerate(lda_model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-11:-1]]))

    # 计算一致性
    similarity_matrix = np.zeros((len(lda_model.components_), len(lda_model.components_)))
    for i in range(len(lda_model.components_)):
        for j in range(len(lda_model.components_)):
            similarity_matrix[i, j] = cosine_similarity(lda_model.components_[i].reshape(1, -1),
                                                        lda_model.components_[j].reshape(1, -1))
    consistency = np.mean(np.triu(similarity_matrix, k=1))
    # print("Consistency: {:.2f}".format(consistency))

    return lda_model, topics, perplexity, consistency




def test(labels, topics):
    X = pd.DataFrame(topics)
    Y = []
    for label in labels:
        for i in range(13):
            Y.append(label)
    Y = pd.Series(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(class_weight='balanced', random_state=37)
    clf = clf.fit(X_train, y_train)  # 拟合训练集
    score_c = clf.score(X_test, y_test)  # 输出测试集准确率
    return score_c


# main
if __name__ == '__main__':
    # paragraph_sampling()
    # 获取labels
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    labels = []
    for path in file_paths:
        path = path.replace('.\\data\\', '', 1)
        labels.append(path.replace('.txt', '', 1))
    paragraph_word = []
    paragraph_char = []
    # 读取保存的json文件, 获取以词、字为单元的段落
    with open('data_final.json', 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        for label in labels:
            for i in range(13):
                paragraph_word.append(json_data[label][str(i)]["paragraph of word"])
                paragraph_char.append(json_data[label][str(i)]["paragraph of char"])
    fp.close()
    scores = []
    perplexitys = []
    consistencys = []
    for i in range(20):
        print("Topic num: ", i+1)
        topics_num = i+1
        lda_model, topics, perplexity, consistency = lda_modeling(paragraph_char, topics_num, "char")
        perplexitys.append(perplexity)
        consistencys.append(consistency)
        scores.append(test(labels, topics))

    for i in range(20):
        print("Topic num: ", i+1)
        print("perplexity: {:.2f}".format(perplexitys[i]),
              "consistency: {:.2f}".format(consistencys[i]),
              "score: {:.2f}".format(scores[i]))

    x = np.arange(1, 21)
    fig, axes = plt.subplots(3, 1)

    axes[0].plot(x, perplexitys)
    axes[0].set_xlabel('topic num')
    axes[0].set_ylabel('困惑度')

    axes[1].plot(x, consistencys)
    axes[1].set_xlabel('topic num')
    axes[1].set_ylabel('一致性')

    axes[2].plot(x, scores)
    axes[2].set_xlabel('topic num')
    axes[2].set_ylabel('模型准确率')

    plt.show()










