import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
from recommenders.utils.timer import Timer

plt.rcParams['font.sans-serif'] = ['SimHei']  # 允许中文标题
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止画图报错


def cosine_similarity(vector1, vector2):
    # 余弦距离的值在 -1 到 1 之间，值越接近 1 表示向量越相似，值越接近 -1 表示向量越不相似。
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


def random_batch():
    random_inputs = []  # 存放两个目标词的one-hot
    random_labels = []  # 存放两个目标词分别对应的某一个上下文中的单词的编号
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)  # 不重复选batch_size个
    for i in random_index:
        # 随机选两个对，作为样本
        # 这里原本是用单位阵来获取onehot，但是内存爆了，所以换了方式
        # random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target   即编号为skip_grams[i][0]的目标词的one-hot向量
        onehot = []
        for j in range(voc_size):
            if j == i:
                onehot.append(1)
            else:
                onehot.append(0)

        random_inputs.append(onehot)
        random_labels.append(skip_grams[i][1])  # context word   上下文单词的字典编号

    return random_inputs, random_labels


def compute_similarity(w1, w2):
    # 函数返回：w1和w2两个单词词向量的余弦相似度
    index_1, index_2 = word_dict[w1], word_dict[w2]
    # 找到编号为index的词的词向量,并转化为array
    tensor_1 = model.W.weight[:, index_1].cpu()  # 从GPU放回CPU，否则无法转换为array
    array_1 = tensor_1.detach().numpy()
    tensor_2 = model.W.weight[:, index_2].cpu()
    array_2 = tensor_2.detach().numpy()
    # 计算相似度
    sim = cosine_similarity(array_1, array_2)
    return sim


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # 这里的参数是（voc_size, embedding_size）（8，2），但是控制台中W的维度是（2，8），原因就是nn在计算时会将W进行转置，这样就解释的通了。
        self.W = nn.Linear(voc_size, embedding_size, bias=False)  # voc_size -> embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)  # embedding_size -> voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X)  # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer)  # output_layer : [batch_size, voc_size]
        return output_layer


if __name__ == '__main__':
    with Timer() as time:
        batch_size = 128  # mini-batch size
        embedding_size = 100  # embedding size
        Epoch = 10000  # 迭代次数
        # 读取数据并处理
        df = pd.read_csv("labeledTrainData.tsv", sep='\t', escapechar='\\')
        sentences = []  # 处理后的句子列表
        for i in range(100):
            # 去掉标点符号, 因为标点附近本就有空格，所以用空字符代替即可
            temp = df["review"][i].replace(",", "").replace(".", "").replace('"', ''). \
                replace('\'', '').replace("(", "").replace(")", "").replace("-", ""). \
                replace("!", "").replace("?", "")
            sentences.append(temp)  # 放入列表

        word_sequence = " ".join(sentences).split()
        word_list = " ".join(sentences).split()
        word_list = list(set(word_list))
        word_dict = {w: i for i, w in enumerate(word_list)}  # 存放元素为{单词：编号}这样的字典，一共8个单词，编号为0~7.  但顺序每次执行都不同，不懂。
        voc_size = len(word_list)  # 8 （一共八种单词）

        # skip gram（窗口大小为1）
        skip_grams = []  # 存储所有中心词和上下文构成的对
        for i in range(1, len(word_sequence) - 1):
            target = word_dict[word_sequence[i]]  # 中心词的编号
            context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]  # 上下文 词的编号（窗口大小为1）
            for w in context:
                skip_grams.append([target, w])  # 目标词和上下文某一个词构成的对儿  [目标词，上下文的某一个]

        model = Word2Vec()
        model = model.to("cuda:0")  # 0 表示第一个GPU

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        l_loss = []  # 记录loss画图
        # Training
        for epoch in range(Epoch):
            input_batch, target_batch = random_batch()  # 随机选取batch sample
            input_batch = torch.Tensor(input_batch)
            target_batch = torch.LongTensor(target_batch)  # 输入模型前 类型都要转换为tensor
            input_batch = input_batch.to("cuda:0")
            target_batch = target_batch.to("cuda:0")  # 将数据放到GPU上

            optimizer.zero_grad()  # 每次epoch都要将梯度重置为0，因为默认梯度是累加的
            output = model(input_batch)

            loss = criterion(output, target_batch)  # 这个loss是batch的
            loss_temp = loss.cpu()  # 放到cpu上，否则后面画不了图
            loss_temp = loss_temp.detach().numpy()  # 转换为ndarry
            l_loss.append(loss_temp)  # 放入列表
            print(f"Epoch: {epoch}, loss = {loss}")
            loss.backward()  # 计算梯度方向传播
            optimizer.step()  # 根据梯度更新参数

        x = np.linspace(1, Epoch, Epoch)  # 取5000个点
        # l_loss = l_loss.cpu()
        plt.plot(x, l_loss)
        plt.xlabel('迭代次数')
        plt.ylabel('loss')
        plt.title('loss变化')
        plt.legend(loc='lower right')  # legend：显示标签
        plt.grid()  # 网格
        plt.show()

    print("运行时间:", time)
