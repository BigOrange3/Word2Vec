# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def random_batch():
    random_inputs = []  # 存放两个目标词的one-hot
    random_labels = []  # 存放两个目标词分别对应的某一个上下文中的单词
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)  # 不重复选batch_size个

    for i in random_index:
        # 随机选两个对，作为样本
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target   即编号为[skip_grams[i][0]的目标词的one-hot向量
        random_labels.append(skip_grams[i][1])  # context word   上下文单词的字典编号

    return random_inputs, random_labels


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        # 这里的参数是（voc_size, embedding_size）（8，2），但是控制台中W的维度是（2，8），原因就是nn在计算时会将W进行转置，这样就解释的通了。
        self.W = nn.Linear(voc_size, embedding_size, bias=False)  # voc_size -> embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)  # embedding_size -> voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X)  # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer)  # output_layer : [batch_size, voc_size]
        return output_layer


if __name__ == '__main__':
    batch_size = 2  # mini-batch size
    embedding_size = 2  # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}  # 存放元素为{单词：编号}这样的字典，一共8个单词，编号为0~7.  但顺序每次执行都不同，不懂。
    voc_size = len(word_list)  # 8 （一共八种单词）

    # Make skip gram of one size window
    skip_grams = []  # 存储所有中心词和上下文构成的对
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]  # 中心词的编号
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]  # 上下文 词的编号（窗口大小为1）
        for w in context:
            skip_grams.append([target, w])  # 目标词和上下文某一个词构成的对儿  [目标词，上下文的某一个]

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)  # 输入模型前 类型都要转换为tensor

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        # output : [2, 8], target_batch : [2,]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        # W矩阵即为词嵌入矩阵，虽然不知道为啥维度和定义linear层时的不一样（刚好互为转置）。
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # plt.show()
