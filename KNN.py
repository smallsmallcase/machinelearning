# @Time    : 2017/9/23 8:56
# @Author  : Jalin Hu
# @File    : KNN.py
# @Software: PyCharm
import numpy
import operator
import matplotlib

def create_data_set():
    group = numpy.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


def classify(inx, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diffmat = numpy.tile(inx, (dataset_size, 1)) - dataset
    sqdiffmat = diffmat ** 2  # 对应元素的内积
    sqdistance = sqdiffmat.sum(axis=1)  # 每一行相加
    distance = sqdistance ** 0.5
    sorteddistance = distance.argsort()  # 返回distances中元素从小到大排序后的索引值
    classCount = {}
    for i in range(k):
        votelabel = labels[sorteddistance[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    inx = numpy.array([[1, 20]])
    result = classify(inx=inx, dataset=group, labels=labels, k=3)
    print(result)
