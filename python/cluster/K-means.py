# -*- coding: utf-8 -*-
"""
Created on 2018/9/16 1:37

@author: wujian
"""

import numpy as np
import pandas as pd
import seaborn

# test
def test():
    # load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    writer = pd.ExcelWriter("output.xlsx")
    dataset.to_excel(writer, "Sheet1")
    writer.save()
    dataset['class'][dataset['class']=='Iris-setosa']=0
    dataset['class'][dataset['class']=='Iris-versicolor']=1
    dataset['class'][dataset['class']=='Iris-virginica']=2
    return dataset.iloc[:, 0:2]

def cal_dist(vec1, vec2):
    """计算距离
    
    Parameters
    ----------
    vec1: numpy.array
    
    vec2: numpy.array
    
    Returns
    -------c
    distance between vec1 and ve2
    """
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def rand_centroid(dataset, k):
    """初始化聚类中心
    
    Parameters
    ----------
    dataset: pandas.core.frame.DataFrame
        origin data
    
    k: int
        the num of centroid
    
    Returns
    -------
    centroid: numpy.matrix
        the init centroids of dataset
    """
    num, dim = dataset.shape
    centroid = np.mat((np.zeros((k, dim))))
    for i in range(dim):
        min_i = min(dataset.iloc[:, i])
        range_i = float(max(dataset.iloc[:, i] - min_i))
        centroid[:, i] = np.mat(min_i + range_i * np.random.rand(k, 1))
    return centroid

def Kmeans(dataset, k):
    num, dim = dataset.shape
    # 分配样本到最近的簇，[簇序号，距离]
    cluster_assment = np.mat(np.zeros((num, 2)))
    
    # step1:
    centroid = rand_centroid(dataset, k)
    print("最初的中心 = {}".format(centroid))
    
    # 标志位，如果迭代前后样本分类发生变化，值为True，否则为False
    cluster_changed = True
    # 迭代次数
    iter_num = 0
    # 所有样本分配结果不再改变，迭代终止
    while cluster_changed:
        cluster_changed = False
        # step2: 分配到最近的聚类中心对应的簇中
        for i in range(num):
            min_dist = np.inf
            min_index = np.inf
            for j in range(k):
                dist_j = cal_dist(centroid[j, :], dataset.values[i, :])
                if min_dist > dist_j:
                    min_dist, min_index = dist_j, j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist
        iter_num += 1
        
        # step3: 更新聚类中心
        for cent in range(k):
            index = np.nonzero(cluster_assment[:, 0].A == cent)[0]
            centroid[cent, :] = np.mean(dataset.iloc[index], axis=0)
    return [centroid, cluster_assment]

#2维数据聚类效果显示
def datashow(dataSet,k,centroids,clusterAssment):  #二维空间显示聚类结果
    from matplotlib import pyplot as plt
    num,dim=dataSet.shape  #样本数num ,维数dim
    
    if dim!=2:
        print('sorry,the dimension of your dataset is not 2!')
        return 1
    marksamples=['or','ob','og','ok','^r','^b','<g'] #样本图形标记
    if k>len(marksamples):
        print('sorry,your k is too large,please add length of the marksample!')
        return 1
        #绘所有样本
    for i in range(num):
        markindex=int(clusterAssment[i,0])#矩阵形式转为int值, 簇序号
        #特征维对应坐标轴x,y；样本图形标记及大小
        plt.plot(dataSet.iat[i,0],dataSet.iat[i,1],marksamples[markindex],markersize=6)
 
    #绘中心点            
    markcentroids=['o','*','^']#聚类中心图形标记
    label=['0','1','2']
    c=['yellow','pink','red']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],markcentroids[i],markersize=15,label=label[i],c=c[i])
        plt.legend(loc = 'upper left')
    plt.xlabel('sepal length')  
    plt.ylabel('sepal width') 
   
    plt.title('k-means cluster result') #标题        
    plt.show()
    
    if dim != 2:
        print('sorry,the dimension of your dataset is not 2!')
        return 1


if __name__ == "__main__":
    dataset = test()
    centroid, cluster_assment = Kmeans(dataset, 3)
    print centroid
    datashow(dataset, 3, centroid, cluster_assment)
