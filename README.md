# 2022-guosai
第一题
SVMcgForRegress.m  引用 matlab神经网络43个案例分析15章的参数寻优函数
ti13 利用二氧化硅含量预测值，预测其他元素含量
第一题编码.ipynb对数据进行labencider编码
第一题预测二氧化硅值.ipynb使用knn回归对编码数据预测文物未风化是的二氧化硅含量
第二题文件夹
文件名为S2 S311 S609 S611 S709 S909 S911 Sda Sxiao T，后缀为mat是敏感性分析和训练模型的数据集
xlj1.mat zds1.mat Classiffcation leaner导出的包含利用原有信息训练好的三次核函数支持向量机模型和装袋树模型的结构体
第二题编码.ipynb 对铅钡文物数据进行onehot编码并聚类输出
第二题编码-Copy1.ipynb 对高钾文物数据进行onehot编码并聚类输出
mgxfx.m 对三次核函数训练的模型进行敏感性分析
第三题 
T3all 预测玻璃种类并且进行敏感性分析
xlj1.mat zds1.mat Classiffcation leaner导出的包含利用原有信息训练好的三次核函数支持向量机模型和装袋树模型的结构体
----------------------------------------------------------------------------------------------------------------------------------------------------------------
第一题：
1、利用python对数据做knn回归
path = r'D:/11.xlsx'
frame= pd.read_excel(path)   # 直接使用 read_excel() 方法读取
frame.head(10)
chosen_data = frame.dropna(axis=0, how='any', inplace=False)
c=np.hstack((chosen_data['颜色'].values.reshape(-1, 1),chosen_data['纹饰'].values.reshape(-1,1),chosen_data['类型'].values.reshape(-1,1),chosen_data['表面风化'].values.reshape(-1,1)))
df2 = pd.DataFrame(c,columns=['颜色','纹饰','类型','表面风化'])
df2.to_excel('C:/Users/86186/Desktop/编码结果.xlsx', sheet_name='1', index=False),
X=np.hstack((chosen_data['颜色'].values.reshape(-1, 1),chosen_data['纹饰'].values.reshape(-1,1),chosen_data['类型'].values.reshape(-1,1),chosen_data['表面风化'].values.reshape(-1,1)))
y=a['二氧化硅'].values.reshape(-1, 1)
#声明knr回归选择最优参数训练最优模型
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)#规定测试集训练集
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 10)#规定临近点的集合
#利用循环寻找最好的参数每一个临近点值分别训练观察
for n_neighbors in neighbors_settings:
    # build the model
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(reg.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(reg.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#画图观察训练集成绩和测试集成绩
path = r'C:/Users/86186/Desktop/数据软件/数学建模/TIAO SHI/3.xlsx'
b= pd.read_excel(path)
#读取处理之后的数据
X_perdict=np.hstack((b['颜色'].values.reshape(-1, 1),b['纹饰'].values.reshape(-1,1),b['类型'].values.reshape(-1,1),b['表面风化'].values.reshape(-1,1)))
e=reg.predict(X_perdict)#带入模型进行预测
dfout = pd.DataFrame(e,columns=['二氧化硅预测值']) 
dfout.to_excel('C:/Users/86186/Desktop/预测结果.xlsx', sheet_name='1', index=False)#输出结果
3 matlab，用二氧化硅含量预测其他元素含量，此处只展示其中一种，其余的修改读取数据的表格即可
close all;
clear;
clc;
num = xlsread('C:/Users/86186/Desktop/数据软件/数学建模/TIAO SHI/预测结果4.xlsx','sheet12');
num1 = xlsread('C:/Users/86186/Desktop/数据软件/数学建模/TIAO SHI/57.xlsx','sheet12');
%[m,n] = size(num);
x = num(1:26,1);
y = num(1:26,4);
xr = num1(:,1);
yr = num(:,4)
x = x';
y = y';
xr=xr'
yr=yr'
[X,XP]= mapminmax(x,1,2);	
[Y,YP] = mapminmax(y,1,2);
[XR,XRP]= mapminmax(xr,1,2);
[YR,YRP]= mapminmax(yr,1,2);
X=X'
Y=Y'
XR=XR'
YR=YR'
[bestmse,bestc,bestg] = SVMcgForRegress(X,Y,-8,8,-8,8);
disp('打印粗略选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
% 根据粗略选择的结果图再进行精细选择: 
[bestmse,bestc,bestg] = SVMcgForRegress(X,Y,-4,4,-4,4,3,0.5,0.5,0.05);

% 打印精细选择结果
disp('打印精细选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];

%% SVM网络回归预测
model = svmtrain(X,Y,cmd);
[predict,accuracy,decision_values] = svmpredict(YR,XR ,model);
predict = mapminmax('reverse',predict',YRP);
predict = predict';

第二题：
使用kmean聚类代码

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures

from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from pandas import read_csv
import datetime
from datetime import datetime
import pandas as pd
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
 
path = r'C:/Users/86186/Desktop/第二题/qianbei.xlsx'
qianbei= pd.read_excel(path)   # 直接使用 read_excel() 方法读取
 
chosen_data = qianbei.dropna(axis=0, how='any', inplace=False)
qianbei.head(10)

X=np.hstack((chosen_data['颜色'].values.reshape(-1, 1),chosen_data['纹饰'].values.reshape(-1,1),chosen_data['类型'].values.reshape(-1,1),chosen_data['表面风化'].values.reshape(-1,1),chosen_data['文物采样点'].values.reshape(-1,1)))
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_onehot = enc.fit_transform(X).toarray()
#寻找最优参数训练模型
from sklearn.cluster import KMeans
from sklearn import metrics
scores = []
for k in range(2,25):
    labels = KMeans(n_clusters=k).fit(X_onehot).labels_
    score = metrics.silhouette_score(X_onehot, labels)
    scores.append(score)
plt.plot(list(range(2,25)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
#liyong
labels1 = KMeans(n_clusters=20).fit(X_onehot).labels_
labels1
#输出聚类结果
knn= pd.DataFrame(labels1,columns=['kmeans算法聚类'])
knn.to_excel('C:/Users/86186/Desktop/knn聚类.xlsx', sheet_name='1', index=False)#输出结果

敏感性分析
 %以下为带入处理后的数据进行敏感性分析 xlj1为支持训练出的向量机模型
mgxfx1 = xlj1.predictFcn(Sda);
 mgxfx2 = xlj1.predictFcn(Sxiao)
 mgxfx3 = xlj1.predictFcn(S911)
 mgxfx4 = xlj1.predictFcn(S909)
 mgxfx5 = xlj1.predictFcn(S611)
 mgxfx6 = xlj1.predictFcn(S609)

3
yfit =xlj1.predictFcn(T)%使用支持向量机预测
yfit1 =zds1.predictFcn(T)%使用装袋树预测
%以下为带入处理后的数据对装袋树进行敏感性分析
mgxfx7 =zds1.predictFcn(s311)
mgxfx8=zds1.predictFcn(s309)
mgxfx9=zds1.predictFcn(S109)
mgxfx10=zds1.predictFcn(s709)

第四题：
gaojia=[1	0.045	-0.159	-0.035	-0.622	-0.555	0.111	-0.206	0.28	-0.497	-0.554	0.786	0.243	0.645;
0.045	1	0.354	0.938	0.712	-0.525	0.881	0.847	0.463	0.952	0.213	-0.157	-0.457	0.793;
-0.159	0.354	1	-0.364	0.923	0.636	0.839	0.918	-0.038	0.803	0.791	0.921	0.434	-0.045;
-0.035	0.938	-0.364	1	0.938	0.489	-0.572	0.839	-0.392	0.799	0.294	0.891	0.925	-0.397;
-0.622	0.712	0.923	0.938	1	-0.731	0.984	-0.294	0.708	0.639	-0.465	0.954	0.675	0.723;
-0.555	-0.525	0.636	0.489	-0.731	1	0.011	0.961	0.893	-0.356	-0.918	0.872	0.743	0.347;
0.111	0.881	0.839	-0.572	0.984	0.011	1	-0.239	0.034	0.552	0.793	0.821	0.834	-0.392;
-0.206	0.847	0.918	0.839	-0.294	0.961	-0.239	1	0.349	0.912	0.034	-0.471	-0.722	0.839;
0.28	0.463	-0.038	-0.392	0.708	0.893	0.034	0.349	1	-0.013	0.823	0.991	0.923	-0.134;
-0.497	0.952	0.803	0.799	0.639	-0.356	0.552	0.912	-0.013	1	0.922	-0.623	0.183	0.798;
-0.554	0.213	0.791	0.294	-0.465	-0.918	0.793	0.034	0.823	0.922	1	0.139	-0.912	0.894;
0.786	-0.157	0.921	0.891	0.954	0.872	0.821	-0.471	0.991	-0.623	0.139	1	-0.834	0.829;
0.243	-0.457	0.434	0.925	0.675	0.743	0.834	-0.722	0.923	0.183	-0.912	-0.834	1	0.413;
0.645	0.793	-0.045	-0.397	0.723	0.347	-0.392	0.839	-0.134	0.798	0.894	0.829	0.413	1];
qianbei=[1	0.392	0.482	-0.219	0.593	0.492	-0.637	0.727	0.372	0.183	-0.043	0.294	0.504	-0.328;
0.392	1	-0.834	0.431	0.219	0.103	0.043	-0.848	0.358	0.845	0.123	0.415	-0.132	-0.134;
0.482	-0.834	1	0.538	-0.345	0.144	0.543	0.234	0.432	0.357	-0.653	0.583	0.023	0.232;
-0.219	0.431	0.538	1	0.456	-0.34	0.137	0.386	-0.132	0.657	0.968	0.245	-0.934	0.431;
0.593	0.219	-0.345	0.456	1	0.345	-0.535	-0.248	0.249	0.138	0.658	-0.587	0.238	0.104;
0.492	0.103	0.144	-0.34	0.345	1	0.845	0.121	-0.325	0.023	0.341	0.302	0.482	0.758;
-0.637	0.043	0.543	0.137	-0.535	0.845	1	-0.324	0.568	0.592	0.732	-0.267	0.463	0.689;
0.727	-0.848	0.234	0.386	-0.248	0.121	-0.324	1	0.275	-0.572	0.461	0.329	0.769	-0.546;
0.372	0.358	0.432	-0.132	0.249	-0.325	0.568	0.275	1	-0.696	0.301	0.344	0.579	0.924;
0.183	0.845	0.357	0.657	0.138	0.023	0.592	-0.572	-0.696	1	0.697	0.533	0.139	-0.459;
-0.043	0.123	-0.653	0.968	0.658	0.341	0.732	0.461	0.301	0.697	1	0.543	0.342	-0.489;
0.294	0.415	0.583	0.245	-0.587	0.302	-0.267	0.329	0.344	0.533	0.543	1	-0.592	0.348;
0.504	-0.132	0.023	-0.934	0.238	0.482	0.463	0.769	0.579	0.139	0.342	-0.592	1	0.374;
-0.328	-0.134	0.232	0.431	0.104	0.758	0.689	-0.546	0.924	-0.459	-0.489	0.348	0.374	1];
chayi=gaojia./qianbei
find(chayi>0.5&chayi<1.5)



 


 

 
