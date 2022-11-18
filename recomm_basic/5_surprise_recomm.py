from surprise import BaselineOnly
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


# Data Load
data = Dataset.load_builtin('ml-100k')

train_set, test_set = train_test_split(data, test_size=0.25)

# 단일 Accuracy Check(KNNWithMeans)
algo = KNNWithMeans()
algo.fit(train_set)
predictions = algo.test(test_set)
print('KNNWithMeans 단일 : ',accuracy.rmse(predictions))

'''
# 다중 Algorithm Accuracy check
algorithms = [BaselineOnly,KNNWithMeans,SVD,SVDpp]
# BaselineOnly : User평점평균 Item평점평균 모델화하여 예측하는 알고리즘
# KNNWithMeans : 사용자의 평가경향을 고려한 알고리즘
# SVD : MF알고리즘
# SVDpp : SVD++알고리즘 --> 사용자의 implicit data를 고려한 알고리즘
names = []
results = []

for single in algorithms:
    algo = single()
    names.append(single.__name__)
    algo.fit(train_set)
    predictions = algo.test(test_set)
    results.append(accuracy.rmse(predictions))

names = np.array(names)
results = np.array(results)

index = np.argsort(results)
plt.ylim(0.8,1)
plt.plot(names[index],results[index])
plt.show()
'''

'''
# 알고리즘 옵션 변경 : KNNWithMeans
result = []
sim_options = {'name':'pearson_baseline',
               'user_based':True}
for n_size in (10,20,30,40,50,60):
    algo = KNNWithMeans(k=n_size,sim_options=sim_options)
    algo.fit(train_set)
    predictions = algo.test(test_set)
    result.append([n_size , accuracy.rmse(predictions)])
print(result)
'''

'''
# GridSearch를 통한 최적 파라미터 추출(KNNWithMeans)
param_grid = {'k' : [5,10,15,20,25,30],
              'sim_options' : {'name' : ['pearson_baseline','cosine'], 
                               'user_based' : [True,False]},     # user_base : True --> UBCF, False --> IBCF
              }
gs = GridSearchCV(KNNWithMeans,param_grid=param_grid,measures=['rmse'],cv=4)
gs.fit(data)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
'''


# GridSearch2(SVD)
param_grid = {'n_epochs' : [70,80,90,100],
              'lr_all' : [0.005,0.006,0.007],    # a : 학습률
              'reg_all' : [0.05,0.07,0.1],      # b : 정규화 계수
             }
gs = GridSearchCV(SVDpp,param_grid=param_grid,measures=['rmse'],cv=4)
gs.fit(data)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

'''
# 데이터셋 로드하여 surprise 작동
r_cols = ['user_id', 'movie_id'， 'rating', 'timestamp']
ratings = pd.read_csv('경로',names=r_cols， sep:'\t'，encoding ='latin-1')
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']] , reader)
'''