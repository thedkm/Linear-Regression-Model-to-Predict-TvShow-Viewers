import pandas as pd
from sklearn import linear_model
import numpy as np

dataset = pd.read_csv('tv_show_viewers_datasest.csv')
print (len(dataset))
print (list(dataset.columns.values))

dataset.describe()
dataset.head()

def split_data(dataset):
    
    features = []
    target = []
    for char1,char2,char3,char4,char5,fight_scene,comedy_scence,romance_scence,viewers_count in zip(
                                dataset['Character1_appeared'],
                                 dataset['Character2_appeared'],
                                 dataset['Character3_appeared'],
                                 dataset['Character4_appeared'],
                                 dataset['Character5_appeared'],
                                 dataset['Fight_scenes'],
                                 dataset['Comedy_scences'],
                                 dataset['Romance_scence'],
                                 dataset['Viewers']
                                ):
        features.append([char1,char2,char3,
                                   char4,char5,fight_scene,
                                   comedy_scence,romance_scence])
        target.append(viewers_count)
        
    return features,target
	
	
train_features,train_target = split_data(dataset)

print (train_features)
print (train_target)

regr = linear_model.LinearRegression()
regr.fit(train_features,train_target)

episode51_features = np.array([4, 7, 8, 6, 3, 4, 9, 9]).reshape(1, -1)
print ("Viewers for episode 51:",regr.predict(episode51_features))

episode52_features = np.array([4, 8, 3, 6, 9, 6, 2, 3]).reshape(1, -1)
print ("Viewers for episode 52:",regr.predict(episode52_features))
