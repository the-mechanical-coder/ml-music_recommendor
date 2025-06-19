import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# import data
music_df = pd.read_csv('music.csv')

# clean data
x = music_df.drop(columns= ['genre'])
y = music_df['genre']

# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create the model
model = DecisionTreeClassifier()

# train the model
model.fit(x_train, y_train)

# Make Predictions
# predictions = model.predict(x_test)
# print(predictions)

# Evaluate and Improve
# score = accuracy_score(y_test, predictions)
# print(score)

#create a new model
model = joblib.dump(model, 'music_recommender.joblib')






