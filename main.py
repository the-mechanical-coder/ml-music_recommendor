import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_test_data():
    music_df = pd.read_csv('music.csv')

    # clean data
    x = music_df.drop(columns=['genre'])
    y = music_df['genre']

    # split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_test, y_test

if __name__ == '__main__':
    #load model
    model = joblib.load('music_recommender.joblib')

    # get xtest data
    x_test, y_test =  get_test_data()

    # Make Predictions
    predictions = model.predict(x_test)
    print(predictions)

    # Evaluate and Improve
    score = accuracy_score(y_test, predictions)
    print(score)