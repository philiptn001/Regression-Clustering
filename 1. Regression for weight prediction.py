import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

if __name__ == '__main__':
    df = pd.read_csv("diet.csv")
    df = shuffle(df)
    df_x = df.drop('weight6weeks', axis=1).values
    df_y = df['weight6weeks'].values

# Split the dataset to test and train set
    split_df = int(len(df_x) * 0.7)
    df_X_train = df_x[:split_df]
    df_y_train = df_y[:split_df]
    df_X_test = df_x[split_df:]
    df_y_test = df_y[split_df:]    
    
#Regression since its a varying value
    model = linear_model.LinearRegression()

    model.fit(df_X_train, df_y_train)

    y_pred = model.predict(df_X_test)

    for i in range(len(df_y_test)):
        print("Expected:", df_y_test[i], "Predicted:", y_pred[i])

    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(df_y_test, y_pred))
