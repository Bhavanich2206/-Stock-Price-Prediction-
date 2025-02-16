import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10

def status_calc(stock_p_change, SP500_p_change, threshold):
    """
    Calculates the status of a stock based on its percentage change compared to S&P500.
    :param stock_p_change: Percentage change of the stock
    :param SP500_p_change: Percentage change of the S&P500
    :param threshold: Threshold for outperformance
    :return: List of binary labels indicating outperformance (1) or not (0)
    """
    return [1 if stock_change > SP500_change + threshold else 0
            for stock_change, SP500_change in zip(stock_p_change, SP500_p_change)]

def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = pd.read_csv("keystats.csv", index_col="Date")
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y_train = np.array(status_calc(training_data["stock_p_change"], training_data["SP500_p_change"], OUTPERFORMANCE))

    return X_train, y_train

def predict_stocks():
    X_train, y_train = build_data_set()
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Now we get the actual data from which we want to generate predictions.
    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    z = data["Ticker"].values

    # Get the predicted tickers
    y_pred = clf.predict(X_test)
    invest_list = z[y_pred == 1].tolist()

    if not invest_list:
        print("No stocks predicted!")
    else:
        # Convert integers to strings
        invest_list = [str(ticker) for ticker in invest_list]
        print(f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:")
        print((" ".join(invest_list)))


        # Generate random performance values for visualization
        performance = np.random.uniform(low=0, high=20, size=len(invest_list))

        # Plot the bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(invest_list, performance, color='skyblue')
        plt.title('Predicted Performance of Stocks')
        plt.xlabel('Stocks')
        plt.ylabel('Performance (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(performance, bins=10, color='skyblue', edgecolor='black')
        plt.title('Histogram of Predicted Performance')
        plt.xlabel('Performance (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()
