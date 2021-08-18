import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime


def plot(df):
    plt.plot(df['Pred'])
    plt.plot(df['close'])
    plt.show()


def main():
    df_cut = pd.read_csv('cut_prediction_dataset.csv')
    df_full = pd.read_csv('full_prediction_dataset.csv')
    plot(df_cut)
    plot(df_full)
    print(df_full)


if __name__ == '__main__':
    main()
