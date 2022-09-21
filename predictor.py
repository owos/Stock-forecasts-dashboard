import pickle
import streamlit as st

import yfinance as yf
from plotly import graph_objs as go

import math # Mathematical functions
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # For analysing and manipulating data
from datetime import date, timedelta # Date Functions
from pandas.plotting import register_matplotlib_converters # Adds plotting functions for calender dates
import matplotlib.pyplot as plt # For visualization
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.preprocessing import MinMaxScaler #to normalize the price data
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential 
from keras.layers import LSTM, Dense 
import tensorflow as tf
from tensorflow import keras
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
import mplfinance as mpf
from sklearn.metrics import precision_score
import snscrape.modules.twitter as sntwitter
import re
from tqdm import tqdm
import string
import yfinance as yf


import flair
def senti():
            return flair.models.TextClassifier.load('en-sentiment')

sentiment_model = senti()

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't", "u" ])

today = date.today()
date_today = today.strftime("%Y-%m-%d")
date_start = '2015-01-01'



st.title('Stock Forecast App')

stockname = ('GOOGLE', 'TESLA', 'MICROSOFT', 'META', 'COCA COLA', 'VISA', 'PFIZER', 'CHEVRON', 'WALMART', 'UNITED PARCEL SERVICES')
select_stockname = st.selectbox('Select Stock', stockname, key='1')

stocks_dict = {'GOOGLE':'GOOG', 'TESLA':'TSLA', 'MICROSOFT':'MSFT', 'META':'META', 'COCA COLA':'KO', 'VISA':'V', 'PFIZER':'PFE', 
                'CHEVRON':'CVX', 'WALMART':'WMT', 'UNITED PARCEL SERVICES':'UPS'}
selected_stock = stocks_dict[select_stockname]


@st.cache
def load_data(ticker):
    data = yf.download(ticker, date_start, date_today)
    data.reset_index(inplace=True)
    return data

download = st.button('Download Data')
if download:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')
    data = data.set_index('Date')
    st.subheader('Date from 2015 - To Date')
    st.write(data.tail())



    st.subheader('Closing Price vs Time Chart')
    def plot_raw_data():
        register_matplotlib_converters()
        years = mdates.YearLocator()
        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax1.xaxis.set_major_locator(years)
        x = data.index
        y = data['Close']
        ax1.fill_between(x, 0, y, color='#b9e1fa')
        ax1.legend([select_stockname], fontsize=12)
        plt.title(select_stockname + ' from '+ date_start + ' to ' + date_today, fontsize=16)
        plt.plot(y, color='#039dfc', label=select_stockname, linewidth=1.0)
        plt.ylabel('Stocks', fontsize=12)
        st.pyplot(fig)
        
    plot_raw_data()

    

    price_model = tf.keras.models.load_model('stock_50epoch.h5')
    #trend_model = pickle.load(open('trend_predictor.pkl'))


    train_df = data.filter(['Open', 'High', 'Low', 'Adj Close', 'Close']) 
    data_unscaled = train_df.values



    train_data_length = math.ceil(len(data_unscaled) * 0.8)

    mmscaler = MinMaxScaler(feature_range=(0, 1))
    np_data = mmscaler.fit_transform(data_unscaled)


    sequence_length = 50

    # Prediction Index
    index_Close = train_df.columns.get_loc("Close")
    train_data_len = math.ceil(np_data.shape[0] * 0.8)
    train_data = np_data[0:train_data_len, :]
    test_data = np_data[train_data_len - sequence_length:, :]


    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 5 features
    def partition_dataset(sequence_length, train_df):
        x, y = [], []
        data_len = train_df.shape[0]
        for i in range(sequence_length, data_len):
            x.append(train_df[i - sequence_length:i, :])
            y.append(train_df[
                        i, index_Close])

        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y


    # Generate training data and test data
    X_train, y_train = partition_dataset(sequence_length, train_data)
    X_test, y_test = partition_dataset(sequence_length, test_data)


    y_pred_scaled = price_model.predict(X_test)

    prediction_copies = np.repeat(y_pred_scaled, X_train.shape[2], axis=-1)
    y_pred_future = mmscaler.inverse_transform(prediction_copies)[:,0]


    test_copy = np.repeat(y_test.reshape(-1, 1), 5, axis=-1)
    y_test_unscaled = mmscaler.inverse_transform(test_copy)[:,0]


    # The date from which on the date is displayed
    display_start_date = "2015-01-01"

    # Add the difference between the valid and predicted prices
    train = train_df[:train_data_length + 1]
    valid = train_df[train_data_length:]
    valid.insert(1, "Predictions", y_pred_future, True)
    valid.insert(1, "Difference", valid["Predictions"] - valid["Close"], True)
    valid = valid[valid.index > display_start_date]
    train = train[train.index > display_start_date]


    def plot_prediction():
    # Visualize the data
        fig, ax = plt.subplots(figsize=(16, 8), sharex=True)

        plt.title("Predictions vs Ground Truth", fontsize=20)
        plt.ylabel(select_stockname, fontsize=18)
        plt.plot(train["Close"], color="#039dfc", linewidth=1.0)
        plt.plot(valid["Predictions"], color="#E91D9E", linewidth=1.0)
        plt.plot(valid["Close"], color="black", linewidth=1.0)
        plt.legend(["Train", "Test Predictions", "Ground Truth"], loc="upper left")


        # Create the bar plot with the differences
        valid.loc[valid["Difference"] >= 0, 'diff_color'] = "#2BC97A"
        valid.loc[valid["Difference"] < 0, 'diff_color'] = "#C92B2B"
        plt.bar(valid.index, valid["Difference"], width=0.8, color=valid['diff_color'])

        st.pyplot(fig)

    plot_prediction()


    df_temp = data[-sequence_length:]
    new_df = df_temp.filter(['Open', 'High', 'Low', 'Adj Close', 'Close'])

    N = sequence_length

    # Get the last N day closing price values and scale the data to be values between 0 and 1
    last_N_days = new_df[-sequence_length:].values
    last_N_days_scaled = mmscaler.transform(last_N_days)

    # Create an empty list and Append past N days
    X_test_new = []
    X_test_new.append(last_N_days_scaled)

    # Convert the X_test data set to a numpy array and reshape the data
    pred_price_scaled = price_model.predict(np.array(X_test_new))
    pred_price_scaled = np.repeat(pred_price_scaled.reshape(-1, 1), 5, axis=-1)
    pred_price_unscaled = mmscaler.inverse_transform(pred_price_scaled)

    # Print last price and predicted price for the next day
    price_today = np.round(new_df['Close'][-1], 2)
    predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
    change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

    plus = '+'; minus = ''
    st.write(f'The close price for {select_stockname} at {today} was {price_today}')
    st.write(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')


    #Trend Prediction

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


    st.subheader('Buy Trend Confidence Score')

    def create_target(df):

        df["Tomorrow"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

        return df


    trend_stock = create_target(data)


    train = trend_stock.iloc[:-100]
    test = trend_stock.iloc[-100:]


    def predictors(df):
        horizons = [5,60,250,1000]
        new_predictors = []

        for horizon in horizons:
            rolling_averages = df.rolling(horizon).mean()
            
            ratio_column = f"Close_Ratio_{horizon}"
            df[ratio_column] = df["Close"] / rolling_averages["Close"]
            
            trend_column = f"Trend_{horizon}"
            df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
            
            new_predictors+= [ratio_column, trend_column]

        return df, new_predictors


    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:,1]
        preds[preds >=.6] = 1
        preds[preds <.6] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        merge = pd.concat([test["Target"], preds], axis=1)
        return merge, preds


    def backtester(data, model, predictors, start=500, step=250):
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions, preds = predict(train, test, predictors, model)
            
            all_predictions.append(predictions)
        if preds[0] >= .6:
            st.write("The trend is a rise")
        else:
            st.write("The trend is a fall")
        return pd.concat(all_predictions)


    new_ft_stock = predictors(data)

    stock = new_ft_stock[0]
    stock = stock.dropna()
    new_predictors = new_ft_stock[1]


    predictions = backtester(stock, model, new_predictors)

    confidence_score = precision_score(predictions["Target"], predictions["Predictions"])

    st.write(f'The buy confidence score for {select_stockname} is {confidence_score}')




    def get_tweets(stock, date, last_date):
        query = "(from:${}) until:{} since:{}".format(stock, date, last_date)
        tweets = []
        limit = 500


        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            
            # print(vars(tweet))
            # break
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.content])
                
        df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

        return df
    df = pd.read_csv(f"stocks {selected_stock}.csv", parse_dates = ['Date'])
    st.write("trying out the head")
    st.write(df.head())
    st.write(df.info())
    last_date= str(df.Date.dt.date[0])
    tweets = get_tweets(selected_stock, date_today, last_date)
    #st.write(tweets)

    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\'u", " you", phrase)
        return phrase


    def process_text(text):
        preprocessed_text = []
        for text in text:
            text = re.sub(r"http\S+", "", text)
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = decontracted(text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub('[‘’“”…]', '', text)
            text = re.sub('\n', '', text)
            text = re.sub("\S*\d\S*", "", text).strip()
            text = re.sub('[^A-Za-z]+', ' ', text)
            # https://gist.github.com/sebleier/554280
            text = ' '.join(e.lower() for e in text.split() if e.lower() not in stopwords)
            preprocessed_text.append(text.strip())
        return preprocessed_text 


    probs = []
    sentiments = []

    # use regex expressions (in clean function) to clean tweets
    tweets['clean'] = process_text(tweets['Tweet'])

    for tweet in tweets['clean'].to_list():
        # make prediction
        sentence = flair.data.Sentence(tweet)
        sentiment_model.predict(sentence)
        # extract sentiment prediction
        probs.append(sentence.labels[0].score)  # numerical score 0-1
        sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

    # add probability and sentiment predictions to tweets dataframe
    tweets['probability'] = probs
    tweets['sentiment'] = sentiments
    tweets = pd.concat([df, tweets])
    tweets.drop_duplicates(inplace=True)
    tweets.to_csv(f"stocks {selected_stock}.csv", index=False)
    st.write(tweets)

    sentiment = tweets['sentiment'].value_counts()
    sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})


    fig = px.bar(sentiment.iloc[:1000, :], x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
    st.plotly_chart(fig)

