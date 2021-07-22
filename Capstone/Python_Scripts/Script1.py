# Get news headlines and stock price data
# import packages
import yfinance as yf
import json
import datetime
import requests
import pandas as pd
import pytz

# Set the start and end date
start_date = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d')
end_date = (datetime.datetime.now(pytz.timezone('US/Pacific')) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

# Opening JSON file - has all the stock tickers
f = open("config.json", )

# returns JSON object as
# a dictionary
config = json.loads(f.read())

# Get stock prices
def get_stock_data(stockticker, startdate, enddate):
    data = yf.download(stockticker, startdate, enddate)
    data['name'] = stockticker
    return data

# Get news headlines
def get_news_data(stockticker, startdate, enddate):
    url = f"https://finnhub.io/api/v1/company-news?symbol={stockticker}&from={startdate}&to={enddate}&token=yourtoken"
    r = requests.get(url)

    response = r.json()
    if not response:
        return pd.DataFrame(index=['datetime', 'headline', 'related', 'source'])

    r2 = pd.DataFrame(response)
    df = r2[['datetime', 'headline', 'related', 'source']]
    return df


# Get stock information about multiple stocks
stock_data_list = []

for ticker in config["stockticker"].split():
    tmp = get_stock_data(ticker, start_date, end_date)
    if not tmp.empty:
        stock_data_list.append(tmp)

stock_data = pd.concat(stock_data_list)

# Get news information about multiple stocks
news_data_list = []

for ticker in config["stockticker"].split():
    tmp = get_news_data(ticker, start_date, end_date)
    if not tmp.empty:
        news_data_list.append(tmp)

news_data = pd.concat(news_data_list)


# upload CSV files to Google cloud

from google.cloud import storage
client = storage.Client.from_service_account_json(json_credentials_path='yourfile.json')
bucket = client.get_bucket('yourbucket1')
object_name_in_gcs_bucket = bucket.blob('stock_data.csv')
df = pd.DataFrame(data=stock_data).to_csv(encoding="UTF-8")
object_name_in_gcs_bucket.upload_from_string(data=df)

object_name_in_gcs_bucket = bucket.blob('news_data.csv')
df = pd.DataFrame(data=news_data).to_csv(encoding="UTF-8")
object_name_in_gcs_bucket.upload_from_string(data=df)
