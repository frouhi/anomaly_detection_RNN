import csv
from binance.client import Client
'''
This script collects financial data that is used by other scripts from Binance exchange.
'''
client = Client("", "")
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "10 years ago UTC","1 month ago UTC")

with open("../data/klines_1h_BTCUSDT.csv","a") as file:
    writer = csv.writer(file)
    writer.writerow(["time","open","high","low","close","volume"])
for kline in klines:
    with open("../data/klines_1h_BTCUSDT.csv","a") as file:
        writer = csv.writer(file)
        writer.writerow(kline[0:6])
