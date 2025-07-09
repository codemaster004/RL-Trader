import yfinance as yf

import json
from os import makedirs
from os.path import join as pjoin


def download_save_ticker(ticker):
	df = yf.download(ticker, start="2000-01-01", end="2025-06-08", interval="1d", group_by="ticker", multi_level_index=False)
	df.reset_index(inplace=True)
	
	df.to_csv(pjoin("data", "tickers", ticker + ".csv"), index=False)


def main():
	makedirs(pjoin("data", "tickers"), exist_ok=True)
	
	with open(pjoin("data", "tickers.json"), "r") as f:
		tickers = json.load(f)["Tickers"]
	
	for ticker in tickers:
		download_save_ticker(ticker)


# Warning please run with working dir as the main project dir
if __name__ == '__main__':
	main()
