import matplotlib.pyplot as plt
import pandas as pd

import json
from os.path import join as pjoin
from os import makedirs


def save_image(df, path):
	fig = plt.figure(figsize=(3.31, 3.33), dpi=100)
	
	plt.plot(df["Close"])
	plt.axis('off')
	plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
	plt.close()


def create_images(df, ticker, n_days=30):
	makedirs(pjoin("data", "images", ticker), exist_ok=True)
	i = 0
	while i + n_days < len(df):
		save_image(df[i:i + n_days], pjoin("data", "images", ticker, f"{ticker}-{str(i)}.jpg"))
		i += 1


def main():
	with open(pjoin("data", "tickers.json"), "r") as f:
		tickers = json.load(f)["Tickers"]

	makedirs(pjoin("data", "images"), exist_ok=True)

	for ticker in tickers:
		print(ticker)
		df = pd.read_csv(pjoin("data", "tickers", f"{ticker}.csv"))
		create_images(df, ticker)


if __name__ == '__main__':
	main()
