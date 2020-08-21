import pandas as pd

class StockEnv():
    def __init__(self, features=["open", "high", "low", "close"], window_len=50,
                 codes=['AAPL', 'V', 'BABA', 'ADBE', 'SNE'], start_date='2015-01-05', end_date="2017-12-29"):
        self.features = features
        self.window_len = window_len
        self.codes = codes
        self.num_asset = len(codes) + 1
        self.num_features = len(features)
        self.start_date = start_date
        self.end_date = end_date
        raw_data = pd.read_csv("AmericaStock.csv", index_col='time', parse_dates=True)
        self.parse_raw_data(raw_data)

    def parse_raw_data(self, raw_data):
        data = raw_data[self.features]
        data = data[self.start_date: self.end_date]
        date_range = pd.date_range(self.start_date, self.end_date)
        asset_dict = dict()
        data_np = np.zeros(shape=(self.num_asset, ))
        for code in self.codes:
            asset_data = data[data['code'] == code]
            # include date that has no data, filled them with NaN value
            asset_data = asset_data.reindex(date_range).sort_index()
            # fill NaN with previously available data for 'close' column
            asset_data['close'].fillna(method='ffill')
            # fill NaN with close price for the other features
            # normalize each feature with the closing price of last day
            base_price = asset_data['close'][-1]
            asset_data['close'] /= base_price
            if 'high' in self.features:
                asset_data['high'].fillna(asset_data['close'], inplace=True)
                asset_data['high'] /= base_price
            if 'low' in self.features:
                asset_data['low'].fillna(asset_data['close'], inplace=True)
                asset_data['low'] /= base_price
            if 'open' in self.features:
                asset_data['open'].fillna(asset_data['close'], inplace=True)
                asset_data['open'] /= base_price
            
            asset_dict[code] = asset_data








    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    ############################# Utility #########################

