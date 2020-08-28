import pandas as pd
from datetime import datetime
import numpy as np

class StockEnv():
    def __init__(self, features, asset_codes, data_path, mu=0.0025,
                 window_len=50, start_date='2015-01-05', end_date="2017-12-29"):
        self.features = features
        self.window_len = window_len
        self.asset_codes = asset_codes
        self.num_assets = len(asset_codes) + 1
        self.num_features = len(features)
        self.start_date = start_date
        self.end_date = end_date
        self.mu = mu
        self.date_range = pd.date_range(self.start_date, self.end_date)
        self.date_diff = len(self.date_range)
        self.t = 0
        self.asset_dict = self.clean_raw_data(data_path)
        self.states = self.get_states(self.asset_dict)
        self.price_change_ratios = self.get_price_change_ratios(self.asset_dict)
        self.alloc_history = []

    def clean_raw_data(self, file_path):
        raw_data = pd.read_csv(file_path, index_col='time', parse_dates=True)
        cols = self.features + ["code"]
        data = raw_data[cols]
        data = data.loc[self.start_date: self.end_date]
        asset_dict = dict()
        for code in self.asset_codes:
            asset_data = data[data['code'] == code][self.features]

            # include date that has no data, filled them with NaN value
            asset_data = asset_data.reindex(self.date_range).sort_index()

            # fill NaN with previously available data for 'close' column
            asset_data['close'].fillna(method='ffill', inplace=True)

            # fill NaN with close price for the other features
            # normalize each feature with the closing price of last day
            base_price = asset_data['close'][-1]
            for feat in self.features:
                if feat != "close":
                    asset_data[feat].fillna(asset_data['close'], inplace=True)
                    asset_data[feat] = asset_data[feat] / base_price
            asset_data["close"] /= base_price
            asset_dict[code] = asset_data
        return asset_dict

    def get_states(self, asset_dict):
        """
        Return a list of state that can feed into agent
        Shape of states (number of assets, window length, number of features)
        Note that the number of assets does not include money (as an asset)
        :param asset_dict:
        :return:
        """
        states = []
        for i in range(self.date_diff - self.window_len):
            # exclude money for state
            state = np.ones(shape=(self.num_assets - 1, self.window_len, self.num_features))
            asset_idx = 0
            for code in self.asset_codes:
                asset_window = asset_dict[code][i: i + self.window_len]
                state[asset_idx, :, :] = asset_window.to_numpy()
                asset_idx += 1
            state.reshape(1, self.num_assets - 1, self.window_len, self.num_features)
            states.append(state)
        return states

    def get_price_change_ratios(self, asset_dict):
        """
        get the daily price change ratio. First date the ratio is set to 1 for all asset
        this is y_t in the paper
        :param asset_dict:
        :return:
        """
        price_change_ratios = np.ones(shape=(self.date_diff, self.num_assets))
        for i in range(1, self.date_diff):
            if i > 0:
                asset_idx = 1
                for code in self.asset_codes:
                    price_change_ratios[i, asset_idx] = \
                        asset_dict[code].iloc[i]['close'] / asset_dict[code].iloc[i - 1]['close']
                    asset_idx += 1
        return price_change_ratios

    def step(self, action):

        # y_t: change of price
        price_change_ratio = self.price_change_ratios[self.t]

        # the allocation at the end of previous period
        prev_trade_end_alloc = self.alloc_history[-1]

        # transaction cost occurs when buy and sell
        trans_cost = self.mu * np.abs(action[1:] - prev_trade_end_alloc[1:]).sum()

        reward = np.log(action * price_change_ratio - trans_cost)

        trade_end_alloc = (action * price_change_ratio) / np.dot(action, price_change_ratio)
        self.alloc_history.append(trade_end_alloc)

        self.t += 1
        next_state = self.states[self.t - self.window_len]
        return reward, next_state

    def reset(self):
        self.t = self.window_len
        cur_date = self.date_range.to_pydatetime()[self.t].strftime('%Y-%m-%d')
        self.alloc_history = []
        # Initial allocation is all in cash / money
        self.alloc_history = []
        init_alloc = np.zeros(self.num_assets)
        init_alloc[0] = 1
        self.alloc_history.append(init_alloc)
        return self.states[self.t - self.window_len]

    def render(self):
        cur_date = self.date_range.to_pydatetime()[self.t].strftime('%Y-%m-%d')
        print(f"We are at date {cur_date}")
        print(f'allocation of portfolio at the end of prev date: {self.alloc_history[-1]}')

        pass

    ############################# Utility #########################

