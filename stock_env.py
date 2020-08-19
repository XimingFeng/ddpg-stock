import pandas as pd

class StockEnv():
    def __init__(self, features=["open", "high", "low", "close"], history_window=50,
                 codes=['AAPL', 'V', 'BABA', 'ADBE', 'SNE']):
        self.data_df = pd.read_csv("AmericaStock.csv")
        self.features = features
        self.history_window = history_window
        self.codes = codes


    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    ############################# Utility #########################

