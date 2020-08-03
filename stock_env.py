import pandas as pd

class StockEnv():
    def __init__(self):
        self.data_df = pd.read_csv("AmericaStock.csv")

    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    ############################# Utility #########################


