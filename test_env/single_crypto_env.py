import numpy as np
import os
import gym
from numpy import random as rd

class CryptoTradingEnv(gym.Env):

    def __init__(self, config, 
                 gamma=0.99, min_stock_rate=0.0001,
                 max_stock=1, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 max_step = 30
                 ):
        # price_ary: open, close, low, high
        price_ary = config['price_array']
        # as many as provided
        tech_ary = config['tech_array']
        #risk_ary = config['risk_array']
        if_train = config['if_train']
        
        # time duration
        n = price_ary.shape[0]
        self.price_ary =  price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        #self.risk_ary = risk_ary
        
        self.tech_ary = self.tech_ary * 2 ** -7
        #self.risk_bool = (risk_ary > risk_thresh).astype(np.float32)
        #self.risk_ary = (self.sigmoid_sign(risk_ary, risk_thresh) * 2 ** -5).astype(np.float32)

        # single crypto
        stock_dim = 1
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        # reset()
        self.day = None
        self.run_index = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'CryptoEnv'
        # amount + price_dim + stock_dim + tech_dim
        self.state_dim = 1 + 4 + 1 + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.action_dim = stock_dim
        
        # max game duration
        self.max_step = max_step
        self.max_datalength = n - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
    def reset(self):
        # random start point
        if self.if_train:
            self.stocks = self.initial_stocks
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05)
            self.day = rd.randint(0, self.max_datalength - self.max_step)
            self.run_index = 0
            price = self.price_ary[self.day]
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.amount = self.initial_capital
            self.day = 0
            self.run_index = 0
            price = self.price_ary[self.day]

        # price[:, 1] --> closing price
        self.total_asset = self.amount + (self.stocks * price[1]).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        def tradable_size(x):
            return (x / self.min_stock_rate).astype(int) * self.min_stock_rate
        
        # order price -> last day (open + close)/2
        order_px = (self.price_ary[self.day + self.run_index, 0] + \
                    self.price_ary[self.day + self.run_index, 1])/2.0
        actions_v = actions[0] * self.max_stock
        
        self.run_index += 1
        price = self.price_ary[self.day + self.run_index]
        
        # within day low-high
        if (order_px > price[2]) and (order_px < price[3]):
            if actions_v > 0:
                buy_num_shares = tradable_size(min(self.amount/order_px, actions_v))
                self.stocks[0] += buy_num_shares
                self.amount -= order_px * buy_num_shares * (1 + self.buy_cost_pct)
            
            if actions_v < 0:
                sell_num_shares = tradable_size(min(self.stocks[0], (-1.0) * actions_v))
                self.stocks[0] -= sell_num_shares
                self.amount += order_px * sell_num_shares * (1 - self.sell_cost_pct)
                
        state = self.get_state(price)
        # in order to maximize the value
        #total_asset = self.amount + (self.stocks * price[1]).sum()
        #reward = (total_asset - self.total_asset) * self.reward_scaling
        #self.total_asset = total_asset
        
        # in order to maximize the holding
        total_asset = self.stocks[0]
        reward = total_asset
        
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.run_index == self.max_step
        if done:
            reward = self.gamma_reward
            #self.episode_return = total_asset / self.initial_total_asset
            self.episode_return = total_asset
            print (self.episode_return)

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        
        return np.hstack((amount,
                          price * scale,
                          self.stocks * scale,
                          self.tech_ary[self.day],
                          ))  # state.astype(np.float32)
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh