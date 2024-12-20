import numpy as np
import os
import gym
from numpy import random as rd

class CryptoTradingEnv(gym.Env):

    def __init__(self, config, 
                 gamma=0.99, min_stock_rate=0.0001,
                 buy_min_value=10.0, sell_min_value = 10.0, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 max_step = 30,
                 start_idx = 0,
                 ):
        # price_ary: open, close, low, high
        price_ary = config['price_array']
        # as many as provided
        tech_ary = config['tech_array']
        #risk_ary = config['risk_array']
        if_sequence = config['if_sequence']
        if_randomInit = config['if_randomV']
        # reward based on value or coin count
        if_value = config['if_value']
        # lookback px history; default 1
        self.lookback_n = config.get('lookback_n', 1)
        self.run_start_idx = start_idx
        
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
        self.buy_min_value = buy_min_value
        self.sell_min_value = sell_min_value
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
        #self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'CryptoEnv'
        
        # version 0: amount + price_dim + stock_dim + tech_dim
        #self.state_dim = 1 + 4 * self.lookback_n + 1 + self.tech_ary.shape[1]
        
        # version 1: cash/(coin_value + cash) + price_dim + tech_dim
        self.state_dim = 1 + 4 * self.lookback_n + self.tech_ary.shape[1]
        
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.action_dim = stock_dim
        
        # max game duration
        self.max_step = max_step
        self.max_datalength = n - 1
        self.if_sequence = if_sequence
        self.if_randomInit = if_randomInit
        self.if_value = if_value
        self.if_discrete = False
        
        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
    def reset(self):
        # random start point
        if not self.if_sequence:
            self.stocks = self.initial_stocks.copy()
            
            if self.if_randomInit:
                # randomly start holdings
                self.amount = self.initial_capital * (rd.rand()*0.6 + 0.2)
                self.stocks = self.stocks + (self.initial_capital - self.amount)/np.mean(self.price_ary[:, 2])
            else:
                self.amount = self.initial_capital
            
            self.day = rd.randint(0, self.max_datalength - self.max_step)
            self.run_index = 0
            price = self.price_ary[self.day]
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.amount = self.initial_capital
            self.day = self.run_start_idx
            self.run_index = 0
            price = self.price_ary[self.day]

        if self.if_value:
            # price[:, 1] --> closing price
            self.total_asset = self.amount + (self.stocks * price[1]).sum()
        else:
            #print ('self.stocks:', self.stocks)
            self.total_asset = self.stocks[0]
        
        self.initial_total_asset = self.total_asset
        self.episode_return = 0.0
        self.gamma_reward = 0.0
        init_state = self.get_state(price)  # state
        
        if self.if_sequence:
            print('initial stock:', self.stocks, 'inital amount: ', self.amount)
            print('initial asset: ', self.initial_total_asset)
        
        return init_state

    def step(self, actions):
        def tradable_size(x):
            return (x / self.min_stock_rate).astype(int) * self.min_stock_rate
        
        # actions -> percentage of stock or cash
        # add clip at 0.9
        actions_v = actions[0] * 0.9
        
        if actions_v == np.nan:
            actions_v = 0.0
        #print (actions_v)
        
        # version 0: order price -> last day (open + close)/2; order can be filled quickly
        order_px = (self.price_ary[self.day + self.run_index, 0] + \
                    self.price_ary[self.day + self.run_index, 1])/2.0
        
        # version 1: order px using last closing px
        #order_px = self.price_ary[self.day + self.run_index, 1]
        
        self.run_index += 1
        price = self.price_ary[self.day + self.run_index]
        
        # within day low-high
        if actions_v > 0:
            if self.amount * actions_v > self.buy_min_value:
                buy_num_shares = tradable_size((self.amount * actions_v/order_px)/(1 + self.buy_cost_pct))

                if self.if_sequence and buy_num_shares != 0.0:
                    print (f'[Day {self.day + self.run_index}] BUY: {buy_num_shares}')

                if order_px > price[2]:
                    actual_order_px = min(order_px, price[3])
                    self.stocks[0] += buy_num_shares
                    self.amount -= actual_order_px * buy_num_shares * (1 + self.buy_cost_pct)
            
        if actions_v < 0:
            sell_num_shares = tradable_size(self.stocks[0] * (-1.0) * actions_v)
            # no short 
            sell_num_shares = min(sell_num_shares, self.stocks[0])
            
            if (order_px * sell_num_shares) > self.sell_min_value:
                if self.if_sequence and sell_num_shares != 0.0:
                    print (f'[Day {self.day + self.run_index}] SELL: {sell_num_shares}')

                if order_px < price[3]:
                    actual_order_px = max(order_px, price[2])
                    self.stocks[0] = self.stocks[0] - sell_num_shares
                    self.amount += actual_order_px * sell_num_shares * (1 - self.sell_cost_pct)
                
        state = self.get_state(price)
        
        if self.if_value:
            # in order to maximize the value
            total_asset = self.amount + (self.stocks * price[1]).sum()
            reward = (total_asset - self.total_asset) * self.reward_scaling
            self.total_asset = total_asset
        else:
            # in order to maximize the holding
            total_asset = self.stocks[0]
            reward = (total_asset - self.total_asset) * self.reward_scaling
            self.total_asset = total_asset
        
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        
        done = self.run_index == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset
            
            if self.if_sequence:
                print ('Episode Return: ', self.episode_return)
            #self.reset()

        return state, reward, done, dict()

    def get_state_v0(self, price):
        amount = np.array(self.amount * (2 ** -12), dtype=np.float32)
        
        if self.lookback_n > 1:
            px_index_st = max(0, self.run_index - self.lookback_n + 1)
            px_index_ed = self.run_index + 1
            new_price = np.zeros((self.lookback_n, 4), dtype = float)
            new_price[(-1 * (px_index_ed - px_index_st)):] = self.price_ary[(self.day + px_index_st):(self.day + px_index_ed)]
            # flatten by row
            price = new_price.flatten()
            
        scale_base_day = max(self.day, self.day + self.run_index - self.lookback_n + 1)
        scale_factor = (-1) * int(np.log(self.price_ary[scale_base_day, 0])/np.log(2))
        px_scale = np.array(2 ** scale_factor, dtype=np.float32)
        
        stock_scale = (2** -5)
        
        return np.hstack((amount,
                          price * px_scale,
                          self.stocks * stock_scale,
                          self.tech_ary[self.day + self.run_index],
                          ))  # state.astype(np.float32)
    
    def get_state(self, price):
        cash_ratio = np.array(self.amount/(self.amount + (self.stocks * price[1]).sum() + 1e-10), dtype=np.float32)
        
        if self.lookback_n > 1:
            px_index_st = max(0, self.day + self.run_index - self.lookback_n + 1)
            px_index_ed = self.day + self.run_index + 1
            new_price = np.zeros((self.lookback_n, 4), dtype = float)
            new_price[(-1 * (px_index_ed - px_index_st)):] = self.price_ary[px_index_st:px_index_ed]
            # flatten by row
            price = new_price.flatten()
            
        # Version 0: scale_base_day might have some bug, to be evaluated
        #scale_base_day = max(self.day, self.day + self.run_index - self.lookback_n + 1)
        #scale_factor = (-1) * int(np.log(self.price_ary[scale_base_day, 0])/np.log(2))
        
        # Version 1: 
        scale_factor = (-1) * int(np.log(price[-1])/np.log(2))
        px_scale = np.array(2 ** scale_factor, dtype=np.float32)
        
        return np.hstack((cash_ratio,
                          price * px_scale,
                          self.tech_ary[self.day + self.run_index],
                          ))  # state.astype(np.float32)
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
