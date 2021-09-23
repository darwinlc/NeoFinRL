#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from datetime import datetime, timedelta


# In[4]:


today_d = datetime.today()
start_d = today_d - timedelta(days = 40)


# In[5]:


query_start = start_d.strftime('%Y-%m-%d')
query_end = today_d.strftime('%Y-%m-%d')


# In[6]:


print (query_start, query_end)


# In[7]:


tic_list = ['DOGE-USD']

tech_indicators = ['cci_30',
 'rsi_30',
 'rsi_14',
 'rsi_6',
 'dx_30', 
 'dx_14']


# ### Daily configuration

# In[8]:

coin_balance = np.array([float(sys.argv[1])], dtype = float)
cash_balance = float(sys.argv[2])
model_file = sys.argv[3]

print (f'Cash Balance: {cash_balance}; Coin Balance: {coin_balance}; Model Fiel: {model_file}')

# ### Pre-train setup

# In[9]:


reward_on_value = True
lookback_n = 2

config_max_step = 15

if reward_on_value:
    reward_scaling = 2 ** -5
else:
    reward_scaling = 2 ** -3


# ### Query last 40 days price

# In[ ]:


from neo_finrl.data_processors.processor_yahoofinance import YahooFinanceProcessor


# In[ ]:


data_downloader = YahooFinanceProcessor()


# In[ ]:


stock_history_df = data_downloader.download_data(query_start, query_end, tic_list, '1D')


# In[ ]:


data_downloader.time_interval = '1D'
stock_history_df = data_downloader.clean_data(stock_history_df)


# In[ ]:


stock_history_df = data_downloader.add_technical_indicator(stock_history_df, tech_indicators)


# In[ ]:

print ("Last Two Data Row:")
print (stock_history_df.tail(2))


# In[ ]:


#stock_history_df.to_csv('./DOGE_px_20210918.csv', index = False)


# ### Model run

# In[10]:


from test_env.single_crypto_env import CryptoTradingEnv

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.logger import configure


# In[11]:


tmp_path = "./tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout"])


# In[12]:


def modelRun(start_idx, px_df, input_amount, input_stocks, last_model):
    def tradable_size(env, x):
            return (x / env.min_stock_rate).astype(int) * env.min_stock_rate
    
    test_config = dict()

    test_config['price_array'] = px_df.iloc[:(start_idx + config_max_step)][['open', 'adjcp', 'low', 'high']].values
    test_config['tech_array'] = px_df.iloc[:(start_idx + config_max_step)][tech_indicators].values

    #randomly start day index for back testing
    test_config['if_sequence'] = True
    # disable random initial capital 
    test_config['if_randomV'] = False

    test_config['if_value'] = reward_on_value
    test_config['lookback_n'] = lookback_n

    max_step = min(config_max_step, px_df.shape[0] - start_idx) - 1
    
    print ('Run model from ', start_idx, ' to ', start_idx + max_step)
    
    test_env = CryptoTradingEnv(test_config,                             initial_capital=input_amount,                             max_step = max_step,                            initial_stocks = input_stocks, 
                           reward_scaling = reward_scaling, \
                            start_idx = start_idx)
    state = test_env.reset()
    
    print (state)
    
    #test_model = PPO.load(cwd)
    test_model = DDPG.load(last_model)
    test_model = test_model.policy.eval()
    
    action = test_model.predict(state)[0]   
    
    # actions -> percentage of stock or cash
    # add clip at 0.9
    actions_v = action[0] * 0.9
    
    if actions_v == np.nan:
        actions_v = 0.0
        
    order_px = (test_env.price_ary[test_env.day + test_env.run_index, 0] +                     test_env.price_ary[test_env.day + test_env.run_index, 1])/2.0
        
    print ('Action value: ', actions_v)
        
    if actions_v > 0:
        buy_num_shares = tradable_size(test_env, (test_env.amount * actions_v/order_px)/(1 + test_env.buy_cost_pct))
        print (f'Buy {buy_num_shares} at price {order_px}')
            
    if actions_v < 0:
        sell_num_shares = tradable_size(test_env, test_env.stocks[0] * (-1.0) * actions_v)
        # no short 
        sell_num_shares = min(sell_num_shares, test_env.stocks[0])
        print (f'Sell {sell_num_shares} at price {order_px}')
        
    print ("\n")
    print ("[!!Warning!!] Order may not be able to placed if it is lower the mininal trade amount!!")
    print ("[!!Warning!!] check current MKT price for better deal!!")
    
    return -1


# In[13]:


#stock_history_df = pd.read_csv('./DOGE_px_20210918.csv')


# In[14]:


modelRun(stock_history_df.shape[0]-1, 
         stock_history_df, 
         cash_balance,
         coin_balance, 
         model_file)


# In[ ]:




