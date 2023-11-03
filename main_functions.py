import ccxt
import streamlit as st
import pandas as pd
# import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import numpy as np
import csv
import os
import time

# def binanceActive():
#     exchange = ccxt.binance({
#         'apiKey': binanceAPI['apiKey'],
#         'secret': binanceAPI['secretKey'],
#         'enableRateLimit': True,
#         'rateLimit': 10000,
#         'options': {
#             # 'recvWindow': 9000,  # replace with your desired recv_window value
#             'test': True,  # use testnet (sandbox) environment
#             # 'adjustForTimeDifference': True,
#         }
#     })
#     exchange.set_sandbox_mode(enable=True)
#     return exchange

# exchange = binanceActive()

# Function to set sandbox_mode in botConfig.py
def set_sandbox_mode(sandbox_mode):
    # Path to your botConfig.py file
    config_path = 'botConfig.py'
    with open(config_path, 'r') as config_file:
        config_lines = config_file.readlines()

    with open(config_path, 'w') as config_file:
        for line in config_lines:
            if line.startswith('sandbox_mode'):
                config_file.write(f'sandbox_mode = {sandbox_mode}\n')
            else:
                config_file.write(line)


def set_api_key_secret(api_key, secret_key, config_path, live_mode=False):
    with open(config_path, 'r') as config_file:
        config_lines = config_file.readlines()

    with open(config_path, 'w') as config_file:
        for line in config_lines:
            if line.startswith('sandbox_mode'):
                if live_mode:
                    config_file.write(f'sandbox_mode = False\n')
                else:
                    config_file.write(f'sandbox_mode = True\n')
            elif line.startswith('live_apiKey'):
                if live_mode:
                    config_file.write(f'live_apiKey = \'{api_key}\'\n')
                else:
                    config_file.write(f'demo_apiKey = \'{api_key}\'\n')
            elif line.startswith('live_secret'):
                if live_mode:
                    config_file.write(f'live_secret = \'{secret_key}\'\n')
                else:
                    config_file.write(f'demo_secret = \'{secret_key}\'\n')
            else:
                config_file.write(line)

def get_api_key_secret(config_path, live_mode=False):
    with open(config_path, 'r') as config_file:
        config_lines = config_file.readlines()
        for line in config_lines:
            if live_mode and line.startswith('live_apiKey'):
                api_key = line.split('=')[1].strip()
            elif not live_mode and line.startswith('demo_apiKey'):
                api_key = line.split('=')[1].strip()
            if live_mode and line.startswith('live_secret'):
                secret_key = line.split('=')[1].strip()
            elif not live_mode and line.startswith('demo_secret'):
                secret_key = line.split('=')[1].strip()
    
    return api_key, secret_key

def check_authentication(exchange):
    try:
        balance = exchange.fetch_balance()  # Replace with an actual API request
        # If the request succeeds, the authentication is correct
        return True

    except ccxt.AuthenticationError as e:
        st.info('Could not authenticate, Authentication error')
        return False

def binanceActive(mode_choice):
    # Set sandbox mode based on the selected mode
    if mode_choice == "Sandbox/Demo":
        sandbox_mode = True
    else:
        sandbox_mode = False
    
    # If not authenticated, perform the authentication process
    config_path = 'botConfig.py'
    live_mode = mode_choice == "Live"
    api_key, secret_key = get_api_key_secret(config_path, live_mode=live_mode)
    sandbox_mode = mode_choice == "Sandbox/Demo"

    # Configure the ccxt.binance instance
    exchange = ccxt.binance({
        'apiKey': api_key.strip("'"),
        'secret': secret_key.strip("'"),
        'verbose': False,  # switch it to False if you don't want the HTTP log
    })

    # Enable or disable sandbox mode based on the selected mode
    exchange.set_sandbox_mode(sandbox_mode)

    # Check if the API key and secret are authenticated
    if check_authentication(exchange):
        # authenticated = True  # Cache the authentication result
        print("Authentication Successful.")
        return exchange
    else:
        print("Authentication failed due to invalid key or secret.")
        return None

# Function to check authentication and display messages
def check_authentication_and_display(exchange):
    # exchange = binanceActive(mode_choice, auth)
    success_message = None
    error_message = None
    
    if exchange:
        _message = 'Authentication Successful'
        success_message = st.success(_message)
    else:
        _message = "Authentication failed due to invalid key or secret."
        error_message = st.error(_message)

    time.sleep(5)
    success_message.empty() if success_message else error_message.empty()

    return exchange

# st.write('mode is set to : ', mode)
# exchange = krakenActive(mode)


def servertime():
    time = exchange.fetch_time()
    time = pd.to_datetime(time, unit ='ms')
    print(time)


# Define the map_timeframe function (you've already provided this)
def map_timeframe(resolution, get_value=True):
    tf_mapping = {
        '1min': '1m',
        '2min': '2m',
        '3min': '3m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '45min': '45m',
        '1hour': '1h',
        '4hour': '4h',
        '6hour': '6h',
        '12hour': '12h',
        '1day': '1d',
        '1week': '1w'
    }
    
    if get_value:
        return tf_mapping.get(resolution, '1d')  # Default to '1d' if not found
    else:
        # If you want to reverse map from value to resolution, you can do this:
        reverse_mapping = {v: k for k, v in tf_mapping.items()}
        return reverse_mapping.get(resolution, '1day')  # Default to '1day' if not found


def getqty(coin):
    params = {'type':'futures'}
    for item in exchange.fetch_balance(params=params)['info']['balances']:
        if item['asset'] == coin:
            qty = float(item['free'])
    return qty

# print(getqty('USDT'))

# exchange.fetch_balance()

# Define function to place buy order
def place_buy_order(symbol, size):
    try:
        buyId = exchange.create_market_buy_order(symbol, size, params={})
        return buyId
    except:
        return False


def place_market_order(symbol, amount, sl_perc, tp_perc, order_type, position_type, filename):
    global in_position # Ensure we modify the global variables

    # # Determine position size
    # amount = calculate_order_size(symbol, usdt_amount)
    # # amount = round(position_size, 3)

    try:
        if order_type == 'buy':
            response = exchange.create_market_buy_order(
                symbol=symbol,
                amount=amount
            )
            buy_pos = True  # Set buy_pos to True for Long
        else:
            response = exchange.create_market_sell_order(
                symbol=symbol,
                amount=amount
            )
            sell_pos = True  # Set sell_pos to True for Short

        # Extract order information
        order_id = response['info']['order_id']
        price = float(response['price'])
        amount = float(response['amount'])
        side = response['side']

        print(f"Order ID: {order_id}")
        print(f"{order_type.capitalize()} Price: {price}")
        print(f"Amount: {amount}")
        print(f"Side: {side}")

        in_position = True
        sl = price * (1 - sl_perc) if order_type == 'buy' else price * (1 + sl_perc)
        tp = price * (1 + tp_perc) if order_type == 'buy' else price * (1 - tp_perc)
        status = "In Position"
        
        print(f"Market {order_type.capitalize()} Order Placed at Market Price")
        print(response)

        # Save the order information
        orderInfo = {
            'order_id': order_id,
            'buyprice' if order_type == 'buy' else 'sellprice': price,
            'amount': amount,
            'side': side,
            'stoploss': sl,
            'takeprofit': tp,
            'status': status,
            'position_type': position_type
        }
        with open(filename, 'w') as f:
            json.dump(orderInfo, f)
        print("Order Info: ", orderInfo)
    except ccxt.BaseError as e:
        print("An error occurred:", e)

# Define a function to close a position
def close_position(symbol, amount, order_type, position_type, filename):
    try:
        if order_type == 'buy':
            response = exchange.create_market_sell_order(
                symbol=symbol,
                amount=amount
            )
        else:
            response = exchange.create_market_buy_order(
                symbol=symbol,
                amount=amount
            )

        # Extract the required information
        order_id = response['info']['order_id']
        price = float(response['price'])
        amount = float(response['amount'])
        side = response['side']

        print(f"Order ID: {order_id}")
        print(f"Price: {price}")
        print(f"Amount: {amount}")
        print(f"Side: {side}")

        print(f"{position_type.capitalize()} Position Closed at Market Price")

        # Save the order information
        order_info = {
            'order_id': order_id,
            'price': price,
            'amount': amount,
            'side': side,
            'status': 'Closed',
            'position_type': position_type
        }
        with open(filename, 'w') as order_file:
            json.dump(order_info, order_file)
    except ccxt.BaseError as e:
        print("An error occurred:", e)

# Define function to place sell order
def place_sell_order(symbol, size):
    # try:
    sellId = exchange.create_market_sell_order(symbol, size)
    return sellId
    # except:
    #     return False
    
def calculate_order_size(symbol, usdt_amount):
    # Get the current market price of the coin
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    # Calculate the order size based on the USDT amount and the market price
    size = usdt_amount / price
    return size

# Function to generate a centered category label
def category_label(category_name, total_width=50, dash_character="-"):
    # Calculate the number of dashes required to center the text
    num_dashes = (total_width - len(category_name)) // 2
    dashes = dash_character * num_dashes

    # Create the centered category label
    return f"{dashes} {category_name} {dashes}"

# { ==========================================================================================
# Load historical price data from kraken exchange, but data is limited to 720 candles 

kraken = ccxt.kraken()
def start_time(days):
    timestamp = kraken.fetch_time() - days*86400*1000
    time = pd.to_datetime(timestamp, unit ='ms')
    print(time)
    return timestamp

def getdata_kraken(symbol, timeframe, days):
    df = pd.DataFrame(kraken.fetch_ohlcv(symbol, timeframe, since = start_time(days)))
    df = df[[0,1,2,3,4,5]]
    df.columns = ['timestamp','Open','High','Low','Close','Volume']
    df = df.set_index('timestamp')
    # Convert the datetime index to date+hh:mm:ss format
    df.index = pd.to_datetime(df.index, unit = 'ms') 
    df= df.astype(float)
    return df
# }============================================================================================

# Load historical price data from binance exchange 
from binance.client import Client
from binance.enums import HistoricalKlinesType

client = Client()
def getdataFutures(ticker, timeframe, day):
    start_str = f"{int(timeframe[:-1]) * day * 3600}m"
    df = pd.DataFrame(client.futures_historical_klines(ticker, timeframe, start_str))
    df = df[[0,1,2,3,4,5]]
    df.columns = ['timestamp','Open','High','Low','Close','Volume']
    df = df.set_index('timestamp')
    # Convert the datetime index to date+hh:mm:ss format
    df.index = pd.to_datetime(df.index, unit = 'ms') 
    df= df.astype(float)
    return df 

def getdataSpot(ticker, interval, start_str, end_str):
    # Fetch historical spot klines from Binance
    klines = client.get_historical_klines(symbol=ticker, interval=interval, start_str=start_str, end_str=end_str, klines_type=HistoricalKlinesType.SPOT)
    # Convert the fetched data to a DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
    # Set the 'timestamp' as the index
    df.set_index('timestamp', inplace=True)
    # Convert the 'timestamp' index to datetime
    df.index = pd.to_datetime(df.index, unit='ms')
    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Define the function for the trading strategy
def getSignals(df, fast_ma_period=20, slow_ma_period=50, rsi_period=14, rsi_value=50):
    # df = getdata(ticker, timeframe, day)  # Assuming you have a getdata function that fetches the data

    # Calculate the moving averages
    df['shifted_open'] = df.Open.shift(-1)
    df['Fast_MA'] = ta.trend.sma_indicator(df['Close'], window=fast_ma_period)
    df['Slow_MA'] = ta.trend.sma_indicator(df['Close'], window=slow_ma_period)
    # Calculate the RSI
    rsi = ta.momentum.RSIIndicator(df['Close'], window=rsi_period)
    df['RSI'] = rsi.rsi()

    df['buySignal'] = np.where(
        (df['Fast_MA'] > df['Slow_MA']) &
        (df['Fast_MA'].shift(1) <= df['Slow_MA'].shift(1)) &
        (df['RSI'] > rsi_value),
        1,
        0
    )
    
    return df

def backtest(ticker, df, fast_ma_period=20, slow_ma_period=50, rsi_period=14, rsi_value=50, sl_perc=(5*0.01), tp_perc=(10*0.01)):
    st.write("Input Parameters Selected")
    inputs = f'''
    fast_ma_period: {fast_ma_period}, slow_ma_period: {slow_ma_period}, rsi_period: {rsi_period},
    rsi_value: {rsi_value}, sl_perc: {sl_perc}, tp_perc: {tp_perc}
    '''
    st.write(inputs)
    
    df = getSignals(df, fast_ma_period=fast_ma_period, slow_ma_period=slow_ma_period, rsi_period=rsi_period, rsi_value=rsi_value)  # Use the trading_strategy function to get signals

    in_position = False
    buy_pos = False
    sell_pos = False

    results_df = pd.DataFrame(columns=['ticker', 'returns', 'winrate', 'trades', 'buy&hold_ret%'])
    buydates, buyprices = [], []
    selldates, sellprices = [], []
    profits = []
    sl_list, tp_list = [], []

    for index, row in df.iterrows():
        # long position block
        if not in_position and row['buySignal'] == 1:
            buyprice = row['shifted_open']
            buydates.append(index)
            buyprices.append(buyprice)
            in_position = True
            sl = buyprice * (1 - sl_perc)
            tp = buyprice * (1 + tp_perc)
            sl_list.append(sl)  # Store stop_loss value
            tp_list.append(tp)  # Store target_price value

        elif in_position:
            # To access the last stop_loss and target_price values
            stop_loss = sl_list[-1]
            target_price = tp_list[-1]
            if row['Low'] <= stop_loss:
                selldates.append(index)
                sellprice = stop_loss
                sellprices.append(sellprice)
                in_position = False
                buy_pos = False
                profits.append((sellprice - buyprice) / buyprice - 0.001)  # Account for a small commission fee (0.1%)

            elif row['High'] >= target_price:
                selldates.append(index)
                sellprice = target_price
                sellprices.append(sellprice)
                in_position = False
                buy_pos = False
                profits.append((sellprice - buyprice) / buyprice - 0.001)  # Account for a small commission fee (0.1%)

    try:
        if len(buydates) == 0:
            st.write("No trades were made.")
        else:
            returns = (pd.Series(profits, dtype=float) + 1).prod() - 1
            returns = round(returns * 100, 2)

            wins = sum(1 for i in profits if i > 0)
            winrate = round((wins / len(buydates)) * 100, 2)
            ct = min(len(buydates), len(selldates))
            buy_hold_ret = (df['Close'][-1] - df['Open'][0]) / df['Open'][0] * 100
            # Calculate your custom score here (e.g., a weighted combination of returns and win rate)
            custom_score = returns * (winrate / 100)
            
            results_df.loc[len(results_df)] = [ticker, returns, winrate, ct, buy_hold_ret]
            st.write(f'{ticker}, winrate={winrate}%, returns={returns}%, no. of trades = {ct}, buy&hold_ret = {buy_hold_ret}%')
    except:
        st.write('Invalid input')

    return {
        'buydates': buydates,
        'buyprices': buyprices,
        'selldates': selldates,
        'sellprices': sellprices,
        'profits': profits,
        'stop_loss': sl_list,  # Include stop_loss values
        'take_profit': tp_list,  # Include target_price values
        # Other results...
    }, results_df, custom_score

def displayTrades(**kwargs):
    # Access the trade data and other results from kwargs
    buydates = kwargs['buydates']
    buyprices = kwargs['buyprices']
    selldates = kwargs['selldates']
    sellprices = kwargs['sellprices']
    profits = kwargs['profits']
    stop_loss = kwargs['stop_loss']
    take_profit = kwargs['take_profit']
    
    ct = min(len(buydates),len(selldates))
    dfr =pd.DataFrame()
    dfr['buydates']= buydates[:ct]
    dfr['buyprice']= buyprices[:ct]
    dfr['selldates'] = selldates[:ct]
    dfr['sellprice'] = sellprices[:ct]
    dfr['profits'] = (profits[:ct])
    dfr['commulative_returns'] = ((pd.Series(profits) + 1).cumprod())
    dfr['tradeSide'] = np.where(dfr['buydates'] < dfr['selldates'], 'Long', 'Short')
    
    dfr['entry_date'] = np.where(dfr['tradeSide'] == 'Long', dfr['buydates'], dfr['selldates'])
    dfr['exit_date'] = np.where(dfr['tradeSide'] == 'Long', dfr['selldates'], dfr['buydates'])
    dfr['entry_price'] = np.where(dfr['tradeSide'] == 'Long', dfr['buyprice'], dfr['sellprice'])
    dfr['exit_price'] = np.where(dfr['tradeSide'] == 'Long', dfr['sellprice'], dfr['buyprice'])
    
    # Create a list of column names with the desired display order
    display_order = ['entry_date', 'entry_price', 'exit_date', 'exit_price', 'profits', 'commulative_returns', 'tradeSide']
    
    # Create a copy of the DataFrame with only the columns to be displayed
    dfr_display = dfr[display_order].copy()
    dfr_display['stop_loss'] = (stop_loss[:ct])
    dfr_display['take_profit'] = (take_profit[:ct])
    
    return dfr, dfr_display

# code for appending a new row to the trades CSV file
def csvlog(df, filename):
    headers = ['timestamp','Open','High','Low','Close','Volume', 'buySignal', 'shortSignal']
    
    if not os.path.isfile(filename):
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(headers)


    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = df.index[-1]
        row_to_write = [timestamp] + df.iloc[-2].tolist()
        writer.writerow(row_to_write)

# code for appending a new row to the trades CSV file
def buycsv(df, buyprice,filename):
    headers = ['timestamp', 'buyprice', 'sellprice', 'profit%']
    
    if not os.path.isfile(filename):
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(headers)


    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        buy_price = buyprice # replace with actual buy price
        sell_price =  "position still open"# replace with actual sell price
        profit_percentage = "nan" #((sell_price - buy_price) / buy_price) * 100
        timestamp = df.index[-1]
        writer.writerow([timestamp,buy_price,sell_price,profit_percentage])
        


def sellcsv(df, buyprice, sellprice, filename):
    headers = ['timestamp', 'buyprice', 'sellprice', 'profit%']
    
    if not os.path.isfile(filename):
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        buy_price = buyprice # replace with actual buy price
        sell_price =  sellprice # replace with actual sell price
        profit_percentage = ((sell_price - buy_price) / buy_price) * 100
        timestamp = df.index[-1]
        writer.writerow([timestamp,buy_price,sell_price,profit_percentage])


# asset = 0
# balance = np.nan
def in_pos(coin):
    balance = exchange.fetch_balance()['info']['balances']
    try:
        asset = float([i['free'] for i in balance if i['asset'] == coin][0])
        if asset > 0:
            in_position = True
        else:
            in_position = False
    except Exception as e:
        print(e)
        in_position = False
        asset = 0
    return in_position, balance, asset


def read_buyprice(filename):
    try:
        trades = pd.read_csv(filename)
        buyprice = trades['buyprice'].iloc[-1]
    except:
        buyprice = np.nan
    return buyprice

import json

def update_dict_value(filename, key, value):
    with open(filename, 'r') as f:
        d = json.load(f)
    d[key] = value
    with open(filename, 'w') as f:
        json.dump(d, f)

def plot_candlestick_with_indicators(df, dfr, dfr_display):
    # Create a subplot with two rows and one column
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_width=[1, 1])

    # Determine the desired y-axis range based on your data
    y_min = df['Low'].min()  # Replace 'Low' with the appropriate column name
    y_max = df['High'].max()  # Replace 'High' with the appropriate column name

    bull_candle = '#e7bf4f'
    bearish_candle = '#497ad2'

    # Add candlestick trace to the main subplot
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color=bull_candle,  # Bull candle outline color
        decreasing_line_color=bearish_candle,    # Bear candle outline color
        increasing_fillcolor=bull_candle,  # Bull candle fill color
        decreasing_fillcolor=bearish_candle,  # Bear candle fill color
        name='Candles',
    ), row=1, col=1)

    # Add Fast_MA and Slow_MA to the main subplot (overlay)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Fast_MA'],
        mode='lines',
        line=dict(color='blue'),
        name='SMA_FAST'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Slow_MA'],
        mode='lines',
        line=dict(color='red'),
        name='SMA_SLOW'
    ), row=1, col=1)

    entry_dates = {"Long": [], "Short": []}

    for index, row in dfr.iterrows():
        if row["tradeSide"] == "Long":
            entry_dates["Long"].append(row['buydates'])
        else:
            entry_dates["Short"].append(row['selldates'])

    # Plot the Long signals as green arrows
    fig.add_trace(go.Scatter(
        x=entry_dates["Long"],
        y=df['Close'].loc[entry_dates["Long"]],
        mode='markers',
        marker=dict(
            symbol='arrow-bar-right',
            size=15,
            color='green',
        ),
        name='Long Signals',
    ))

    # Plot the Short signals as red arrows
    fig.add_trace(go.Scatter(
        x=entry_dates["Short"],
        y=df['Close'].loc[entry_dates["Short"]],
        mode='markers',
        marker=dict(
            symbol='arrow-bar-right',
            size=15,
            color='red',
        ),
        name='Short Signals',
    ))

    # Plot stop_loss and target_price polygons
    for index, row in dfr_display.iterrows():
        if not pd.isna(row['stop_loss']) and not pd.isna(row['take_profit']):
            fig.add_shape(
                type="rect",
                x0=row['entry_date'],
                x1=row['exit_date'],
                y0=row['stop_loss'],
                y1=row['entry_price'],
                fillcolor="rgba(255, 0, 0, 0.3)",  # Red polygon for stop_loss
                line=dict(width=0),
                xref="x",
                yref="y",
            )
            
            fig.add_shape(
                type="rect",
                x0=row['entry_date'],
                x1=row['exit_date'],
                y0=row['entry_price'],
                y1=row['take_profit'],
                fillcolor="rgba(0, 255, 0, 0.3)",  # Green polygon for target_price
                line=dict(width=0),
                xref="x",
                yref="y",
            )

    # Customize the layout
    # Update layout to customize the appearance (optional)
    fig.update_layout(
        yaxis=dict(
            # Remove the 'range' parameter to let Plotly automatically determine the y-axis range
            range=[y_min, y_max],  # adjust scaling at y-axis
            # autorange=True,  # Set autorange to True to enable autoscaling
        ),
        height=800,  # Set the desired height in pixels
        title_text='Candlestick Chart with SMA and RSI',
        xaxis_rangeslider_visible=False,
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        hovermode='x'
    )

    # Set the hovermode to 'x unified' to enable a vertical line on hover
    fig.update_layout(hovermode='x unified')

    # Add RSI to a separate subplot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        line=dict(color='green'),
        name='RSI'
    ), row=2, col=1)

    # Add RSI upper, middle, and lower bands
    fig.add_shape(
        type="line",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=70,
        y1=70,
        line=dict(color="purple", width=1, dash="dot"),
        xref="x",
        yref="y",
        row=2,
        col=1
    )

    fig.add_shape(
        type="line",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=50,
        y1=50,
        line=dict(color="purple", width=1, dash="dot"),
        xref="x",
        yref="y",
        row=2,
        col=1
    )

    fig.add_shape(
        type="line",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=30,
        y1=30,
        line=dict(color="purple", width=1, dash="dot"),
        xref="x",
        yref="y",
        row=2,
        col=1
    )

    # Add a filled area between the bands
    fig.add_shape(
        type="rect",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=70,
        y1=30,
        fillcolor="rgba(128, 0, 128, 0.2)",  # Light purple color
        line=dict(width=0),
        xref="x",
        yref="y",
        row=2,
        col=1
    )

    # Add horizontal line for the crosshair in the subplot
    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=1,
        y1=4,
        line=dict(color='purple', width=1, dash='dot'),
    )

    return fig


# Function to calculate position size
def calculate_position_size(
    method_type, price_type, value, sl_price, 
    symbol, exchange, backtest=False, current_balance=None, 
    entry_price=None
    ):
    ticker= None
    if not backtest:
        # Get the current ticker for the symbol
        ticker = exchange.fetch_ticker(symbol)
    
    entry_price = entry_price if backtest else ticker['last']  # Use provided entry_price during backtest
    sym_quote = symbol[-4:]
    _bal = current_balance if backtest else exchange.fetch_balance()['free'][sym_quote]  # Use provided total_bal during backtest
    print('Entry Price: ', entry_price)
    print('SL Price: ', sl_price)
    print('Total Balance: ', total_bal)
    # print('Symbol: ', symbol)
    # print('Qoote for Symbol is: ', sym_quote)
    # Calculate position size
    
    if method_type == 'Fixed':
        if price_type == 'Quote':
            qty_ = (value / entry_price)
        elif price_type == 'Base':
            qty_ = value
        elif price_type == 'Percentage':
            qty_ = (total_bal * (value * 0.01)) / entry_price
    elif method_type == 'Dynamic':
        if price_type == 'Quote':
            qty_quote = (entry_price / ((abs(entry_price - sl_price)) / value))
            qty_ = qty_quote / entry_price
        elif price_type == 'Base':
            qty_ = (entry_price / ((abs(entry_price - sl_price)) / (value * entry_price))) / entry_price
        elif price_type == 'Percentage':
            qty_quote = (entry_price / ((abs(entry_price - sl_price)) / (total_bal * (value * 0.01))))
            qty_ = qty_quote / entry_price

    qty = round(qty_, 3)
    print('Quantity Calculated: ', qty)
    # Calculate money needed
    money_needed = qty * entry_price
    print(f"Money ${money_needed} is needed to buy qty : {qty}")

    # Check if position size exceeds available equity and adjust if needed
    if money_needed > total_bal:
        qty_ = total_bal / entry_price
        qty = round(qty_, 3)
        print(f"Money needed to buy quantity Exceeded Total Balance, So Adjusting Qty : {qty}")
    print('--------------------------------------------------------------------------\n')
    return qty

# # update the json files
# pos = in_pos("BTC")
# in_position = pos[0]
# update_dict_value('pos.json', 'btc4h', in_position)

# qty = pos[2]
# update_dict_value('qty.json', 'btc4h', qty)

# trades = exchange.fetch_trades('ETHUSDT')[-2:]
# print(trades)


# async def get_qty(coin):
#     balance = exchange.fetch_balance()
#     qty = [float(item['free']) for item in balance['info']['balances'] if item['asset'] == coin][0]
#     return qty

# async def main():
#     qty_task = asyncio.create_task(get_qty(coin))
#     qty_task2 =  asyncio.create_task(getqty(coin))
#     qty = await qty_task
#     print(servertime())
#     qty2 = await qty_task2
#     print(servertime())
#     print(qty)
#     print(f"qty = {qty2}")


# asyncio.run(main())