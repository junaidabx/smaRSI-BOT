import datetime as dt
import time
from main_functions import *
import json
import os
import warnings
import importlib
import botConfig

# Filter out the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Record the start time
start_time = time.time()

# If you want to update the module:
importlib.reload(botConfig)
from botConfig import *

print("Current SandBox Mode is: ", mode)
# st.write('mode is set to : ', mode)
exchange = binanceActive(mode)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the script's directory
os.chdir(script_dir)

# Define your trade records filename
tradesfile = "tradesHistory_bot2.csv"
logfile = "loggile_bot2.csv"
orderInfofile = "orderInfo_bot2.json"
statusfile = "flag_status_bot2.json"

# Load the JSON data from the file
# with open('qty.json', 'r') as f:
#     json_qty = f.read()
with open("Bot2.json", "r") as f:
    json_params = f.read()
with open(orderInfofile, "r") as f:
    json_orderInfo = f.read()

# Convert the JSON data back to a dictionary
# qty = json.loads(json_qty)
info = json.loads(json_orderInfo)
params = json.loads(json_params)

time.sleep(5)

print("\nImported Optimized Parameters")
print("------------------------------")
for key, value in params.items():
    # Create variables with the key as the variable name using globals()
    globals()[key] = value
    print("%s : %s" % (key, value))

currency_code = "USD"  # Replace with the desired currency code
# amount_to_use, usd_free_balance, btc_free_balance = calculate_balance(exchange, currency_code)

interval = map_timeframe(timeframe, get_value=True)

# Define trading variables
# usdt_amount = amount_to_use # 20% amount available balance of client account
# print("Balance to Use: ", usdt_amount)

# Check if the state file exists
if os.path.exists(statusfile):
    # Load state from JSON files at the beginning of the script
    with open(statusfile, "r") as state_file:
        state = json.load(state_file)
else:
    # Initialize default values if the file doesn't exist
    state = {"in_position": False}

in_position = state["in_position"]

# sell_pos = state["sell_pos"]
print("\nCurrent Position States:")
print("=========================")
print("In Position", in_position)

# Define the end date as today
end_date = datetime.now()
start_date = end_date - timedelta(days=1000)    # Calculate the start date by subtracting 1000 days from the end date
# Format the start and end dates as strings
start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")
print("Start Date", start_str)
print("End Date", end_str)

try:
    try:
        # Fetch the latest candlestick data
        print('Fetching ticker data from exchange')
        df = getdataSpot(ticker, interval, start_str, end_str)

    except Exception as e:
        # Handle exceptions raised by getdata_kraken
        print(f"An error occurred while fetching data: {e}")

    if len(df) > 0:
        # Get the latest closing price
        close_price = df["Close"].iloc[-1]
        print('Close Price: ', close_price)
        print('\nFetching Signals Data...')
        df = getSignals(df, fast_ma_period=fast_ma_period, slow_ma_period=slow_ma_period, rsi_period=rsi_period, rsi_value=rsi_value)
        # Fetch the latest buy and sell signals, as well as stop loss levels, from your DataFrame 'df'
        row = df.iloc[-1]  # Assuming the last row contains the latest data
        print(row)
        # Inside the 'if Not in_position:' block
        # --------------------------------------------- position check------------------------------
        if not in_position:
            print('\nIn "Not In Position" Block')
            # print("Postion Method Type: ", method_type)
            # print("Price Type: ", price_type)
            # print("Position Size Value: ", pos_size_value)

            if row["buySignal"] == 1:
                
                print("Got Long Position Signal, Taking Long Position")
                # Calculate position size
                qty = calculate_position_size(
                    method_type, price_type, pos_size_value, ticker, exchange
                )

                print(f"Position Size: {qty}")
                # Place a market buy order for a long position
                place_market_order(ticker, qty, sl_perc, tp_perc, "buy", "long", filename=orderInfofile)
                with open(orderInfofile, "r") as f:
                    json_orderInfo = f.read()

                # Convert the JSON data back to a dictionary
                info = json.loads(json_orderInfo)
                buyCSV(df, buyprice=info["buyprice"], sellprice=0, filename=tradesfile)

        # Inside the 'if in_position:' block
        # --------------------------------------------- position close check------------------------------
        elif in_position:
            print("IN POSITION BLOCK")
            amount = info["amount"]
            print("Side: ", amount)
            stoploss = info["stoploss"]
            takeprofit = info["takeprofit"]
            
            if row["Low"] < stoploss:
                close_position(ticker, amount, "buy", "Long", filename=orderInfofile)
                in_position = False
                
                with open(orderInfofile, "r") as f:
                    json_orderInfo = f.read()
                # Convert the JSON data back to a dictionary
                info = json.loads(json_orderInfo)
                sellcsv(
                    df,
                    buyprice=read_tradefile(tradesfile, "long"),
                    sellprice=info["price"],
                    filename=tradesfile,
                )
            elif row["High"] >= tp:
                close_position(ticker, amount, "buy", "Long", filename=orderInfofile)
                in_position = False
                
                with open(orderInfofile, "r") as f:
                    json_orderInfo = f.read()
                # Convert the JSON data back to a dictionary
                info = json.loads(json_orderInfo)
                sellcsv(
                    df,
                    buyprice=read_tradefile(tradesfile, "long"),
                    sellprice=info["price"],
                    filename=tradesfile,
                )
            
        else:
            print("No Signals!, Exiting")
    csvlog(df, logfile)
except Exception as ex:
    print("An error occurred:", ex)

# Save the updated state to JSON files at the end of the script
state = {"in_position": in_position}

with open(statusfile, "w") as state_file:
    json.dump(state, state_file)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

print(f"Execution Time: {execution_time} seconds")
print(
    "========================================== End of Code ============================================"
)
