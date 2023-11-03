import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import optuna
import main_functions
import importlib
import subprocess
import warnings
import botConfig

def init_session_state():
    if not hasattr(st, 'session_state'):
        st.session_state.my_session_state = None

# Initialize session state
st.session_state.authenticated = False
# 
# Filter out the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# If you want to update the module:
importlib.reload(main_functions)
from main_functions import *

# If you want to update the module:
importlib.reload(botConfig)
from botConfig import *

binance_tickers= ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'BCCUSDT', 'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'TUSDUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VENUSDT', 'NULSUSDT', 'VETUSDT', 'PAXUSDT', 'BCHABCUSDT', 'BCHSVUSDT', 'USDCUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'USDSUSDT', 'ONGUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'NANOUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT', 'MITHUSDT', 'MATICUSDT', 'ATOMUSDT', 'TFUELUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'USDSBUSDT', 'GTOUSDT', 'ERDUSDT', 'DOGEUSDT', 'DUSKUSDT', 'ANKRUSDT', 'WINUSDT', 'COSUSDT', 'NPXSUSDT', 'COCOSUSDT', 'MTLUSDT', 'TOMOUSDT', 'PERLUSDT', 'DENTUSDT', 'MFTUSDT', 'KEYUSDT', 'STORMUSDT', 'DOCKUSDT', 'WANUSDT', 'FUNUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'BUSDUSDT', 'BEAMUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HCUSDT', 'HBARUSDT', 'NKNUSDT', 'STXUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'MCOUSDT', 'CTXCUSDT', 'BCHUSDT', 'TROYUSDT', 'VITEUSDT', 'FTTUSDT', 'EURUSDT', 'OGNUSDT', 'DREPUSDT', 'BULLUSDT', 'BEARUSDT', 'ETHBULLUSDT', 'ETHBEARUSDT', 'TCTUSDT', 'WRXUSDT', 'BTSUSDT', 'LSKUSDT', 'BNTUSDT', 'LTOUSDT', 'EOSBULLUSDT', 'EOSBEARUSDT', 'XRPBULLUSDT', 'XRPBEARUSDT', 'STRATUSDT', 'AIONUSDT', 'MBLUSDT', 'COTIUSDT', 'BNBBULLUSDT', 'BNBBEARUSDT', 'STPTUSDT', 'WTCUSDT', 'DATAUSDT', 'XZCUSDT', 'SOLUSDT', 'CTSIUSDT', 'HIVEUSDT', 'CHRUSDT', 'BTCUPUSDT', 'BTCDOWNUSDT', 'GXSUSDT', 'ARDRUSDT', 'LENDUSDT', 'MDTUSDT', 'STMXUSDT', 'KNCUSDT', 'REPUSDT', 'LRCUSDT', 'PNTUSDT', 'COMPUSDT', 'BKRWUSDT', 'SCUSDT', 'ZENUSDT', 'SNXUSDT', 'ETHUPUSDT', 'ETHDOWNUSDT', 'ADAUPUSDT', 'ADADOWNUSDT', 'LINKUPUSDT', 'LINKDOWNUSDT', 'VTHOUSDT', 'DGBUSDT', 'GBPUSDT', 'SXPUSDT', 'MKRUSDT', 'DAIUSDT', 'DCRUSDT', 'STORJUSDT', 'BNBUPUSDT', 'BNBDOWNUSDT', 'XTZUPUSDT', 'XTZDOWNUSDT', 'MANAUSDT', 'AUDUSDT', 'YFIUSDT', 'BALUSDT', 'BLZUSDT', 'IRISUSDT', 'KMDUSDT', 'JSTUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'NMRUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT', 'PAXGUSDT', 'WNXMUSDT', 'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'DIAUSDT', 'RUNEUSDT', 'FIOUSDT', 'UMAUSDT', 'EOSUPUSDT', 'EOSDOWNUSDT', 'TRXUPUSDT', 'TRXDOWNUSDT', 'XRPUPUSDT', 'XRPDOWNUSDT', 'DOTUPUSDT', 'DOTDOWNUSDT', 'BELUSDT', 'WINGUSDT', 'LTCUPUSDT', 'LTCDOWNUSDT', 'UNIUSDT', 'NBSUSDT', 'OXTUSDT', 'SUNUSDT', 'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'UNIUPUSDT', 'UNIDOWNUSDT', 'ORNUSDT', 'UTKUSDT', 'XVSUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'SXPUPUSDT', 'SXPDOWNUSDT', 'FILUSDT', 'FILUPUSDT', 'FILDOWNUSDT', 'YFIUPUSDT', 'YFIDOWNUSDT', 'INJUSDT', 'AUDIOUSDT', 'CTKUSDT', 'BCHUPUSDT', 'BCHDOWNUSDT', 'AKROUSDT', 'AXSUSDT', 'HARDUSDT', 'DNTUSDT', 'STRAXUSDT', 'UNFIUSDT', 'ROSEUSDT', 'AVAUSDT', 'XEMUSDT', 'AAVEUPUSDT', 'AAVEDOWNUSDT', 'SKLUSDT', 'SUSDUSDT', 'SUSHIUPUSDT', 'SUSHIDOWNUSDT', 'XLMUPUSDT', 'XLMDOWNUSDT', 'GRTUSDT', 'JUVUSDT', 'PSGUSDT', '1INCHUSDT', 'REEFUSDT', 'OGUSDT', 'ATMUSDT', 'ASRUSDT', 'CELOUSDT', 'RIFUSDT', 'BTCSTUSDT', 'TRUUSDT', 'CKBUSDT', 'TWTUSDT', 'FIROUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'CAKEUSDT', 'ACMUSDT', 'BADGERUSDT', 'FISUSDT', 'OMUSDT', 'PONDUSDT', 'DEGOUSDT', 'ALICEUSDT', 'LINAUSDT', 'PERPUSDT', 'RAMPUSDT', 'SUPERUSDT', 'CFXUSDT', 'EPSUSDT', 'AUTOUSDT', 'TKOUSDT', 'PUNDIXUSDT', 'TLMUSDT', '1INCHUPUSDT', '1INCHDOWNUSDT', 'BTGUSDT', 'MIRUSDT', 'BARUSDT', 'FORTHUSDT', 'BAKEUSDT', 'BURGERUSDT', 'SLPUSDT', 'SHIBUSDT', 'ICPUSDT', 'ARUSDT', 'POLSUSDT', 'MDXUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'XVGUSDT', 'ATAUSDT', 'GTCUSDT', 'TORNUSDT', 'KEEPUSDT', 'ERNUSDT', 'KLAYUSDT', 'PHAUSDT', 'BONDUSDT', 'MLNUSDT', 'DEXEUSDT', 'C98USDT', 'CLVUSDT', 'QNTUSDT', 'FLOWUSDT', 'TVKUSDT', 'MINAUSDT', 'RAYUSDT', 'FARMUSDT', 'ALPACAUSDT', 'QUICKUSDT', 'MBOXUSDT', 'FORUSDT', 'REQUSDT', 'GHSTUSDT', 'WAXPUSDT', 'TRIBEUSDT', 'GNOUSDT', 'XECUSDT', 'ELFUSDT', 'DYDXUSDT', 'POLYUSDT', 'IDEXUSDT', 'VIDTUSDT', 'USDPUSDT', 'GALAUSDT', 'ILVUSDT', 'YGGUSDT', 'SYSUSDT', 'DFUSDT', 'FIDAUSDT', 'FRONTUSDT', 'CVPUSDT', 'AGLDUSDT', 'RADUSDT', 'BETAUSDT', 'RAREUSDT', 'LAZIOUSDT', 'CHESSUSDT', 'ADXUSDT', 'AUCTIONUSDT', 'DARUSDT', 'BNXUSDT', 'RGTUSDT', 'MOVRUSDT', 'CITYUSDT', 'ENSUSDT', 'KP3RUSDT', 'QIUSDT', 'PORTOUSDT', 'POWRUSDT', 'VGXUSDT', 'JASMYUSDT', 'AMPUSDT', 'PLAUSDT', 'PYRUSDT', 'RNDRUSDT', 'ALCXUSDT', 'SANTOSUSDT', 'MCUSDT', 'ANYUSDT', 'BICOUSDT', 'FLUXUSDT', 'FXSUSDT', 'VOXELUSDT', 'HIGHUSDT', 'CVXUSDT', 'PEOPLEUSDT', 'OOKIUSDT', 'SPELLUSDT', 'USTUSDT', 'JOEUSDT', 'ACHUSDT', 'IMXUSDT', 'GLMRUSDT', 'LOKAUSDT', 'SCRTUSDT', 'API3USDT', 'BTTCUSDT', 'ACAUSDT', 'ANCUSDT', 'XNOUSDT', 'WOOUSDT', 'ALPINEUSDT', 'TUSDT', 'ASTRUSDT', 'NBTUSDT', 'GMTUSDT', 'KDAUSDT', 'APEUSDT', 'BSWUSDT', 'BIFIUSDT', 'MULTIUSDT', 'STEEMUSDT', 'MOBUSDT', 'NEXOUSDT', 'REIUSDT', 'GALUSDT', 'LDOUSDT', 'EPXUSDT', 'OPUSDT', 'LEVERUSDT', 'STGUSDT', 'LUNCUSDT', 'GMXUSDT', 'NEBLUSDT', 'POLYXUSDT', 'APTUSDT', 'OSMOUSDT', 'HFTUSDT', 'PHBUSDT', 'HOOKUSDT', 'MAGICUSDT', 'HIFIUSDT', 'RPLUSDT', 'PROSUSDT', 'AGIXUSDT', 'GNSUSDT', 'SYNUSDT', 'VIBUSDT', 'SSVUSDT', 'LQTYUSDT', 'AMBUSDT', 'BETHUSDT', 'USTCUSDT', 'GASUSDT', 'GLMUSDT', 'PROMUSDT', 'QKCUSDT', 'UFTUSDT', 'IDUSDT', 'ARBUSDT', 'LOOMUSDT', 'OAXUSDT', 'RDNTUSDT', 'WBTCUSDT', 'EDUUSDT', 'SUIUSDT', 'AERGOUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'ASTUSDT', 'SNTUSDT', 'COMBOUSDT', 'MAVUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'WBETHUSDT', 'WLDUSDT', 'FDUSDUSDT', 'SEIUSDT', 'CYBERUSDT', 'ARKUSDT', 'CREAMUSDT', 'GFTUSDT', 'IQUSDT', 'NTRNUSDT'
]

# Define a list of bot names
bot_names = ["Bot1", "Bot2", "Bot3"]

# Create dictionaries to store parameters for each bot
bot_parameters = {bot_name: {} for bot_name in bot_names}

# Constants for position size types and price types
pos_size_types = ['Fixed']
pos_size_price = ['Quote', 'Base', 'Percentage']

# Define tooltips
tooltip_base = "Position size value is specified in base currency (e.g. BTC)."
tooltip_quote = "Position size value is specified in quote currency (i.e. USD)."
tooltip_percentage = "Position size value is specified as a percentage of available equity."
tt_method_type = """Fixed P/S: $1K, or 0.1 BTC
"""
pos_label = "Position Size Calculator" 

# Main Streamlit app
def main():
    st.title("Backtesting and Optimization App")

    bot_name = st.selectbox("Select Bot for Parameter Optimization:", bot_names)
    st.session_state.bot_name = bot_name
    # Sidebar with input parameters
    st.subheader(f"**Input Parameters for {bot_name}**")

    ticker = st.selectbox("Select Ticker Symbol:", binance_tickers, index=binance_tickers.index("BTCUSDT"))
    # Input the timeframe as a selectbox
    tf = ['1min', '5min', '15min', '30min', '1hour', '4hour', '12hour', '1day', '1week']  # Add more timeframes as needed
    sel_tf = st.selectbox("Select Timeframe:", tf, index=tf.index('1day'))
    # Map the selected timeframe to its value
    timeframe = map_timeframe(sel_tf)
    # st.write("Mapped Value:", timeframe)
    
    # You can use a date input widget for the start date
    start_str = str(st.date_input("Select Start Date:", datetime(2019, 9, 8)))
    # start_str = datetime.strptime(start_date_input, "%Y-%m-%d")
    end_str = str(st.date_input("Select End Date:", datetime.now()))
    # end_str = datetime.strptime(end_date_input, "%Y-%m-%d")
    # current_date = str(datetime.now())
    # Calculate the difference in days
    # day = (datetime.now() - start_date).days
    with st.sidebar:
            
        st.title("Strategy Parameters")
        st.markdown(category_label(pos_label))
        # initial_capital = st.number_input("Initial Capital (USD)")
        # Strategy inputs
        method_type = st.selectbox("Position Size", pos_size_types, index=0, help=tt_method_type)
        
        if method_type == "Fixed":
            price_type = st.selectbox("Price Type", pos_size_price, index=0, help=tooltip_quote)
            if price_type == "Quote":
                value_label = "$1000"
                value = st.number_input("Position Size Value (USD)", value=1000.0, help="Enter the position size value in quote currency (USD).")
            elif price_type == "Base":
                value_label = "0.1 BTC"
                value = st.number_input("Position Size Value (BTC)", value=0.1, help="Enter the position size value in base currency (BTC).")
            else:
                value_label = "Percentage"
                value = st.number_input("Position Size Value (%)", value=20, help="Enter the percentage value for position size.")
    
    st.sidebar.markdown(category_label("Optimization Parameters"))
    fast_ma_period = st.sidebar.number_input("Fast MA Period", min_value=1, max_value=100, value=20)
    slow_ma_period = st.sidebar.number_input("Slow MA Period", min_value=1, max_value=100, value=50)
    rsi_period = st.sidebar.number_input("RSI Period", min_value=1, max_value=50, value=14)
    rsi_value = st.sidebar.number_input("RSI Value", min_value=1, max_value=100, value=50)
    sl_perc = st.sidebar.number_input("Stop Loss Percentage", min_value=1, max_value=20, value=5)
    tp_perc = st.sidebar.number_input("Take Profit Percentage", min_value=1, max_value=20, value=10)

    # Convert start date to a string
    # start_str = start.strftime("%Y-%m-%d")

    if st.sidebar.button("Run Backtest"):
        # Call your backtest function with the selected parameters
        st.header("Backtest Results & Display Trades")
        df = getdataSpot(ticker, timeframe, start_str, end_str)
        st.session_state.df = df
        results_data, results_df, score = backtest(ticker, df, fast_ma_period, slow_ma_period, rsi_period, rsi_value, sl_perc*0.01, tp_perc*0.01)
        st.session_state.results_data = results_data
        st.session_state.results_df = results_df

        dfr, dfr_display = displayTrades(**results_data)
        st.session_state.dfr_display = dfr_display
        # Display the results in the main window
        # Display backtest results here
        # st.write(results_df)
        st.dataframe(dfr_display)
        
        # Plot the candlestick chart with indicators
        fig = plot_candlestick_with_indicators(df, dfr, dfr_display)
        st.session_state.fig = fig
        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

    if st.sidebar.button("Run Optimization"):
        # Call your optimization function using Optuna with selected parameters
        def objective(trial):
            fast_ma_period = trial.suggest_int("fast_ma_period", 15, 30)
            slow_ma_period = trial.suggest_int("slow_ma_period", 50, 75)
            rsi_period = trial.suggest_int("rsi_period", 11, 24)
            rsi_value = trial.suggest_int("rsi_value", 45, 60)
            sl_perc = trial.suggest_int("sl_perc", 3, 9)
            tp_perc = trial.suggest_int("tp_perc", 5, 12)
            
            # Run your strategy with the trial parameters and calculate a score
            _, __, score = backtest(ticker, st.session_state.df, fast_ma_period, slow_ma_period, rsi_period, sl_perc=sl_perc*0.01, tp_perc=tp_perc*0.01)
            return score
        
        st.header("Backtest Results & Display Trades")
        st.write(st.session_state.results_df)
        st.dataframe(st.session_state.dfr_display)
        st.plotly_chart(st.session_state.fig)
        
        # Create an Optuna study and run optimization
        study = optuna.create_study(direction="maximize")  # You can also use "minimize" depending on your objective
        study.optimize(objective, n_trials=100)

        # Retrieve the best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        st.session_state.best_score = best_score
        # Update the dictionary for the selected bot
        bot_parameters[st.session_state.bot_name] = {
            "ticker": ticker,
            "timeframe": timeframe,
            "method_type": method_type,
            "price_type": price_type,
            "pos_size_value": value,
            **best_params  # Add the optimized parameters
        }
        st.session_state.best_params = bot_parameters[st.session_state.bot_name]
        
        st.subheader(f"Optimization Results for {st.session_state.bot_name}")
        # Display optimization results here
        st.write("Best Parameters:", st.session_state.best_params)
        st.write("Best Score:", best_score)
        # Display the optimized parameters and score in the main window
        
    if st.sidebar.button("Save and Copy Parameters"):
        # Save the parameters to a JSON file
        file_name = f"{st.session_state.bot_name}.json"
        with open(file_name, "w") as json_file:
            json.dump(st.session_state.best_params, json_file, indent=4)
        
        st.subheader(f"{st.session_state.bot_name} Best Parameters")
        st.write(st.session_state.best_params)
        st.write("Best Score", st.session_state.best_score)
        
        st.success(f"Parameters for {st.session_state.bot_name} saved and copied to JSON.")
    
    with st.sidebar:
        authen_label = "Login info"
        # Create the centered category label
        st.markdown(category_label(authen_label))
        # Check the value of sandbox_mode in kraken_config.py
        mode = botConfig.sandbox_mode
        # Radio button to choose mode (sandbox/demo or live)
        mode_options = ["Sandbox/Demo", "Live"]

        # Check if mode_choice is in session state and set the mode accordingly
        init_session_state()
        # Initialize session state
        if 'mode_choice' not in st.session_state:
            # Set the initial mode_choice based on the value in kraken_config.py
            st.session_state.mode_choice = "Sandbox/Demo" if mode else "Live"

        # Get the selected mode_choice from session state
        mode_choice = st.radio("Select Mode:", mode_options, index=mode_options.index(st.session_state.mode_choice))

        # Set the mode_choice in session state
        st.session_state.mode_choice = mode_choice
        # Check if authentication is already done
        if not st.session_state.authenticated:
            # Display authentication widget
            # When authentication is successful:
            st.session_state.authenticated = True
            exchange = binanceActive(st.session_state.mode_choice)
            # Call the authentication function and display messages
            check_authentication_and_display(exchange)
        else:
            # If already authenticated, return the cached exchange
            st.info("Using cached authentication result.")

        # Set sandbox_mode based on mode_choice and update it in kraken_config.py
        sandbox_mode = (mode_choice == "Sandbox/Demo")
        set_sandbox_mode(sandbox_mode)
        # Print the updated mode for debugging
        st.write("Updated SandBox Mode to: ", sandbox_mode)
        
        # Get the existing API key and secret from config file
        config_path = 'botConfig.py'
        live_mode = mode_choice == "Live"  # Check the mode from kraken_config.py
        api_key, secret_key = get_api_key_secret(config_path, live_mode=live_mode)

        if mode_choice == "Sandbox/Demo":
            st.write("Using Sandbox/Demo Mode")
        elif mode_choice == "Live":
            if not api_key or not secret_key:
                st.warning("Please provide your API key and secret for Live Mode below.")
                api_key = st.text_input("Enter Live API Key:")
                secret_key = st.text_input("Enter Live Secret Key:")
                if st.button("Save Live API Key and Secret"):
                    set_api_key_secret(api_key, secret_key, config_path, live_mode=True)
                    st.success("Live API Key and Secret saved successfully.")
            else:
                st.write("Using Live Mode")
                
                # Option to change API key and secret for Live Mode
                if st.checkbox("Change Live API Key and Secret"):
                    st.warning("You can change your Live API key and secret below.")
                    new_api_key = st.text_input("Enter New Live API Key:")
                    new_secret_key = st.text_input("Enter New Live Secret Key:")
                    if st.button("Update Live API Key and Secret"):
                        set_api_key_secret(new_api_key, new_secret_key, config_path, live_mode=True)
                        st.success("Live API Key and Secret updated successfully.")
    
if __name__ == "__main__":
    main()
