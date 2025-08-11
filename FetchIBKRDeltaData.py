import os
import pytz
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import sys
import signal
import psutil
import subprocess
from datetime import datetime, timezone
import warnings
import concurrent.futures
import logging
logging.disable(logging.CRITICAL)

warnings.filterwarnings("ignore")
os.chdir("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/")
import threading

# Global counter and lock for generating unique reqIds
# req_id_lock = threading.Lock()
# global_req_id = 1

# def get_unique_reqId():
#     global global_req_id
#     with req_id_lock:
#         req = global_req_id
#         global_req_id += 1
#     return req

# --- Holiday Check ---
# Define holiday dates (YYYY-MM-DD)
holidays = [
    "2025-01-01",  # New Year’s Day (Wednesday)
    "2025-01-20",  # Martin Luther King Jr. Day (Monday)
    "2025-02-17",  # Presidents’ Day (Monday)
    "2025-04-18",  # Good Friday (Friday)
    "2025-05-26",  # Memorial Day (Monday)
    "2025-06-19",  # Juneteenth National Independence Day (Thursday)
    "2025-07-04",  # Independence Day (Friday)
    "2025-09-01",  # Labor Day (Monday)
    "2025-11-27",  # Thanksgiving Day (Thursday)
    "2025-12-25"   # Christmas Day (Thursday)
]

# Get today's date in Europe/Berlin timezone and format as YYYY-MM-DD
today_str = datetime.now(pytz.timezone("Europe/Berlin")).strftime("%Y-%m-%d")
if today_str in holidays:
    print(f"Today ({today_str}) is a holiday. Exiting script.")
    sys.exit(0)

# Define the timezone and record the start time
berlin_tz = pytz.timezone("Europe/Berlin")
start_time = time.time()
# Redirect stdout to null
sys.stdout = open(os.devnull, 'w')

def kill_stuck_processes():
    """
    Kills all Python processes running any of the target scripts:
    FetchIBKRDeltaData.py, ImplementBuySellLogic.py, PlaceOrderIBKRNew1.py
    """
    current_pid = os.getpid()
    target_scripts = [
        "FetchIBKRDeltaData.py",
        "ImplementBuySellLogic.py",
        "PlaceIBKRPortfolioOrder.py"
    ]
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline_str = " ".join(proc.info['cmdline'])
                # If any target script is found in the command line, kill the process
                if any(target in cmdline_str for target in target_scripts) and proc.info['pid'] != current_pid:
                    print(f"Killing stuck process {proc.info['pid']} running one of {target_scripts}...")
                    os.kill(proc.info['pid'], signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

kill_stuck_processes()


end_time = time.time()
execution_time = end_time - start_time
print(f"Script finished at: {datetime.now()}")
print(f"Script executed in {execution_time:.2f} seconds.")
# Restore stdout when done
sys.stdout = sys.__stdout__

python_path = sys.executable

###### SCRIPT 1 #############################
print("Executing Alpaca Pre-MarketData...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/FetchAlpacaDataPreMarket.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("FetchAlpacaDataPreMarket.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ExitPosition.py: {e}", flush=True)
    exit(1)

############ Script 2 ###########################
print("Executing Alpaca Data...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/FetchAlpacaData.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("FetchAlpacaData.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ExitPosition.py: {e}", flush=True)
    exit(1)
############ Script 3 ############################
print("Getting Long List Data...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/GetLongListFiltered.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("GetLongListFiltered.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ExitPosition.py: {e}", flush=True)
    exit(1)

############## Script 4 ############################

print("Getting PortfolioTickersData...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/GetPortfolioTickersLongShort.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("GetPortfolioTickersLongShort.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ExitPosition.py: {e}", flush=True)
    exit(1)

############ Script 5 ################################
print("Executing Alpaca Data for Only LongList...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/FetchAlpacaDataForLong.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("FetchAlpacaDataForLong.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ExitPosition.py: {e}", flush=True)
    exit(1)

############ Script 6 ################################
print("Executing ImplementBuySellLogic.py...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ImplementBuySellLogic.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("ImplementBuySellLogic.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ImplementBuySellLogic.py: {e}", flush=True)
    exit(1)

########### Script 7 #################################

print("Executing PositionDetailsWithTime.py...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/PositionDetailsWithTime.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("Reclassification of Top10 List has been done and executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing ImplementBuySellLogic.py: {e}", flush=True)
    exit(1)

############# Script 8 #############################

print("Executing PlaceIBKRPortfolioOrder.py for Long...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/PlaceIBKRPortfolioOrder.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("PlaceIBKRPortfolioOrder.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing PlaceIBKRPortfolioOrder.py: {e}", flush=True)
    exit(1)
############# Script 9 #############################
print("Executing PlaceAlpacaPortfolioOrder.py for Short...")
try:
    subprocess.run([python_path, "-u", "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/PlaceAlpacaPortfolioOrder.py"], 
                   check=True,
                   stdout=sys.stdout,
                   stderr=sys.stderr)
    print("PlaceAlpacaPortfolioOrder.py executed successfully.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing PlaceAlpacaPortfolioOrder.py: {e}", flush=True)
    exit(1)

end_time_total = time.time()
execution_time_total = end_time_total - start_time
print(f"Script finished at: {datetime.now()}")
print(f"Script executed in {execution_time_total:.2f} seconds.")
