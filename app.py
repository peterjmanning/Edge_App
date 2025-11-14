from flask import Flask, render_template, jsonify, send_file,  request
import pandas as pd
from pathlib import Path
import time
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re


app = Flask(__name__)

EDGE_FILE = Path("/mnt/c/Users/pmanning/Desktop/LogEdges.xlsx")


def load_data(sheet=None):
    # Open and close ExcelFile properly to ensure fresh reads
    with pd.ExcelFile(EDGE_FILE, engine='openpyxl') as xls:
        sheets = xls.sheet_names
        latest = sorted(sheets)[-1]
        sheet = sheet or latest
    
    # Read the data fresh each time (don't use the cached ExcelFile)
    df = pd.read_excel(EDGE_FILE, sheet_name=sheet, header=None, skiprows=1, engine='openpyxl')

    if df.empty:
        return pd.DataFrame(columns=["Time","Ticker","Bid Edge","Ask Edge"]), sheet

    df.columns = ["Time","Ticker","Bid Edge","Ask Edge"]

    # Convert types
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Bid Edge"] = pd.to_numeric(df["Bid Edge"], errors="coerce")
    df["Ask Edge"] = pd.to_numeric(df["Ask Edge"], errors="coerce")

    # Drop rows where Time or Ticker is missing
    df = df.dropna(subset=["Time","Ticker"])

    # Compute average edge
    df["AvgEdge"] = df[["Bid Edge","Ask Edge"]].mean(axis=1)

    return df, sheet

@app.route("/")
def index():
    """Home route with dropdown of all sheets."""
    if not EDGE_FILE.exists():
        return "EdgeLog.xlsx not found", 404
    
    # Open and close ExcelFile properly to ensure fresh reads
    with pd.ExcelFile(EDGE_FILE, engine='openpyxl') as xls:
        sheets = sorted(xls.sheet_names)
    
    # Find EdgeLog_YYYY_MM_DD sheets and select the most recent one
    edge_log_sheets = []
    for sheet in sheets:
        match = re.match(r'EdgeLog_(\d{4})_(\d{2})_(\d{2})', sheet)
        if match:
            year, month, day = match.groups()
            edge_log_sheets.append((sheet, (int(year), int(month), int(day))))
    
    if edge_log_sheets:
        # Sort by date (year, month, day) and get the most recent
        edge_log_sheets.sort(key=lambda x: x[1])
        latest = edge_log_sheets[-1][0]
        print(f"Selected most recent EdgeLog sheet: {latest}")
    else:
        # Fall back to last sheet alphabetically if no EdgeLog sheets found
        latest = sheets[-1]
        print(f"No EdgeLog sheets found, using: {latest}")
    
    return render_template("index.html", sheets=sheets, current=latest)

@app.route("/data")
def get_data():
    sheet = request.args.get("sheet")
    minutes_str = request.args.get("minutes", "20")  # Default to 20 minutes
    show_separate = request.args.get("show_separate", "false").lower() == "true"
    filter_outliers = request.args.get("filter_outliers", "false").lower() == "true"
    
    try:
        minutes = int(minutes_str)
    except (ValueError, TypeError):
        minutes = 20  # Default to 20 if invalid
    
    print(f"Received minutes parameter: '{minutes_str}' -> {minutes}")
    
    df, sheet_name = load_data(sheet)

    if df.empty:
        return jsonify({"times": [], "values": {}, "bid_values": {}, "ask_values": {}})

    # Filter by time range if minutes is specified (0 means all data - no filtering)
    if minutes > 0:
        if not df["Time"].empty and len(df) > 0:
            latest_time = df["Time"].max()
            cutoff_time = latest_time - timedelta(minutes=minutes)
            df = df[df["Time"] >= cutoff_time]
            print(f"Filtered to last {minutes} minutes: {len(df)} rows")
    else:
        print(f"Showing all data (no time filter): {len(df)} rows")

    if df.empty:
        return jsonify({"times": [], "values": {}, "bid_values": {}, "ask_values": {}})

    # Convert NaN to None (null in JSON) for Chart.js compatibility
    def nan_to_none(val):
        return None if pd.isna(val) else float(val)
    
    # Filter outliers function - removes values more than 5 std devs from mean
    def filter_outliers_in_list(values, std_devs=5):
        """Filter outliers from a list of values, setting them to None."""
        # Filter out None values for calculation
        valid_values = [v for v in values if v is not None and not pd.isna(v)]
        if len(valid_values) < 2:
            return values  # Not enough data to calculate std dev
        
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        if std_val == 0:
            return values  # No variation, nothing to filter
        
        threshold = std_devs * std_val
        lower_bound = mean_val - threshold
        upper_bound = mean_val + threshold
        
        # Create new list with outliers set to None
        filtered = []
        for v in values:
            if v is None or pd.isna(v):
                filtered.append(None)
            elif v < lower_bound or v > upper_bound:
                filtered.append(None)  # Outlier - set to None
            else:
                filtered.append(v)
        
        return filtered
    
    if show_separate:
        # Read Bid Edge and Ask Edge directly from the dataframe columns
        # Group by Time and Ticker, taking the mean if there are multiple entries for same time/ticker
        bid_per_ticker = df.groupby(["Time", "Ticker"])["Bid Edge"].mean().unstack()
        ask_per_ticker = df.groupby(["Time", "Ticker"])["Ask Edge"].mean().unstack()
        
        # Get all unique times from the dataframe (not just from the grouped data)
        all_times = sorted(df["Time"].unique())
        times = [t.strftime("%Y-%m-%d %H:%M:%S") for t in all_times]
        
        # Build bid and ask values dictionaries
        bid_values = {}
        ask_values = {}
        
        # Get all unique tickers from the dataframe
        all_tickers = sorted(df["Ticker"].unique())
        
        if len(all_times) == 0:
            print("Warning: No times found in dataframe for bid/ask separate mode")
            return jsonify({"times": [], "values": {}, "bid_values": {}, "ask_values": {}})
        
        if len(all_tickers) == 0:
            print("Warning: No tickers found in dataframe for bid/ask separate mode")
            return jsonify({"times": [], "values": {}, "bid_values": {}, "ask_values": {}})
        
        for ticker in all_tickers:
            bid_list = []
            ask_list = []
            
            for t in all_times:
                # Get bid value if exists in the grouped data
                bid_val = None
                try:
                    if ticker in bid_per_ticker.columns and t in bid_per_ticker.index:
                        val = bid_per_ticker.loc[t, ticker]
                        if pd.notna(val):
                            bid_val = nan_to_none(val)
                except (KeyError, IndexError):
                    pass
                
                # Get ask value if exists in the grouped data
                ask_val = None
                try:
                    if ticker in ask_per_ticker.columns and t in ask_per_ticker.index:
                        val = ask_per_ticker.loc[t, ticker]
                        if pd.notna(val):
                            ask_val = nan_to_none(val)
                except (KeyError, IndexError):
                    pass
                
                bid_list.append(bid_val)
                ask_list.append(ask_val)
            
            bid_values[ticker] = bid_list
            ask_values[ticker] = ask_list
        
        # Filter outliers if requested
        if filter_outliers:
            for ticker in all_tickers:
                if ticker in bid_values:
                    bid_values[ticker] = filter_outliers_in_list(bid_values[ticker])
                if ticker in ask_values:
                    ask_values[ticker] = filter_outliers_in_list(ask_values[ticker])
            print(f"Applied outlier filtering to bid/ask data")
        
        print(f"Bid/Ask separate mode: {len(all_tickers)} tickers, {len(times)} time points")
        print(f"  Bid values keys: {list(bid_values.keys())}")
        print(f"  Ask values keys: {list(ask_values.keys())}")
        
        return jsonify({
            "times": times,
            "values": {},  # Empty when showing separate
            "bid_values": bid_values,
            "ask_values": ask_values
        })
    else:
        # Original behavior: show average
        avg_per_ticker = df.groupby(["Time","Ticker"])["AvgEdge"].mean().unstack()
        times = [t.strftime("%Y-%m-%d %H:%M:%S") for t in avg_per_ticker.index]
        values = {
            ticker: [nan_to_none(v) for v in avg_per_ticker[ticker].tolist()] 
            for ticker in avg_per_ticker.columns
        }
        
        # Filter outliers if requested
        if filter_outliers:
            for ticker in values:
                values[ticker] = filter_outliers_in_list(values[ticker])
            print(f"Applied outlier filtering to average data")
        
        return jsonify({
            "times": times,
            "values": values,
            "bid_values": {},
            "ask_values": {}
        })
    
@app.route("/plot")
def plot_image():
    sheet = request.args.get("sheet") or None
    df, sheet_name = load_data(sheet)

    if df.empty:
        return "No data for this sheet", 404

    # Compute avg per ticker
    avg_per_ticker = df.groupby(["Time", "Ticker"])["AvgEdge"].mean().unstack()

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    for ticker in avg_per_ticker.columns:
        ax.plot(avg_per_ticker.index, avg_per_ticker[ticker], label=ticker, linewidth=1.5)
    ax.set_title("Bid/Ask Edge Average by Ticker Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Edge")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    # Run with Flask dev server
    app.run(host="0.0.0.0", port=5001, debug=False)
