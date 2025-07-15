import pandas as pd 
import os
from datetime import datetime
import matplotlib.pyplot as plt



def preprocess_for_backtest(year=None):


    # if you want to run this on your machine you must replace the file paths below
    # gas loads day ahead gas prices
    gas = pd.read_csv("C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/Gas Prices/Gas_D1.csv")


    # each of the file paths loads intraday data for 2023-2025
    ID_2023 = pd.read_csv("C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/Tick Level/intraday_2023_clean.csv")
    ID_2024 = pd.read_csv("C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/Tick Level/intraday_2024_clean.csv")
    ID_2025 = pd.read_csv("C:/data/DATA/Continous Intraday/intraday_2025_clean.csv")


    # the format of execution should be like the following
    ID_2023['execution'] = pd.to_datetime(ID_2023['execution'], format='ISO8601')
    ID_2024['execution'] = pd.to_datetime(ID_2024['execution'], format='ISO8601')
    ID_2025['execution'] = pd.to_datetime(ID_2025['execution'], format='ISO8601')


    # same with Delivery Start
    ID_2023['DeliveryStart'] = pd.to_datetime(ID_2023['DeliveryStart'], format='ISO8601')
    ID_2024['DeliveryStart'] = pd.to_datetime(ID_2024['DeliveryStart'], format='ISO8601')
    ID_2025['DeliveryStart'] = pd.to_datetime(ID_2025['DeliveryStart'], format='ISO8601')


    # values are sorted by execution and then Delivery Start
    # the idea here is that a 15 minute product can be traded some hours before
    # in our backtesting strategy we look for all possible opportunities of buy and sell before the Delivery of the product arrives
    ID_2023_sorted = ID_2023.sort_values(by=['execution', 'DeliveryStart'])
    ID_2024_sorted = ID_2024.sort_values(by=['execution', 'DeliveryStart'])
    ID_2025_sorted = ID_2025.sort_values(by=['execution', 'DeliveryStart'])

    # this is a helper column to join with gas prices and later on calculate the buy threshold
    ID_2023_sorted['DeliveryDate'] = ID_2023_sorted['DeliveryStart'].dt.date
    ID_2024_sorted['DeliveryDate'] = ID_2024_sorted['DeliveryStart'].dt.date
    ID_2025_sorted['DeliveryDate'] = ID_2025_sorted['DeliveryStart'].dt.date

    ID_2023_sorted['DeliveryDate'] = pd.to_datetime(ID_2023_sorted['DeliveryDate'])
    ID_2024_sorted['DeliveryDate'] = pd.to_datetime(ID_2024_sorted['DeliveryDate'])
    ID_2025_sorted['DeliveryDate'] = pd.to_datetime(ID_2025_sorted['DeliveryDate'])

    # same here
    gas['Date'] = pd.to_datetime(gas['Date'])

    # we merge for all 3 datasets to later on calculate the buy threshold
    df_2023 = ID_2023_sorted.merge(gas, left_on='DeliveryDate', right_on='Date', how='left')
    df_2024 = ID_2024_sorted.merge(gas, left_on='DeliveryDate', right_on='Date', how='left')
    df_2025 = ID_2025_sorted.merge(gas, left_on='DeliveryDate', right_on='Date', how='left')

    # columns are dropped after
    df_2023 = df_2023.drop(columns=['DeliveryDate', 'Date'])
    df_2024 = df_2024.drop(columns=['DeliveryDate', 'Date'])
    df_2025 = df_2025.drop(columns=['DeliveryDate', 'Date'])


    # if you're running this particular script you got to make sure that you can this here and also in the front end.
    # here parameters calculate the buy threshold
    efficiency_parameter = 0.708
    certificates = 10.175

    df_2023['th'] = efficiency_parameter * (certificates + df_2023['Settlement'])
    df_2024['th'] = efficiency_parameter * (certificates + df_2024['Settlement'])
    df_2025['th'] = efficiency_parameter * (certificates + df_2025['Settlement'])

    # Return data based on specified year
    if year == 2023:
        return df_2023
    elif year == 2024:
        return df_2024
    elif year == 2025:
        return df_2025
    else:
        return df_2023, df_2024, df_2025



# this is the crocs 
def run_trading_strategy(delivery_groups, efficiency_parameter, certificates, log_to_file=True, buy_threshold_pct=0, sell_threshold_pct=10):
    #log
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"trading_log_{log_timestamp}.txt"
    log_file_path = os.path.join(os.getcwd(), log_filename)
    
    # Function to log messages only to file
    def log_message(message):
        if log_to_file:
            with open(log_file_path, 'a') as log_file:
                log_file.write(message + '\n')
    
    # Initialize trading variables
    entry = None
    total_pnl = 0
    electrolyser_profits = 0
    electrolyser_count = 0  # Counter for electrolyser operations. Devide it by 4 to get the total operating hours
    trade_count = 0
    hold_durations = []
    entry_time = None
    trades_data = []  # To store trade data for later analysis
    
    # New dictionary to track trades per delivery start
    trades_per_delivery = {}
    # Dictionary to track electrolyser operations per delivery start
    electrolyser_per_delivery = {}
    
    log_message(f"Trading log started at {datetime.now()}")
    log_message(f"Log file: {log_file_path}")
    log_message(f"Strategy parameters: Buy threshold: {buy_threshold_pct}% below threshold, Sell threshold: {sell_threshold_pct}% above entry")
    log_message("=" * 80)
    
    for delivery_start, group in delivery_groups:
        entry = None
        entry_time = None
        electrolyser_operate = 0
        group = group.sort_values('execution')
        
        # Initialize counts for this delivery start
        delivery_start_str = str(delivery_start)
        if delivery_start_str not in trades_per_delivery:
            trades_per_delivery[delivery_start_str] = 0
            electrolyser_per_delivery[delivery_start_str] = 0
            
        log_message(f"\n--- New Delivery Window: {delivery_start} ---")
        
        # Trading phase - before delivery starts
        for i in range(len(group)):
            row = group.iloc[i]
            
            # Only perform trading operations before delivery
            if row['execution'] < delivery_start:
                # Calculate buy price with custom threshold
                buy_price = row['th'] * (1 - buy_threshold_pct/100)
                
                # If no position yet and price is favorable (below adjusted threshold), enter position
                if entry is None and row['Price'] <= buy_price:
                    entry = row['Price']
                    entry_time = row['execution']
                    log_message(f"[{row['execution']}] ENTRY: Price={entry:.2f}, Threshold={row['th']:.2f}, Adjusted Buy Threshold={buy_price:.2f}")
                
                # If we have a position and can exit profitably based on custom sell threshold
                elif entry is not None and row['Price'] >= row['th'] * (1 + sell_threshold_pct/100):
                    trade_pnl = (row['Price'] - entry) / 4
                    total_pnl += trade_pnl
                    hold_duration = row['execution'] - entry_time
                    hold_durations.append(hold_duration)
                    price_change_pct = ((row['Price']/entry)-1)*100
                    
                    log_message(f"[{row['execution']}] EXIT: Price={row['Price']:.2f}, Profit={trade_pnl:.2f}")
                    log_message(f"      Hold duration: {hold_duration}, Price change: {price_change_pct:.2f}%")
                    
                    # Store trade data
                    trades_data.append({
                        'entry_time': entry_time,
                        'exit_time': row['execution'],
                        'entry_price': entry,
                        'exit_price': row['Price'],
                        'profit': trade_pnl,
                        'hold_duration': hold_duration,
                        'price_change_pct': price_change_pct,
                        'type': 'trade',
                        'delivery_start': delivery_start
                    })
                    
                    # Increment trade counts
                    trades_per_delivery[delivery_start_str] += 1
                    
                    entry = None
                    entry_time = None
                    trade_count += 1
            
        # After going through all data points, check if we still have an open position
        # If so, send to electrolyser (as we've reached delivery time)
        if entry is not None:
            # Calculate electrolyser profit
            el_profit = (row['th'] - entry) / 4
            hold_duration = delivery_start - entry_time
            
            log_message(f"[{delivery_start}] ELECTROLYSE: Entry price={entry:.2f}")
            log_message(f"      Hold duration until delivery: {hold_duration}, Profit={el_profit:.2f}")
            
            # Store electrolyser data
            trades_data.append({
                'entry_time': entry_time,
                'exit_time': delivery_start,
                'entry_price': entry,
                'exit_price': None,
                'profit': el_profit,
                'hold_duration': hold_duration,
                'price_change_pct': None,
                'type': 'electrolyser',
                'delivery_start': delivery_start
            })
            
            # Increment electrolyser counts
            electrolyser_count += 1
            electrolyser_per_delivery[delivery_start_str] += 1
            electrolyser_profits += el_profit
            entry = None
            entry_time = None
    
    avg_hold_time = pd.to_timedelta(pd.Series(hold_durations)).mean() if hold_durations else pd.Timedelta(0)
    
    # Create DataFrame for trades per delivery
    trades_per_delivery_df = pd.DataFrame({
        'delivery_start': list(trades_per_delivery.keys()),
        'num_trades': list(trades_per_delivery.values())
    })
    
    # Create DataFrame for electrolyser operations per delivery
    electrolyser_per_delivery_df = pd.DataFrame({
        'delivery_start': list(electrolyser_per_delivery.keys()),
        'num_electrolyser': list(electrolyser_per_delivery.values())
    })
    
    # Merge the two dataframes
    delivery_summary = trades_per_delivery_df.merge(
        electrolyser_per_delivery_df, 
        on='delivery_start', 
        how='outer'
    ).fillna(0)
    
    # Sort by total operations (trades + electrolyser)
    delivery_summary['total_operations'] = delivery_summary['num_trades'] + delivery_summary['num_electrolyser']
    delivery_summary = delivery_summary.sort_values('total_operations', ascending=False)
    
    # Log summary of trades and electrolyser operations per delivery
    log_message("\n=== Activity per Delivery Window ===")
    for _, row in delivery_summary.iterrows():
        log_message(f"Delivery Start: {row['delivery_start']}, Trades: {int(row['num_trades'])}, Electrolyser: {int(row['num_electrolyser'])}")
    
    # Add summary statistics
    log_message(f"\nTotal delivery windows: {len(delivery_summary)}")
    log_message(f"Total trades executed: {trade_count}")
    log_message(f"Total electrolyser operations: {electrolyser_count}")
    log_message(f"Average trades per delivery window: {trade_count / len(delivery_summary) if len(delivery_summary) > 0 else 0:.2f}")
    log_message(f"Average electrolyser operations per delivery window: {electrolyser_count / len(delivery_summary) if len(delivery_summary) > 0 else 0:.2f}")
    
    summary = f"\n=== Trading Summary ===\n"
    summary += f"Total trading PnL: {total_pnl:.2f}\n"
    summary += f"Total electrolyser profits: {electrolyser_profits:.2f}\n"
    summary += f"Combined total: {total_pnl + electrolyser_profits:.2f}\n"
    summary += f"Total trades executed: {trade_count}\n"
    summary += f"Total electrolyser operations: {electrolyser_count}\n"
    summary += f"Average holding time for trades: {avg_hold_time}"
    log_message(summary)
    
    trades_df = pd.DataFrame(trades_data)
    if len(trades_df) > 0 and log_to_file:
        trades_csv_filename = f"trades_data_{log_timestamp}.csv"
        trades_df.to_csv(trades_csv_filename, index=False)
        log_message(f"\nTrade data exported to {trades_csv_filename}")
        
        # Export delivery activity summary data
        delivery_summary_csv = f"delivery_activity_{log_timestamp}.csv"
        delivery_summary.to_csv(delivery_summary_csv, index=False)
        log_message(f"Delivery activity data exported to {delivery_summary_csv}")
    
    results = {
        'total_pnl': total_pnl,
        'electrolyser_profits': electrolyser_profits, 
        'electrolyser_count': electrolyser_count,
        'combined_profit': total_pnl + electrolyser_profits,
        'trade_count': trade_count,
        'avg_hold_time': avg_hold_time,
        'buy_threshold_pct': buy_threshold_pct,
        'sell_threshold_pct': sell_threshold_pct,
        'timestamp': log_timestamp,
        'delivery_summary': delivery_summary
    }
    
    return results, trades_df, log_file_path if log_to_file else None


def monthly_performance_summary(trades_df, save_plot=False, output_dir=None):
    # Ensure entry_time is datetime
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    
    # Extract month and year
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    
    # Group by month and type, summing profits
    monthly_summary = trades_df.groupby(['month', 'type'])['profit'].sum().unstack(fill_value=0)
    
    # Ensure both 'trade' and 'electrolyser' columns exist
    if 'trade' not in monthly_summary.columns:
        monthly_summary['trade'] = 0
    if 'electrolyser' not in monthly_summary.columns:
        monthly_summary['electrolyser'] = 0
    
    # Calculate total profit
    monthly_summary['total'] = monthly_summary['trade'] + monthly_summary['electrolyser']
    
    # Format month names
    monthly_summary = monthly_summary.reset_index()
    monthly_summary['month_str'] = monthly_summary['month'].dt.strftime('%b %Y')
    
    # Add electrolyser count per month (count of rows where type is 'electrolyser')
    electrolyser_counts = trades_df[trades_df['type'] == 'electrolyser'].groupby(
        trades_df['entry_time'].dt.to_period('M')
    ).size()
    
    # Add the counts to monthly_summary
    monthly_summary['electrolyser_count'] = monthly_summary['month'].map(electrolyser_counts).fillna(0).astype(int)
    
    return monthly_summary