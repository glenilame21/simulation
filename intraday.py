import pandas as pd
import numpy as np

def intraday_strategy(
    df,
    price_column,
    settlement_column,
    efficiency_parameter,
    certificates,
    time_interval,
    power_capacity,
    manual_threshold=None,
    price_jump_percentage=10
):
    """
    Intraday trading strategy that buys when price is below threshold and either:
    1. Sells when price increases by specified percentage
    2. Uses the electricity in electrolyser if no selling opportunity arises
    """
    # Create a copy of the input dataframe
    result_df = df.copy()
    
    # Calculate the time interval factor (for hourly rate adjustment)
    hour_factor = time_interval / 60
    
    # Calculate buy threshold if not provided
    if manual_threshold is not None:
        result_df['buy_threshold'] = manual_threshold
    elif settlement_column is not None:
        # Calculate threshold from gas price: gas_price / efficiency
        result_df['buy_threshold'] = (result_df[settlement_column] / efficiency_parameter) - certificates
    else:
        raise ValueError("Either manual_threshold or settlement_column must be provided")
    
    # Prepare columns for results
    result_df['operation'] = 0  # Initialize to not operating
    result_df['profit'] = 0.0
    result_df['trade_type'] = ''
    
    # Calculate price jump threshold factor
    jump_factor = 1 + (price_jump_percentage / 100)
    
    # Group by delivery start date
    if 'DeliveryStart' in result_df.columns:
        # Ensure DeliveryStart is datetime
        result_df['DeliveryStart'] = pd.to_datetime(result_df['DeliveryStart'])
        delivery_groups = result_df.groupby('DeliveryStart')
    else:
        # Create a dummy group with all data
        result_df['dummy_group'] = 1
        delivery_groups = [(None, result_df)]
    
    total_trading_pnl = 0
    total_electrolyser_profits = 0
    
    for group_key, group in delivery_groups:
        # Ensure we have execution time information or create it
        if 'execution' not in group.columns:
            if isinstance(group_key, pd.Timestamp):
                # Create execution times based on time_interval
                group['execution'] = group_key - pd.Timedelta(minutes=time_interval)
        
        entry_price = None
        entry_index = None
        
        # Sort by execution time if available, otherwise use dataframe order
        if 'execution' in group.columns:
            group = group.sort_values('execution')
        
        # Process each time period
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i+1]
            
            current_price = current_row[price_column]
            next_price = next_row[price_column]
            threshold = current_row['buy_threshold']
            
            current_idx = current_row.name
            next_idx = next_row.name
            
            # Check if we should enter a position
            if entry_price is None and current_price <= threshold * 0.9:
                entry_price = current_price
                entry_index = current_idx
                result_df.at[current_idx, 'operation'] = 1  # Set to operating
                result_df.at[current_idx, 'trade_type'] = 'BUY'
            
            # If we have an entry position, check if we can exit
            elif entry_price is not None:
                if next_price > entry_price * jump_factor:
                    # Sell with profit
                    trade_pnl = (next_price - entry_price) * power_capacity * hour_factor
                    result_df.at[next_idx, 'operation'] = 1  # Set to operating
                    result_df.at[next_idx, 'profit'] = trade_pnl
                    result_df.at[next_idx, 'trade_type'] = 'SELL'
                    total_trading_pnl += trade_pnl
                    
                    # Reset entry position
                    entry_price = None
                    entry_index = None
    
    # Process any remaining open positions at the end - send to electrolyser
    for group_key, group in delivery_groups:
        for i, row in group.iterrows():
            if row['operation'] == 1 and row['trade_type'] == 'BUY' and row['profit'] == 0:  # Changed from 'operation' to 'operate'
                # Calculate electrolyser profit for this position
                entry_price = row[price_column]
                el_profit = (efficiency_parameter * (certificates + entry_price) - entry_price) * power_capacity * hour_factor
                
                result_df.at[i, 'profit'] = el_profit
                result_df.at[i, 'trade_type'] = 'ELECTROLYSER'
                total_electrolyser_profits += el_profit
    
    # Calculate total metrics
    total_profit = total_trading_pnl + total_electrolyser_profits
    operating_hours = result_df['operation'].sum() * (time_interval / 60)  # Changed from 'operation' to 'operate'
    
    # Add summary columns
    result_df['trading_pnl'] = 0
    result_df['electrolyser_profit'] = 0
    
    # Fill in the specific profit columns
    result_df.loc[result_df['trade_type'] == 'SELL', 'trading_pnl'] = result_df.loc[result_df['trade_type'] == 'SELL', 'profit']
    result_df.loc[result_df['trade_type'] == 'ELECTROLYSER', 'electrolyser_profit'] = result_df.loc[result_df['trade_type'] == 'ELECTROLYSER', 'profit']

    result_df['operate'] = result_df['operate']
    
    return result_df, operating_hours, total_profit, total_trading_pnl, total_electrolyser_profits