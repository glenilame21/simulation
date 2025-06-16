import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import calendar


st.set_page_config(
    page_title="Electrolyzer Trading Dashboard",
    layout="wide"
)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'total_profit' not in st.session_state:
    st.session_state.total_profit = 125680

def generate_price_data(hours=24, current_time=None):
    """Generate dummy electricity price data"""
    if current_time is None:
        current_time = datetime.now()
    
    data = []
    for i in range(hours):
        time_point = current_time - timedelta(hours=hours-i-1)
        # Create realistic price pattern with some volatility
        base_price = 52 + np.sin(i * 0.3) * 8 + np.random.normal(0, 3)
        price = max(20, base_price)  # Ensure price doesn't go negative
        
        data.append({
            'timestamp': time_point,
            'price': round(price, 2),
            'hour': time_point.hour
        })
    
    return pd.DataFrame(data)

def calculate_strategy_decision(price, buy_threshold, sell_threshold):
    """Determine what action to take based on current price"""
    if price < buy_threshold:
        return "ELECTROLYZER", "Convert to Energy Storage (0.7 efficiency)"
    elif price <= sell_threshold:
        return "HOLD", "Hold Position"
    else:
        return "SELL", "Sell Power"

def simulate_trading_performance(df, buy_threshold, sell_threshold, electrolyzer_capacity):
    """Simulate position-based trading performance with simplified electrolyzer logic"""
    df_copy = df.copy()
    
    # Initialize state variables
    position_open = False
    buy_price = 0
    actions = []
    profits = []
    energy_converted = []
    
    for i, row in df_copy.iterrows():
        current_price = row['price']
        
        if not position_open:
            # WAIT STATE - No position open
            if current_price <= buy_threshold:
                # Price <= Buy Threshold → Enter BUY STATE
                position_open = True
                buy_price = current_price
                action = "BUY"
                profit = 0
                energy_mwh = 0
            else:
                # Price > Buy Threshold → Stay in WAIT STATE
                action = "WAIT"
                profit = 0
                energy_mwh = 0
        else:
            # Position is open - check exit conditions
            if current_price >= sell_threshold:
                # Price >= Sell Threshold → SELL STATE
                profit = (current_price - buy_price) * electrolyzer_capacity
                position_open = False
                buy_price = 0
                action = "SELL"
                energy_mwh = 0
            elif current_price < buy_price:
                # Price < Entry Price → ELECTROLYZER STATE (forced exit)
                # Convert electricity to energy storage with 0.7 efficiency
                energy_input = electrolyzer_capacity  # MWh input
                energy_output = energy_input * 0.7    # MWh output (70% efficiency)
                
                # Calculate the effective profit/loss
                # We lose 30% of the energy, but we paid buy_price and avoid current lower price
                electricity_cost = buy_price * electrolyzer_capacity
                energy_value = energy_output * current_price  # Value of stored energy at current price
                
                # The "profit" is avoiding the loss vs selling at current price
                # Plus we have 0.7 MWh of stored energy worth current_price per MWh
                profit = max(0, energy_value - electricity_cost)
                
                position_open = False
                buy_price = 0
                action = "ELECTROLYZER"
                energy_mwh = energy_output
            else:
                # Entry Price <= Price < Sell Threshold → HOLD/BUY STATE (continue holding)
                action = "HOLD"
                profit = 0
                energy_mwh = 0
        
        actions.append(action)
        profits.append(max(0, profit))
        energy_converted.append(energy_mwh)
    
    df_copy['action'] = actions
    df_copy['profit'] = profits
    df_copy['energy_stored_mwh'] = energy_converted
    df_copy['position_open'] = [action in ['BUY', 'HOLD'] for action in actions]
    df_copy['buy_price'] = [buy_price if pos_open else 0 for pos_open in df_copy['position_open']]
    
    return df_copy

def load_data_from_file(uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        # Try to read the CSV file with different delimiters
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
        except:
            # If semicolon doesn't work, try comma delimiter
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file)
        
        # Check if we have the required columns
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            # Try to auto-detect columns - assume first column is timestamp, second is price
            if len(df.columns) >= 2:
                df.columns = ['timestamp', 'price'] + list(df.columns[2:])
                st.sidebar.info("Auto-detected columns: first as 'timestamp', second as 'price'")
            else:
                st.sidebar.error("CSV must contain at least 2 columns (timestamp and price)")
                return None
        
        # Convert timestamp column to datetime with multiple format attempts
        timestamp_formats = [
            # Try common date formats
            None,  # Let pandas infer the format
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%d-%m-%Y %H:%M',
            '%Y%m%d%H%M',
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%m/%d/%Y'
        ]
        
        success = False
        for date_format in timestamp_formats:
            try:
                if date_format is None:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=date_format)
                success = True
                st.sidebar.success(f"Successfully parsed timestamps")
                break
            except:
                continue
        
        if not success:
            # Show sample of problematic data to help user
            sample_timestamps = df['timestamp'].head(3).tolist()
            st.sidebar.error(f"Could not parse timestamp column. Examples of your data: {sample_timestamps}")
            st.sidebar.info("Try reformatting your dates to YYYY-MM-DD HH:MM:SS format")
            return None
        
        # Ensure price is numeric
        try:
            # Replace comma with dot for numeric values (European format)
            if df['price'].dtype == object:  # If price is string
                df['price'] = df['price'].str.replace(',', '.').astype(float)
            else:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Drop rows with invalid prices
            invalid_count = df['price'].isna().sum()
            if invalid_count > 0:
                st.sidebar.warning(f"Removed {invalid_count} rows with invalid price values")
                df = df.dropna(subset=['price'])
        except:
            st.sidebar.error("Could not parse price column. Please ensure it contains numeric values.")
            return None
        
        # Add hour column for analysis
        df['hour'] = df['timestamp'].dt.hour
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.sidebar.info("Please check your file format and try again")
        return None
    

    
# Main Dashboard
st.title("Electrolyzer Trading Strategy Dashboard")
st.markdown("Real-time simulation of power trading with energy storage optimization (70% efficiency)")

# Sidebar for controls
st.sidebar.header("Strategy Parameters")

# Trading thresholds
buy_threshold = st.sidebar.number_input("Buy Threshold (€/MWh)", min_value=0.0, max_value=2000.0, value=50.0, step=1.0)

# Sell threshold options
sell_threshold_mode = st.sidebar.radio(
    "Sell Threshold Mode",
    ["Absolute Value", "Percentage Above Buy Threshold"],
    help="Choose how to set the sell threshold"
)

if sell_threshold_mode == "Absolute Value":
    sell_threshold = st.sidebar.number_input(
        "Sell Threshold (€/MWh)", 
        min_value=0.0, 
        max_value=2000.0, 
        value=55.0, 
        step=1.0
    )
else:
    sell_percentage = st.sidebar.number_input(
        "Sell Threshold (% above Buy Threshold)", 
        min_value=0.1, 
        max_value=100.0, 
        value=10.0, 
        step=0.1,
        help="Percentage above buy threshold (e.g., 10% means sell at 110% of buy price)"
    )
    sell_threshold = buy_threshold * (1 + sell_percentage / 100)
    st.sidebar.info(f"Calculated Sell Threshold: €{sell_threshold:.2f}/MWh")

# Electrolyzer parameters
electrolyzer_capacity = st.sidebar.number_input("Power Capacity (MW)", min_value=1, max_value=168, value=10, step=1)

# Validation
if sell_threshold <= buy_threshold:
    st.sidebar.warning("Sell threshold should be higher than buy threshold!")

# Add efficiency info
st.sidebar.info("⚡ Energy Conversion: 1 MWh input → 0.7 MWh stored energy")

# Data Upload Section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload electricity price data (CSV)", 
    type=['csv'],
    help="CSV should contain 'timestamp' and 'price' columns. If columns have different names, the first column will be treated as timestamp and second as price."
)

# Sample CSV format info
with st.sidebar.expander("Expected CSV Format"):
    st.text("""
Required columns:
- timestamp: Date/time (various formats supported)
- price: Electricity price (numeric)

Example:
timestamp,price
2024-01-01 00:00:00,45.2
2024-01-01 01:00:00,42.8
2024-01-01 02:00:00,38.5
...
    """)

# Load data
df = None
if uploaded_file is not None:
    df = load_data_from_file(uploaded_file)
    if df is not None:
        st.sidebar.success(f"Loaded {len(df):,} data points")
        st.sidebar.info(f"Period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # Backtesting period selection
        st.sidebar.header("Backtesting Period")
        
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on selected period
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            df_filtered = df[mask].copy()
            
            if len(df_filtered) == 0:
                st.sidebar.warning(f"No data available for selected period")
                df_filtered = df
        else:
            df_filtered = df
    else:
        st.sidebar.error("Failed to load data. Using sample data instead.")
        df_filtered = generate_price_data(48)
else:
    # Use sample data when no file is uploaded
    df_filtered = generate_price_data(48)
    st.sidebar.info("Using sample data. Upload a CSV file to backtest with your own data.")

# Simulation controls
st.sidebar.header("Simulation Controls")
if st.sidebar.button("Start/Stop Real-time Simulation"):
    st.session_state.simulation_running = not st.session_state.simulation_running

if st.sidebar.button("Reset Metrics"):
    st.session_state.total_profit = 0

# Process data with strategy
df_with_strategy = simulate_trading_performance(df_filtered, buy_threshold, sell_threshold, electrolyzer_capacity)

# Current price and status (use last data point)
current_price = df_with_strategy.iloc[-1]['price']
current_action, action_description = calculate_strategy_decision(current_price, buy_threshold, sell_threshold)

# Main metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    if len(df_with_strategy) > 1:
        price_delta = current_price - df_with_strategy.iloc[-2]['price']
    else:
        price_delta = 0
    
    st.metric(
        "Current Price",
        f"€{current_price:.2f}/MWh",
        delta=f"{price_delta:.2f}",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Current Strategy",
        current_action,
        help=action_description
    )

with col3:
    total_profit = df_with_strategy['profit'].sum()
    st.metric(
        "Total Profit",
        f"€{total_profit:,.0f}",
        delta=f"{total_profit * 0.05:.0f}"
    )

with col4:
    total_energy_stored = df_with_strategy['energy_stored_mwh'].sum()
    st.metric(
        "Energy Stored",
        f"{total_energy_stored:.1f} MWh",
        delta=f"{total_energy_stored * 0.1:.1f}"
    )

# Charts section
st.subheader("Position-Based Trading Strategy")

fig = go.Figure()

# Separate data by strategy
buy_periods = df_with_strategy[df_with_strategy['action'] == 'BUY']
sell_periods = df_with_strategy[df_with_strategy['action'] == 'SELL']
hold_periods = df_with_strategy[df_with_strategy['action'] == 'HOLD']
electrolyzer_periods = df_with_strategy[df_with_strategy['action'] == 'ELECTROLYZER']
wait_periods = df_with_strategy[df_with_strategy['action'] == 'WAIT']

# Main price line
fig.add_trace(go.Scatter(
    x=df_with_strategy['timestamp'],
    y=df_with_strategy['price'],
    mode='lines',
    name='Electricity Price',
    line=dict(color='darkblue', width=2),
    hovertemplate='<b>Price</b>: €%{y:.2f}/MWh<br><b>Time</b>: %{x}<extra></extra>'
))

# Add position background shading
position_open = False
position_start = None

for i, row in df_with_strategy.iterrows():
    if row['action'] == 'BUY' and not position_open:
        position_open = True
        position_start = row['timestamp']
    elif row['action'] in ['SELL', 'ELECTROLYZER'] and position_open:
        position_open = False
        if position_start is not None:
            # Shade the position period
            fig.add_vrect(
                x0=position_start, 
                x1=row['timestamp'],
                fillcolor='lightblue', 
                opacity=0.3,
                layer="below", 
                line_width=0,
                annotation_text="Position Open",
                annotation_position="top left"
            )

# Add strategy markers with clear labels
if not buy_periods.empty:
    fig.add_trace(go.Scatter(
        x=buy_periods['timestamp'],
        y=buy_periods['price'],
        mode='markers',
        name='BUY Signal (Open Position)',
        marker=dict(color='blue', size=12, symbol='circle'),
        hovertemplate='<b>BUY SIGNAL</b><br>Price: €%{y:.2f}/MWh<br>Time: %{x}<br>Action: Open Position<extra></extra>'
    ))

if not sell_periods.empty:
    fig.add_trace(go.Scatter(
        x=sell_periods['timestamp'],
        y=sell_periods['price'],
        mode='markers',
        name='SELL Signal (Close Position)',
        marker=dict(color='red', size=12, symbol='triangle-up'),
        hovertemplate='<b>SELL SIGNAL</b><br>Price: €%{y:.2f}/MWh<br>Time: %{x}<br>Action: Close Position (Profit)<extra></extra>'
    ))

if not electrolyzer_periods.empty:
    fig.add_trace(go.Scatter(
        x=electrolyzer_periods['timestamp'],
        y=electrolyzer_periods['price'],
        mode='markers',
        name='ENERGY STORAGE (Close Position)',
        marker=dict(color='green', size=12, symbol='diamond'),
        hovertemplate='<b>ENERGY STORAGE</b><br>Price: €%{y:.2f}/MWh<br>Time: %{x}<br>Action: Convert to Stored Energy (0.7 efficiency)<extra></extra>'
    ))

if not hold_periods.empty:
    fig.add_trace(go.Scatter(
        x=hold_periods['timestamp'],
        y=hold_periods['price'],
        mode='markers',
        name='HOLD Position',
        marker=dict(color='orange', size=8, symbol='square'),
        hovertemplate='<b>HOLDING POSITION</b><br>Price: €%{y:.2f}/MWh<br>Time: %{x}<br>Action: Wait for Exit Signal<extra></extra>'
    ))

if not wait_periods.empty:
    fig.add_trace(go.Scatter(
        x=wait_periods['timestamp'],
        y=wait_periods['price'],
        mode='markers',
        name='WAIT (No Position)',
        marker=dict(color='gray', size=6, symbol='x'),
        hovertemplate='<b>WAITING</b><br>Price: €%{y:.2f}/MWh<br>Time: %{x}<br>Action: Wait for Buy Signal<extra></extra>'
    ))

# Threshold lines
fig.add_hline(
    y=buy_threshold, 
    line_dash="dash", 
    line_color="blue", 
    line_width=3,
    annotation_text=f"BUY Threshold (€{buy_threshold:.1f})",
    annotation_position="top right"
)
fig.add_hline(
    y=sell_threshold, 
    line_dash="dash", 
    line_color="red", 
    line_width=3,
    annotation_text=f"SELL Threshold (€{sell_threshold:.1f})",
    annotation_position="bottom right"
)

fig.update_layout(
    title="Position-Based Trading Strategy with Energy Storage",
    xaxis_title="Time",
    yaxis_title="Price (€/MWh)",
    height=500,
    hovermode='closest',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Strategy Statistics
st.subheader("Strategy Statistics")

if len(df_with_strategy) > 0:
    stats_data = df_with_strategy.groupby('action').agg({
        'price': ['mean', 'min', 'max', 'count'],
        'profit': ['sum', 'mean'],
        'energy_stored_mwh': 'sum'
    }).round(2)
    
    # Flatten column names
    stats_data.columns = ['_'.join(col).strip() for col in stats_data.columns]
    stats_data = stats_data.reset_index()
    
    # Rename columns for display
    display_columns = {
        'action': 'Strategy',
        'price_count': 'Hours',
        'price_mean': 'Avg Price (€/MWh)',
        'price_min': 'Min Price (€/MWh)',
        'price_max': 'Max Price (€/MWh)',
        'profit_sum': 'Total Profit (€)',
        'profit_mean': 'Avg Hourly Profit (€)',
        'energy_stored_mwh_sum': 'Energy Stored (MWh)'
    }
    
    stats_data = stats_data.rename(columns=display_columns)
    st.dataframe(stats_data, use_container_width=True)

# Key Insights
st.subheader("Insights")

col1, col2, col3 = st.columns(3)

with col1:
    storage_hours = len(df_with_strategy[df_with_strategy['action'] == 'ELECTROLYZER'])
    total_hours = len(df_with_strategy)
    utilization = (storage_hours / total_hours) * 100 if total_hours > 0 else 0
    st.info(f"**Energy Storage Utilization**: {utilization:.1f}% ({storage_hours}/{total_hours} hours)")

with col2:
    avg_profit_per_hour = df_with_strategy['profit'].mean() if len(df_with_strategy) > 0 else 0
    st.success(f"**Average Hourly Profit**: €{avg_profit_per_hour:.2f}")

with col3:
    efficiency_loss = df_with_strategy['energy_stored_mwh'].sum() * 0.3  # 30% loss
    st.warning(f"**Energy Loss (30%)**: {efficiency_loss:.1f} MWh")

# Data Export
if uploaded_file is not None:
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df_with_strategy.to_csv(index=False)
        st.download_button(
            label="Download Detailed Results (CSV)",
            data=csv_data,
            file_name=f"energy_storage_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_data = {
            'Metric': [
                'Total Profit (€)',
                'Energy Stored (MWh)',
                'Energy Storage Utilization (%)',
                'Average Hourly Profit (€)',
                'Buy Threshold (€/MWh)',
                'Sell Threshold (€/MWh)',
                'Power Capacity (MW)',
                'Energy Conversion Efficiency (%)'
            ],
            'Value': [
                f"{total_profit:.2f}",
                f"{total_energy_stored:.2f}",
                f"{utilization:.1f}",
                f"{avg_profit_per_hour:.2f}",
                f"{buy_threshold:.2f}",
                f"{sell_threshold:.2f}",
                f"{electrolyzer_capacity}",
                "70.0"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="Download Summary Report (CSV)",
            data=summary_csv,
            file_name=f"energy_storage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )