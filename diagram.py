import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
import os

# Global variable to store gas prices
gas_prices_df = None

def load_gas_prices(gas_file_option="Month Ahead"):
    global gas_prices_df
    try:
        if gas_file_option == "Month Ahead":
            gas_path = "C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/gas_streamlit.csv"
        else:  # Spot
            gas_path = "C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/gas_streamlit1.csv"
        
        if os.path.exists(gas_path):
            gas_prices_df = pd.read_csv(gas_path)
            gas_prices_df = gas_prices_df.set_index('Date')
            # Convert gas prices index to datetime
            gas_prices_df.index = pd.to_datetime(gas_prices_df.index)
            st.success(f"Gas prices loaded successfully from {gas_file_option} file: {len(gas_prices_df)} records")
            return True
        else:
            st.error(f"Gas prices file not found at: {gas_path}")
            return False
    except Exception as e:
        st.error(f"Error loading gas prices: {str(e)}")
        return False

def preprocess_with_gas_prices(df, use_manual_threshold=False, manual_threshold_value=None):
    """
    Preprocess uploaded data by merging with gas prices OR using manual threshold
    Expected: df must contain a column named 'Delivery day'
    """
    global gas_prices_df
        
    # Preprocess the uploaded data - following your working prototype
    processed_df = df.copy()
    processed_df = processed_df.set_index('Delivery day')
    processed_df.index.name = 'Date'
    processed_df = processed_df.sort_index()
    
    # Convert index to datetime (this was the main issue in your original code)
    processed_df.index = pd.to_datetime(processed_df.index, dayfirst=True)
    
    # Handle gas prices - either merge with actual gas prices or use manual threshold
    if use_manual_threshold and manual_threshold_value is not None:
        # Create a Settlement column with manual threshold value for all rows
        merged = processed_df.copy()
        merged['Settlement'] = manual_threshold_value
        st.info(f"Using manual threshold of €{manual_threshold_value:.2f}/MWh as Settlement price for all periods")
    else:
        # Merge with gas prices
        try:
            merged = pd.merge(processed_df, gas_prices_df, how="inner", on="Date")
        except Exception as e:
            st.error(f"Merge error: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    # Rename Hour 3A to Hour 3 if it exists
    if 'Hour 3A' in merged.columns:
        merged = merged.rename(columns={'Hour 3A': 'Hour 3'})
    
    # Reshape the data - using your working prototype logic
    hour_columns = [f'Hour {i}' for i in range(1, 25)]
    # Filter to only existing columns
    hour_columns = [col for col in hour_columns if col in merged.columns]
    
    if not hour_columns:
        # If no hour columns found, assume it's already in the right format
        st.warning("No hour columns found for reshaping. Using data as-is.")
        return merged
    
    # Reset index for melting
    merged_reset = merged.reset_index()
    
    reshaped_df = pd.melt(
        merged_reset, 
        id_vars=['Date', 'Settlement'],
        value_vars=hour_columns,
        var_name='Hour',
        value_name='Price'
    )
    
    # Extract hour number and sort
    reshaped_df['Hour_num'] = reshaped_df['Hour'].str.extract('(\d+)').astype(int)
    reshaped_df = reshaped_df.sort_values(['Date', 'Hour_num'])
    reshaped_df = reshaped_df.drop('Hour_num', axis=1)
    reshaped_df = reshaped_df.set_index('Date')
    
    return reshaped_df

def spot(df, price_col, settlement_col, efficiency_parameter, certificates, time_interval_minutes, power_energy=1.0, manual_threshold=None):
    """
    Fixed spot function with better handling for Series values
    """
    result_df = df.copy()
    
    # Initialize columns
    result_df['operate'] = False
    result_df['profit'] = 0.0
    result_df['buy_threshold'] = None
    result_df['MegaWatt'] = 0.0
    result_df['profit_per_mw'] = 0.0
    
    time_factor = time_interval_minutes / 60.0  
    missing_data = 0
    
    # Handle different data structures
    if price_col in result_df.columns:
        price_series = result_df[price_col]
    elif 'Price' in result_df.columns:
        price_series = result_df['Price']
    else:
        st.error(f"Cannot find price data in column '{price_col}'")
        return result_df, 0, 0
    
    # Process each row
    for i in range(len(result_df)):
        try:
            # Get price value - handle both single values and Series
            price_val = price_series.iloc[i]
            if isinstance(price_val, pd.Series):
                price = price_val.iloc[0] if len(price_val) > 0 else None
            else:
                price = price_val
            
            # Skip if price is missing
            if pd.isna(price):
                missing_data += 1
                continue
                
            # Calculate buy threshold
            if manual_threshold is not None:
                buy_threshold = manual_threshold
                el_profit = buy_threshold - price
            else:
                # Get gas price - handle both single values and Series
                if settlement_col and settlement_col in result_df.columns:
                    gas_val = result_df[settlement_col].iloc[i]
                elif 'Settlement' in result_df.columns:
                    gas_val = result_df['Settlement'].iloc[i]
                else:
                    missing_data += 1
                    continue
                
                # Handle Series values
                if isinstance(gas_val, pd.Series):
                    gas_price = gas_val.iloc[0] if len(gas_val) > 0 else None
                else:
                    gas_price = gas_val
                    
                if pd.isna(gas_price):
                    missing_data += 1
                    continue
                    
                buy_threshold = efficiency_parameter * (certificates + gas_price)
                el_profit = buy_threshold - price
            
            # Use iloc for setting values
            result_df.iloc[i, result_df.columns.get_loc('buy_threshold')] = buy_threshold
            
            # Only operate if price is below threshold (profit > 0)
            if el_profit > 0:
                result_df.iloc[i, result_df.columns.get_loc('operate')] = True
                result_df.iloc[i, result_df.columns.get_loc('profit')] = el_profit * time_factor * power_energy
                result_df.iloc[i, result_df.columns.get_loc('profit_per_mw')] = el_profit * time_factor
                result_df.iloc[i, result_df.columns.get_loc('MegaWatt')] = power_energy
                
        except Exception as e:
            missing_data += 1
            if st.checkbox("Show detailed error information", value=False):
                st.error(f"Error processing row {i}: {str(e)}")
    
    if missing_data > 0:
        st.warning(f"Missing or invalid data for {missing_data} rows")
    
    # Calculate totals
    total_operating_hours = result_df['operate'].sum() * time_factor
    total_profit = result_df['profit'].sum()
    
    # Debug info
    operating_count = result_df['operate'].sum()
    st.info(f"Operating in {operating_count} periods out of {len(result_df)} total periods")
    
    return result_df, total_operating_hours, total_profit

def parse_date_column(df, date_col):
    """Enhanced date parsing with better error handling"""
    try:
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d.%m.%Y %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M'
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(df[date_col], format=fmt)
            except:
                continue
        
        return pd.to_datetime(df[date_col], infer_datetime_format=True)
    
    except Exception as e:
        st.error(f"Could not parse date column: {str(e)}")
        return None

def create_enhanced_visualization(df_result, price_col, time_interval_minutes):
    """Create a visualization showing price data and operating decisions"""
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result[price_col],
            mode='lines',
            name='Electricity Price',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False,
    )
    
    # Add buy threshold line
    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result['buy_threshold'],
            mode='lines',
            name='Buy Threshold',
            line=dict(color='red', width=1, dash='dash')
        ),
        secondary_y=False,
    )
    
    # Add operating points
    operating_points = df_result[df_result['operate'] == True]
    if not operating_points.empty:
        fig.add_trace(
            go.Scatter(
                x=operating_points.index,
                y=operating_points[price_col],
                mode='markers',
                name='Operating Hours',
                marker=dict(
                    color='green',
                    size=8,
                    symbol='circle'
                ),
                text=[f"Profit: €{profit:.2f} ({time_interval_minutes}min)" for profit in operating_points['profit']],
                hovertemplate="<b>Operating Period</b><br>" +
                            "Price: %{y:.2f}<br>" +
                            "%{text}<br>" +
                            "<extra></extra>"
            ),
            secondary_y=False,
        )
    
    # Update layout
    fig.update_layout(
        title=f"Electrolyser Operation Strategy ({time_interval_minutes}-minute intervals)",
        xaxis_title="Time Period",
        height=600,
        hovermode='x unified'
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (€/MWh)", secondary_y=False)
    
    return fig

def calculate_advanced_metrics(df_result, time_interval_minutes, power_energy):
    """Calculate additional performance metrics"""
    metrics = {}
    
    # Basic metrics
    operating_periods = df_result['operate'].sum()
    total_periods = len(df_result)
    
    metrics['operating_periods'] = operating_periods
    metrics['total_periods'] = total_periods
    metrics['utilization_rate'] = (operating_periods / total_periods * 100) if total_periods > 0 else 0
    
    # Profit metrics
    total_profit = df_result['profit'].sum()
    operating_profit = df_result[df_result['operate']]['profit'].sum()
    
    metrics['total_profit'] = total_profit
    metrics['operating_profit'] = operating_profit
    metrics['avg_profit_per_operating_period'] = (operating_profit / operating_periods) if operating_periods > 0 else 0
    
    # Energy metrics
    time_factor = time_interval_minutes / 60.0
    total_energy_consumed = operating_periods * time_factor * power_energy
    
    metrics['total_energy_consumed_mwh'] = total_energy_consumed
    metrics['profit_per_mwh'] = (total_profit / total_energy_consumed) if total_energy_consumed > 0 else 0
    
    # Price statistics - Fixed to handle different column names
    if operating_periods > 0:
        operating_data = df_result[df_result['operate']]
        # Find the price column
        price_col = None
        for col in ['Price', 'price']:
            if col in operating_data.columns:
                price_col = col
                break
        
        if price_col:
            metrics['avg_operating_price'] = operating_data[price_col].mean()
            metrics['min_operating_price'] = operating_data[price_col].min()
            metrics['max_operating_price'] = operating_data[price_col].max()
    
    return metrics

def main():
    st.set_page_config(
        page_title="Enhanced Electrolyser Energy Trading Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("Enhanced Electrolyser Energy Trading Simulation")
    st.markdown("Upload your data and configure parameters to simulate electrolyser operations on the energy market.")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")
        
        # Gas file selection first (before loading)
        st.subheader("Gas Price Source")
        gas_file_option = st.selectbox(
            "Select Gas Price File:",
            options=["Month Ahead", "Spot"],
            index=0,
            help="Choose which gas price dataset to use for calculations"
        )
        
        if gas_file_option == "Month Ahead":
            st.info("Using: gas_streamlit.csv")
        else:
            st.info("Using: gas_streamlit1.csv")
    
    # Load gas prices based on selection
    with st.spinner(f"Loading {gas_file_option.lower()} gas prices..."):
        gas_loaded = load_gas_prices(gas_file_option)
    
    if not gas_loaded:
        st.error("Cannot proceed without gas prices. Please check the gas prices file.")
        return
    
    with st.expander("Gas Prices Information"):
        global gas_prices_df
        st.write(f"**Gas price source:** {gas_file_option}")
        st.write(f"**Gas prices loaded:** {len(gas_prices_df)} records")
        st.write(f"**Date range:** {gas_prices_df.index.min()} to {gas_prices_df.index.max()}")
        st.write(f"**Columns:** {', '.join(gas_prices_df.columns.tolist())}")
        st.write("**Preview:**")
        st.dataframe(gas_prices_df.head())
    
    with st.sidebar:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Data",
            type=['csv'],
            help="Upload a CSV file containing a 'Delivery day' column and hourly price data or standard price/settlement data"
        )
        
        st.divider()
        
        st.subheader("System Parameters")
        
        efficiency_parameter = st.number_input(
            "Efficiency Parameter",
            min_value=0.00,
            max_value=1.00,
            value=0.700,
            step=0.001,
            format="%.3f",
            help="Electrolyser efficiency parameter (typically 0.6-0.8)"
        )

        power_energy = st.number_input(
            "Power Capacity (MW)",
            min_value=0.1,
            max_value=9999.99,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Electrolyser power capacity in MW"
        )
        
        certificates = st.number_input(
            "Green Certificates (€/MWh)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            help="Green certificate price in €/MWh"
        )

        st.divider()

        st.subheader("Operating Threshold")
        threshold_option = st.radio(
            "Threshold Method:",
            options=["Calculate from Gas Price", "Set Manual Threshold"],
            index=0,
            help="Choose how to determine when to operate"
        )

        manual_threshold = None
        if threshold_option == "Set Manual Threshold":
            manual_threshold = st.number_input(
                "Manual Threshold (€/MWh)",
                min_value=0.0,
                value=100.0,
                step=1.0,
                help="Fixed price threshold for operation"
            )
            
            st.info(f"With manual threshold of €{manual_threshold:.2f}/MWh, the system will operate when electricity price is below this value.")
        else:
            st.info(f"Gas prices from {gas_file_option} file will be automatically loaded and merged with your data if 'Delivery day' column is present.")
        
        st.divider()
        
        st.subheader("Time Resolution")
        time_interval = st.selectbox(
            "Data Time Interval",
            options=[15, 30, 60],
            index=2,
            help="Time interval of your data in minutes"
        )
        
        if time_interval != 60:
            st.info(f"Profit calculations adjusted for {time_interval}-minute intervals")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            
            # Check if we have 'Delivery day' column - this determines preprocessing path
            has_delivery_day = 'Delivery day' in df_raw.columns
            
            if has_delivery_day:
                with st.expander("Debug Information", expanded=False):
                    st.write("**Raw data preview:**")
                    st.dataframe(df_raw.tail())
                    st.write("**Delivery day column sample:**")
                    st.write(df_raw['Delivery day'].head(10).tolist())
                
                # Determine preprocessing parameters based on threshold option
                if threshold_option == "Calculate from Gas Price":
                    with st.spinner(f"Loading and preprocessing data with {gas_file_option.lower()} gas prices..."):
                        df = preprocess_with_gas_prices(df_raw, use_manual_threshold=False)
                        
                        if df.empty:
                            st.error("Data preprocessing failed. Please check the debug information and ensure your data format is correct.")
                            return
                        
                        st.success("Data preprocessed and merged with gas prices successfully!")
                else:  # Manual threshold
                    with st.spinner("Preprocessing data with manual threshold..."):
                        df = preprocess_with_gas_prices(df_raw, use_manual_threshold=True, manual_threshold_value=manual_threshold)
                        
                        if df.empty:
                            st.error("Data preprocessing failed. Please check the debug information and ensure your data format is correct.")
                            return
                        
                        st.success("Data preprocessed with manual threshold successfully!")
                
                with st.expander("Preprocessing Results", expanded=True):
                    st.write(f"**Original data shape:** {df_raw.shape}")
                    st.write(f"**Processed data shape:** {df.shape}")
                    st.write(f"**Available columns:** {', '.join(df.columns.tolist())}")
                    
                    if len(df) == 0:
                        st.error("No data remaining after preprocessing.")
                        return
                    
                    st.write("**Processed data preview:**")
                    st.dataframe(df.head(10))
                
                # Automatically detect Price and Settlement columns if they exist
                price_default_idx = 0
                settlement_default_idx = 0
                
                if 'Price' in df.columns:
                    price_default_idx = df.columns.tolist().index('Price')
                if 'Settlement' in df.columns:
                    settlement_default_idx = df.columns.tolist().index('Settlement')
                    
            else:
                # Use standard processing for data without 'Delivery day'
                df = df_raw.copy()
                
                # Smart column detection for any data format
                price_default_idx = 0
                settlement_default_idx = 0
                
                # Look for common price column names
                price_keywords = ['price', 'electricity', 'power', 'energy', 'spot', 'dam']
                settlement_keywords = ['settlement', 'gas', 'fuel', 'cost']
                
                for i, col in enumerate(df.columns):
                    col_lower = col.lower()
                    # Skip 'Delivery day' for price selection
                    if 'delivery' in col_lower or 'date' in col_lower:
                        continue
                    # Look for price columns
                    if any(keyword in col_lower for keyword in price_keywords):
                        price_default_idx = i
                        break
                
                for i, col in enumerate(df.columns):
                    col_lower = col.lower()
                    # Skip 'Delivery day' for settlement selection
                    if 'delivery' in col_lower or 'date' in col_lower:
                        continue
                    # Look for settlement/gas columns
                    if any(keyword in col_lower for keyword in settlement_keywords):
                        settlement_default_idx = i
                        break
                
                # If no keywords found, default to first non-date column
                if price_default_idx == 0:
                    for i, col in enumerate(df.columns):
                        col_lower = col.lower()
                        if 'delivery' not in col_lower and 'date' not in col_lower:
                            price_default_idx = i
                            break
            
            # Column selection section
            st.header("Column Selection")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_column = st.selectbox(
                    "Electricity Price Column",
                    options=df.columns.tolist(),
                    index=price_default_idx
                )
            
            with col2:
                if threshold_option == "Calculate from Gas Price":
                    settlement_column = st.selectbox(
                        "Gas Price Column", 
                        options=df.columns.tolist(),
                        index=settlement_default_idx
                    )
                else:
                    settlement_column = st.selectbox(
                        "Gas Price Column (optional)", 
                        options=['None'] + df.columns.tolist(),
                        index=0
                    )
                    if settlement_column == 'None':
                        settlement_column = None
            
            with col3:
                date_column = st.selectbox(
                    "Date/Time Column (optional)",
                    options=['None'] + df.columns.tolist(),
                    index=0
                )
            
            # Validate required columns
            if threshold_option == "Calculate from Gas Price" and settlement_column is None:
                st.error("Gas price column is required when calculating threshold from gas price!")
                return
            
            # Data filtering and processing
            processed_df = df.copy()
            
            # Date filtering - handle both preprocessed and standard data
            if date_column != 'None' and not has_delivery_day:
                st.header("Date Range Selection")
                
                parsed_dates = parse_date_column(processed_df, date_column)
                if parsed_dates is not None:
                    processed_df['parsed_date'] = parsed_dates
                    
                    min_date = processed_df['parsed_date'].min().date()
                    max_date = processed_df['parsed_date'].max().date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
                    
                    if start_date <= end_date:
                        date_mask = (processed_df['parsed_date'].dt.date >= start_date) & (processed_df['parsed_date'].dt.date <= end_date)
                        processed_df = processed_df[date_mask]
                        st.success(f"Filtered to {len(processed_df):,} rows from {start_date} to {end_date}")
                    else:
                        st.error("End date must be after start date!")
                        return
            elif has_delivery_day:
                # Date filtering for preprocessed data
                st.header("Date Range Selection")
                
                # Ensure the index is datetime and handle potential errors
                try:
                    if hasattr(processed_df.index, 'date'):
                        min_date = processed_df.index.min().date()
                        max_date = processed_df.index.max().date()
                    else:
                        # If index is not datetime, convert it
                        processed_df.index = pd.to_datetime(processed_df.index)
                        min_date = processed_df.index.min().date()
                        max_date = processed_df.index.max().date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
                    
                    if start_date <= end_date:
                        # Convert dates to datetime for comparison
                        start_datetime = pd.to_datetime(start_date)
                        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        
                        date_mask = (processed_df.index >= start_datetime) & (processed_df.index <= end_datetime)
                        original_length = len(processed_df)
                        processed_df = processed_df[date_mask]
                        
                        if len(processed_df) < original_length:
                            st.success(f"Date filter applied: {len(processed_df):,} rows selected from {original_length:,} total rows")
                        else:
                            st.info("No filtering applied - selected range includes all data")
                    else:
                        st.error("End date must be after start date!")
                        return
                        
                except Exception as date_error:
                    st.warning(f"Date filtering skipped due to date format issues: {str(date_error)}")
                    st.info("Proceeding without date filtering. You can still run the simulation with all available data.")
            
            # Display data overview
            with st.expander("Data Overview", expanded=True):
                st.write(f"**Data shape:** {processed_df.shape}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(processed_df))
                    if pd.api.types.is_numeric_dtype(processed_df[price_column]):
                        st.write(f"**{price_column}:**")
                        st.write(f"• Range: €{processed_df[price_column].min():.2f} - €{processed_df[price_column].max():.2f}")
                        st.write(f"• Average: €{processed_df[price_column].mean():.2f}")
                
                with col2:
                    equivalent_hours = len(processed_df) * time_interval / 60
                    st.metric("Equivalent Hours", f"{equivalent_hours:.1f}")
                    
                    if settlement_column and pd.api.types.is_numeric_dtype(processed_df[settlement_column]):
                        st.write(f"**{settlement_column}:**")
                        st.write(f"• Range: €{processed_df[settlement_column].min():.2f} - €{processed_df[settlement_column].max():.2f}")
                        st.write(f"• Average: €{processed_df[settlement_column].mean():.2f}")
                
                with col3:
                    missing_values = processed_df.isnull().sum().sum()
                    st.metric("Missing Values", missing_values)
                
                st.dataframe(processed_df.head(), use_container_width=True)
            
            # Validate numeric columns
            if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
                st.error(f"Price column '{price_column}' is not numeric!")
                return
            
            if settlement_column and not pd.api.types.is_numeric_dtype(processed_df[settlement_column]):
                st.error(f"Settlement column '{settlement_column}' is not numeric!")
                return
            
            # Run simulation
            if st.button("Run Simulation", type="primary", use_container_width=True):
                with st.spinner("Running simulation..."):
                    result_df, operating_hours, total_profit = spot(
                        processed_df, price_column, settlement_column, 
                        efficiency_parameter, certificates, time_interval, 
                        power_energy, manual_threshold
                    )
                
                # Calculate advanced metrics
                metrics = calculate_advanced_metrics(result_df, time_interval, power_energy)
                
                # Display results
                st.header("Simulation Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Operating Hours", f"{operating_hours:,.1f}")
                
                with col2:
                    st.metric("Total Profit", f"€{total_profit:,.2f}")
                
                with col3:
                    st.metric("Utilization Rate", f"{metrics['utilization_rate']:.1f}%")
                
                with col4:
                    st.metric("Energy Consumed", f"{metrics['total_energy_consumed_mwh']:.1f} MWh")
                
                # Additional metrics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric("Profit per MWh", f"€{metrics['profit_per_mwh']:.2f}")
                
                with col6:
                    if metrics['operating_periods'] > 0:
                        st.metric("Avg Profit/Period", f"€{metrics['avg_profit_per_operating_period']:.2f}")
                    else:
                        st.metric("Avg Profit/Period", "€0.00")
                
                with col7:
                    st.metric("Operating Periods", f"{metrics['operating_periods']:,}")
                
                with col8:
                    total_possible_hours = len(processed_df) * time_interval / 60
                    st.metric("Total Possible Hours", f"{total_possible_hours:.1f}")
                
                # Visualization
                st.header("Analysis Dashboard")
                
                if operating_hours > 0:
                    fig = create_enhanced_visualization(result_df, price_column, time_interval)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    st.subheader("Key Insights")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.info(f"**Strategy Performance:**\n- System operates {metrics['utilization_rate']:.1f}% of the time\n- Generates €{metrics['profit_per_mwh']:.2f} profit per MWh consumed\n- Average profit of €{metrics['avg_profit_per_operating_period']:.2f} per operating period")
                    
                    with insights_col2:
                        if 'avg_operating_price' in metrics:
                            st.success(f"**Price Analysis:**\n- Average operating price: €{metrics['avg_operating_price']:.2f}/MWh\n- Price range during operation: €{metrics['min_operating_price']:.2f} - €{metrics['max_operating_price']:.2f}/MWh")
                        
                        if time_interval != 60:
                            st.info(f"**Time Resolution:**\nProfit adjusted for {time_interval}-minute intervals ({time_interval/60:.2f}x hourly rate)")
                        
                else:
                    st.warning("No profitable operating periods found with current parameters.")
                    st.info("Try adjusting the efficiency parameter, certificate price, or threshold value.")
                    
                    # Show basic price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result_df.index,
                        y=result_df[price_column],
                        mode='lines',
                        name='Electricity Price',
                        line=dict(color='blue')
                    ))
                    if 'buy_threshold' in result_df.columns:
                        fig.add_trace(go.Scatter(
                            x=result_df.index,
                            y=result_df['buy_threshold'],
                            mode='lines',
                            name='Buy Threshold',
                            line=dict(color='red', dash='dash')
                        ))
                    fig.update_layout(
                        title="Price vs Threshold Analysis",
                        xaxis_title="Time Period",
                        yaxis_title="Price (€/MWh)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with numeric data in the selected columns.")
            # Show the full error traceback for debugging
            import traceback
            with st.expander("Full Error Details", expanded=False):
                st.code(traceback.format_exc())
    
    else:
        st.info("Please upload a CSV file to get started.")
        
        # Enhanced example section
        with st.expander("Expected Data Format & Features", expanded=False):
            st.markdown("""
            ### Data Requirements
            
            **Option 1: Standard Format**
            - **Electricity Price**: Numeric values (e.g., €/MWh)
            - **Gas Settlement Price**: Numeric values (e.g., €/MWh) - *only if calculating threshold from gas price*
            - **Date/Time**: For filtering specific time periods (optional)
            
            **Option 2: Gas Price Preprocessing (Automatic)**
            - **'Delivery day' column** (required): Date information for merging with gas prices
            - **Hourly price columns**: Hour 1, Hour 2, ..., Hour 24 (or similar hourly data)
            
            ### Operating Strategies
            
            **1. Gas-Based Threshold (Recommended)**
            - Threshold = Efficiency × (Certificates + Gas Price)
            - Operates when: Electricity Price < Threshold
            - Most realistic approach for hydrogen production
            - Supports automatic gas price loading and preprocessing
            
            **2. Manual Threshold**
            - Fixed price threshold (e.g., €100/MWh)
            - Operates when: Electricity Price < Manual Threshold
            - Useful for sensitivity analysis
            - **Now supports data preprocessing even with manual threshold!**
            
            ### Automatic Data Preprocessing
            
            **What happens when 'Delivery day' column is detected:**
            1. Your data is loaded and indexed by the 'Delivery day' column
            2. Data is reshaped from wide format (hourly columns) to long format
            3. **For Gas-Based Threshold**: Gas prices are automatically merged from the system database
            4. **For Manual Threshold**: Only data reshaping is performed (no gas price merging)
            5. You can then select the appropriate price and settlement columns
            
            ### Time Intervals Supported
            - **15 minutes**: Ultra-high frequency trading
            - **30 minutes**: High frequency analysis  
            - **60 minutes**: Standard hourly analysis
            
            ### Key Metrics Calculated
            - Total profit and operating hours
            - Utilization rate and energy consumption
            - Profit per MWh and per operating period
            - Price statistics during operation
            
            ### Example Data Formats
            
            **Standard Format:**
            ```csv
            DateTime,Product,Electricity_Price,Gas_Settlement
            2024-01-01 00:00:00,DAM,45.2,30.5
            2024-01-01 01:00:00,DAM,52.1,31.2
            2024-01-01 02:00:00,DAM,38.7,29.8
            ```
            
            **Gas Preprocessing Format:**
            ```csv
            Delivery day,Hour 1,Hour 2,Hour 3,...,Hour 24
            01/01/2024,45.2,52.1,38.7,...,42.3
            02/01/2024,47.5,49.8,41.2,...,45.1
            ```
            
            Column names can be customized - you'll select them after upload!
            """)

if __name__ == "__main__":
    main()