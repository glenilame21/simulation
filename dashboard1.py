import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
import requests
from io import StringIO

# Global variable to store gas prices
gas_prices_df = None

# GitHub repository configuration
GITHUB_BASE_URL = "https://github.com/glenilame21/simulation/"
# Alternative: Use specific commit hash for version control
# GITHUB_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/COMMIT_HASH/"

# CSV file paths on GitHub
GAS_FILES = {
    "Month Ahead": "https://raw.githubusercontent.com/glenilame21/simulation/refs/heads/main/gas_streamlit.csv",  # Replace with your actual filename
    "Spot": "https://raw.githubusercontent.com/glenilame21/simulation/refs/heads/main/Gas_D1.csv"                # Replace with your actual filename
}

@st.cache_data
def load_gas_prices_from_github(gas_file_option="Month Ahead"):
    """Load gas prices from GitHub repository"""
    try:
        filename = GAS_FILES[gas_file_option]
        url = GITHUB_BASE_URL + filename
        
        # Download the CSV file
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Read CSV from string
        csv_data = StringIO(response.text)
        gas_prices_df = pd.read_csv(csv_data)
        
        # Process the dataframe
        gas_prices_df = gas_prices_df.set_index('Date')
        gas_prices_df.index = pd.to_datetime(gas_prices_df.index)
        
        return gas_prices_df, True
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading gas prices from GitHub: {str(e)}")
        return None, False
    except Exception as e:
        st.error(f"Error processing gas prices: {str(e)}")
        return None, False

def load_gas_prices(gas_file_option="Month Ahead"):
    """Updated function to load gas prices from GitHub"""
    global gas_prices_df
    
    gas_prices_df, success = load_gas_prices_from_github(gas_file_option)
    return success

def preprocess_with_gas_prices(df, use_manual_threshold=False, manual_threshold_value=None):
    """
    Traders want to have the option of either using a manual threshold for gas Price or a dropdown that let's them pick M1 or D1
    For this reason, by default the function above will load Month Ahead in memory and use that as the default value for the threshold
    If users choose otherwise then this function comes in play
    Preprocess uploaded data by merging with gas prices OR using manual threshold
    Expected: df must contain a column named 'Delivery day'
    """
    global gas_prices_df
    
    processed_df = df.copy()
    processed_df = processed_df.set_index('Delivery day')
    processed_df.index.name = 'Date'
    processed_df = processed_df.sort_index()
    
    processed_df.index = pd.to_datetime(processed_df.index, dayfirst=True)
    
    if use_manual_threshold and manual_threshold_value is not None:
        merged = processed_df.copy()
        merged['Settlement'] = manual_threshold_value
        st.info(f"Using manual threshold of €{manual_threshold_value:.2f}/MWh as Settlement price for all periods")
    else:
        try:
            merged = pd.merge(processed_df, gas_prices_df, how="inner", on="Date")
        except Exception as e:
            st.error(f"Merge error: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    if 'Hour 3A' in merged.columns:
        merged = merged.rename(columns={'Hour 3A': 'Hour 3'})
    
    hour_columns = [f'Hour {i}' for i in range(1, 25)]
    hour_columns = [col for col in hour_columns if col in merged.columns]
    
    if not hour_columns:
        st.warning("No hour columns found for reshaping. Using data as-is.")
        return merged
    
    merged_reset = merged.reset_index()
    
    reshaped_df = pd.melt(
        merged_reset, 
        id_vars=['Date', 'Settlement'],
        value_vars=hour_columns,
        var_name='Hour',
        value_name='Price'
    )
    
    reshaped_df['Hour_num'] = reshaped_df['Hour'].str.extract('(\d+)').astype(int)
    reshaped_df = reshaped_df.sort_values(['Date', 'Hour_num'])
    reshaped_df = reshaped_df.drop('Hour_num', axis=1)
    reshaped_df = reshaped_df.set_index('Date')
    
    return reshaped_df

def spot(df, price_col, settlement_col, efficiency_parameter, certificates, time_interval_minutes, power_energy=1.0, manual_threshold=None):
    """
    Here the idea is that we buy electricity when the threshold is hit so that we send it to the electrolyser
    Ideally this works best with negative power prices where you get paid for getting energy
    """
    result_df = df.copy()
    
    # Here we initialize the columns we will plot as results in the end
    result_df['operate'] = False # a yes or no variable that will count how many times the threshold was hit and therefore the electolyser was used
    result_df['profit'] = 0.0 # easy to understand, a column for profit
    result_df['buy_threshold'] = None
    result_df['MegaWatt'] = 0.0 # Traders also want to see the amount of MW/h generated
    result_df['profit_per_mw'] = 0.0 #idk why this is here
    result_df['gas_generated_mwh'] = 0.0 # Gas generated in MWh
    

    # this is an input field from the user - I can maybe make this a constant so that based on the product being traded the time_interval_minutes is populated
    time_factor = time_interval_minutes / 60.0  # when we sell a 15-minute product in intraday the profit needs to be devided by 0.25
    missing_data = 0
    
    
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
            
            result_df.iloc[i, result_df.columns.get_loc('buy_threshold')] = buy_threshold
            
            if el_profit > 0:
                result_df.iloc[i, result_df.columns.get_loc('operate')] = True
                result_df.iloc[i, result_df.columns.get_loc('profit')] = el_profit * time_factor * power_energy
                result_df.iloc[i, result_df.columns.get_loc('profit_per_mw')] = el_profit * time_factor
                result_df.iloc[i, result_df.columns.get_loc('MegaWatt')] = power_energy
                
                # Calculate gas generation: electricity input * efficiency
                electricity_consumed_mwh = power_energy * time_factor  # MWh of electricity consumed
                gas_generated_mwh = electricity_consumed_mwh * efficiency_parameter  # MWh of gas generated
                
                result_df.iloc[i, result_df.columns.get_loc('gas_generated_mwh')] = gas_generated_mwh
                
        except Exception as e:
            missing_data += 1
            if st.checkbox("Show detailed error information", value=False):
                st.error(f"Error processing row {i}: {str(e)}")
    
    if missing_data > 0:
        st.warning(f"Missing or invalid data for {missing_data} rows")
    

    total_operating_hours = result_df['operate'].sum() * time_factor
    total_profit = result_df['profit'].sum()
    
   
    operating_count = result_df['operate'].sum()
    st.info(f"Operating in {operating_count} periods out of {len(result_df)} total periods")
    
    return result_df, total_operating_hours, total_profit

def parse_date_column(df, date_col):
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
    """Calculate additional performance metrics including gas production"""
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
    
    # Gas generation metrics
    if 'gas_generated_mwh' in df_result.columns:
        metrics['total_gas_generated_mwh'] = df_result['gas_generated_mwh'].sum()
        metrics['gas_generation_rate_mwh_per_hour'] = df_result[df_result['operate']]['gas_generated_mwh'].mean() / time_factor if operating_periods > 0 else 0
        
        # Profit per MWh of gas generated
        if metrics['total_gas_generated_mwh'] > 0:
            metrics['profit_per_mwh_gas'] = total_profit / metrics['total_gas_generated_mwh']
    
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
        page_title="Electrolyser Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Electrolyser Energy Trading Simulation")
    st.markdown("Upload your data and configure parameters. Gas prices loaded from GitHub repository.")
    
    # Configuration section for GitHub repository
    with st.expander("📂 GitHub Repository Configuration", expanded=False):
        st.markdown("""
        **Current Configuration:**
        - Repository: `YOUR_USERNAME/YOUR_REPO`
        - Branch: `main`
        - Gas Files: `gas_streamlit.csv` (Month Ahead), `Gas_D1.csv` (Spot)
        
        **To customize:** Update the `GITHUB_BASE_URL` and `GAS_FILES` variables in the code.
        """)
        
        # Show current URLs for debugging
        st.code(f"Month Ahead URL: {GITHUB_BASE_URL + GAS_FILES['Month Ahead']}")
        st.code(f"Spot URL: {GITHUB_BASE_URL + GAS_FILES['Spot']}")
    
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
            st.info("Using: Gas Month Ahead")
        else:
            st.info("Using: Gas Spot")
    
    # Load gas prices based on selection
    with st.spinner(f"Loading {gas_file_option.lower()} gas prices from GitHub..."):
        gas_loaded = load_gas_prices(gas_file_option)
    
    if not gas_loaded:
        st.error("Cannot proceed without gas prices. Please check the GitHub repository configuration and ensure the CSV files are accessible.")
        st.markdown("""
        **Troubleshooting:**
        1. Verify the GitHub repository URL is correct
        2. Ensure the CSV files exist in the repository
        3. Check that the repository is public or accessible
        4. Verify the file names match exactly (case-sensitive)
        """)
        return
    
    # Display gas prices info
    with st.expander("📊 Gas Prices Information"):
        global gas_prices_df
        st.write(f"**Gas price source:** {gas_file_option}")
        st.write(f"**Gas prices loaded:** {len(gas_prices_df)} records")
        st.write(f"**Date range:** {gas_prices_df.index.min()} to {gas_prices_df.index.max()}")
        st.write(f"**Columns:** {', '.join(gas_prices_df.columns.tolist())}")
        st.write("**Preview:**")
        st.dataframe(gas_prices_df.tail())
    
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
            help="Electrolyser efficiency parameter (typically 0.6-0.8). This determines gas generation: 1 MWh electricity → efficiency × 1 MWh gas"
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
                
                # Key metrics (first row)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Profit", f"€{total_profit:,.2f}")
                
                with col2:
                    st.metric("Energy Consumed", f"{metrics['total_energy_consumed_mwh']:.1f} MWh")

                
                with col3:
                    if 'total_gas_generated_mwh' in metrics:
                        st.metric("Gas Generated", f"{metrics['total_gas_generated_mwh']:.1f} MWh")
                    else:
                        st.metric("Avg Profit/Period", f"€{metrics['avg_profit_per_operating_period']:.2f}")
                
                # Additional metrics
                col4, col5 = st.columns(2)
                
                with col4:
                    st.metric("Operating Hours", f"{operating_hours:,.1f}")
                
                with col5:
                    st.metric("Utilization Rate", f"{metrics['utilization_rate']:.1f}%")
                
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
                    
                    # Add gas generation insights
                    if 'total_gas_generated_mwh' in metrics and metrics['total_gas_generated_mwh'] > 0:
                        st.subheader("Gas Generation Summary")
                        
                        gas_col1, gas_col2 = st.columns(2)
                        
                        with gas_col1:
                            st.info(f"""**Production Summary:**
- Total gas generated: {metrics['total_gas_generated_mwh']:.1f} MWh
- Average generation rate: {metrics['gas_generation_rate_mwh_per_hour']:.2f} MWh/hour
- Efficiency factor used: {efficiency_parameter:.1%}""")
                        
                        with gas_col2:
                            efficiency_check = metrics['total_gas_generated_mwh'] / metrics['total_energy_consumed_mwh'] if metrics['total_energy_consumed_mwh'] > 0 else 0
                            st.success(f"""**Efficiency Verification:**
- Electricity consumed: {metrics['total_energy_consumed_mwh']:.1f} MWh
- Gas generated: {metrics['total_gas_generated_mwh']:.1f} MWh  
- Actual efficiency: {efficiency_check:.1%}
- Profit per MWh gas: €{metrics['profit_per_mwh_gas']:.2f}""")
                        
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
        
        # Show example of expected data format
        with st.expander("📋 Expected Data Format", expanded=False):
            st.markdown("""
            **Option 1: Standard Format**
            ```
            Date,Price,Settlement
            2024-01-01,45.50,30.25
            2024-01-02,52.75,31.10
            ```
            
            **Option 2: Hourly Format with Delivery Day**
            ```
            Delivery day,Hour 1,Hour 2,Hour 3,...,Hour 24
            01/01/2024,45.50,46.25,47.00,...,52.75
            02/01/2024,52.75,53.25,54.00,...,48.50
            ```
            """)

if __name__ == "__main__":
    main()