import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
import os

# Global variable to store gas prices
gas_prices_df = None

def load_gas_prices():
    global gas_prices_df
    try:
        gas_path = "C:/Users/Z_LAME/Desktop/Crawler/Electrolyser/gas_streamlit.csv"
        if os.path.exists(gas_path):
            gas_prices_df = pd.read_csv(gas_path)
            gas_prices_df = gas_prices_df.set_index('Date')
            # Convert gas prices index to datetime
            gas_prices_df.index = pd.to_datetime(gas_prices_df.index)
            st.success(f"Gas prices loaded successfully: {len(gas_prices_df)} records")
            return True
        else:
            st.error(f"Gas prices file not found at: {gas_path}")
            return False
    except Exception as e:
        st.error(f"Error loading gas prices: {str(e)}")
        return False

def preprocess_with_gas_prices(df):
    """
    Preprocess uploaded data by merging with gas prices
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

def spot(df, price_col, settlement_col, efficiency_parameter, certificates, time_interval_minutes):
    result_df = df.copy()
    result_df['operate'] = False
    result_df['profit'] = 0.0
    result_df['buy_threshold'] = None
    
    time_factor = time_interval_minutes / 60.0  
    missing_gas_prices = 0
    
    for idx, row in result_df.iterrows():
        try:
            gas_price = row[settlement_col]
            
            if pd.isna(gas_price):
                missing_gas_prices += 1
                continue
            
            buy_threshold = efficiency_parameter * (certificates + gas_price)
            price = row[price_col]  
            el_profit = efficiency_parameter * (certificates + gas_price) - price
            
            result_df.at[idx, 'buy_threshold'] = buy_threshold
            
            if el_profit > 0:
                result_df.at[idx, 'operate'] = True
                result_df.at[idx, 'profit'] = el_profit * time_factor
                
        except (KeyError, TypeError) as e:
            missing_gas_prices += 1
    
    if missing_gas_prices > 0:
        st.warning(f"Missing or invalid gas prices for {missing_gas_prices} rows")
    
    total_operating_hours = result_df['operate'].sum() * time_factor
    total_profit = result_df['profit'].sum()
    
    return result_df, total_operating_hours, total_profit

def parse_date_column(df, date_col):
    """Try to parse date column with multiple formats"""
    try:
        # Try common date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y'
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(df[date_col], format=fmt)
            except:
                continue
        
        return pd.to_datetime(df[date_col])
    
    except Exception as e:
        st.error(f"Could not parse date column: {str(e)}")
        return None

def create_visualization(df_result, price_col, time_interval_minutes):
    """Create a visualization showing price data and operating decisions"""
    
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

# Streamlit App
def main():
    st.set_page_config(
        page_title="Electrolyser Energy Trading Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("Electrolyser Energy Trading Simulation")
    st.markdown("Upload your data and configure parameters to simulate electrolyser operations on the energy market.")
    
    # Load gas prices at startup
    with st.spinner("Loading gas prices..."):
        gas_loaded = load_gas_prices()
    
    if not gas_loaded:
        st.error("Cannot proceed without gas prices. Please check the gas prices file.")
        return
    
    # Show gas prices info
    with st.expander("Gas Prices Information"):
        global gas_prices_df
        st.write(f"**Gas prices loaded:** {len(gas_prices_df)} records")
        st.write(f"**Date range:** {gas_prices_df.index.min()} to {gas_prices_df.index.max()}")
        st.write(f"**Columns:** {', '.join(gas_prices_df.columns.tolist())}")
        st.write("**Preview:**")
        st.dataframe(gas_prices_df.head())
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Data",
            type=['csv'],
            help="Upload a CSV file containing a 'Delivery day' column and hourly price data"
        )
        
        # Parameters
        efficiency_parameter = st.number_input(
            "Efficiency Parameter",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.01,
            help="Electrolyser efficiency parameter (typically 0.6-0.8)"
        )
        
        certificates = st.number_input(
            "Certificates (€/MWh)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            help="Certificate price in €/MWh"
        )
        
        # Time interval selection
        st.subheader("Time Resolution")
        time_interval = st.selectbox(
            "Data Time Interval",
            options=[15, 30, 60],
            index=2,  # Default to 60 minutes
            help="Time interval of your data in minutes"
        )
        
        if time_interval != 60:
            st.info(f"Profit will be adjusted for {time_interval}-minute intervals")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            with st.spinner("Loading and preprocessing data..."):
                df_raw = pd.read_csv(uploaded_file)
                
                # Check for required column
                if 'Delivery day' not in df_raw.columns:
                    st.error("⚠️ The uploaded CSV file must contain a column named 'Delivery day'")
                    st.info("Please ensure your CSV file has the required column structure.")
                    return
                
                # Show debug information
                with st.expander("Debug Information", expanded=False):
                    st.write("**Raw data preview:**")
                    st.dataframe(df_raw.head())
                    st.write("**Delivery day column sample:**")
                    st.write(df_raw['Delivery day'].head(10).tolist())
                
                # Preprocess with gas prices
                df = preprocess_with_gas_prices(df_raw)
                
                if df.empty:
                    st.error("Data preprocessing failed. Please check the debug information and ensure your data format is correct.")
                    return
                
                st.success("✅ Data preprocessed and merged with gas prices successfully!")
            
            # Show preprocessing results
            with st.expander("Preprocessing Results", expanded=True):
                st.write(f"**Original data shape:** {df_raw.shape}")
                st.write(f"**Processed data shape:** {df.shape}")
                st.write(f"**Available columns:** {', '.join(df.columns.tolist())}")
                
                if len(df) == 0:
                    st.error("No data remaining after preprocessing. Check date alignment with gas prices.")
                    return
                
                st.write("**Processed data preview:**")
                st.dataframe(df.head(10))
            
            # Column selection section
            st.header("Column Selection")
            st.markdown("Select which columns contain your data:")
            
            # Automatically detect Price and Settlement columns if they exist
            price_default_idx = 0
            settlement_default_idx = 0
            
            if 'Price' in df.columns:
                price_default_idx = df.columns.tolist().index('Price')
            if 'Settlement' in df.columns:
                settlement_default_idx = df.columns.tolist().index('Settlement')
            
            col1, col2 = st.columns(2)
            
            with col1:
                price_column = st.selectbox(
                    "Select Price Column",
                    options=df.columns.tolist(),
                    index=price_default_idx,
                    help="Column containing electricity prices (should be 'Price' after preprocessing)"
                )
            
            with col2:
                settlement_column = st.selectbox(
                    "Select Gas Settlement Column", 
                    options=df.columns.tolist(),
                    index=settlement_default_idx,
                    help="Column containing gas settlement prices (should be 'Settlement' after preprocessing)"
                )
            
            # Validate column selection
            if price_column == settlement_column:
                st.warning("Please select different columns for price and settlement data!")
                return
            
            # Optional date filtering
            st.header("Date Range Selection")
            
            # Get date range from index
            min_date = df.index.min().date()
            max_date = df.index.max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Apply date filter
            if start_date <= end_date:
                date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
                original_length = len(df)
                df = df[date_mask]
                
                if len(df) < original_length:
                    st.success(f"Date filter applied: {len(df):,} rows selected from {original_length:,} total rows")
                else:
                    st.info("No filtering applied - selected range includes all data")
            else:
                st.error("End date must be after start date!")
                return
            
            # Display data info
            with st.expander("Data Overview", expanded=True):
                st.write(f"**Final data shape:** {df.shape}")
                st.write(f"**Selected Price Column:** {price_column}")
                st.write(f"**Selected Settlement Column:** {settlement_column}")
                st.write(f"**Date Range:** {start_date} to {end_date}")
                st.write(f"**Time Interval:** {time_interval} minutes")
                
                # Show column statistics
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.write(f"**{price_column} Statistics:**")
                    if pd.api.types.is_numeric_dtype(df[price_column]):
                        st.write(f"- Min: {df[price_column].min():.2f}")
                        st.write(f"- Max: {df[price_column].max():.2f}")
                        st.write(f"- Mean: {df[price_column].mean():.2f}")
                        st.write(f"- Missing values: {df[price_column].isnull().sum()}")
                    else:
                        st.write("Column appears to be non-numeric")
                
                with col_stats2:
                    st.write(f"**{settlement_column} Statistics:**")
                    if pd.api.types.is_numeric_dtype(df[settlement_column]):
                        st.write(f"- Min: {df[settlement_column].min():.2f}")
                        st.write(f"- Max: {df[settlement_column].max():.2f}")
                        st.write(f"- Mean: {df[settlement_column].mean():.2f}")
                        st.write(f"- Missing values: {df[settlement_column].isnull().sum()}")
                    else:
                        st.write("Column appears to be non-numeric")
                
                st.write("**Sample data:**")
                st.dataframe(df.head())
                
                # Summary metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Total Records", len(df))
                with metrics_col2:
                    st.metric("Total Missing Values", df.isnull().sum().sum())
                with metrics_col3:
                    if time_interval == 60:
                        st.metric("Equivalent Hours", len(df))
                    else:
                        equivalent_hours = len(df) * time_interval / 60
                        st.metric("Equivalent Hours", f"{equivalent_hours:.1f}")
            
            # Validate that selected columns are numeric
            numeric_issues = []
            if not pd.api.types.is_numeric_dtype(df[price_column]):
                numeric_issues.append(f"Price column '{price_column}' is not numeric")
            if not pd.api.types.is_numeric_dtype(df[settlement_column]):
                numeric_issues.append(f"Settlement column '{settlement_column}' is not numeric")
            
            if numeric_issues:
                st.error("Column Type Issues:")
                for issue in numeric_issues:
                    st.write(f"- {issue}")
                st.info("Please select columns that contain numeric data for the simulation to work properly.")
                return
            
            # Run simulation
            if st.button("Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    result_df, operating_hours, total_profit = spot(
                        df, price_column, settlement_column, efficiency_parameter, certificates, time_interval
                    )
                
                # Display results
                st.header("Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Operating Hours", 
                        f"{operating_hours:,.1f}",
                        help=f"Number of hours the electrolyser would operate (adjusted for {time_interval}-minute intervals)"
                    )
                
                with col2:
                    st.metric(
                        "Total Profit", 
                        f"€{total_profit:,.2f}",
                        help=f"Total profit from the strategy (adjusted for {time_interval}-minute intervals)"
                    )
                
                with col3:
                    total_possible_hours = len(df) * time_interval / 60
                    utilization = (operating_hours / total_possible_hours) * 100 if total_possible_hours > 0 else 0
                    st.metric(
                        "Utilization Rate",
                        f"{utilization:.1f}%",
                        help="Percentage of time the electrolyser operates"
                    )
                
                # Additional metrics for different time intervals
                if time_interval != 60:
                    st.info(f"**Time Resolution Impact:** With {time_interval}-minute intervals, profit is calculated as {time_interval}/60 = {time_interval/60:.2f} of hourly profit per period.")
                
                # Visualization
                st.header("Price Analysis & Operations")
                
                if operating_hours > 0:
                    fig = create_visualization(result_df, price_column, time_interval)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights
                    st.subheader("Key Insights")
                    avg_profit_per_hour = total_profit / operating_hours if operating_hours > 0 else 0
                    
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        st.info(f"Average profit per operating hour: €{avg_profit_per_hour:.2f}")
                    with insight_col2:
                        st.info(f"System operates {utilization:.1f}% of the time")
                    
                    # Time-specific insights
                    operating_periods = result_df['operate'].sum()
                    st.info(f"Operating periods: {operating_periods:,} ({time_interval}-minute intervals)")
                        
                else:
                    st.warning("No profitable operating hours found with current parameters. Try adjusting the efficiency parameter or certificate price.")
                    
                    # Still show price chart for reference
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result_df.index,
                        y=result_df[price_column],
                        mode='lines',
                        name='Electricity Price',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=result_df.index,
                        y=result_df['buy_threshold'],
                        mode='lines',
                        name='Buy Threshold',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"Price vs Buy Threshold ({time_interval}-minute intervals)",
                        xaxis_title="Time Period",
                        yaxis_title="Price (€/MWh)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Optional: Download results
                if st.checkbox("Show detailed results"):
                    st.subheader("Detailed Results")
                    display_columns = [price_column, settlement_column, 'buy_threshold', 'operate', 'profit']
                    result_display = result_df[display_columns].copy()
                    
                    # Add time interval info to column names
                    result_display = result_display.rename(columns={
                        'profit': f'profit_({time_interval}min)',
                        'operate': f'operate_({time_interval}min)'
                    })
                    
                    st.dataframe(result_display)
                    
                    # Download button
                    csv_data = result_display.to_csv(index=True)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name=f"electrolyser_simulation_results_{time_interval}min.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required 'Delivery day' column and is properly formatted.")
            # Show the full error traceback for debugging
            import traceback
            with st.expander("Full Error Details", expanded=False):
                st.code(traceback.format_exc())
    
    else:
        st.info("Please upload a CSV file to get started.")
        
        # Show example of expected data format
        with st.expander("Expected Data Format", expanded=True):
            st.markdown("""
            **🔄 Automatic Data Preprocessing Enabled!**
            
            Your CSV file must contain:
            - **'Delivery day' column** (required): Date information that will be used for merging with gas prices
            - **Hourly price columns**: Hour 1, Hour 2, ..., Hour 24 (or similar hourly data)
            
            **What happens automatically:**
            1. Your data is loaded and indexed by the 'Delivery day' column
            2. Gas prices are automatically merged from the system database
            3. Data is reshaped from wide format (hourly columns) to long format
            4. You can then select the appropriate price and settlement columns
            
            **Example Input Format:**
            ```
            Delivery day,Hour 1,Hour 2,Hour 3,...,Hour 24
            01/01/2024,45.2,52.1,38.7,...,42.3
            02/01/2024,47.5,49.8,41.2,...,45.1
            ...
            ```
            
            **After Preprocessing:**
            ```
            Date,Hour,Price,Settlement
            2024-01-01,Hour 1,45.2,30.5
            2024-01-01,Hour 2,52.1,30.5
            2024-01-01,Hour 3,38.7,30.5
            ...
            ```
            
            **Time Intervals Supported:**
            - 15 minutes: Profit = (hourly_profit × 15/60)
            - 30 minutes: Profit = (hourly_profit × 30/60)  
            - 60 minutes: Standard hourly calculation
            
            **Note:** Gas prices are automatically loaded from the system database and merged with your data based on the delivery date.
            """)

if __name__ == "__main__":
    main()