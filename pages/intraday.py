import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from logic import (
    load_gas_prices,
    process_data,
    parse_date_column
)

from vizualization import(
    calculate_advanced_metrics,
    create_enhanced_visualization,
    generate_pdf_report
)

st.set_page_config(page_title="Intraday Trading Strategy", page_icon="📊", layout="wide")

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
    result_df['operation'] = 0
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
                result_df.at[current_idx, 'operation'] = 1
                result_df.at[current_idx, 'trade_type'] = 'BUY'
            
            # If we have an entry position, check if we can exit
            elif entry_price is not None:
                if next_price > entry_price * jump_factor:
                    # Sell with profit
                    trade_pnl = (next_price - entry_price) * power_capacity * hour_factor
                    result_df.at[next_idx, 'operation'] = 1
                    result_df.at[next_idx, 'profit'] = trade_pnl
                    result_df.at[next_idx, 'trade_type'] = 'SELL'
                    total_trading_pnl += trade_pnl
                    
                    # Reset entry position
                    entry_price = None
                    entry_index = None
    
    # Process any remaining open positions at the end - send to electrolyser
    for group_key, group in delivery_groups:
        for i, row in group.iterrows():
            if row['operation'] == 1 and row['trade_type'] == 'BUY' and row['profit'] == 0:
                # Calculate electrolyser profit for this position
                entry_price = row[price_column]
                el_profit = (efficiency_parameter * (certificates + entry_price) - entry_price) * power_capacity * hour_factor
                
                result_df.at[i, 'profit'] = el_profit
                result_df.at[i, 'trade_type'] = 'ELECTROLYSER'
                total_electrolyser_profits += el_profit
    
    # Calculate total metrics
    total_profit = total_trading_pnl + total_electrolyser_profits
    operating_hours = result_df['operation'].sum() * (time_interval / 60)
    
    # Add summary columns
    result_df['trading_pnl'] = 0
    result_df['electrolyser_profit'] = 0
    
    # Fill in the specific profit columns
    result_df.loc[result_df['trade_type'] == 'SELL', 'trading_pnl'] = result_df.loc[result_df['trade_type'] == 'SELL', 'profit']
    result_df.loc[result_df['trade_type'] == 'ELECTROLYSER', 'electrolyser_profit'] = result_df.loc[result_df['trade_type'] == 'ELECTROLYSER', 'profit']
    
    return result_df, operating_hours, total_profit, total_trading_pnl, total_electrolyser_profits

# Main UI
st.title("Intraday Trading Strategy")
st.markdown("""
This strategy buys electricity when the price falls below a threshold and either:
1. Sells when price rises by a specified percentage
2. Uses the electricity in the electrolyser if no selling opportunity arises
""")

# If gas_prices_df doesn't exist in session state, initialize it
if 'gas_prices_df' not in st.session_state:
    st.session_state.gas_prices_df = None

# Function to load gas prices and store in session state
def load_and_store_gas_prices(gas_file_option):
    """Load gas prices and store in session state"""
    success = load_gas_prices(gas_file_option)
    if success:
        # Get the global gas_prices_df from the logic module
        from logic import gas_prices_df
        st.session_state.gas_prices_df = gas_prices_df.copy() if gas_prices_df is not None else None
    return success

# Initialize gas prices if not already loaded
if st.session_state.gas_prices_df is None:
    with st.spinner("Loading default gas prices..."):
        load_and_store_gas_prices("Month Ahead")

# Sidebar for parameters
with st.sidebar:
    st.header("Configuration")
    
    # Strategy selection
    st.subheader("Strategy Parameters")
    price_jump_percentage = st.slider(
        "Price Jump % (for selling)",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Percentage price increase needed to sell position",
    )

    # Gas file selection
    st.subheader("Gas Price Source")
    gas_file_option = st.selectbox(
        "Select Gas Price File:",
        options=["Month Ahead", "Spot"],
        index=0,
        help="Choose which gas price dataset to use for calculations",
    )

# Load gas prices based on selection
with st.spinner(f"Loading {gas_file_option.lower()} gas prices from Google Sheets..."):
    gas_loaded = load_and_store_gas_prices(gas_file_option)

if not gas_loaded:
    st.error("Cannot proceed without gas prices. Please check your internet connection.")
    st.info("Make sure the Google Sheets are publicly accessible and the URLs are correct.")
    st.stop()

with st.sidebar:
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV Data",
        type=["csv"],
        help="Upload a CSV file with electricity price data.",
    )

    st.divider()

    st.subheader("Parameters")

    efficiency_parameter = st.number_input(
        "Efficiency Parameter",
        min_value=0.00,
        max_value=1.00,
        value=0.708,
        step=0.001,
        format="%.3f",
        help="Electrolyser efficiency parameter (typically 0.6-0.8)",
    )

    power_energy = st.number_input(
        "Power Capacity (MW)",
        min_value=0.1,
        max_value=9999.99,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Electrolyser power capacity in MW",
    )

    certificates = st.number_input(
        "Green Certificates (€/MWh)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="Green certificate price in €/MWh",
    )

    st.divider()

    st.subheader("Operating Threshold")
    threshold_option = st.radio(
        "Threshold Method:",
        options=["Calculate from Gas Price", "Set Manual Threshold"],
        index=0,
        help="Choose how to determine when to operate",
    )

    manual_threshold = None
    if threshold_option == "Set Manual Threshold":
        manual_threshold = st.number_input(
            "Manual Threshold (€/MWh)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Fixed price threshold for operation",
        )

    st.divider()

    st.subheader("Time Resolution")
    time_interval = st.selectbox(
        "Data Time Interval",
        options=[15, 30, 60],
        index=2,
        help="Time interval of your data in minutes",
    )

# Process the uploaded data
if uploaded_file is not None:
    try:
        # First load the raw data
        df_raw = pd.read_csv(uploaded_file)
        
        # Process data using the unified function
        with st.spinner("Processing data..."):
            # Determine gas source based on threshold option
            if threshold_option == "Calculate from Gas Price":
                gas_source = "api"
            else:
                gas_source = "manual"
            
            # Process data with the consolidated function
            df = process_data(
                df_raw,
                gas_source=gas_source,
                manual_threshold_value=manual_threshold
            )
            
            if not df.empty:
                if gas_source == "api":
                    st.success(f"Data processed and merged with {gas_file_option.lower()} gas prices successfully!")
                else:
                    st.success("Data processed with manual threshold successfully!")
            else:
                st.error("Processing returned empty dataset. Please check your data format.")
                st.stop()
            
        # Determine default column indices for selection
        price_keywords = ["price", "electricity", "power", "energy", "spot", "low", "high"]
        settlement_keywords = ["settlement", "gas", "fuel", "cost"]

        price_default_idx = 0
        settlement_default_idx = 0

        # Look for price columns
        for i, col in enumerate(df.columns):
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in price_keywords):
                price_default_idx = i
                break

        # Look for settlement columns
        for i, col in enumerate(df.columns):
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in settlement_keywords):
                settlement_default_idx = i
                break

        # Column selection section
        st.header("Column Selection")

        col1, col2, col3 = st.columns(3)

        with col1:
            price_column = st.selectbox(
                "Electricity Price Column",
                options=df.columns.tolist(),
                index=price_default_idx,
            )

        with col2:
            if threshold_option == "Calculate from Gas Price":
                settlement_column = st.selectbox(
                    "Gas Price Column",
                    options=df.columns.tolist(),
                    index=settlement_default_idx,
                )
            else:
                settlement_column = st.selectbox(
                    "Gas Price Column (optional)",
                    options=["None"] + df.columns.tolist(),
                    index=settlement_default_idx + 1 if settlement_default_idx >= 0 else 0,
                )
                if settlement_column == "None":
                    settlement_column = None

        with col3:
            # For date filtering, detect if we have a datetime index or DeliveryStart column
            if "DeliveryStart" in df.columns:
                date_column_options = ["DeliveryStart"] + [col for col in df.columns.tolist() if col != "DeliveryStart"]
                date_default_idx = 0
            else:
                date_column_options = ["None"] + df.columns.tolist()
                date_default_idx = 0

            date_column = st.selectbox(
                "Date/Time Column (optional)",
                options=date_column_options,
                index=date_default_idx,
            )

        # Validate required columns
        if threshold_option == "Calculate from Gas Price" and settlement_column is None:
            st.error("Gas price column is required when calculating threshold from gas price!")
            st.stop()

        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.sample(n=10))

        # Data filtering
        processed_df = df.copy()

        # Date filtering
        if date_column and date_column != "None":
            st.header("Date Range Selection")

            try:
                if date_column == "DeliveryStart":
                    # Use DeliveryStart column
                    processed_df[date_column] = pd.to_datetime(processed_df[date_column])
                    min_date = processed_df[date_column].min().date()
                    max_date = processed_df[date_column].max().date()
                    date_values = processed_df[date_column]
                elif hasattr(processed_df.index, "date"):
                    # Use index for date filtering
                    min_date = processed_df.index.min().date()
                    max_date = processed_df.index.max().date()
                    date_values = processed_df.index
                else:
                    # Parse the selected date column
                    parsed_dates = parse_date_column(processed_df, date_column)
                    if parsed_dates is not None:
                        processed_df["parsed_date"] = parsed_dates
                        min_date = processed_df["parsed_date"].min().date()
                        max_date = processed_df["parsed_date"].max().date()
                        date_values = processed_df["parsed_date"]
                    else:
                        st.warning("Could not parse the selected date column")
                        date_values = None

                if date_values is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                        )
                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                        )

                    if start_date <= end_date:
                        if date_column == "DeliveryStart":
                            date_mask = (date_values.dt.date >= start_date) & (date_values.dt.date <= end_date)
                        elif hasattr(processed_df.index, "date"):
                            date_mask = (date_values.date >= start_date) & (date_values.date <= end_date)
                        else:
                            date_mask = (date_values.dt.date >= start_date) & (date_values.dt.date <= end_date)
                        
                        original_length = len(processed_df)
                        processed_df = processed_df[date_mask]
                        
                        if len(processed_df) < original_length:
                            st.success(
                                f"Date filter applied: {len(processed_df):,} rows selected from {original_length:,} total rows"
                            )
                    else:
                        st.error("End date must be after start date!")
                        st.stop()
            except Exception as e:
                st.warning(f"Date filtering failed: {str(e)}")

        # Validate numeric columns
        if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
            st.error(f"Price column '{price_column}' is not numeric!")
            st.stop()

        if settlement_column and settlement_column in processed_df.columns and not pd.api.types.is_numeric_dtype(processed_df[settlement_column]):
            st.error(f"Settlement column '{settlement_column}' is not numeric!")
            st.stop()

        # Strategy explanation
        st.header("Strategy Details")
        st.markdown(f"""
        ### Intraday Trading Strategy Logic
        
        This strategy follows these rules:
        
        1. **Buy Position**: Enter when price is below 90% of the threshold
        2. **Sell Position**: Exit when price rises by {price_jump_percentage}% from entry point
        3. **Electrolyser Usage**: If price doesn't rise enough to trigger a sell, use the electricity in the electrolyser
        
        The threshold is {"calculated from gas prices" if threshold_option == "Calculate from Gas Price" else f"manually set to €{manual_threshold}/MWh"}.
        """)

        # Run simulation
        if st.button("Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running intraday trading simulation..."):
                result_df, operating_hours, total_profit, trading_pnl, electrolyser_profits = intraday_strategy(
                    processed_df,
                    price_column,
                    settlement_column,
                    efficiency_parameter,
                    certificates,
                    time_interval,
                    power_energy,
                    manual_threshold,
                    price_jump_percentage
                )

            metrics = calculate_advanced_metrics(result_df, time_interval, power_energy)

            # Add intraday-specific metrics
            metrics['trading_pnl'] = trading_pnl
            metrics['electrolyser_profits'] = electrolyser_profits

            # Display results
            st.header("Simulation Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Profit", f"€{total_profit:,.2f}")

            with col2:
                st.metric(
                    "Energy Consumed",
                    f"{metrics['total_energy_consumed_mwh']:.1f} MWh",
                )

            with col3:
                st.metric(
                    "Operating Hours",
                    f"{operating_hours:.1f} hours",
                )

            col4, col5 = st.columns(2)
            with col4:
                st.metric("Trading PnL", f"€{trading_pnl:,.2f}")
            with col5:
                st.metric("Electrolyser Profits", f"€{electrolyser_profits:,.2f}")

            # Visualization
            st.header("Analysis Dashboard")

            if operating_hours > 0:
                fig = create_enhanced_visualization(
                    result_df, price_column, time_interval
                )
                
                # Add specific traces for intraday strategy
                buy_points = result_df[result_df['trade_type'] == 'BUY']
                sell_points = result_df[result_df['trade_type'] == 'SELL']
                electrolyser_points = result_df[result_df['trade_type'] == 'ELECTROLYSER']
                
                if len(buy_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_points.index,
                        y=buy_points[price_column],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='circle'),
                        name='Buy Points'
                    ))
                
                if len(sell_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_points.index,
                        y=sell_points[price_column],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='circle'),
                        name='Sell Points'
                    ))
                
                if len(electrolyser_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=electrolyser_points.index,
                        y=electrolyser_points[price_column],
                        mode='markers',
                        marker=dict(color='blue', size=10, symbol='star'),
                        name='Electrolyser Usage'
                    ))
                
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.subheader("Key Insights")

                insights_col1, insights_col2 = st.columns(2)

                with insights_col1:
                    st.info(
                        f"**Strategy Performance:**\n- Trading profit: €{trading_pnl:,.2f}\n- Electrolyser profit: €{electrolyser_profits:,.2f}\n- Total energy consumed: {metrics['total_energy_consumed_mwh']:.1f} MWh"
                    )

                with insights_col2:
                    if "avg_operating_price" in metrics:
                        st.success(
                            f"**Price Analysis:**\n- Average operating price: €{metrics['avg_operating_price']:.2f}/MWh\n- Price range during operation: €{metrics['min_operating_price']:.2f} - €{metrics['max_operating_price']:.2f}/MWh"
                        )

                    if time_interval != 60:
                        st.info(
                            f"**Time Resolution:**\nProfit adjusted for {time_interval}-minute intervals ({time_interval/60:.2f}x hourly rate)"
                        )

                # Trade statistics
                st.subheader("Intraday Trading Details")
                st.info(
                    f"""
                    **Trading Strategy:**
                    - Buy when price falls below {90}% of threshold
                    - Sell when price rises by {price_jump_percentage}%
                    - Send to electrolyser if no selling opportunity
                    """
                )
                
                # Calculate trade statistics
                buy_count = len(result_df[result_df['trade_type'] == 'BUY'])
                sell_count = len(result_df[result_df['trade_type'] == 'SELL'])
                electrolyser_count = len(result_df[result_df['trade_type'] == 'ELECTROLYSER'])
                
                trade_stats_col1, trade_stats_col2 = st.columns(2)
                with trade_stats_col1:
                    st.success(
                        f"""
                        **Trade Counts:**
                        - Buy positions: {buy_count}
                        - Sell positions: {sell_count}
                        - Electrolyser use: {electrolyser_count}
                        """
                    )
                
                with trade_stats_col2:
                    success_rate = sell_count / buy_count * 100 if buy_count > 0 else 0
                    avg_trading_profit = trading_pnl / sell_count if sell_count > 0 else 0
                    avg_electrolyser_profit = electrolyser_profits / electrolyser_count if electrolyser_count > 0 else 0
                    
                    st.success(
                        f"""
                        **Performance Metrics:**
                        - Trading success rate: {success_rate:.1f}%
                        - Avg. trading profit: €{avg_trading_profit:.2f} per trade
                        - Avg. electrolyser profit: €{avg_electrolyser_profit:.2f} per use
                        """
                    )

                # Generate PDF report
                simulation_params = {
                    'strategy': "Intraday Trading",
                    'efficiency_parameter': efficiency_parameter,
                    'power_energy': power_energy,
                    'certificates': certificates,
                    'time_interval': time_interval,
                    'threshold_method': 'manual' if threshold_option == "Set Manual Threshold" else 'gas_based',
                    'manual_threshold': manual_threshold if threshold_option == "Set Manual Threshold" else None,
                    'data_format': 'auto_detected',
                    'price_jump_percentage': price_jump_percentage
                }

                dataset_metadata = {
                    'dataset_name': uploaded_file.name if uploaded_file else 'Unknown Dataset',
                    'time_resolution': time_interval,
                    'gas_price_source': gas_file_option if threshold_option == "Calculate from Gas Price" else "Manual",
                }
                
                if threshold_option == "Set Manual Threshold":
                    dataset_metadata['manual_threshold_value'] = manual_threshold
                
                if date_column and date_column != "None" and 'start_date' in locals() and 'end_date' in locals():
                    dataset_metadata['date_range_start'] = start_date
                    dataset_metadata['date_range_end'] = end_date

                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_report = generate_pdf_report(
                            result_df, 
                            metrics, 
                            simulation_params,
                            price_column,
                            fig,
                            dataset_metadata
                        )
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_report,
                            file_name=f"intraday_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="pdf_download",
                            help="Download a PDF report with simulation results",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
            else:
                st.warning("No profitable operating periods found with current parameters.")
                st.info("Try adjusting the price jump percentage, efficiency parameter, or threshold value.")

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=result_df.index,
                        y=result_df[price_column],
                        mode="lines",
                        name="Electricity Price",
                        line=dict(color="blue"),
                    )
                )
                if "buy_threshold" in result_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=result_df.index,
                            y=result_df["buy_threshold"],
                            mode="lines",
                            name="Buy Threshold",
                            line=dict(color="red", dash="dash"),
                        )
                    )
                fig.update_layout(
                    title="Price vs Threshold Analysis",
                    xaxis_title="Time Period",
                    yaxis_title="Price (€/MWh)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted with numeric data in the selected columns.")
        import traceback

        with st.expander("Full Error Details", expanded=False):
            st.code(traceback.format_exc())

else:
    st.info("Please upload a CSV file to get started.")
    
    with st.expander("About this Strategy", expanded=True):
        st.markdown("""
        ## Intraday Trading + Electrolyser Strategy
        
        This strategy combines the benefits of intraday electricity trading with electrolyser operation:
        
        1. **Buy Low:** Enter positions when electricity prices are below the threshold
        2. **Sell High:** Exit with profit when prices increase by the specified percentage
        3. **Fallback to Electrolyser:** If price doesn't rise enough to trigger selling, use the electricity in the electrolyser
        
        ### Key Benefits
        
        - **Dual Revenue Streams:** Generate profits from both trading and hydrogen production
        - **Risk Mitigation:** Electrolyser provides a safety net for unprofitable trades
        - **Higher Utilization:** Achieve better utilization of available capacity
        
        ### Getting Started
        
        Upload a CSV file with electricity price data to test this strategy with your parameters.
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Electrolysis_of_Water.svg/640px-Electrolysis_of_Water.svg.png", caption="Electrolyser Process")