import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tempfile
from upsell_and_electrolyse import preprocess_for_backtest, run_trading_strategy, monthly_performance_summary

st.set_page_config(page_title="Electrolyser Trading Strategy", layout="wide")

def create_streamlit_interface():
    st.title("Electrolyser Trading Strategy Backtest")
    st.markdown("Configure parameters and run backtests to evaluate the electrolyser trading strategy.")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")

        # Year selection
        st.subheader("Data Source")
        year_options = [2023, 2024, 2025, "All Years"]
        selected_year = st.selectbox(
            "Select Year for Backtest",
            options=year_options,
            index=0,
            help="Choose which year's data to use for backtesting"
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
            help="Electrolyser efficiency parameter (typically 0.6-0.8). This determines gas generation: 1 MWh electricity → efficiency × 1 MWh gas",
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
            value=10.175,
            step=0.025,
            help="Green certificate price in €/MWh",
        )

        st.divider()

        st.subheader("Trading Thresholds")
        buy_threshold = st.number_input(
            "Buy Threshold (%)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            format="%.1f",
            help="Price must be this percentage below threshold to enter position"
        )
        
        sell_threshold = st.number_input(
            "Sell Threshold (%)",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            format="%.1f",
            help="Price must be this percentage above entry to exit position"
        )
        
        st.divider()
        
        # Enable/disable logging
        log_to_file = st.checkbox("Log to File", value=True)

    # Load the data based on selection
    data_loaded = False
    if st.button("Load Data", type="primary", use_container_width=True):
        with st.spinner(f"Loading data for {selected_year}..."):
            try:
                if selected_year == 2023:
                    data = preprocess_for_backtest(2023)
                    st.session_state.data = data
                    st.write(f"Loaded data for 2023: {len(data):,} records")
                    data_loaded = True
                elif selected_year == 2024:
                    data = preprocess_for_backtest(2024)
                    st.session_state.data = data
                    st.write(f"Loaded data for 2024: {len(data):,} records")
                    data_loaded = True
                elif selected_year == 2025:
                    data = preprocess_for_backtest(2025)
                    st.session_state.data = data
                    st.write(f"Loaded data for 2025: {len(data):,} records")
                    data_loaded = True
                else:  # All years
                    data_2023, data_2024, data_2025 = preprocess_for_backtest()
                    data = pd.concat([data_2023, data_2024, data_2025])
                    st.session_state.data = data
                    st.write(f"Loaded data for all years: {len(data):,} records")
                    data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
    
    # If data was just loaded or already in session state
    if data_loaded or 'data' in st.session_state:
        data = st.session_state.data
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(5))
        
        # Date filtering
        st.header("Date Range Selection")
        
        try:
            # Get min and max dates
            min_date = data['DeliveryStart'].min().date()
            max_date = data['DeliveryStart'].max().date()
            
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
                # Filter data by date range
                date_mask = (data['DeliveryStart'].dt.date >= start_date) & (data['DeliveryStart'].dt.date <= end_date)
                filtered_data = data[date_mask]
                
                st.write(f"Selected {len(filtered_data):,} records from {start_date} to {end_date}")
            else:
                st.error("End date must be after start date!")
                filtered_data = data
        except Exception as e:
            st.warning(f"Date filtering failed: {str(e)}")
            filtered_data = data
        
        # Run backtest button
        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                try:
                    # Group data by delivery start
                    delivery_groups = filtered_data.groupby('DeliveryStart')
                    delivery_groups_list = list(delivery_groups)
                    
                    # Run the trading strategy
                    results, trades_df, log_path = run_trading_strategy(
                        delivery_groups=delivery_groups_list,
                        efficiency_parameter=efficiency_parameter,
                        certificates=certificates,
                        log_to_file=log_to_file,
                        buy_threshold_pct=buy_threshold,
                        sell_threshold_pct=sell_threshold
                    )
                    
                    # Display results
                    st.header("Trading Results")
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trading PnL", f"€{results['total_pnl']:.2f}")
                    with col2:
                        st.metric("Electrolyser Profits", f"€{results['electrolyser_profits']:.2f}")
                    with col3:
                        st.metric("Combined Profit", f"€{results['combined_profit']:.2f}")
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trade Count", results['trade_count'])
                    with col2:
                        st.metric("Electrolyser Count", results['electrolyser_count'])
                    with col3:
                        st.metric("Avg Hold Time", f"{results['avg_hold_time']}")
                    
                    # Monthly performance
                    if len(trades_df) > 0:
                        st.header("Monthly Performance")
                        monthly_data = monthly_performance_summary(trades_df)
                        
                        # Display monthly data as a table
                        st.dataframe(monthly_data[['month_str', 'trade', 'electrolyser', 'total', 'electrolyser_count']])
                        
                        # Link to log file if logging is enabled
                        if log_to_file and log_path:
                            with open(log_path, 'r') as file:
                                log_content = file.read()
                            
                            st.download_button(
                                "Download Log File", 
                                data=log_content, 
                                file_name=os.path.basename(log_path),
                                mime="text/plain",
                                use_container_width=True
                            )
                            
                        # Option to download trade data
                        if len(trades_df) > 0:
                            csv = trades_df.to_csv(index=False)
                            st.download_button(
                                "Download Trade Data CSV",
                                data=csv,
                                file_name=f"trades_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Click 'Load Data' to begin the analysis")

if __name__ == "__main__":
    create_streamlit_interface()