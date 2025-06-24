from datetime import datetime, date
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import tempfile
import os

from strategies import spot

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

st.set_page_config(page_title="Electrolyser Dashboard", layout="wide")


if 'gas_prices_df' not in st.session_state:
    st.session_state.gas_prices_df = None

def load_and_store_gas_prices(gas_file_option):
    """Load gas prices and store in session state"""
    success = load_gas_prices(gas_file_option)
    if success:
        # Get the global gas_prices_df from the logic module
        from logic import gas_prices_df
        st.session_state.gas_prices_df = gas_prices_df.copy() if gas_prices_df is not None else None
    return success


def main():

    st.title("Electrolyser Trading Simulation")
    st.markdown("Upload your data and configure parameters.")
    
    # Initialize gas prices if not already loaded
    if st.session_state.gas_prices_df is None:
        with st.spinner("Loading default gas prices..."):
            load_and_store_gas_prices("Month Ahead")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")

        # Data format selection - add this at the top of sidebar
        st.subheader("Data Format")
        data_format = st.radio(
            "Select your data format:",
            options=["Wide Format", "Long Format"],
            index=0,
            help="Wide Format: Data with 'DeliveryStart' column and hourly price columns (Hour 1, Hour 2, etc.)\nLong Format: Data with individual rows for each hour/period"
        )

        # Gas file selection first (before loading)
        st.subheader("Gas Price Source")
        gas_file_option = st.selectbox(
            "Select Gas Price File:",
            options=["Month Ahead", "Spot"],
            index=0,
            help="Choose which gas price dataset to use for calculations",
        )

    # Here We load gas prices based on selection
    with st.spinner(
        f"Loading {gas_file_option.lower()} gas prices from Google Sheets..."
    ):
        gas_loaded = load_and_store_gas_prices(gas_file_option)

    if not gas_loaded:
        st.error(
            "Cannot proceed without gas prices. Please check your internet connection and try again."
        )
        st.info(
            "Make sure the Google Sheets are publicly accessible and the URLs are correct."
        )
        return

    with st.sidebar:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Data",
            type=["csv"],
            help="Upload a CSV file. Format depends on your selection above:\n- Wide Format: 'DeliveryStart' column + hourly columns\n- Long Format: Individual rows for each time period",
        )

        st.divider()

        st.subheader("Parameters")

        efficiency_parameter = st.number_input(
            "Efficiency Parameter",
            min_value=0.00,
            max_value=1.00,
            value=0.700,
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
                    return
                
            # Determine default column indices for selection
            if data_format == "Wide Format":
                price_default_idx = 0
                settlement_default_idx = 0
                
                if "Price" in df.columns:
                    price_default_idx = df.columns.tolist().index("Price")
                if "Settlement" in df.columns:
                    settlement_default_idx = df.columns.tolist().index("Settlement")
            else:  # Long Format
                # For long format, set default column indices
                price_keywords = ["price", "electricity", "power", "energy", "spot"]
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

            # Column selection section (common for both formats)
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
                # For long format, we might want to use DeliveryStart as date column
                if data_format == "Long Format" and "DeliveryStart" in df.columns:
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
            if (
                threshold_option == "Calculate from Gas Price"
                and settlement_column is None
            ):
                st.error(
                    "Gas price column is required when calculating threshold from gas price!"
                )
                return

            # Data filtering and processing
            processed_df = df.copy()

            # Date filtering - handle both formats
            if date_column != "None" and date_column is not None:
                st.header("Date Range Selection")

                # For long format with DeliveryStart, handle datetime parsing
                if data_format == "Long Format" and date_column == "DeliveryStart":
                    try:
                        processed_df["parsed_date"] = pd.to_datetime(processed_df[date_column])
                    except:
                        st.error(f"Could not parse {date_column} as datetime. Please check your date format.")
                        return
                elif data_format == "Wide Format" and not processed_df.index.name:
                    # For wide format without datetime index
                    parsed_dates = parse_date_column(processed_df, date_column)
                    if parsed_dates is not None:
                        processed_df["parsed_date"] = parsed_dates
                else:
                    # Use index for wide format with datetime index
                    try:
                        if hasattr(processed_df.index, "date"):
                            processed_df["parsed_date"] = processed_df.index
                        else:
                            processed_df.index = pd.to_datetime(processed_df.index)
                            processed_df["parsed_date"] = processed_df.index
                    except Exception as date_error:
                        st.warning(f"Date filtering skipped due to date format issues: {str(date_error)}")
                        processed_df["parsed_date"] = None

                if "parsed_date" in processed_df.columns and processed_df["parsed_date"] is not None:
                    try:
                        min_date = processed_df["parsed_date"].min().date()
                        max_date = processed_df["parsed_date"].max().date()

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
                            if data_format == "Long Format":
                                # Filter by date for long format
                                date_mask = (
                                    processed_df["parsed_date"].dt.date >= start_date
                                ) & (processed_df["parsed_date"].dt.date <= end_date)
                            else:
                                # Filter by date for wide format
                                date_mask = (
                                    processed_df["parsed_date"].dt.date >= start_date
                                ) & (processed_df["parsed_date"].dt.date <= end_date)
                            
                            original_length = len(processed_df)
                            processed_df = processed_df[date_mask]
                            
                            if len(processed_df) < original_length:
                                st.success(
                                    f"Date filter applied: {len(processed_df):,} rows selected from {original_length:,} total rows"
                                )
                        else:
                            st.error("End date must be after start date!")
                            return
                    except Exception as e:
                        st.warning(f"Date filtering failed: {str(e)}")

            # Validate numeric columns
            if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
                st.error(f"Price column '{price_column}' is not numeric!")
                return

            if settlement_column and not pd.api.types.is_numeric_dtype(
                processed_df[settlement_column]
            ):
                st.error(f"Settlement column '{settlement_column}' is not numeric!")
                return

            # Run simulation
            if st.button("Run Simulation", type="primary", use_container_width=True):
                with st.spinner("Running simulation..."):
                    result_df, operating_hours, total_profit = spot(
                        processed_df,
                        price_column,
                        settlement_column,
                        efficiency_parameter,
                        certificates,
                        time_interval,
                        power_energy,
                        manual_threshold,
                    )

                
                metrics = calculate_advanced_metrics(
                    result_df, time_interval, power_energy
                )

                
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
                    if "total_gas_generated_mwh" in metrics:
                        st.metric(
                            "Gas Generated",
                            f"{metrics['total_gas_generated_mwh']:.1f} MWh",
                        )
                    else:
                        st.metric(
                            "Avg Profit/Period",
                            f"€{metrics['avg_profit_per_operating_period']:.2f}",
                        )

              
                col4, col5 = st.columns(2)

                with col4:
                    st.metric("Utilization Rate", f"{metrics['utilization_rate']:.1f}%")

                with col5:
                    if "gas_generation_rate_mwh_per_hour" in metrics:
                        st.metric(
                            "Gas Rate",
                            f"{metrics['gas_generation_rate_mwh_per_hour']:.2f} MWh/h",
                        )
                    else:
                        st.metric(
                            "Operating Periods", f"{metrics['operating_periods']:,}"
                        )

                # Visualization
                st.header("Analysis Dashboard")

                if operating_hours > 0:
                    fig = create_enhanced_visualization(
                        result_df, price_column, time_interval
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    st.subheader("Key Insights")

                    insights_col1, insights_col2 = st.columns(2)

                    with insights_col1:
                        st.info(
                            f"**Strategy Performance:**\n- System operates {metrics['utilization_rate']:.1f}% of the time\n- Generates €{metrics['profit_per_mwh']:.2f} profit per MWh consumed\n- Average profit of €{metrics['avg_profit_per_operating_period']:.2f} per operating period"
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

                    
                    if (
                        "total_gas_generated_mwh" in metrics
                        and metrics["total_gas_generated_mwh"] > 0
                    ):
                        st.subheader("Gas Generation Summary")

                        gas_col1, gas_col2 = st.columns(2)

                        with gas_col1:
                            st.info(
                                f"""**Production Summary:**
                                - Total gas generated: {metrics['total_gas_generated_mwh']:.1f} MWh
                                - Average generation rate: {metrics['gas_generation_rate_mwh_per_hour']:.2f} MWh/hour
                                - Efficiency factor used: {efficiency_parameter:.1%}"""
                                                            )

                        with gas_col2:
                            efficiency_check = (
                                metrics["total_gas_generated_mwh"]
                                / metrics["total_energy_consumed_mwh"]
                                if metrics["total_energy_consumed_mwh"] > 0
                                else 0
                            )
                            st.success(
                                f"""**Efficiency Verification:**
                                - Electricity consumed: {metrics['total_energy_consumed_mwh']:.1f} MWh
                                - Gas generated: {metrics['total_gas_generated_mwh']:.1f} MWh  
                                - Actual efficiency: {efficiency_check:.1%}
                                - Profit per MWh gas: €{metrics['profit_per_mwh_gas']:.2f}"""
                                                            )

                else:
                    st.warning(
                        "No profitable operating periods found with current parameters."
                    )
                    st.info(
                        "Try adjusting the efficiency parameter, certificate price, or threshold value."
                    )

                    
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

                
                if operating_hours > 0:
                    
                    simulation_params = {
                        'efficiency_parameter': efficiency_parameter,
                        'power_energy': power_energy,
                        'certificates': certificates,
                        'time_interval': time_interval,
                        'threshold_method': 'manual' if threshold_option == "Set Manual Threshold" else 'gas_based',
                        'manual_threshold': manual_threshold if threshold_option == "Set Manual Threshold" else None,
                        'data_format': data_format
                    }
                
                    
                    with st.spinner("Generating PDF report..."):
                        pdf_report = generate_pdf_report(
                            result_df, 
                            metrics, 
                            simulation_params,
                            price_column,
                            fig  
                        )
                        
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_report,
                            file_name=f"electrolyser_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="pdf_download",
                            help="Download a PDF report with simulation results",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info(
                "Please ensure your CSV file is properly formatted with numeric data in the selected columns."
            )
            import traceback

            with st.expander("Full Error Details", expanded=False):
                st.code(traceback.format_exc())

    else:
        st.info("Upload a CSV file to get started.")
        
        
        with st.expander("Data Format Guide", expanded=False):
            st.markdown("""
            **Wide Format:**
            - Must contain a 'DeliveryStart' column with dates
            - Hour columns: 'Hour 1', 'Hour 2', ..., 'Hour 24'
            - Each row represents one day
            
            **Long Format:**
            - Must contain a 'DeliveryStart' column with datetime values
            - Each row represents one time period
            - Columns for price, settlement data, etc.
            """)


if __name__ == "__main__":
    main()