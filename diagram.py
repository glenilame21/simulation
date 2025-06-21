from datetime import datetime, date
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from logic import (
    load_gas_prices,
    preprocess_with_gas_prices,
    test_google_sheets_connection,
    parse_date_column,
    spot,
    calculate_advanced_metrics,
    create_enhanced_visualization,
    generate_pdf_report
)

st.set_page_config(page_title="Electrolyser Dashboard", page_icon="⚡", layout="wide")


gas_prices_df = None

# Load default gas prices on startup
load_gas_prices("Month Ahead")


def main():

    st.title("Electrolyser Trading Simulation")
    st.markdown("Upload your data and configure parameters.")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")

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
        gas_loaded = load_gas_prices(gas_file_option)

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
            help="Upload a CSV file containing a 'Delivery day' column and hourly price data or standard price/settlement data",
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
            df_raw = pd.read_csv(uploaded_file)

            # Check if we have 'Delivery day' column - this determines preprocessing path
            has_delivery_day = "Delivery day" in df_raw.columns

            if has_delivery_day:
                with st.expander("Debug Information", expanded=False):
                    st.write("**Raw data preview:**")
                    st.dataframe(df_raw.tail())
                    st.write("**Delivery day column sample:**")
                    st.write(df_raw["Delivery day"].head(10).tolist())

                # Determine preprocessing parameters based on threshold option
                if threshold_option == "Calculate from Gas Price":
                    with st.spinner(
                        f"Loading and preprocessing data with {gas_file_option.lower()} gas prices..."
                    ):
                        df = preprocess_with_gas_prices(
                            df_raw, use_manual_threshold=False
                        )

                        if df.empty:
                            st.error(
                                "Data preprocessing failed. Please check the debug information and ensure your data format is correct."
                            )
                            return

                        st.success(
                            "Data preprocessed and merged with gas prices successfully!"
                        )
                else:  # Manual threshold
                    with st.spinner("Preprocessing data with manual threshold..."):
                        df = preprocess_with_gas_prices(
                            df_raw,
                            use_manual_threshold=True,
                            manual_threshold_value=manual_threshold,
                        )

                        if df.empty:
                            st.error(
                                "Data preprocessing failed. Please check the debug information and ensure your data format is correct."
                            )
                            return

                        st.success(
                            "Data preprocessed with manual threshold successfully!"
                        )

                with st.expander("Preprocessing Results", expanded=True):
                    st.write(f"**Original data shape:** {df_raw.shape}")
                    st.write(f"**Processed data shape:** {df.shape}")
                    st.write(f"**Available columns:** {', '.join(df.columns.tolist())}")

                    if len(df) == 0:
                        st.error("No data remaining after preprocessing.")
                        return

                    st.write("**Processed data preview:**")
                    st.dataframe(df.tail(10))

                # Automatically detect Price and Settlement columns if they exist
                price_default_idx = 0
                settlement_default_idx = 0

                if "Price" in df.columns:
                    price_default_idx = df.columns.tolist().index("Price")
                if "Settlement" in df.columns:
                    settlement_default_idx = df.columns.tolist().index("Settlement")

            else:
                # Use standard processing for data without 'Delivery day'
                df = df_raw.copy()

                # Smart column detection for any data format
                price_default_idx = 0
                settlement_default_idx = 0

                # Look for common price column names
                price_keywords = [
                    "price",
                    "electricity",
                    "power",
                    "energy",
                    "spot",
                    "dam",
                ]
                settlement_keywords = ["settlement", "gas", "fuel", "cost"]

                for i, col in enumerate(df.columns):
                    col_lower = col.lower()
                    # Skip 'Delivery day' for price selection
                    if "delivery" in col_lower or "date" in col_lower:
                        continue
                    # Look for price columns
                    if any(keyword in col_lower for keyword in price_keywords):
                        price_default_idx = i
                        break

                for i, col in enumerate(df.columns):
                    col_lower = col.lower()
                    # Skip 'Delivery day' for settlement selection
                    if "delivery" in col_lower or "date" in col_lower:
                        continue
                    # Look for settlement/gas columns
                    if any(keyword in col_lower for keyword in settlement_keywords):
                        settlement_default_idx = i
                        break

                # If no keywords found, default to first non-date column
                if price_default_idx == 0:
                    for i, col in enumerate(df.columns):
                        col_lower = col.lower()
                        if "delivery" not in col_lower and "date" not in col_lower:
                            price_default_idx = i
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
                        index=0,
                    )
                    if settlement_column == "None":
                        settlement_column = None

            with col3:
                date_column = st.selectbox(
                    "Date/Time Column (optional)",
                    options=["None"] + df.columns.tolist(),
                    index=0,
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

            # Date filtering - handle both preprocessed and standard data
            if date_column != "None" and not has_delivery_day:
                st.header("Date Range Selection")

                parsed_dates = parse_date_column(processed_df, date_column)
                if parsed_dates is not None:
                    processed_df["parsed_date"] = parsed_dates

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
                        date_mask = (
                            processed_df["parsed_date"].dt.date >= start_date
                        ) & (processed_df["parsed_date"].dt.date <= end_date)
                        processed_df = processed_df[date_mask]
                        st.success(
                            f"Filtered to {len(processed_df):,} rows from {start_date} to {end_date}"
                        )
                    else:
                        st.error("End date must be after start date!")
                        return
            elif has_delivery_day:
                # Date filtering for preprocessed data
                st.header("Date Range Selection")

                # Ensure the index is datetime and handle potential errors
                try:
                    if hasattr(processed_df.index, "date"):
                        min_date = processed_df.index.min().date()
                        max_date = processed_df.index.max().date()
                    else:
                        # If index is not datetime, convert it
                        processed_df.index = pd.to_datetime(processed_df.index)
                        min_date = processed_df.index.min().date()
                        max_date = processed_df.index.max().date()

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
                        # Convert dates to datetime for comparison
                        start_datetime = pd.to_datetime(start_date)
                        end_datetime = (
                            pd.to_datetime(end_date)
                            + pd.Timedelta(days=1)
                            - pd.Timedelta(seconds=1)
                        )

                        date_mask = (processed_df.index >= start_datetime) & (
                            processed_df.index <= end_datetime
                        )
                        original_length = len(processed_df)
                        processed_df = processed_df[date_mask]

                        if len(processed_df) < original_length:
                            st.success(
                                f"Date filter applied: {len(processed_df):,} rows selected from {original_length:,} total rows"
                            )
                        else:
                            st.info(
                                "No filtering applied - selected range includes all data"
                            )
                    else:
                        st.error("End date must be after start date!")
                        return

                except Exception as date_error:
                    st.warning(
                        f"Date filtering skipped due to date format issues: {str(date_error)}"
                    )
                    st.info(
                        "Proceeding without date filtering. You can still run the simulation with all available data."
                    )

            # Display data overview
            with st.expander("Data Overview", expanded=True):
                st.write(f"**Data shape:** {processed_df.shape}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Records", len(processed_df))
                    if pd.api.types.is_numeric_dtype(processed_df[price_column]):
                        st.write(f"**{price_column}:**")
                        st.write(
                            f"• Range: €{processed_df[price_column].min():.2f} - €{processed_df[price_column].max():.2f}"
                        )
                        st.write(f"• Average: €{processed_df[price_column].mean():.2f}")

                with col2:
                    equivalent_hours = len(processed_df) * time_interval / 60
                    st.metric("Equivalent Hours", f"{equivalent_hours:.1f}")

                    if settlement_column and pd.api.types.is_numeric_dtype(
                        processed_df[settlement_column]
                    ):
                        st.write(f"**{settlement_column}:**")
                        st.write(
                            f"• Range: €{processed_df[settlement_column].min():.2f} - €{processed_df[settlement_column].max():.2f}"
                        )
                        st.write(
                            f"• Average: €{processed_df[settlement_column].mean():.2f}"
                        )

                with col3:
                    missing_values = processed_df.isnull().sum().sum()
                    st.metric("Missing Values", missing_values)

                st.dataframe(processed_df.head(), use_container_width=True)

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

                # Calculate advanced metrics
                metrics = calculate_advanced_metrics(
                    result_df, time_interval, power_energy
                )

                # Display results
                st.header("Simulation Results")

                # Key metrics (first row)
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

                # Additional metrics (second row)
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

                    # Add gas generation insights
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

                    # Show basic price chart
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

                # Add the PDF download button right here inside the if block
                if operating_hours > 0:
                    # Create a dictionary with simulation parameters for the PDF report
                    simulation_params = {
                        'efficiency_parameter': efficiency_parameter,
                        'power_energy': power_energy,
                        'certificates': certificates,
                        'time_interval': time_interval,
                        'threshold_method': 'manual' if threshold_option == "Set Manual Threshold" else 'gas_based',
                        'manual_threshold': manual_threshold if threshold_option == "Set Manual Threshold" else None
                    }
                
                # Generate the PDF report
                with st.spinner("Generating PDF report..."):
                    pdf_report = generate_pdf_report(
                        result_df, 
                        metrics, 
                        simulation_params,
                        price_column,
                        fig  # Pass the visualization figure
                    )
                    
                    # Create a download button for the PDF
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
            # Show the full error traceback for debugging
            import traceback

            with st.expander("Full Error Details", expanded=False):
                st.code(traceback.format_exc())

    else:
        st.info("Upload a CSV file to get started.")


if __name__ == "__main__":
    main()