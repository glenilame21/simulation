import pandas as pd
import streamlit as st

def spot(
    df,
    price_col,
    settlement_col,
    efficiency_parameter,
    certificates,
    time_interval_minutes,
    power_energy=1.0,
    manual_threshold=None,
):
    """
    Here the idea is that we buy electricity when the threshold is hit so that we send it to the electrolyser
    Ideally this works best with negative power prices where you get paid for getting energy
    """
    result_df = df.copy()

    # Here we initialize the columns we will plot as results in the end
    result_df["operate"] = (
        False  # a yes or no variable that will count how many times the threshold was hit and therefore the electolyser was used
    )
    result_df["profit"] = 0.0  # easy to understand, a column for profit
    result_df["buy_threshold"] = None
    result_df["MegaWatt"] = 0.0  # Traders also want to see the amount of MW/h generated
    result_df["profit_per_mw"] = 0.0  # idk why this is here
    result_df["gas_generated_mwh"] = 0.0  # Gas generated in MWh

    # this is an input field from the user - I can maybe make this a constant so that based on the product being traded the time_interval_minutes is populated
    time_factor = (
        time_interval_minutes / 60.0
    )  # when we sell a 15-minute product in intraday the profit needs to be devided by 0.25
    missing_data = 0

    if price_col in result_df.columns:
        price_series = result_df[price_col]
    elif "Price" in result_df.columns:
        price_series = result_df["Price"]
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
                elif "Settlement" in result_df.columns:
                    gas_val = result_df["Settlement"].iloc[i]
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

            result_df.iloc[i, result_df.columns.get_loc("buy_threshold")] = (
                buy_threshold
            )

            if el_profit > 0:
                result_df.iloc[i, result_df.columns.get_loc("operate")] = True
                result_df.iloc[i, result_df.columns.get_loc("profit")] = (
                    el_profit * time_factor * power_energy
                )
                result_df.iloc[i, result_df.columns.get_loc("profit_per_mw")] = (
                    el_profit * time_factor
                )
                result_df.iloc[i, result_df.columns.get_loc("MegaWatt")] = power_energy

                # Calculate gas generation: electricity input * efficiency
                electricity_consumed_mwh = (
                    power_energy * time_factor
                )  # MWh of electricity consumed
                gas_generated_mwh = (
                    electricity_consumed_mwh * efficiency_parameter
                )  # MWh of gas generated

                result_df.iloc[i, result_df.columns.get_loc("gas_generated_mwh")] = (
                    gas_generated_mwh
                )

        except Exception as e:
            missing_data += 1
            if st.checkbox("Show detailed error information", value=False):
                st.error(f"Error processing row {i}: {str(e)}")

    if missing_data > 0:
        st.warning(f"Missing or invalid data for {missing_data} rows")

    total_operating_hours = result_df["operate"].sum() * time_factor
    total_profit = result_df["profit"].sum()

    operating_count = result_df["operate"].sum()
    st.info(
        f"Operating in {operating_count} periods out of {len(result_df)} total periods"
    )

    return result_df, total_operating_hours, total_profit