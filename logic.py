import base64
from datetime import date, datetime
from io import BytesIO, StringIO
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

def load_gas_prices(gas_file_option="Month Ahead"):
    # by default the month ahead gas prices are loaded.
    # if you want to change the source you will need to create a google spreadsheet and insert the data.
    # You can viuew the url on how the format is expeted to be, just remove the /export?format=csv to view the url in a browser but make sure these parameters stay here to make the request to the correct url
    global gas_prices_df
    if gas_file_option == "Month Ahead":
            gas_url = "https://docs.google.com/spreadsheets/d/1scfhBGJNj1s7P2jJQseeR8zlaN26Dyekpm8gbE3pYzY/export?format=csv"
    else:
            gas_url = "https://docs.google.com/spreadsheets/d/12ESt-pVMbA7Q7YcnRfd8-fgqL8qb0l4vev8-uqwpLMY/export?format=csv"
            
    response = requests.get(gas_url, timeout=30)
    response.raise_for_status()

    csv_content = StringIO(response.text)
    gas_prices_df = pd.read_csv(csv_content)

    if gas_prices_df.empty:
        st.error("Downloaded gas prices file is empty")
        return False
    if "Date" not in gas_prices_df.columns:
        st.error(
            f"'Date' column not found in gas prices. Available columns: {list(gas_prices_df.columns)}")
        return False
    gas_prices_df = gas_prices_df.set_index("Date")
    gas_prices_df.index = pd.to_datetime(gas_prices_df.index)
    return True

def preprocess_with_gas_prices(df, use_manual_threshold=False, manual_threshold_value=None):
    global gas_prices_df

    processed_df = df.copy()
    processed_df = processed_df.set_index("DeliveryStart")
    processed_df.index.name = "Date"
    processed_df = processed_df.sort_index()

    processed_df.index = pd.to_datetime(processed_df.index, dayfirst=True)

    if "Date" in processed_df.columns:
        processed_df = processed_df.drop(columns=["Date"])

    processed_df = processed_df.reset_index() 


    if use_manual_threshold and manual_threshold_value is not None:
        merged = processed_df.copy()
        merged["Settlement"] = manual_threshold_value
        st.info(
            f"Using manual threshold of €{manual_threshold_value:.2f}/MWh as Settlement price for all periods"
        )
    else:
        try:
            merged = pd.merge(processed_df, gas_prices_df, how="inner", on="Date")
        except Exception as e:
            st.error(f"Merge error: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    if "Hour 3A" in merged.columns:
        merged = merged.rename(columns={"Hour 3A": "Hour 3"})

    hour_columns = [f"Hour {i}" for i in range(1, 25)]
    hour_columns = [col for col in hour_columns if col in merged.columns]

    if not hour_columns:
        st.warning("No hour columns found for reshaping. Using data as-is.")
        return merged

    merged_reset = merged.reset_index()

    reshaped_df = pd.melt(
        merged_reset,
        id_vars=["Date", "Settlement"],
        value_vars=hour_columns,
        var_name="Hour",
        value_name="Price",
    )

    reshaped_df["Hour_num"] = reshaped_df["Hour"].str.extract("(\d+)").astype(int)
    reshaped_df = reshaped_df.sort_values(["Date", "Hour_num"])
    reshaped_df = reshaped_df.drop("Hour_num", axis=1)
    reshaped_df = reshaped_df.set_index("Date")

    return reshaped_df

def add_settlement_to_long_format(df, gas_month_ahead_path=None, gas_spot_path=None, gas_threshold=None):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure DeliveryStart is datetime
    result_df['DeliveryStart'] = pd.to_datetime(result_df['DeliveryStart'])
    
    # Create date column from DeliveryStart for easier joining
    result_df['date'] = result_df['DeliveryStart'].dt.date
    
    # Load gas price data if paths are provided
    gas_month_ahead = None
    gas_spot = None
    
    if gas_month_ahead_path:
        gas_month_ahead = pd.read_csv(gas_month_ahead_path)
        gas_month_ahead['date'] = pd.to_datetime(gas_month_ahead['date']).dt.date
        gas_month_ahead = gas_month_ahead[['date', 'price']].rename(columns={'price': 'month_ahead_price'})
    
    if gas_spot_path:
        gas_spot = pd.read_csv(gas_spot_path)
        gas_spot['date'] = pd.to_datetime(gas_spot['date']).dt.date
        gas_spot = gas_spot[['date', 'price']].rename(columns={'price': 'spot_price'})
    
    # Join gas price data to the result dataframe
    if gas_month_ahead is not None:
        result_df = result_df.merge(gas_month_ahead, on='date', how='left')
    
    if gas_spot is not None:
        result_df = result_df.merge(gas_spot, on='date', how='left')
    
    # Determine settlement based on available data
    if gas_month_ahead is not None and gas_spot is not None:
        result_df['settlement'] = result_df.apply(
            lambda row: row['spot_price'] if pd.notna(row['spot_price']) else row['month_ahead_price'], 
            axis=1
        )
    elif gas_month_ahead is not None:
        result_df['settlement'] = result_df['month_ahead_price']
    elif gas_spot is not None:
        result_df['settlement'] = result_df['spot_price']
    elif gas_threshold is not None:
        result_df['settlement'] = gas_threshold
    else:
        raise ValueError("No settlement data source provided. Please provide at least one of: gas_month_ahead_path, gas_spot_path, or gas_threshold")
    
    # Drop intermediate columns
    columns_to_drop = ['date']
    if 'month_ahead_price' in result_df.columns:
        columns_to_drop.append('month_ahead_price')
    if 'spot_price' in result_df.columns:
        columns_to_drop.append('spot_price')
    
    result_df = result_df.drop(columns=columns_to_drop)
    
    return result_df


def parse_date_column(df, date_col):
    try:
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
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

def process_data(df, gas_source="api", gas_month_ahead_path=None, gas_spot_path=None, 
                manual_threshold_value=None):
     
    # Make a copy of the input dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Check if we have a wide format with hour columns or long format with DeliveryStart
    is_wide_format = any(col.startswith("Hour ") for col in processed_df.columns)
    
    # Process wide format data
    if is_wide_format:
        # Set up DeliveryStart as the index
        processed_df = processed_df.set_index("DeliveryStart")
        processed_df.index.name = "Date"
        processed_df = processed_df.sort_index()
        
        # Ensure index is datetime
        processed_df.index = pd.to_datetime(processed_df.index, dayfirst=True)
        
        # Drop Date column if it exists
        if "Date" in processed_df.columns:
            processed_df = processed_df.drop(columns=["Date"])
        
        # Reset index to have Date as a column for merging
        processed_df = processed_df.reset_index()
        
        # Handle "Hour 3A" column if present
        if "Hour 3A" in processed_df.columns:
            processed_df = processed_df.rename(columns={"Hour 3A": "Hour 3"})
    else:
        # For long format, ensure DeliveryStart is datetime
        processed_df['DeliveryStart'] = pd.to_datetime(processed_df['DeliveryStart'])
        
        # Create date column from DeliveryStart for easier joining
        if 'Date' not in processed_df.columns:
            processed_df['Date'] = processed_df['DeliveryStart'].dt.normalize()
    
    # Add settlement prices based on the specified source
    if gas_source == "api":
        # Use the global gas_prices_df
        global gas_prices_df
        
        try:
            # Make a copy of gas_prices_df to avoid modifying the global variable
            gas_df_copy = gas_prices_df.copy()
            
            # Ensure Date columns have compatible datetime formats (no timezone)
            if 'Date' in processed_df.columns:
                # Convert to naive datetime (remove timezone) if needed
                if hasattr(processed_df['Date'].dtype, 'tz'):
                    processed_df['Date'] = processed_df['Date'].dt.tz_localize(None)
            
            # Convert gas_prices_df index to naive datetime if needed
            if hasattr(gas_df_copy.index.dtype, 'tz'):
                gas_df_copy.index = gas_df_copy.index.tz_localize(None)
                
            # Reset index to get Date as column for merging
            gas_df_copy = gas_df_copy.reset_index()
            
            # Ensure both DataFrames have 'Date' as datetime64[ns] (no timezone)
            processed_df['Date'] = pd.to_datetime(processed_df['Date']).dt.normalize()
            gas_df_copy['Date'] = pd.to_datetime(gas_df_copy['Date']).dt.normalize()
            
            # Perform the merge
            processed_df = pd.merge(processed_df, gas_df_copy, how="inner", on="Date")
            
            if processed_df.empty:
                st.error("Merge resulted in empty DataFrame. Check if dates overlap between your data and gas prices.")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Merge error with API gas prices: {str(e)}")
            st.info("Try using manual threshold instead, or check your data format.")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    elif gas_source == "files":
        # Load gas price data if paths are provided
        gas_month_ahead = None
        gas_spot = None
        
        if gas_month_ahead_path:
            try:
                gas_month_ahead = pd.read_csv(gas_month_ahead_path)
                gas_month_ahead['date'] = pd.to_datetime(gas_month_ahead['date']).dt.date
                gas_month_ahead = gas_month_ahead[['date', 'price']].rename(columns={'price': 'month_ahead_price'})
                processed_df['date'] = pd.to_datetime(processed_df['Date']).dt.date
                processed_df = processed_df.merge(gas_month_ahead, on='date', how='left')
            except Exception as e:
                st.error(f"Error loading month ahead gas prices: {str(e)}")
        
        if gas_spot_path:
            try:
                gas_spot = pd.read_csv(gas_spot_path)
                gas_spot['date'] = pd.to_datetime(gas_spot['date']).dt.date
                gas_spot = gas_spot[['date', 'price']].rename(columns={'price': 'spot_price'})
                if 'date' not in processed_df.columns:
                    processed_df['date'] = pd.to_datetime(processed_df['Date']).dt.date
                processed_df = processed_df.merge(gas_spot, on='date', how='left')
            except Exception as e:
                st.error(f"Error loading spot gas prices: {str(e)}")
        
        # Determine settlement based on available data
        if 'month_ahead_price' in processed_df.columns and 'spot_price' in processed_df.columns:
            processed_df['Settlement'] = processed_df.apply(
                lambda row: row['spot_price'] if pd.notna(row['spot_price']) else row['month_ahead_price'], 
                axis=1
            )
        elif 'month_ahead_price' in processed_df.columns:
            processed_df['Settlement'] = processed_df['month_ahead_price']
        elif 'spot_price' in processed_df.columns:
            processed_df['Settlement'] = processed_df['spot_price']
        else:
            st.error("No gas price data available from files")
            return pd.DataFrame()
        
        # Drop intermediate columns
        columns_to_drop = []
        if 'date' in processed_df.columns:
            columns_to_drop.append('date')
        if 'month_ahead_price' in processed_df.columns:
            columns_to_drop.append('month_ahead_price')
        if 'spot_price' in processed_df.columns:
            columns_to_drop.append('spot_price')
        
        if columns_to_drop:
            processed_df = processed_df.drop(columns=columns_to_drop)
    
    elif gas_source == "manual":
        # Use manual threshold
        if manual_threshold_value is not None:
            processed_df["Settlement"] = manual_threshold_value
            st.info(f"Using manual threshold of €{manual_threshold_value:.2f}/MWh as Settlement price for all periods")
        else:
            st.error("Manual threshold selected but no value provided")
            return pd.DataFrame()
    
    # For wide format, reshape to long format
    if is_wide_format:
        hour_columns = [f"Hour {i}" for i in range(1, 25)]
        hour_columns = [col for col in hour_columns if col in processed_df.columns]
        
        if not hour_columns:
            st.warning("No hour columns found for reshaping. Using data as-is.")
            return processed_df
        
        # Reset index if it's not already
        if processed_df.index.name == 'Date':
            processed_df = processed_df.reset_index()
        
        # Ensure Settlement column exists
        if "Settlement" not in processed_df.columns:
            st.error("Settlement column not found after data processing")
            return pd.DataFrame()
            
        # Reshape wide format to long format
        reshaped_df = pd.melt(
            processed_df,
            id_vars=["Date", "Settlement"],
            value_vars=hour_columns,
            var_name="Hour",
            value_name="Price",
        )
        
        # Extract numeric hour and sort
        reshaped_df["Hour_num"] = reshaped_df["Hour"].str.extract("(\d+)").astype(int)
        reshaped_df = reshaped_df.sort_values(["Date", "Hour_num"])
        reshaped_df = reshaped_df.drop("Hour_num", axis=1)
        
        # Set index to Date
        reshaped_df = reshaped_df.set_index("Date")
        
        return reshaped_df
    else:
        # For data already in long format, just return the processed dataframe
        return processed_df
