import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date

def spot(df, price_col, settlement_col, efficiency_parameter, certificates, time_interval_minutes):
    result_df = df.copy()
    result_df['operate'] = False
    result_df['profit'] = 0.0
    result_df['buy_threshold'] = None
    
    # Time interval factor for profit calculation
    time_factor = time_interval_minutes / 60.0  # Convert to hours
    
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
                # Adjust profit based on time interval
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
        
        # If none work, try pandas auto-parsing
        return pd.to_datetime(df[date_col])
    
    except Exception as e:
        st.error(f"Could not parse date column: {str(e)}")
        return None

def create_visualization(df_result, price_col, time_interval_minutes):
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

# Streamlit App
def main():
    st.set_page_config(
        page_title="Electrolyser Energy Trading Dashboard",
        page_icon="",
        layout="wide"
    )
    
    st.title("Electrolyser Energy Trading Simulation")
    st.markdown("Upload your data and configure parameters to simulate electrolyser operations on the energy market.")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Data",
            type=['csv'],
            help="Upload a CSV file containing price, settlement, and date data"
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
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Column selection section
            st.header("Column Selection")
            st.markdown("Select which columns contain your data:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_column = st.selectbox(
                    "Select Price Column",
                    options=df.columns.tolist(),
                    index=0,
                    help="Column containing electricity prices"
                )
            
            with col2:
                settlement_column = st.selectbox(
                    "Select Gas Price Column", 
                    options=df.columns.tolist(),
                    index=1 if len(df.columns) > 1 else 0,
                    help="Column containing gas settlement prices"
                )
            
            with col3:
                date_column = st.selectbox(
                    "Select Date/Time Column",
                    options=['None'] + df.columns.tolist(),
                    index=0,
                    help="Column containing date/time information (optional)"
                )
            
            # Product column selection
            col4, col5 = st.columns(2)
            with col4:
                product_column = st.selectbox(
                    "Select Product Column",
                    options=['None'] + df.columns.tolist(),
                    index=0,
                    help="Column containing product information (optional)"
                )
            
            with col5:
                if product_column != 'None':
                    unique_products = df[product_column].dropna().unique()
                    selected_product = st.selectbox(
                        "Select Product",
                        options=['All Products'] + list(unique_products),
                        index=0,
                        help="Choose specific product to analyze"
                    )
                else:
                    selected_product = None
            
            # Validate column selection
            selected_cols = [price_column, settlement_column]
            if date_column != 'None':
                selected_cols.append(date_column)
            if product_column != 'None':
                selected_cols.append(product_column)
            
            # Remove duplicates and check if all selections are unique
            unique_selected = [col for col in selected_cols if col != 'None']
            if len(set(unique_selected)) != len(unique_selected):
                st.warning("Please select different columns for each field!")
                return
            
            # Product filtering (if product column is selected)
            product_filter_applied = False
            if product_column != 'None' and selected_product != 'All Products':
                st.header("Product Filtering")
                
                product_mask = df[product_column] == selected_product
                original_length = len(df)
                df = df[product_mask]
                product_filter_applied = True
                
                if len(df) > 0:
                    st.success(f"Product filter applied: {len(df):,} rows for '{selected_product}' from {original_length:,} total rows")
                else:
                    st.error(f"No data found for product '{selected_product}'")
                    return
            
            # Date range selection (if date column is selected)
            date_filter_applied = False
            if date_column != 'None':
                st.header("Date Range Selection")
                
                # Parse date column
                parsed_dates = parse_date_column(df, date_column)
                if parsed_dates is not None:
                    df['parsed_date'] = parsed_dates
                    
                    # Get date range
                    min_date = df['parsed_date'].min().date()
                    max_date = df['parsed_date'].max().date()
                    
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
                        date_mask = (df['parsed_date'].dt.date >= start_date) & (df['parsed_date'].dt.date <= end_date)
                        original_length = len(df)
                        df = df[date_mask]
                        date_filter_applied = True
                        
                        if len(df) < original_length:
                            st.success(f"Date filter applied: {len(df):,} rows selected from {original_length:,} total rows")
                        else:
                            st.info("No filtering applied - selected range includes all data")
                    else:
                        st.error("End date must be after start date!")
                        return
                else:
                    st.error("Could not parse the selected date column. Please check the date format.")
                    return
            
            # Display data info
            with st.expander("Data Overview", expanded=True):
                st.write(f"**Data shape:** {df.shape}")
                st.write(f"**Selected Price Column:** {price_column}")
                st.write(f"**Selected Settlement Column:** {settlement_column}")
                if date_column != 'None':
                    st.write(f"**Selected Date Column:** {date_column}")
                    if date_filter_applied:
                        st.write(f"**Date Range:** {start_date} to {end_date}")
                if product_column != 'None':
                    st.write(f"**Selected Product Column:** {product_column}")
                    if product_filter_applied:
                        st.write(f"**Selected Product:** {selected_product}")
                    elif selected_product == 'All Products':
                        st.write(f"**Product Filter:** All Products")
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
                
                # Product statistics (if product column is selected)
                if product_column != 'None':
                    st.write(f"**{product_column} Statistics:**")
                    product_counts = df[product_column].value_counts()
                    st.write(f"- Unique products: {len(product_counts)}")
                    st.write(f"- Most common: {product_counts.index[0]} ({product_counts.iloc[0]} records)")
                    if product_filter_applied:
                        st.write(f"- Selected product records: {len(df)}")
                
                st.write("**First few rows:**")
                display_df = df.head()
                if 'parsed_date' in display_df.columns:
                    display_df = display_df.drop('parsed_date', axis=1)
                st.dataframe(display_df)
                
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
                    csv_data = result_display.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name=f"electrolyser_simulation_results_{time_interval}min.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains numeric data in the selected columns.")
    
    else:
        st.info("Please upload a CSV file to get started.")
        
        # Show example of expected data format
        with st.expander("Expected Data Format", expanded=False):
            st.markdown("""
            Your CSV file should contain at least two numeric columns and optionally date and product columns:
            - **Price Column**: Electricity prices (e.g., in €/MWh)
            - **Settlement Column**: Gas settlement prices (e.g., in €/MWh)
            - **Date Column**: Date/time information (optional, for filtering)
            - **Product Column**: Product categories (optional, for filtering)
            
            Example:
            ```
            DateTime,Product,Electricity_Price,Gas_Settlement,Other_Data
            2024-01-01 00:00:00,Product_A,45.2,30.5,some_value
            2024-01-01 00:15:00,Product_A,52.1,31.2,some_value
            2024-01-01 00:30:00,Product_B,38.7,29.8,some_value
            ...
            ```
            
            **Filtering Options:**
            - **Date Range**: Filter data to specific time periods
            - **Product**: Analyze specific products or all products combined
            
            **Time Intervals Supported:**
            - 15 minutes: Profit = (hourly_profit × 15/60)
            - 30 minutes: Profit = (hourly_profit × 30/60)  
            - 60 minutes: Standard hourly calculation
            
            Column names can be anything - you'll select them after uploading!
            """)

if __name__ == "__main__":
    main()