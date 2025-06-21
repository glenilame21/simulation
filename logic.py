import base64
from datetime import date, datetime
from io import BytesIO, StringIO
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from PIL import Image as PILImage
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def load_gas_prices(gas_file_option="Month Ahead"):
    global gas_prices_df
    try:
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
                f"'Date' column not found in gas prices. Available columns: {list(gas_prices_df.columns)}"
            )
            return False

        gas_prices_df = gas_prices_df.set_index("Date")
        gas_prices_df.index = pd.to_datetime(gas_prices_df.index)

        # st.success(f"Successfully loaded {len(gas_prices_df)} gas price records from Google Sheets")
        return True

    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading gas prices: {str(e)}")
        return False
    except pd.errors.EmptyDataError:
        st.error("Gas prices file is empty or corrupted")
        return False
    except Exception as e:
        st.error(f"Error loading gas prices: {str(e)}")
        return False


def preprocess_with_gas_prices(
    df, use_manual_threshold=False, manual_threshold_value=None
):
    """
    Traders want to have the option of either using a manual threshold for gas Price or a dropdown that let's them pick M1 or D1
    For this reason, by default the function above will load Month Ahead in memory and use that as the default value for the threshold
    If users choose otherwise then this function comes in play
    Preprocess uploaded data by merging with gas prices OR using manual threshold
    Expected: df must contain a column named 'Delivery day'
    """
    global gas_prices_df

    processed_df = df.copy()
    processed_df = processed_df.set_index("Delivery day")
    processed_df.index.name = "Date"
    processed_df = processed_df.sort_index()

    processed_df.index = pd.to_datetime(processed_df.index, dayfirst=True)

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


def test_google_sheets_connection(url):
    """Test if a Google Sheets URL is accessible"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if len(response.text) > 0:
            return True, "Connection successful"
        else:
            return False, "Empty response"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"


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


def calculate_advanced_metrics(df_result, time_interval_minutes, power_energy):
    """Calculate additional performance metrics including gas production"""
    metrics = {}

    # Basic metrics
    operating_periods = df_result["operate"].sum()
    total_periods = len(df_result)

    metrics["operating_periods"] = operating_periods
    metrics["total_periods"] = total_periods
    metrics["utilization_rate"] = (
        (operating_periods / total_periods * 100) if total_periods > 0 else 0
    )

    # Profit metrics
    total_profit = df_result["profit"].sum()
    operating_profit = df_result[df_result["operate"]]["profit"].sum()

    metrics["total_profit"] = total_profit
    metrics["operating_profit"] = operating_profit
    metrics["avg_profit_per_operating_period"] = (
        (operating_profit / operating_periods) if operating_periods > 0 else 0
    )

    # Energy metrics
    time_factor = time_interval_minutes / 60.0
    total_energy_consumed = operating_periods * time_factor * power_energy

    metrics["total_energy_consumed_mwh"] = total_energy_consumed
    metrics["profit_per_mwh"] = (
        (total_profit / total_energy_consumed) if total_energy_consumed > 0 else 0
    )

    # Gas generation metrics
    if "gas_generated_mwh" in df_result.columns:
        metrics["total_gas_generated_mwh"] = df_result["gas_generated_mwh"].sum()
        metrics["gas_generation_rate_mwh_per_hour"] = (
            df_result[df_result["operate"]]["gas_generated_mwh"].mean() / time_factor
            if operating_periods > 0
            else 0
        )

        # Profit per MWh of gas generated
        if metrics["total_gas_generated_mwh"] > 0:
            metrics["profit_per_mwh_gas"] = (
                total_profit / metrics["total_gas_generated_mwh"]
            )

    if operating_periods > 0:
        operating_data = df_result[df_result["operate"]]
        price_col = None
        for col in ["Price", "price"]:
            if col in operating_data.columns:
                price_col = col
                break

        if price_col:
            metrics["avg_operating_price"] = operating_data[price_col].mean()
            metrics["min_operating_price"] = operating_data[price_col].min()
            metrics["max_operating_price"] = operating_data[price_col].max()

    return metrics


def create_enhanced_visualization(df_result, price_col, time_interval_minutes):
    """Create a visualization showing price data and operating decisions"""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result[price_col],
            mode="lines",
            name="Electricity Price",
            line=dict(color="blue", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result.index,
            y=df_result["buy_threshold"],
            mode="lines",
            name="Buy Threshold",
            line=dict(color="red", width=1, dash="dash"),
        ),
        secondary_y=False,
    )

    operating_points = df_result[df_result["operate"] == True]
    if not operating_points.empty:
        fig.add_trace(
            go.Scatter(
                x=operating_points.index,
                y=operating_points[price_col],
                mode="markers",
                name="Operating Hours",
                marker=dict(color="green", size=8, symbol="circle"),
                text=[
                    f"Profit: €{profit:.2f} ({time_interval_minutes}min)"
                    for profit in operating_points["profit"]
                ],
                hovertemplate="<b>Operating Period</b><br>"
                + "Price: %{y:.2f}<br>"
                + "%{text}<br>"
                + "<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title=f"Electrolyser Operation Strategy ({time_interval_minutes}-minute intervals)",
        xaxis_title="Time Period",
        height=600,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price (€/MWh)", secondary_y=False)

    return fig


def generate_pdf_report(result_df, metrics, simulation_params, price_col, fig=None):
    """
    Generate a PDF report summarizing electrolyser simulation results.
    
    Parameters:
    - result_df: DataFrame containing simulation results
    - metrics: Dictionary of calculated metrics
    - simulation_params: Dictionary with simulation parameters
    - price_col: Column name for price data
    - fig: Optional Plotly figure to include in report
    
    Returns:
    - PDF file contents as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title="Electrolyser Simulation Report",
        author="Electrolyser Dashboard",
        leftMargin=inch*0.5,
        rightMargin=inch*0.5,
        topMargin=inch*0.75,
        bottomMargin=inch*0.75
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading1_style = styles["Heading1"]
    heading2_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    elements = []
    elements.append(Paragraph("Electrolyser Simulation Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    elements.append(Paragraph("Simulation Parameters", heading1_style))
    
    param_data = [
        ["Parameter", "Value"],
        ["Efficiency Parameter", f"{simulation_params.get('efficiency_parameter', 'N/A'):.3f}"],
        ["Power Capacity (MW)", f"{simulation_params.get('power_energy', 'N/A'):.1f}"],
        ["Green Certificates (€/MWh)", f"€{simulation_params.get('certificates', 'N/A'):.2f}"],
        ["Time Interval (min)", f"{simulation_params.get('time_interval', 'N/A')}"],
    ]
    
    threshold_method = simulation_params.get('threshold_method', 'Unknown')
    if threshold_method == "manual":
        param_data.append(["Threshold Method", "Manual"])
        param_data.append(["Manual Threshold Value", f"€{simulation_params.get('manual_threshold', 'N/A'):.2f}"])
    else:
        param_data.append(["Threshold Method", "Gas Price Based"])
    
    param_table = Table(param_data, colWidths=[doc.width*0.4, doc.width*0.4])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(param_table)
    elements.append(Spacer(1, 0.25*inch))
    
    elements.append(Paragraph("Key Results", heading1_style))
    
    results_data = [
        ["Metric", "Value"],
        ["Total Profit", f"€{metrics.get('total_profit', 0):,.2f}"],
        ["Operating Periods", f"{metrics.get('operating_periods', 0):,} out of {metrics.get('total_periods', 0):,}"],
        ["Utilization Rate", f"{metrics.get('utilization_rate', 0):.1f}%"],
        ["Total Energy Consumed", f"{metrics.get('total_energy_consumed_mwh', 0):.1f} MWh"],
        ["Profit per MWh Consumed", f"€{metrics.get('profit_per_mwh', 0):.2f}"],
    ]
    
    if 'total_gas_generated_mwh' in metrics:
        results_data.extend([
            ["Total Gas Generated", f"{metrics.get('total_gas_generated_mwh', 0):.1f} MWh"],
            ["Gas Generation Rate", f"{metrics.get('gas_generation_rate_mwh_per_hour', 0):.2f} MWh/h"],
            ["Profit per MWh Gas", f"€{metrics.get('profit_per_mwh_gas', 0):.2f}"],
        ])
    
    results_table = Table(results_data, colWidths=[doc.width*0.4, doc.width*0.4])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(results_table)
    elements.append(Spacer(1, 0.25*inch))
    
    if 'avg_operating_price' in metrics:
        elements.append(Paragraph("Price Analysis", heading2_style))
        
        price_data = [
            ["Metric", "Value"],
            ["Average Operating Price", f"€{metrics.get('avg_operating_price', 0):.2f}"],
            ["Minimum Operating Price", f"€{metrics.get('min_operating_price', 0):.2f}"],
            ["Maximum Operating Price", f"€{metrics.get('max_operating_price', 0):.2f}"],
        ]
        
        price_table = Table(price_data, colWidths=[doc.width*0.4, doc.width*0.4])
        price_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(price_table)
        elements.append(Spacer(1, 0.25*inch))
    
    if fig is not None:
        elements.append(Paragraph("Visualization", heading1_style))
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'temp_plot_{timestamp}.png')
            try:
                fig.write_image(img_path, width=800, height=400)
            except Exception as write_err:
                try:
                    from kaleido.scopes.plotly import PlotlyScope
                    scope = PlotlyScope()
                    with open(img_path, "wb") as f:
                        f.write(scope.transform(fig, format="png", width=800, height=400))
                except Exception as kaleido_err:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 5))
                    plt.title("Electrolyser Operation Strategy")
                    plt.xlabel("Time Period")
                    plt.ylabel("Price (€/MWh)")
                    plt.text(0.5, 0.5, "Visualization could not be generated with Plotly.\nSee interactive chart in dashboard.", 
                            horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                    plt.savefig(img_path)
                    plt.close()
            
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                img = Image(img_path, width=doc.width*0.9, height=3*inch)
                elements.append(img)
            else:
                elements.append(Paragraph("Could not generate visualization image: File was not created successfully", normal_style))
                
        except Exception as e:
            elements.append(Paragraph(f"Could not include visualization due to an error: {str(e)}", normal_style))
            import traceback
            elements.append(Paragraph(f"Error details: {traceback.format_exc()[:200]}...", normal_style))
    
    elements.append(Paragraph("Data Sample", heading1_style))

    if len(result_df) > 10:
        sample_df = result_df.sample(n=10).reset_index()
        elements.append(Paragraph("Random sample of 10 data points:", normal_style))
    else:
        sample_df = result_df.reset_index()
        elements.append(Paragraph(f"Complete dataset ({len(result_df)} data points):", normal_style))

    if 'Date' in sample_df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(sample_df['Date']):
                sample_df['Date'] = sample_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            else:
                try:
                    sample_df['Date'] = pd.to_datetime(sample_df['Date']).dt.strftime('%Y-%m-%d %H:%M')
                except:
                    sample_df['Date'] = sample_df['Date'].astype(str)
        except:
            sample_df['Date'] = sample_df['Date'].astype(str)

    if 'Date' in sample_df.columns:
        try:
            sample_df = sample_df.sort_values('Date')
        except:
            pass 

    report_columns = ['Date', 'Hour', price_col, 'buy_threshold', 'operate', 'profit']
    report_columns = [col for col in report_columns if col in sample_df.columns]
    
    sample_data = [report_columns] 
    
    for _, row in sample_df[report_columns].iterrows():
        formatted_row = []
        for col in report_columns:
            if col in row:
                if col == 'operate':
                    formatted_row.append('Yes' if row[col] else 'No')
                elif col in ['profit', price_col, 'buy_threshold']:
                    formatted_row.append(f"€{row[col]:.2f}" if pd.notnull(row[col]) else 'N/A')
                else:
                    formatted_row.append(str(row[col]))
            else:
                formatted_row.append('N/A')
        sample_data.append(formatted_row)
    
    sample_table = Table(sample_data)
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(sample_table)
    
    try:
        doc.build(elements)
        
        pdf = buffer.getvalue()
        buffer.close()
    
    finally:
        try:
            if 'img_path' in locals() and os.path.exists(img_path):
                os.unlink(img_path)
        except:
            pass 
    
    return pdf