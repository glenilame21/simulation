import base64
from datetime import date, datetime
from io import BytesIO, StringIO
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def generate_pdf_report(result_df, metrics, simulation_params, price_col, fig=None, dataset_metadata=None):

    buffer = BytesIO()
    
    def add_page_elements(canvas, doc):
        # Save the state of the canvas
        canvas.saveState()
    
        logo_path = "C:/Users/Z_LAME/Desktop/Papers/logo.png"
        if os.path.exists(logo_path):
            # Logo positioning: right side with margins
            logo_width = 3 * inch  # Set your preferred width
            logo_height = 2.2 * inch  # Maintain aspect ratio as needed
            x = doc.pagesize[0] - logo_width + 0.5*inch  # Right margin
            y = doc.pagesize[1] - logo_height + 0.5*inch  # Top margin
            
            canvas.drawImage(logo_path, x, y, width=logo_width, height=logo_height, preserveAspectRatio=True)
        
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title="Electrolyser Simulation Report",
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
    
    if dataset_metadata:
        elements.append(Paragraph("Dataset Information", heading1_style))
        
        dataset_data = [
            ["Property", "Value"],
        ]
        
        # Add dataset name if provided
        if dataset_metadata.get('dataset_name'):
            dataset_data.append(["Dataset Name", dataset_metadata['dataset_name']])
        
        # Add date range if provided
        if dataset_metadata.get('date_range_start') and dataset_metadata.get('date_range_end'):
            start_date = dataset_metadata['date_range_start']
            end_date = dataset_metadata['date_range_end']
            
            # Format dates if they're datetime objects
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)
                
            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)
                
            dataset_data.append(["Date Range", f"{start_str} to {end_str}"])
        
        # Add time resolution
        if dataset_metadata.get('time_resolution'):
            dataset_data.append(["Time Resolution", f"{dataset_metadata['time_resolution']} minutes"])
        
        # Add gas price source
        if dataset_metadata.get('gas_price_source'):
            gas_source = dataset_metadata['gas_price_source']
            if gas_source == "Month Ahead":
                gas_source_text = "Month Ahead Gas Prices"
            elif gas_source == "Spot":
                gas_source_text = "Spot Gas Prices"
            elif gas_source == "Manual":
                manual_value = dataset_metadata.get('manual_threshold_value', 'N/A')
                gas_source_text = f"Manual Threshold (€{manual_value}/MWh)"
            else:
                gas_source_text = gas_source
            
            dataset_data.append(["Gas Price Source", gas_source_text])
        
        # Add total data points
        dataset_data.append(["Total Data Points", f"{len(result_df):,}"])
        
        dataset_table = Table(dataset_data, colWidths=[doc.width*0.4, doc.width*0.5])
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(dataset_table)
        elements.append(Spacer(1, 0.25*inch))
    
    # Simulation Parameters section
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
    
    # Key Results section
    elements.append(Paragraph("Key Results", heading1_style))
    
    results_data = [
        ["Metric", "Value"],
        ["Total Profit", f"€{metrics.get('total_profit', 0):,.2f}"],
        ["Operating Periods", f"{metrics.get('operating_periods', 0):,} out of {metrics.get('total_periods', 0):,}"],
        ["Utilization Rate", f"{metrics.get('utilization_rate', 0):.1f}%"],
        ["Total Energy Consumed", f"{metrics.get('total_energy_consumed_mwh', 0):.1f} MWh"],
    ]
    
    if 'total_gas_generated_mwh' in metrics:
        results_data.extend([
            ["Total Gas Generated", f"{metrics.get('total_gas_generated_mwh', 0):.1f} MWh"],
            ["Gas Generation Rate", f"{metrics.get('gas_generation_rate_mwh_per_hour', 0):.2f} MWh/h"],
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
    
    # Price Analysis section (if available)
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
    
    # Visualization section (if provided)
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
    
    # Data Sample section
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
        # This is the key change - use onFirstPage and onLaterPages to add the logo to all pages
        doc.build(elements, onFirstPage=add_page_elements, onLaterPages=add_page_elements)
        
        pdf = buffer.getvalue()
        buffer.close()
    
    finally:
        try:
            if 'img_path' in locals() and os.path.exists(img_path):
                os.unlink(img_path)
        except:
            pass 
    
    return pdf