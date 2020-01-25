import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


THRESHOLD_LINE_COLOR = 'red'
THRESHOLD_LINE_WIDTH = 4


def line_chart(base, y, y_name, threshold):
    base.rename(columns={y: y_name, '200101': 'Population'}, inplace=True)
    fig = px.line(
        base.fillna(0), 
        x='TIME', 
        y=y_name, 
        hover_name='LOCATION', 
        color='continent',
    )
    fig.add_shape(get_threshold_line(1970, 2019, threshold))
    fig.update_layout(yaxis_type="log")
    return fig

def combined_line_chart(base): 
    base.rename(columns={'NY_GNP_PCAP_CD': 'target', '200101': 'Population'}, inplace=True)

    fig = px.line(
        base.fillna(0), 
        x="TIME", 
        y="target", 
        hover_name="LOCATION", 
        color="continent",
    )
    fig.add_shape(get_threshold_line(1970, 1996, 1))
    fig.add_shape(get_threshold_line(1996, 2005, 1.25))
    fig.add_shape(get_threshold_line(2005, 2019, 1.9))
    fig.update_layout(yaxis_type="log")
    return fig


def get_threshold_line(start, end, height):
    return go.layout.Shape(
        type='line',
        x0=start ,
        y0=height,
        x1=end,
        y1=height,
        line=dict(
            color=THRESHOLD_LINE_COLOR,
            width=THRESHOLD_LINE_WIDTH,
            dash='dashdot'
        )
    )

def pie_chart(base):
    #base['poverty_count'] = base['TIME']
    fig = px.pie(
        base, 
        names='LOCATION', 
        values='poverty',
        #animation_frame='TIME'
        #color_continuous_scale=px.colors.sequential.Viridis, 
        #render_mode="webgl"
    )
    return fig

def scatter_poor_rich(base):
    base.rename(columns={
        '200151': 'Population aged 65 or older',
        'SP_DYN_LE00_IN': 'Life expectancy at birth', 
        '200101': 'Population'
    }, 
    inplace=True)
    base['Population'].fillna(1, inplace=True)

    fig = px.scatter(
            base, 
            x='LOCATION', 
            y='Life expectancy at birth',
            facet_col="poverty",
            animation_frame='TIME', 
            size='Population',
            color='continent'
        )
    return fig


def scatter(base, x, x_name, y, y_name):
    base.rename(columns={x: x_name, y: y_name, '200101': 'Population'}, inplace=True)
    base['Population'].fillna(1, inplace=True)

    fig = px.scatter(
        data_frame=base, 
        x=x_name, 
        y=y_name, 
        animation_frame='TIME',
        hover_name='LOCATION',
        size='Population',
        color='continent',
        size_max=60
    )
    return fig


def world_map(base, y, y_name):
    #base.rename(columns={y: y_name, '200101': 'Population'}, inplace=True)
    base[y_name] = base[y].apply(lambda x: (x / 365))
    fig = px.choropleth(    
        base,
        locations="LOCATION",
        color=y,
        hover_name="LOCATION",
        animation_frame="TIME"
    )
    return fig