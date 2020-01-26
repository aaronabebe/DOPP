import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


THRESHOLD_LINE_COLOR = 'red'
THRESHOLD_LINE_WIDTH = 4


def question1(base):
    poverty_percent = {'Year': [], '% of population in extreme poverty': []}
    for i in range(1970, 2019):
        poverty_percent['Year'].append(i)
        poverty_percent['% of population in extreme poverty'].append(base[(base['TIME']==i) & (base['poverty'])].shape[0] / base[base['TIME']==i].shape[0])

    pp = pd.DataFrame(poverty_percent)

    fig = px.line(
        pp, 
        x='Year', 
        y='% of population in extreme poverty', 
    )
    return fig


def line_chart(base, y, y_name, threshold):
    base.rename(columns={y: y_name, '200101': 'Population'}, inplace=True)
    base = merge_continents(base)

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


def combined_line_chart(base, y): 

    # COUNT DAILY NOT YEARLY
    base[y] = base['NY_GNP_PCAP_CD'].apply(lambda x: (x / 365))

    base = merge_continents(base)

    fig = px.line(
        base, 
        x="TIME", 
        y=y, 
        hover_name="LOCATION", 
        color="continent",
    )

    # ADD THRESHOLD LINE
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

def merge_continents(base):
    # ADD CONTINENTS FOR PLOTTING
    continents = pd.read_csv('continents.csv')
    base = pd.merge(base, continents, left_on='LOCATION', right_on='LOCATION')
    return base


def scatter_poor_rich(base, x, x_name, y, y_name):
    base.rename(columns={
        x: x_name,
        y: y_name, 
        '200101': 'Population'
    }, inplace=True)

    # fill missing values with 1 to get shown on the scatter plot
    base['Population'].fillna(1, inplace=True)
    
    base = merge_continents(base)

    fig = px.scatter(
            base, 
            x=x_name, 
            y=y_name,
            facet_col="poverty",
            animation_frame='TIME', 
            hover_name='LOCATION',
            size='Population',
            color='continent'
        )
    return fig


def scatter(base, x, x_name, y, y_name):
    base.rename(columns={
        x: x_name, 
        y: y_name, 
        '200101': 'Population'}, 
    inplace=True)

    # fill missing values with 1 to get shown on the scatter plot
    base['Population'].fillna(1, inplace=True)

    base = merge_continents(base)

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


def world_map(base, y, y_name, yearly_feature=False):
    if yearly_feature:
        base[y_name] = base[y].apply(lambda x: (x / 365))
    else:
        base.rename(columns={y: y_name}, inplace=True)
    fig = px.choropleth(    
        base,
        locations="LOCATION",
        color=y_name,
        hover_name="LOCATION",
        animation_frame="TIME"
    )
    return fig