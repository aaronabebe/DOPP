import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


THRESHOLD_LINE_COLOR = 'red'
THRESHOLD_LINE_WIDTH = 4


def ppp_line_chart(base): 
    base.rename(columns={'target': 'PPP', '200101': 'Population'}, inplace=True)
    fig = px.line(
        base.fillna(0), 
        x="TIME", 
        y="PPP", 
        hover_name="LOCATION", 
        color="continent",
    )
    fig.add_shape(get_threshold_line(1970, 2019, 1.9))
    fig.update_layout(yaxis_type="log")
    return st.plotly_chart(fig)

def atlas_line_chart(base):
    base.rename(columns={'target': 'Atlas', '200101': 'Population'}, inplace=True)
    fig = px.line(
        base.fillna(0), 
        x="TIME", 
        y="Atlas", 
        hover_name="LOCATION", 
        color="continent",
    )
    fig.add_shape(get_threshold_line(1970, 2019, 1.5))
    fig.update_layout(yaxis_type="log")
    return st.plotly_chart(fig)

def lcu_line_chart(base): 
    base.rename(columns={'target': 'LCU', '200101': 'Population'}, inplace=True)
    fig = px.line(
        base.fillna(0), 
        x="TIME", 
        y="LCU", 
        hover_name="LOCATION", 
        color="continent",
    )
    fig.add_shape(get_threshold_line(1970, 2019, 1.25))
    fig.update_layout(yaxis_type="log")
    return st.plotly_chart(fig)

def combined_line_chart(base): 
    base.rename(columns={'NY_GNP_PCAP_CD': 'target', '200101': 'Population'}, inplace=True)

    fig = px.line(
        base.fillna(0), 
        x="TIME", 
        y="target", 
        hover_name="LOCATION", 
        color="LOCATION",
    )
    fig.add_shape(get_threshold_line(1970, 1996, 1))
    fig.add_shape(get_threshold_line(1996, 2005, 1.25))
    fig.add_shape(get_threshold_line(2005, 2019, 1.9))
    fig.update_layout(yaxis_type="log")
    return st.plotly_chart(fig)


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
    return st.plotly_chart(fig)

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
            #color_continuous_scale=px.colors.sequential.Viridis, 
            #render_mode="webgl"
        )
    return st.plotly_chart(fig)



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
    return st.plotly_chart(fig)

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
    return st.plotly_chart(fig)
'''

#poor = pd.concat([poor_ppp, poor_atlas, poor_lcu])
#st.write(poor_plot)
fig = px.scatter(
    data_frame=poor, 
    x="TIME", 
    y="target", 
    #animation_frame="TIME", 
    #animation_group="LOCATION",
    #size="200101", 
    color="LOCATION", 
    #hover_name="LOCATION", 
    facet_col="type",
    #size_max=45,
    #category_orders={'Type': poor['type']}     
)
fig = px.line(
    poor_plot, 
    x="TIME", 
    y="PPP", 
    hover_name="LOCATION", 
    color="LOCATION"
    #animation_frame="LOCATION"
)
st.plotly_chart(fig)




fig = px.choropleth(    
    base,
    locations="LOCATION",
    color="NY_GNP_PCAP_PP_CD",
    hover_name="LOCATION",
    animation_frame="TIME"
)
st.plotly_chart(fig)







DATA_URL='https://raw.githubusercontent.com/johan/world.geo.json/master/countries/AUT.geo.json'

st.markdown("## Interactive Plot Test")
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
st.plotly_chart(fig)

def get_plot(country_data, year):
    return st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        mapbox_key='pk.eyJ1IjoiYWFyb25hYmViZSIsImEiOiJjam9kMGoyZHcxYWM3M29ud3Nuc3pxYWxjIn0.OPb01Qq3C4VM_vns9T-WXw',
        initial_view_state=pdk.ViewState(
            latitude=30,
            longitude=0,
            zoom=1,
        ), 
        layers=[get_country_layers(country_data, year)]
))

def get_country_layers(data, year):
    layers = []

    data = data[data['TIME']==year]

    for index, row in data.iterrows():
        x = map_to_layer(row)
        layers.append(x)

    return layers

def map_to_layer(datapoint):
    data_url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries/' + datapoint['LOCATION'] + '.geo.json'
    #todo calculate percentage
    fill = (datapoint['NY_GNP_PCAP_CD'] / 365) * 100 / 255
    return pdk.Layer(
        'GeoJsonLayer',
        data_url,
        opacity=1,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color='[fill*10, 0, 0]',
        pickable=True
    )

get_plot([], 1980)
'''