from dash import html, dcc
import dash_bootstrap_components as dbc
from .chart_container_callbacks import register_chart_container_callbacks

confidence_distribution = dcc.Graph(id='confidence-distribution')
prediction_time_analysis = dcc.Graph(id='prediction-time-analysis')
accuracy_vs_confidence = dcc.Graph(id='accuracy-vs-confidence')

label_distribution = dbc.Container(
    id="label-distribution",
    children=[
        dbc.Container("Distribution of Predicted Labels"),
        html.Br(),
        dcc.Tabs(id='label-distribution-tabs', value='tab-pie-chart', children=[
            dcc.Tab(label='Bar Chart', value='tab-bar-chart'),
            dcc.Tab(label='Pie Chart', value='tab-pie-chart')
        ]),
        dcc.Graph(id='label-chart')  # A single graph for displaying the selected chart
    ],
    style={
        'direction': 'ltr',
        'text-anchor': 'start',
        'font-family': '"Open Sans", verdana, arial, sans-serif',
        'font-size': '17px',
        'fill': 'rgb(42, 63, 95)',
        'opacity': 1,
        'font-weight': 'normal',
        'white-space': 'pre',
        'margin': '0% 0% 5% 5%'
    }
)

chart_group = dbc.Container(
    id="chart-container",
    children=[
        html.H1("Sentiment Analysis Insights"),
        confidence_distribution,
        prediction_time_analysis,
        accuracy_vs_confidence,
        label_distribution
    ],
    style={'display': 'none'}
)

