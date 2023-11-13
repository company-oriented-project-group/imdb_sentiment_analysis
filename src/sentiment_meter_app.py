### sentiment_meter_app.py
##import dash
##from dash import dcc, html
##from dash.dependencies import Input, Output
##import dash_core_components as dcc
##import dash_html_components as html
##import dash_bootstrap_components as dbc
##
##app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
##
##app.layout = html.Div([
##    html.H1("Sentiment Analysis Meter"),
##    dcc.Slider(id='sentiment-slider', min=0, max=100, step=1, value=50),
##    dcc.Graph(id='sentiment-meter'),
##    html.Audio(id='sound-effect', controls=True),
##    html.Button('Analyze', id='analyze-button'),
##    html.Img(src='/assets/needle.png', id='needle-img', style={'position': 'relative', 'top': '30px'}),
##    html.Script(src='/assets/needle_animation.js')
##])
##
##if __name__ == '__main__':
##    app.run_server(debug=True)


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.Div([
        html.Img(id='bg-image', src='/assets/sentiment_meter.png', style={'width': '496px', 'height': 'auto'}),
        html.Div(
            id='needle-image', 
            style={'width': '146px', 'height': '99px', 'backgroundImage': 'url(/assets/needle.png)', 'backgroundSize': 'contain', 'backgroundRepeat': 'no-repeat',  },
            children=[
                # html.H1(" aaa", style={'backgroundImage': 'url(/assets/needle.png)'})
                html.Div()
            ]
            # style={'width': '300px', 'height': 'auto', 'transform': 'rotate(-180deg)', 'backgroundImage': 'url(needle.png)'},
         ),
    ],
    style={'display': 'flex'},
    ),
    dcc.Interval(
        id='interval-component',
        interval=170,  # in milliseconds
        n_intervals=0
    ),
])

@app.callback(
    Output('needle-image', 'style'),
    Input('interval-component', 'n_intervals')
)
def update_needle(n_intervals):
    angel = -180 + (n_intervals % 18) * 10
    # sign = 1 if (n_intervals // 20) % 2 == 0 else -1
    # coefficient = n_intervals % 20
    # if sign > 0:
    #     angel = -180 + sign * coefficient * 12 
    # else:
    #     angel = 60 + sign * coefficient * 12 
    # return {'transform': f'rotate({angel}deg)'}
    # return f"(sign, coefficient) = ({sign}, {coefficient})", {'transform': f'rotate({angel}deg)', 'width': '200px', 'height': '200px', 'backgroundImage': 'url(/assets/needle.png)', 'backgroundSize': 'contain', 'backgroundRepeat': 'no-repeat'}
    return {'transformOrigin': 'bottom center', 'transform': f'rotate({angel}deg)', 'marginLeft': '-250px', 'marginTop': '50px', 'width': '146px', 'height': '99px', 'backgroundImage': 'url(/assets/needle.png)', 'backgroundSize': 'contain', 'backgroundRepeat': 'no-repeat', }
    # return {'postiion': 'absolute', 'width': '300px', 'height': 'auto', 'transform': f'rotate({angel}deg)'}
    # return {'transform': f'translate(-50%, -50%) translate({x * 100}%, {y * 100}%) rotate({angle}deg)', 'width': '100%', 'height': 'auto'}

if __name__ == '__main__':
    app.run_server(debug=True)
