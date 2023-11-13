from dash import html, dcc
import dash_bootstrap_components as dbc
from .try_model_container_callbacks import register_try_model_container_callbacks
from models.keras import predict_sentiment
import os

btn_analyze = dbc.Button("Analyze",
    id='btn-analyze',
    n_clicks=0,
)


script_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_dir, 'try_model_container.js')


try_model_content = dbc.Container(
    id="try-model-content",
    children=[
        dbc.Textarea(
            id="try-model-input",
            size="lg",
            placeholder="Type a movie review...",
            className="m-2",
        ),
        btn_analyze,
        html.P(' ', id='analyzing-text', className="mt-4"),
        html.P(' ', id='sentiment-result', className="mt-4")
    ],
)

try_model_container = dbc.Container(
    id="try-model-container",
    children=[
        # btn_try_model,
        try_model_content
    ],
    className="mt-4",
    style = {'display': 'none'},
)

