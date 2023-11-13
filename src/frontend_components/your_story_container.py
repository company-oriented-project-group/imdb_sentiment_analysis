from dash import html, dcc
import dash_bootstrap_components as dbc
from .your_story_container_callbacks import register_your_story_container_callbacks

your_story_container = dbc.Container(
    id="your-story-container",
    children=[
    ],
    style = {'display': 'none'},
)

