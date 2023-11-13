from dash import html, dcc
import dash_bootstrap_components as dbc
from .chart_container import chart_group
from .our_story_container_callbacks import register_our_story_container_callbacks

btn_our_story = dbc.Button("Show Story", id="btn-our-story", className="mt-4")

btn_hide_our_story = dbc.Button("Hide Story", id="btn-hide-our-story" ,className="mt-2 mb-2")

our_story_text = dbc.Container("The story begins", id="our-story-text")

our_story_block_1 = dbc.Container(
    id="our-story-block-1",
    children=[btn_our_story]
)

our_story_block_2 = dbc.Container(
    id="our-story-block-2",
    children=[our_story_text, chart_group, btn_hide_our_story],
    style={'display': 'none'}
)

our_story_container = dbc.Container(
    id="our-story-container",
    children=[
        our_story_block_1,
        our_story_block_2,
    ],
    style = {'display': 'none'},
)

