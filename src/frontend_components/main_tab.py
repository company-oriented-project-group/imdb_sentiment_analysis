from dash import html, dcc
import dash_bootstrap_components as dbc
from .our_story_container import our_story_container
from .try_model_container import try_model_container
from .your_story_container import your_story_container
from .main_tab_callbacks import register_main_tab_callbacks

main_tab = dbc.Container(
    id="main-tab",

    children=[
        dcc.Tabs(
            id='main-tabs',
            value='our-story-container',
            children=[
                dcc.Tab(label='Our Story', value='our-story-container'),
                dcc.Tab(label='Try The Model', value='try-model-container'),
                # dcc.Tab(label='Your Story', value='your-story-container'),
        ]),
        dbc.Container(
            id='main-tabs-content',
                children=[our_story_container, try_model_container, your_story_container],
        ),
    ]
)

