from dash import Dash, Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import ast, time

# 'predictions' is a DataFrame containing label, score, and elapsed_time columns
df = pd.read_csv(r'./data/movie_reviews_with_model_prediction.csv')
df['model_prediction'] = df['model_prediction'].apply(ast.literal_eval)
predictions = pd.DataFrame(df['model_prediction'].tolist(), index=df.index)
predictions['score'] = pd.to_numeric(predictions['score'])
predictions['elapsed_time'] = pd.to_numeric(predictions['elapsed_time'])

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

btn_our_story = dbc.Button("Start", id="btn-our-story", className="mt-4")

our_story_block_1 = dbc.Container(
    id="our-story-block-1",
    children=[btn_our_story]
)

btn_hide_our_story = dbc.Button("Hide Our Story", id="btn-hide-our-story" ,className="mt-2 mb-2")

interval = dcc.Interval(id="countdown-interval", interval=2000, n_intervals=0, disabled=True)

our_story_text_list = [
    "The story begins in 5 seconds",
    "The story begins in 3 seconds",
    "The story begins in 1 seconds",
    "The story begins",
    "Accessing the internet...",
    "Finding a good labeled training data...",
    "Found one on kaggle.com!",
    "Applying a mysterious algorithm to train a model...",
    "Training the model...",
    "Applying speep hack...",
    "Training finished!",
    "Now, use the trained model to analyze live data...",
    "Fetching IMDb reviews...",
    "500 reviews fetched!",
    "Applying sentiment analysys...",
    "Done!",
    "The result is..."
]
our_story_text = dbc.Container("The story begins", id="our-story-text")

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

our_story_block_2 = dbc.Container(
    id="our-story-block-2",
    children=[btn_hide_our_story, our_story_text, chart_group, btn_hide_our_story],
    style={'display': 'none'}
)

# btn_try_model = dbc.Button("Try The Model", id="btn-try-model", n_clicks=0)

try_model_content = dbc.Container(
    id="try-model-content",
    children=[
        dbc.Textarea(
            id="try-model-input",
            size="lg",
            placeholder="Type a movie review...",
            className="m-2",
        ),
        dbc.Button("Analyze")
    ],
    # style={'display': 'none'},
)

btn_your_story = dbc.Container()

our_story_container = dbc.Container(
    id="our-story-container",
    children=[
        our_story_block_1,
        our_story_block_2,
    ],
    style = {'display': 'none'},
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

your_story_container = dbc.Container(
    id="your-story-container",
    children=[
    ],
    style = {'display': 'none'},
)

# Define app layout
# app.layout = dbc.Container(children=[chart_group])
app.layout = dbc.Container(
    children=[
        interval,
        dcc.Tabs(id='main-tabs', value='our-story-container', children=[
            dcc.Tab(label='Our Story', value='our-story-container'),
            dcc.Tab(label='Try The Model', value='try-model-container'),
            dcc.Tab(label='Your Story', value='your-story-container'),
        ]),
        dbc.Container(
            id='main-tabs-content',
            children=[our_story_container, try_model_container, your_story_container], # hidden by default
        ),
        
    ],
    className="mt-4"
)

@app.callback(
    # Output('main-tabs-content', 'children'),
    Output('our-story-container', 'style'),
    Output('try-model-container', 'style'),
    Output('your-story-container', 'style'),
    Input('main-tabs', 'value'),
)
def switch_main_tabs(value):
    if value == 'our-story-container':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif value == 'try-model-container':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif value == 'your-story-container':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return dbc.Container("Something wrong with the main tabs")

@app.callback(
    Output('our-story-block-2', 'style', allow_duplicate=True),
    Output('btn-our-story', 'style', allow_duplicate=True),
    Output('countdown-interval', 'disabled'),
    Input('btn-our-story', 'n_clicks'),
    prevent_initial_call=True,
)
def show_our_story_block_2(n_clicks):
    """ Show our-story-block-2 and hide btn-our-story """
    return {'display': 'block'}, {'display': 'none'}, False

@app.callback(
    Output('btn-our-story', 'style', allow_duplicate=True),
    Output('our-story-block-2', 'style', allow_duplicate=True),
    Input('btn-hide-our-story', 'n_clicks'),
    prevent_initial_call=True,
)
def hide_our_story(n_clicks):
    return {'display': 'block'}, {'display': 'none'}

@app.callback(
    Output('our-story-text', 'children'),
    Output('chart-container', 'style'),
    Input('countdown-interval', 'n_intervals')
)
def update_story_text_and_charts(n_intervals):
    if n_intervals > len(our_story_text_list) - 1:
        return our_story_text_list[len(our_story_text_list) - 1], {'display': 'block'}
    else:
        return our_story_text_list[n_intervals], {'display': 'none'}

# @app.callback(
#     Output('try-model-content', 'style'),
#     Input('btn-try-model', 'n_clicks')
# )
# def show_try_model(n_clicks):
#     if n_clicks > 0:
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}

# Define callbacks for interactive updates
@app.callback(
    Output('confidence-distribution', 'figure'),
    Output('prediction-time-analysis', 'figure'),
    Output('accuracy-vs-confidence', 'figure'),
    # Output('label-distribution', 'figure'),
    Output('label-chart', 'figure'),  # Single graph for displaying selected chart
    Input('confidence-distribution', 'relayoutData'),
    Input('prediction-time-analysis', 'relayoutData'),
    Input('accuracy-vs-confidence', 'relayoutData'),
    # Input('label-distribution', 'relayoutData'),
    Input('label-distribution-tabs', 'value')  # New input for tab selection
)
def update_graphs(confidence_layout, time_layout, accuracy_layout, selected_tab):
    # Perform updates based on user interactions
    # You can use the layout data to customize the displayed content

    # Example: Update confidence distribution chart
    confidence_fig = px.histogram(
        predictions,
        x='score',
        nbins=20,
        title='Confidence Score Distribution'
    )

    # Example: Update prediction time analysis chart
    time_fig = px.box(
        predictions,
        x='elapsed_time',
        title='Prediction Time Analysis'
    )

    # Example: Update accuracy vs. confidence scatter plot
    accuracy_vs_confidence_fig = px.scatter(
        predictions,
        x='score',
        y='elapsed_time',
        title='Accuracy vs. Confidence'
    )

    # Example: Update label distribution chart
    label_distribution_fig = px.bar(
        predictions['label'].value_counts(),
    )

    # Example: Add a pie chart for label info
    label_pie_chart_fig = px.pie(
        predictions,
        names='label',
    )

    # Determine which chart to display based on the selected tab
    if selected_tab == 'tab-bar-chart':
        selected_fig = label_distribution_fig
    else:
        selected_fig = label_pie_chart_fig

    return confidence_fig, time_fig, accuracy_vs_confidence_fig, selected_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


