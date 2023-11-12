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
app = Dash(__name__)

btn_our_story = dbc.Button("Our Story", size="lg", color="primary")

our_story_block_1 = html.Div(
    children=[btn_our_story]
)

btn_hide_story = html.Button("Hide Our Story")

our_story_text = html.Div("our story text")

our_story_block_2 = html.Div(
    children=[btn_hide_story, our_story_text, btn_hide_story],
    style={'display': 'none'}
)

confidence_distribution = dcc.Graph(id='confidence-distribution')
prediction_time_analysis = dcc.Graph(id='prediction-time-analysis')
accuracy_vs_confidence = dcc.Graph(id='accuracy-vs-confidence')

label_distribution = html.Div(
    id="label-distribution",
    children=[
        html.Div("Distribution of Predicted Labels"),
        html.Br(),
        dcc.Tabs(id='tabs', value='tab-pie-chart', children=[
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

chart_group = html.Div(
    id="chart-container",
    children=[
        html.H1("Sentiment Analysis Insights"),
        confidence_distribution,
        prediction_time_analysis,
        accuracy_vs_confidence,
        label_distribution
    ],
    style={'display': 'block'}
)

btn_try_model = html.Div()

btn_your_story = html.Div()

our_story_container = html.Div(
    id="our-story-container",
    children=[
        our_story_block_1,
        our_story_block_2,
        chart_group,
    ]
)

try_model_container = html.Div(
    id="try-model-container",
    children=[
    ]
)

your_story_container = html.Div(
    id="your-story-container",
    children=[
    ]
)

# Define app layout
app.layout = html.Div(children=[chart_group])

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
    Input('tabs', 'value')  # New input for tab selection
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


