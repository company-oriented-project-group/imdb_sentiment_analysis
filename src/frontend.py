import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import ast

# Assume 'predictions' is a DataFrame containing label, score, and time columns
# Replace it with your actual data
df = pd.read_csv('./data/movie_reviews_with_model_prediction.csv')
predictions = pd.DataFrame()
predictions[['label', 'score', 'elapsed_time']] = df['model_prediction'].apply(lambda x: pd.Series(ast.literal_eval(x)))
predictions['score'] = pd.to_numeric(predictions['score'])
predictions['elapsed_time'] = pd.to_numeric(predictions['elapsed_time'])
print(predictions.head())

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(children=[
    html.H1("Sentiment Analysis Insights"),
    
    dcc.Graph(id='confidence-distribution'),
    dcc.Graph(id='prediction-time-analysis'),
    dcc.Graph(id='accuracy-vs-confidence'),
    dcc.Graph(id='label-distribution')
])

# Define callbacks for interactive updates
@app.callback(
    Output('confidence-distribution', 'figure'),
    Output('prediction-time-analysis', 'figure'),
    Output('accuracy-vs-confidence', 'figure'),
    Output('label-distribution', 'figure'),
    Input('confidence-distribution', 'relayoutData'),
    Input('prediction-time-analysis', 'relayoutData'),
    Input('accuracy-vs-confidence', 'relayoutData'),
    Input('label-distribution', 'relayoutData')
)
def update_graphs(confidence_layout, time_layout, accuracy_layout, label_layout):
    # Perform updates based on user interactions
    # You can use the layout data to customize the displayed content
    
    # Example: Update confidence distribution chart
    confidence_fig = px.histogram(predictions, x='score', nbins=20, title='Confidence Score Distribution')
    
    # Example: Update prediction time analysis chart
    time_fig = px.box(predictions, x='elapsed_time', title='Prediction Time Analysis')

    # Example: Update accuracy vs. confidence scatter plot
    accuracy_vs_confidence_fig = px.scatter(predictions, x='score', y='elapsed_time', title='Accuracy vs. Confidence')

    # Example: Update label distribution chart
    label_distribution_fig = px.bar(predictions['label'].value_counts(), title='Distribution of Predicted Labels')

    return confidence_fig, time_fig, accuracy_vs_confidence_fig, label_distribution_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
