from dash import Input, Output
import plotly.express as px
import pandas as pd
import ast

# 'predictions' is a DataFrame containing label, score, and elapsed_time columns
df = pd.read_csv(r'./data/movie_reviews_with_model_prediction.csv')
df['model_prediction'] = df['model_prediction'].apply(ast.literal_eval)
predictions = pd.DataFrame(df['model_prediction'].tolist(), index=df.index)
predictions['score'] = pd.to_numeric(predictions['score'])
predictions['elapsed_time'] = pd.to_numeric(predictions['elapsed_time'])

def register_chart_container_callbacks(app):
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