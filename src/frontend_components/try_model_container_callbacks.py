from dash import Input, Output, State
from models.keras import predict_sentiment

def register_try_model_container_callbacks(app):
    @app.callback(
        # Output('analyzing-text', 'children'),
        Output('sentiment-result', 'children'),
        Input('btn-analyze', 'n_clicks'),
        State('try-model-input', 'value'),
    )
    def analyze_custom_review(n_clicks, custom_review):
        if n_clicks > 0:
            sentiment_result = predict_sentiment(custom_review).label
            # return 'Analyzing, please wait...', 'Your review looks ' + sentiment_result
            return 'Your review looks ' + sentiment_result
