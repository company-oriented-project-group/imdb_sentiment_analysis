from dash import Input, Output

our_story_text_list = [
    "The story begins in 3 seconds",
    # "The story begins in 2 seconds",
    # "The story begins in 1 seconds",
    # "The story begins",
    # "Accessing the internet...",
    # "Finding a good labeled training data...",
    # "Found one on kaggle.com!",
    # "Applying a mysterious algorithm to train a model...",
    # "Training the model...",
    # "Applying speep hack...",
    # "Training finished!",
    # "Now, use the trained model to analyze live data...",
    # "Fetching IMDb reviews...",
    # "500 reviews fetched!",
    # "Applying sentiment analysys...",
    # "Done!",
    "The result is...",
]

def register_our_story_container_callbacks(app):
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
        Output('our-story-text', 'children'),
        Output('chart-container', 'style'),
        Input('countdown-interval', 'n_intervals')
    )
    def update_story_text_and_charts(n_intervals):
        if n_intervals > len(our_story_text_list) - 1:
            return our_story_text_list[len(our_story_text_list) - 1], {'display': 'block'}
        else:
            return our_story_text_list[n_intervals], {'display': 'none'}


    @app.callback(
        Output('btn-our-story', 'style', allow_duplicate=True),
        Output('our-story-block-2', 'style', allow_duplicate=True),
        Input('btn-hide-our-story', 'n_clicks'),
        prevent_initial_call=True,
    )
    def hide_our_story(n_clicks):
        return {'display': 'block'}, {'display': 'none'}
