from dash import Input, Output

def register_main_tab_callbacks(app):
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
            return dbc.Container("Something wrong with the main tabs. Please contact the administrator")
