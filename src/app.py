from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from frontend_components.our_story_container import our_story_container
from frontend_components.your_story_container import your_story_container
from frontend_components.try_model_container import try_model_container
from frontend_components.main_tab import main_tab
from frontend_components.main_tab_callbacks import register_main_tab_callbacks
from register_all_callbacks import register_all_callbacks

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

interval = dcc.Interval(id="countdown-interval", interval=500, n_intervals=0, disabled=True)

# Define app layout
app.layout = dbc.Container(
    children=[
        interval,
        main_tab,
    ],
    className="mt-4"
)

register_all_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
