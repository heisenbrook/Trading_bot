import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Dash(__name__)
app.title = "Prediction vs real candles"


data_folder = os.path.join(os.path.dirname(__file__), '../data')
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]


app.layout = html.Div([
    html.H4("Candlestick Chart Viewer", style={'textAlign': 'center'}),
    dcc.Checklist(
        id='toggle-rangeslider',
        options=[{'label': file, 'value': file} for file in csv_files]
    ),
    dcc.Graph(id='candlestick-graph')
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('toggle-rangeslider', 'value')
)
def update_graph(csv_files):
    fig = make_subplots(rows=1, cols=2,
                        vertical_spacing=0.1,
                        subplot_titles=[file for file in csv_files],
                        specs=[[{'type': 'candlestick'}, {'type': 'candlestick'}]])
    col = 1

    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)


        fig.add_trace(go.Candlestick(x = df.index,
                                         high= df['next_high'],
                                         low= df['next_low'],
                                         open= df['next_open'],
                                         close= df['next_close']),
                                         row=1, col=col)
        col += 1

        fig.update_layout(xaxis_rangeslider_visible=False in csv_files,
                          xaxis1=dict(rangeslider=dict(visible=False)),
                          xaxis2=dict(rangeslider=dict(visible=False)),
                          height=600, width=1200)
    

    return fig

if __name__ == '__main__':
    app.run(debug=True)