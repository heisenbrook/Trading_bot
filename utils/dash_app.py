from utils.keys import data_folder
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Dash(__name__)
app.title = "Prediction vs real candles"

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
csv_names = [os.path.splitext(f)[0] for f in csv_files]
csv = dict(zip(csv_names, csv_files))


app.layout = html.Div([
    html.H1('BTCUSDT Candlestick Chart 4h', style={'textAlign': 'center'}),
    dcc.Checklist(
        id='toggle-rangeslider',
        options=[{'label': filename, 'value': file} for filename, file in csv.items()],
    ),
    dcc.Graph(id='candlestick-graph')
])

@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('toggle-rangeslider', 'value')
)
def update_graph(csv):
    fig = make_subplots(rows=1, cols=2)
    
    col = 1

    for file in csv:
        file_path = os.path.join(data_folder, file)
        filename = os.path.splitext(file)[0]
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)


        fig.add_trace(go.Candlestick(x = df.index,
                                         high= df['next_high'],
                                         low= df['next_low'],
                                         open= df['next_open'],
                                         close= df['next_close']),
                                         row=1, col=col)
        
        fig.add_annotation(text=filename,
                           row=1, col=col,               
                           xref='x domain', yref='y domain',
                           x=0.5, y=1.2, showarrow=False,
                           font=dict(size=16, color='black'))
        
        fig.update_xaxes(rangeslider= {'visible':True}, row=1, col=col)
                          
        col += 1

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False
    )
    

    return fig

if __name__ == '__main__':
    app.run(debug=True)