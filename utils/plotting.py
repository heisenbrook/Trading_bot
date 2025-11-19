import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import datetime as dt
import os
from tqdm import tqdm

#============================================
# Plotting functions 
#============================================

def plot_closes(targets, preds, folder):
    """
    Plots real vs predicted closes and saves the figure.
    """

    fig = make_subplots(rows=1, cols=2, column_titles=['Real closes','Predicted closes'])

    fig.add_trace(go.Scatter(x = targets.index,
                                         y= targets['next_close'],
                                         mode='markers',
                                         name='Real closes',
                                         line=dict(color='blue')),
                                         row=1, col=1)

    fig.add_trace(go.Scatter(x = targets.index,
                                         y= preds['next_close'],
                                         mode='markers',
                                         name='Predicted closes',
                                         line=dict(color='red')),
                                         row=1, col=2)
    
    fig.update_layout(height=600, 
                      width=1200,
                      title='Real vs predictions',
                      xaxis1=dict(rangeslider=dict(visible=False)),
                      xaxis2=dict(rangeslider=dict(visible=False)))
    

    fig.write_image(os.path.join(folder,'Pred_vs_real_candles.png'))


def plot_predictions(btcusdt, preds_df, folder, LSTM=False):
    """
    Plots the newly predicted candles on the current data.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x = btcusdt.index,
                                         high= btcusdt['high'],
                                         low= btcusdt['low'],
                                         open= btcusdt['open'],
                                         close= btcusdt['close'],
                                         name='Historical candles'))

    fig.add_trace(go.Scatter(x = preds_df.index,
                                         y= preds_df['close'],
                                         mode='lines',
                                         name='Predicted closes',
                                         line=dict(color='red')))
    
    fig.add_trace(go.Scatter(x = preds_df.index.tolist() + preds_df.index.tolist()[::-1],
                                         y= preds_df['range_high'].tolist() + preds_df['range_low'].tolist()[::-1],
                                         fill='toself',
                                         fillcolor='rgba(255, 182, 193, 0.5)',
                                         mode='lines',
                                         name='Predicted ranges',
                                         line=dict(color='red', dash='dash')))
    
    fig.update_layout(height=600, 
                      width=800,
                      title='Predicted candles',
                      xaxis1=dict(rangeslider=dict(visible=False)))
    if LSTM:
        fig.write_image(os.path.join(folder,'New_predicted_candles_LSTM.png'))
    else:
        fig.write_image(os.path.join(folder,'New_predicted_candles_transformer.png'))


def plot_loss(train_losses, test_losses, folder):
    """
    Plots training and test loss over epochs and saves the figure.
    """

    df = pd.DataFrame(dict(train_loss=train_losses, test_loss=test_losses))
    fig = px.line(df, labels={'index': 'Epochs', 'value': 'Loss'},
                  title='Training and Test Loss Over Epochs')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss') 
    fig.write_image(os.path.join(folder, 'Training_loss.png'))


def plot_loss_fine_tuning(train_losses, test_losses, folder):
    """
    Plots training and test loss over epochs and saves the figure.
    Modified for fine-tuning context.
    """
    date_now = dt.now().strftime('%Y%m%d_%H%M%S')

    df = pd.DataFrame(dict(train_loss=train_losses, test_loss=test_losses))
    fig = px.line(df, labels={'index': 'Epochs', 'value': 'Loss'},
                  title='Training and Test Loss Over Epochs')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss') 
    fig.write_image(os.path.join(folder, f'Training_loss_fine_tuning_{date_now}.png'))
