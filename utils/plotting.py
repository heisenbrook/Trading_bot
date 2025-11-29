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
    if 'LSTM' in folder:
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

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
    

    fig.write_image(os.path.join(folder,f'Pred_vs_real_candles_{model_type}.png'))


def plot_predictions(btcusdt, preds_df, folder, input_model):
    """
    Plots the newly predicted candles on the current data.
    """
    if input_model == 'lstm':
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

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

    fig.write_image(os.path.join(folder,f'New_predicted_candles_{model_type}.png'))


def plot_loss(train_losses, test_losses, folder, timeframe):
    """
    Plots training and test loss over epochs and saves the figure.
    """
    if 'LSTM' in folder:
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

    df = pd.DataFrame(dict(train_loss=train_losses, test_loss=test_losses))
    fig = px.line(df, labels={'index': 'Epochs', 'value': 'Loss'},
                  title='Training and Test Loss Over Epochs')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss') 
    fig.write_image(os.path.join(folder, f'Training_loss_{model_type}_{timeframe}.png'))


def plot_loss_fine_tuning(train_losses, test_losses, folder, timeframe):
    """
    Plots training and test loss over epochs and saves the figure.
    Modified for fine-tuning context.
    """
    if 'LSTM' in folder:
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

    date_now = dt.now().strftime('%Y%m%d_%H%M%S')

    df = pd.DataFrame(dict(train_loss=train_losses, test_loss=test_losses))
    fig = px.line(df, labels={'index': 'Epochs', 'value': 'Loss'},
                  title='Training and Test Loss Over Epochs')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss') 
    fig.write_image(os.path.join(folder, f'Training_loss_fine_tuning_{model_type}_{timeframe}_{date_now}.png'))


def plot_class_metrics(metrics_dict: dict, folder, timeframe):
    """
    Plots classification metrics and saves the figure.
    Displays Confusion Matrix and ROC Curve.
    """
    if 'LSTM' in folder:
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Confusion Matrix', f'ROC Curve (AUC = {metrics_dict["roc_auc"]:.2f})'),
                        specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
                        )
    
    # Confusion Matrix
    cm = metrics_dict['confusion_matrix']
    fig.add_trace(go.Heatmap(z=cm,
                             x=['Predicted Down (0)', 'Predicted Up (1)'],
                             y=['Actual Down (0)', 'Actual Up (1)'], 
                             colorscale='Blues',
                             showscale=True,
                             text=cm,
                             texttemplate="%{text}"),
                             row=1, col=1)
    # ROC Curve
    fpr = metrics_dict['fpr']
    tpr = metrics_dict['tpr']
    fig.add_trace(go.Scatter(x=fpr, 
                             y=tpr, 
                             mode='lines',
                             name='ROC Curve',
                             line=dict(color='red')),
                             row=1, col=2)
    
    # Random Classifier Line
    fig.add_trace(go.Scatter(x=[0, 1], 
                             y=[0, 1], 
                             mode='lines',
                             name='Random Classifier',
                             line=dict(color='blue', dash='dash')),
                             row=1, col=2)
    
    fig.update_layout(height=600, 
                      width=1000,
                      title='Classification Metrics',
                      showlegend=True)

    fig.update_xaxes(title_text='False Positive Rate', row=1, col=2)
    fig.update_yaxes(title_text='True Positive Rate', row=1, col=2) 
    fig.write_image(os.path.join(folder,f'Classification_Metrics_{model_type}_{timeframe}.png'))   


def plot_class_signals(btcusdt, preds, folder, input_model):
    """
    Plots real vs predicted signals and saves the figure.
    """
    if input_model == 'lstm':
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x = btcusdt.index,
                                 high= btcusdt['high'],
                                 low= btcusdt['low'],
                                 open= btcusdt['open'],
                                 close= btcusdt['close'],
                                 name='Pricess'))
    
    common_idx = btcusdt.index.intersection(preds.index)

    if len(common_idx) > 0:
        valid_preds = preds.loc[common_idx]
        valid_prices = btcusdt.loc[common_idx]

        buy_signals = valid_preds[valid_preds['signal'] == 'BUY']
        sell_signals = valid_preds[valid_preds['signal'] == 'SELL']

        if not buy_signals.empty:
            buy_signals = valid_prices.loc[buy_signals.index]['low']

            fig.add_trace(go.Scatter(x = buy_signals.index,
                                     y= buy_signals * 0.998,
                                     mode='markers',
                                     name='BUY Signals',
                                     marker=dict(color='green', 
                                                 symbol='triangle-up', 
                                                 size=10,
                                                 line=dict(color='darkgreen', width=1)),
                                     text=[f'Prob Up: {prob:.2%}' for prob in buy_signals['prob_up']],
                                     hoverinfo='text+x+y'))
        
        if not sell_signals.empty:
            buy_signals = valid_prices.loc[buy_signals.index]['low']

            fig.add_trace(go.Scatter(x = sell_signals.index,
                                     y= sell_signals * 0.998,
                                     mode='markers',
                                     name='SELL Signals',
                                     marker=dict(color='red', 
                                                 symbol='triangle-down', 
                                                 size=10,
                                                 line=dict(color='darkred', width=1)),
                                     text=[f'Prob Down: {prob:.2%}' for prob in buy_signals['prob_up']],
                                     hoverinfo='text+x+y'))
        
    fig.update_layout(height=600, 
                      width=800,
                      title=f'Trading signals {model_type} model',
                      xaxis1=dict(rangeslider=dict(visible=False)))
    
    fig.write_image(os.path.join(folder,f'Trading_signals_{model_type}.png'))

   

