from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from functools import lru_cache

app = Flask(__name__)

# MODELLERİ VE VERİLERİ LAZY LOAD YAP (gerektiğinde yükle)

@lru_cache()
def load_model():
    return joblib.load('REGRESYON_MODEL.joblib')

@lru_cache()
def load_main_data():
    df = pd.read_csv('kpmart.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

@lru_cache()
def load_dfc_data():
    df = pd.read_csv('dfc_output.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df[df['Datetime'].dt.year == 2007]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_variable = None
    fig1 = None
    fig2 = None

    data = load_main_data()
    dfc_data = load_dfc_data()
    model = load_model()

    variable_list = dfc_data.columns.tolist()

    # Grafik 1 - Gerçek ve Tahmin Kp Değeri
    fig1 = px.line(data, x='Datetime', y=['Kps'], 
                   title='Gerçek ve Tahmin Kp Değeri (3 saat sonrası)')
    fig1.update_traces(name='Gerçek Kp', line=dict(color='blue'))
    fig1.add_scatter(x=data['Datetime'], y=data['Predicted_Kp'], mode='lines', name='Tahmin Kp', line=dict(color='red'))
    fig1.update_layout(
        plot_bgcolor="#101c33", 
        paper_bgcolor="#101c33", 
        font_color="white",
        xaxis_title="Datetime",
        yaxis_title="Kps / Predicted Kps",
        legend_title="Değişkenler", 
        xaxis=dict(
            showgrid=False,
            showline=True,
            tickangle=45,
            tickmode='linear',
            ticks='outside'
        ),
        yaxis=dict(showgrid=True)
    )

    if request.method == 'POST':
        if request.form.get('input_1'):
            inputs = [float(request.form[f'input_{i}']) for i in range(1, 17)]
            prediction = model.predict([inputs])[0]

    selected_variable = request.args.get('variable_select')

    if selected_variable:
        fig2 = px.line(dfc_data, x='Datetime', y=selected_variable, title=f'{selected_variable} Zaman Grafiği')
        fig2.update_traces(line=dict(color='#75b8e5'))
        fig2.update_layout(
            plot_bgcolor="#101c33", 
            paper_bgcolor="#101c33", 
            font_color="white",
            xaxis_title="Datetime",
            yaxis_title=selected_variable,
            xaxis=dict(
                showgrid=False,
                showline=True,
                tickangle=45,
                tickmode='linear',
                tickformat="%Y-%m-%d",
                dtick="M1",
                ticks='outside'
            ),
            yaxis=dict(showgrid=True)
        )

    graph_html1 = pio.to_html(fig1, full_html=False) if fig1 else None
    graph_html2 = pio.to_html(fig2, full_html=False) if fig2 else None
    
    return render_template('index.html', prediction=prediction, graph_html1=graph_html1, graph_html2=graph_html2,
                           variable_list=variable_list, selected_variable=selected_variable)

if __name__ == "__main__":
    app.run(debug=False)
