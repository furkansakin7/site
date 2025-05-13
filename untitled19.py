# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:20:04 2024

@author: sakin
"""

from flask import Flask, render_template
import plotly.express as px
import pandas as pd

app = Flask(__name__)

# Örnek veri seti
data = {
    'Time': ['8 Dec 00-03h', '8 Dec 03-06h', '8 Dec 06-09h', '8 Dec 09-12h'],
    'Kp': [1, 2, 3, 2]
}
df = pd.DataFrame(data)

# Plotly ile grafik oluşturma
def create_graph():
    fig = px.bar(df, x='Time', y='Kp', title="Kp Index", labels={'Time': 'Zaman', 'Kp': 'Kp Değeri'})
    graph_html = fig.to_html(full_html=False)
    return graph_html

@app.route('/')
def index():
    graph_html = create_graph()
    return render_template("index.html", graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
