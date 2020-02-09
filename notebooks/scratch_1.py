# import matplotlib.pyplot as plt
import plotly.offline as pyoff
import plotly.graph_objs as go

def plot_rev(df, col1, col2, cat, title):
    plot_data = [
        go.Scatter(
            x=df[col1],
            y=df[col2],
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": cat},
        title=title
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)