# import matplotlib.pyplot as plt
import plotly.offline as pyoff
import plotly.graph_objs as go


def plot_rev(df, col1, col2, cat, title, plot_type=go.Scatter):
    plot_data = [
        plot_type(
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

def query_plot(x, y, cat, title):
    plot_data = [
        go.Scatter(
            x=x,
            y=y,
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": cat},
        title=title
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)