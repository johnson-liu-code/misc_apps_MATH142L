import numpy as np
from scipy.stats import t, norm
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
server = app.server

x = np.linspace(-5, 5, 1000)

app.layout = html.Div([
    html.H2("t-test for two populations", style={'fontSize': '30px'}),

    html.Div([
        html.Div([
            html.H4("Population 1", style={'fontSize': '22px'}),
            html.Label("Mean:", style={'fontSize': '20px'}),
            dcc.Input(id='mean1', type='number', value=34.5, step=0.1,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px', 'marginRight': '10px'}),
            html.Label("Std Dev:", style={'fontSize': '20px'}),
            dcc.Input(id='std1', type='number', value=2.1, step=0.1,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px', 'marginRight': '10px'}),
            html.Label("Sample Size:", style={'fontSize': '20px'}),
            dcc.Input(id='n1', type='number', value=15, min=2,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px'}),
        ], style={'marginBottom': '20px', 'width': '45%', 'display': 'inline-block'}),

        html.Div([
            html.H4("Population 2", style={'fontSize': '22px'}),
            html.Label("Mean:", style={'fontSize': '20px'}),
            dcc.Input(id='mean2', type='number', value=36.3, step=0.1,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px', 'marginRight': '10px'}),
            html.Label("Std Dev:", style={'fontSize': '20px'}),
            dcc.Input(id='std2', type='number', value=3.8, step=0.1,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px', 'marginRight': '10px'}),
            html.Label("Sample Size:", style={'fontSize': '20px'}),
            dcc.Input(id='n2', type='number', value=13, min=2,
                      style={'width': '160px', 'height': '40px', 'fontSize': '18px'}),
        ], style={'marginBottom': '20px', 'width': '45%', 'display': 'inline-block'}),
    ]),

    html.Label("Test Type:", style={'fontSize': '20px'}),
    dcc.RadioItems(
        id='test-type',
        options=[
            {'label': 'μ₁ < μ₂  (Left-tailed)', 'value': 'left'},
            {'label': 'μ₁ > μ₂  (Right-tailed)', 'value': 'right'},
            {'label': 'μ₁ ≠ μ₂  (Two-tailed)', 'value': 'two'},
        ],
        value='two',
        labelStyle={'display': 'inline-block', 'margin-right': '15px'},
        style={'fontSize': '18px'}
    ),

    html.Div(id='decision-output', style={
        'fontSize': '24px',
        'fontWeight': 'bold',
        'color': '#222',
        'marginTop': '10px',
        'marginBottom': '20px'
    }),

    html.Label("Confidence Level (%):", style={'fontSize': '20px'}),
    dcc.Slider(
        id='conf-level',
        min=50,
        max=99.9,
        step=0.1,
        value=95,
        marks={i: f'{i}%' for i in range(50, 100, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Br(),
    dcc.Graph(id='pdf-plot'),
    dcc.Graph(id='sample-plot')
])


@app.callback(
    Output('pdf-plot', 'figure'),
    Output('sample-plot', 'figure'),
    Output('decision-output', 'children'),
    Input('mean1', 'value'),
    Input('std1', 'value'),
    Input('n1', 'value'),
    Input('mean2', 'value'),
    Input('std2', 'value'),
    Input('n2', 'value'),
    Input('test-type', 'value'),
    Input('conf-level', 'value')
)
def update_plots(mean1, std1, n1, mean2, std2, n2, test_type, conf_level):
    df = max(min(n1, n2) - 1, 1)
    pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    t_value = (mean1 - mean2) / pooled_se
    alpha = 1 - conf_level / 100
    y_pdf = t.pdf(x, df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_pdf, mode='lines',
                             name='t-distribution',
                             line=dict(color='magenta', width=4)))

    if test_type == 'left':
        p_value = t.cdf(t_value, df)
        crit_t = t.ppf(alpha, df)
        x_p = x[x <= t_value]
        x_crit = x[x <= crit_t]
    elif test_type == 'right':
        p_value = 1 - t.cdf(t_value, df)
        crit_t = t.ppf(1 - alpha, df)
        x_p = x[x >= t_value]
        x_crit = x[x >= crit_t]
    else:
        p_value = t.cdf(-abs(t_value), df) + (1 - t.cdf(abs(t_value), df))
        crit_t = t.ppf(1 - alpha / 2, df)
        x_p_left = x[x <= -abs(t_value)]
        x_p_right = x[x >= abs(t_value)]
        x_crit_left = x[x <= -crit_t]
        x_crit_right = x[x >= crit_t]

    if test_type in ['left', 'right']:
        fig.add_trace(go.Scatter(x=x_p, y=t.pdf(x_p, df), fill='tozeroy', mode='none',
                                 name=f"p = {p_value:.4f}", fillcolor='rgba(255, 0, 0, 0.4)'))
    else:
        fig.add_trace(go.Scatter(x=x_p_left, y=t.pdf(x_p_left, df), fill='tozeroy', mode='none',
                                 name=f"Left p = {p_value/2:.4f}", fillcolor='rgba(255, 0, 0, 0.4)'))
        fig.add_trace(go.Scatter(x=x_p_right, y=t.pdf(x_p_right, df), fill='tozeroy', mode='none',
                                 name=f"Right p = {p_value/2:.4f}", fillcolor='rgba(255, 0, 0, 0.4)'))

    def add_vertical_fill(fig, x_fill):
        for xi in np.linspace(x_fill[0], x_fill[-1], 50):
            yi = t.pdf(xi, df)
            fig.add_shape(
                type='line',
                x0=xi, x1=xi,
                y0=0, y1=yi,
                line=dict(color='green', width=2, dash='dot'),
                layer='below'
            )

    if test_type == 'left':
        add_vertical_fill(fig, x_crit)
        fig.add_vline(x=crit_t, line_dash='dash', line_color='green', line_width=4,
                      annotation_text=f"<b>critical t value = {crit_t:.2f}</b>",
                      annotation_position="top left", annotation_standoff=20,
                      annotation_yanchor="bottom")
    elif test_type == 'right':
        add_vertical_fill(fig, x_crit)
        fig.add_vline(x=crit_t, line_dash='dash', line_color='green', line_width=4,
                      annotation_text=f"<b>critical t value = {crit_t:.2f}</b>",
                      annotation_position="top right", annotation_standoff=20,
                      annotation_yanchor="bottom")
    else:
        add_vertical_fill(fig, x_crit_left)
        add_vertical_fill(fig, x_crit_right)
        fig.add_vline(x=-crit_t, line_dash='dash', line_color='green', line_width=4,
                      annotation_text=f"<b>critical t value = {-crit_t:.2f}</b>",
                      annotation_position="top left", annotation_standoff=20,
                      annotation_yanchor="bottom")
        fig.add_vline(x=crit_t, line_dash='dash', line_color='green', line_width=4,
                      annotation_text=f"<b>critical t value = {crit_t:.2f}</b>",
                      annotation_position="top right", annotation_standoff=20,
                      annotation_yanchor="bottom")

    fig.add_vline(
        x=t_value,
        line_dash='solid',
        line_color='red',
        line_width=4,
        # annotation_text=f"<b>t score = {t_value:.2f}</b>",
        # annotation_position="top",
        # annotation_standoff=20
    )

    fig.add_shape(
        type="line",
        x0=t_value, x1=t_value,
        y0=-0.075, y1=max(t.pdf(t_value, df), 0.1),  # extends below x-axis
        line=dict(color="red", width=4),
        layer="above"
    )

    fig.add_annotation(
        x=t_value,
        y=-0.025,
        text=f"<b>t score = {t_value:.2f}</b>",
        showarrow=False,
        font=dict(size=16),
        yanchor="top",
        xanchor="left" if t_value < 0 else "right"
    )

    fig.update_layout(
        title=f"{test_type} test: p = {p_value:.4f}, t = {t_value:.2f}, df = {df}, α = {alpha:.3f}",
        xaxis_title='t',
        yaxis_title='Probability Density',
        template='plotly_white',
        font=dict(size=18),
        showlegend=True
    )

    decision = "✅ Reject H₀" if p_value < alpha else "❌ Fail to reject H₀"

    x1 = np.linspace(mean1 - 4 * std1 / np.sqrt(n1), mean1 + 4 * std1 / np.sqrt(n1), 300)
    x2 = np.linspace(mean2 - 4 * std2 / np.sqrt(n2), mean2 + 4 * std2 / np.sqrt(n2), 300)
    y1 = norm.pdf(x1, mean1, std1 / np.sqrt(n1))
    y2 = norm.pdf(x2, mean2, std2 / np.sqrt(n2))

    sample_fig = go.Figure()
    sample_fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Sample 1',
                                    line=dict(color='royalblue', width=4)))
    sample_fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='Sample 2',
                                    line=dict(color='darkorange', width=4)))

    sample_fig.add_vline(
        x=mean1,
        line_color='royalblue',
        line_dash='dot',
        line_width=5,
        annotation_text=f"<b>Mean 1 = {mean1:.2f}</b>",
        annotation_position='top left',
        annotation_standoff=30,
        annotation_yanchor='bottom'
    )
    sample_fig.add_vline(
        x=mean2,
        line_color='darkorange',
        line_dash='dot',
        line_width=5,
        annotation_text=f"<b>Mean 2 = {mean2:.2f}</b>",
        annotation_position='top right',
        annotation_standoff=30,
        annotation_yanchor='bottom'
    )

    sample_fig.update_layout(
        title="Sample Distributions (Normal Approximation)",
        xaxis_title="Sample Value",
        yaxis_title="Probability Density",
        template='plotly_white',
        font=dict(size=18)
    )

    return fig, sample_fig, decision


if __name__ == '__main__':
    app.run(debug=True)
