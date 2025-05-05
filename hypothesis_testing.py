import numpy as np
from scipy.stats import norm, t

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, ctx


app = Dash(__name__)
server = app.server

mu_null = 0

def generate_figure(df, alpha, delta, sigma, show_beta=True, show_power=True):
    mu_alt = mu_null + delta
    x = np.linspace(mu_null - 4 * sigma, mu_alt + 4 * sigma, 1000)
    null_dist = t(df, loc=mu_null, scale=sigma)
    alt_dist = norm(loc=mu_null + delta, scale=sigma)

    crit_val = t.ppf(1 - alpha, df)

    x_alpha = x[x > crit_val]
    x_beta = x[x <= crit_val]
    x_power = x[x > crit_val]

    fig = go.Figure()

    # Main distributions
    fig.add_trace(go.Scatter(x=x, y=null_dist.pdf(x), mode='lines', name='H₀',
                             line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=alt_dist.pdf(x), mode='lines', name='H₁',
                             line=dict(color='orange', width=4, dash='dash')))

    # Alpha region
    fig.add_trace(go.Scatter(x=x_alpha, y=null_dist.pdf(x_alpha), fill='tozeroy', mode='none',
                             name='α ("Alpha")', fillcolor='rgba(255,0,0,0.4)'))

    # Beta region
    if show_beta:
        for x_val in x_beta[::5]:
            fig.add_shape(type='line', x0=x_val, x1=x_val, y0=0, y1=alt_dist.pdf(x_val),
                          line=dict(color='rgba(0,0,150,0.6)', width=2, dash='dot'), layer='below')
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='β (Type II error)',
                                 line=dict(color='rgba(0,0,150,0.8)', dash='dot', width=2)))

    # Power region
    if show_power:
        y_vals = np.linspace(0, max(alt_dist.pdf(x_power)), 20)
        for y_val in y_vals:
            x_clip = x_power[alt_dist.pdf(x_power) >= y_val]
            if len(x_clip) > 0:
                fig.add_shape(type='line',
                              x0=x_clip[0], x1=x_clip[-1],
                              y0=y_val, y1=y_val,
                              line=dict(color='rgba(0,150,0,0.6)', width=2, dash='dot'),
                              layer='below')
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='1 - β (Power)',
                                 line=dict(color='rgba(0,150,0,1)', dash='dot', width=2)))

    # Curve labels (non-overlapping)
    fig.add_annotation(x=mu_null - 1.5 * sigma,
                       y=null_dist.pdf(mu_null - 1.5 * sigma) * 0.5,
                       text="<b>H₀</b>",
                       showarrow=False, font=dict(size=16, color='blue'))

    fig.add_annotation(x=mu_alt + 1.5 * sigma,
                       y=alt_dist.pdf(mu_alt + 1.5 * sigma) * 0.5,
                       text="<b>H₁</b>",
                       showarrow=False, font=dict(size=16, color='orange'))

    # Critical value vertical line
    fig.add_vline(x=crit_val, line_color='black', line_dash='dot', line_width=3,
                  annotation=dict(
                      text=f'<b>t = {crit_val:.2f} | α = {alpha:.3f}</b>',
                      font=dict(size=16, color='black'),
                      showarrow=False, yanchor='bottom', y=1.05
                    )
                )

    fig.update_layout(
        title='',
        xaxis_title='t Score',
        yaxis_title='Probability Density',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'),
        template='plotly_white',
        font=dict(size=20)
    )
    return fig

# Add 4 subplots below the existing plot showing alpha, beta, 1-alpha, and 1-beta regions
def generate_subplots(df, alpha, delta, sigma):
    mu_null = 0
    mu_alt = mu_null + delta
    x = np.linspace(mu_null - 4 * sigma, mu_alt + 4 * sigma, 1000)
    null_pdf = t(df, loc=mu_null, scale=sigma).pdf(x)
    alt_pdf = norm(loc=mu_null + delta, scale=sigma).pdf(x)

    crit_val = t.ppf(1 - alpha, df)

    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[
        "α (Type I error)", "1 − β (Power)",
        "1 − α", "β (Type II Error)"
    ])

    # Region masks
    x_alpha = x > crit_val
    x_beta = x <= crit_val

    # Type I Error (α)
    fig2.add_trace(go.Scatter(x=x, y=null_pdf, mode='lines', line=dict(color='blue')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=x, y=alt_pdf, mode='lines', line=dict(color='orange', dash='dash')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=x[x_alpha], y=null_pdf[x_alpha], fill='tozeroy', mode='none',
                             fillcolor='rgba(255,0,0,0.4)'), row=1, col=1)

    # Power (1 − β)
    fig2.add_trace(go.Scatter(x=x, y=null_pdf, mode='lines', line=dict(color='blue')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=x, y=alt_pdf, mode='lines', line=dict(color='orange', dash='dash')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=x[x_alpha], y=alt_pdf[x_alpha], fill='tozeroy', mode='none',
                             fillcolor='rgba(0,255,0,0.3)'), row=1, col=2)

    # True Negative (1 − α)
    fig2.add_trace(go.Scatter(x=x, y=null_pdf, mode='lines', line=dict(color='blue')), row=2, col=1)
    fig2.add_trace(go.Scatter(x=x, y=alt_pdf, mode='lines', line=dict(color='orange', dash='dash')), row=2, col=1)
    fig2.add_trace(go.Scatter(x=x[x_beta], y=null_pdf[x_beta], fill='tozeroy', mode='none',
                             fillcolor='rgba(0,200,255,0.3)'), row=2, col=1)

    # Type II Error (β)
    fig2.add_trace(go.Scatter(x=x, y=null_pdf, mode='lines', line=dict(color='blue')), row=2, col=2)
    fig2.add_trace(go.Scatter(x=x, y=alt_pdf, mode='lines', line=dict(color='orange', dash='dash')), row=2, col=2)
    fig2.add_trace(go.Scatter(x=x[x_beta], y=alt_pdf[x_beta], fill='tozeroy', mode='none',
                             fillcolor='rgba(128,0,255,0.3)'), row=2, col=2)

    fig2.update_layout(height=800, width=950, title_text="Error Region Subplots", title_x=0.5,
                      showlegend=False, font=dict(size=16))
    return fig2

# Layout
# app.layout = html.Div([
#     html.H2("Fully Synchronized Hypothesis Testing App", style={'textAlign': 'center', 'fontSize': '30px'}),

#     html.Div([
#         html.Label("Critical t-value (syncs with α and slider):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#         dcc.Input(id='crit-input', type='number', debounce=True,
#                   style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#         dcc.Slider(id='crit-slider', min=-2, max=5, step=0.01, value=1.96,
#                    marks={i: f"{i}" for i in range(-2, 6)},
#                    tooltip={"placement": "bottom", "always_visible": True}),
#     ], style={'marginBottom': '20px'}),

#     html.Div([
#         html.Label("Alpha (syncs with t):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#         dcc.Input(id='alpha-input', type='number', min=0.0001, max=0.9999, step=0.0001,
#                   debounce=True, style={'width': '100px', 'fontSize': '20px'}),
#     ], style={'marginBottom': '20px'}),

#     html.Div([
#         html.Label("Effect Size (Δ = μ₁ - μ₀):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#         dcc.Input(id='delta-input', type='number', value=1.5, debounce=True,
#                   style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#         dcc.Slider(id='delta-slider', min=-4, max=4, step=0.1, value=1.5,
#                    marks={i: f"{i}" for i in range(-4, 5)},
#                    tooltip={"placement": "bottom", "always_visible": True}),
#     ], style={'marginBottom': '20px'}),

#     html.Div([
#         html.Label("Standard Deviation (σ):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#         dcc.Input(id='sigma-input', type='number', value=1.2, debounce=True,
#                   style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#         dcc.Slider(id='sigma-slider', min=0.1, max=3, step=0.1, value=1.2,
#                    marks={i: f"{i}" for i in range(1, 4)},
#                    tooltip={"placement": "bottom", "always_visible": True}),
#     ], style={'marginBottom': '20px'}),

#     html.Button("Toggle Beta Region", id='toggle-beta', n_clicks=0, style={'fontSize': '18px', 'marginRight': '15px'}),
#     html.Button("Toggle Power Region", id='toggle-power', n_clicks=0, style={'fontSize': '18px'}),

#     html.Br(), html.Br(),
#     dcc.Graph(id='hypothesis-plot'),
#     dcc.Graph(id='subplots')
# ])

# app.layout = html.Div([
#     html.H2("Fully Synchronized Hypothesis Testing App", style={'textAlign': 'center', 'fontSize': '30px'}),

#     html.Div([
#         html.Div([
#             dcc.Graph(id='subplots')  # Subplots on top-right
#         ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

#         html.Div([
#             html.Div([
#                 html.Label("Critical t-value (syncs with α and slider):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='crit-input', type='number', debounce=True,
#                           style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='crit-slider', min=-2, max=5, step=0.01, value=1.96,
#                            marks={i: f"{i}" for i in range(-2, 6)},
#                            tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px'}),

#             html.Div([
#                 html.Label("Alpha (syncs with t):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='alpha-input', type='number', min=0.0001, max=0.9999, step=0.0001,
#                           debounce=True, style={'width': '100px', 'fontSize': '20px'}),
#             ], style={'marginBottom': '20px'}),

#             html.Div([
#                 html.Label("Effect Size (Δ = μ₁ - μ₀):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='delta-input', type='number', value=1.5, debounce=True,
#                           style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='delta-slider', min=-4, max=4, step=0.1, value=1.5,
#                            marks={i: f"{i}" for i in range(-4, 5)},
#                            tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px'}),

#             html.Div([
#                 html.Label("Standard Deviation (σ):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='sigma-input', type='number', value=1.2, debounce=True,
#                           style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='sigma-slider', min=0.1, max=3, step=0.1, value=1.2,
#                            marks={i: f"{i}" for i in range(1, 4)},
#                            tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px'}),

#             html.Button("Toggle Beta Region", id='toggle-beta', n_clicks=0,
#                         style={'fontSize': '18px', 'marginRight': '15px'}),
#             html.Button("Toggle Power Region", id='toggle-power', n_clicks=0, style={'fontSize': '18px'})
#         ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
#     ], style={'display': 'flex', 'justifyContent': 'space-between'}),

#     html.Br(), html.Br(),
#     dcc.Graph(id='hypothesis-plot')
# ])

# app.layout = html.Div([
#     html.H2("Fully Synchronized Hypothesis Testing App", style={'textAlign': 'center', 'fontSize': '30px'}),

#     html.Div([
#         html.Div([
#             html.Div([
#                 html.Label("Critical t-value (syncs with α and slider):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='crit-input', type='number', debounce=True,
#                         style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='crit-slider', min=-4, max=4, step=0.01, value=1.9738243523417665,
#                         marks={i: f"{i}" for i in range(-4, 5)},
#                         tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px', 'marginRight': '1400px'}),

#             html.Div([
#                 html.Label("α (syncs with critical t-value):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='alpha-input', type='number', min=0.0001, max=0.9999, step=0.0001, value=0.05,
#                         debounce=True, style={'width': '100px', 'fontSize': '20px'}),
#             ], style={'marginBottom': '20px', 'marginRight': '1400px'}),

#             html.Div([
#                 html.Label("Effect Size (Δ = μ₁ - μ₀):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='delta-input', type='number', value=1.5, debounce=True,
#                         style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='delta-slider', min=-4, max=4, step=0.1, value=1.5,
#                         marks={i: f"{i}" for i in range(-4, 5)},
#                         tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px', 'marginRight': '1400px'}),

#             html.Div([
#                 html.Label("Standard Deviation (σ):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
#                 dcc.Input(id='sigma-input', type='number', value=1.2, debounce=True,
#                         style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
#                 dcc.Slider(id='sigma-slider', min=0.1, max=3, step=0.1, value=1.2,
#                         marks={i: f"{i}" for i in range(1, 4)},
#                         tooltip={"placement": "bottom", "always_visible": True}),
#             ], style={'marginBottom': '20px', 'marginRight': '1400px'}),

#             html.Button("Toggle Beta Region", id='toggle-beta', n_clicks=0,
#                         style={'fontSize': '18px', 'marginRight': '15px'}),
#             html.Button("Toggle Power Region", id='toggle-power', n_clicks=0, style={'fontSize': '18px'})
#         ]),

#         html.Div([
#             html.Img(src='/assets/confusion_matrix.png', style={'height': '300px', 'marginTop': '20px'})
#         ])
#     ]),

#     html.Div([
#         html.Div([dcc.Graph(id='hypothesis-plot')], style={'width': '48%', 'display': 'inline-block'}),
#         html.Div([dcc.Graph(id='subplots')], style={'width': '48%', 'display': 'inline-block', 'paddingLeft': '2%'})
#     ], style={'display': 'flex', 'justifyContent': 'space-between'})
# ])

app.layout = html.Div([
    html.H2("Hypothesis Testing", style={'textAlign': 'center', 'fontSize': '30px'}),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Critical t-value (syncs with α and slider):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
                dcc.Input(id='crit-input', type='number', debounce=True,
                          style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
                dcc.Slider(id='crit-slider', min=-4, max=4, step=0.01, value=1.66,
                           marks={i: f"{i}" for i in range(-4, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={'marginBottom': '20px', 'marginRight': '1000px'}),

            html.Div([
                html.Label("α (syncs with critical t-value):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
                dcc.Input(id='alpha-input', type='number', min=0.0001, max=0.9999, step=0.0001, value=0.05,
                          debounce=True, style={'width': '100px', 'fontSize': '20px'}),
            ], style={'marginBottom': '20px', 'marginRight': '1000px'}),

            html.Div([
                html.Label("Effect Size (Δ = μ₁ - μ₀):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
                dcc.Input(id='delta-input', type='number', value=1.5, debounce=True,
                          style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
                dcc.Slider(id='delta-slider', min=-4, max=4, step=0.1, value=1.5,
                           marks={i: f"{i}" for i in range(-4, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={'marginBottom': '20px', 'marginRight': '1000px'}),

            html.Div([
                html.Label("Standard Deviation (σ):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
                dcc.Input(id='sigma-input', type='number', value=1.2, debounce=True,
                          style={'marginRight': '10px', 'width': '100px', 'fontSize': '20px'}),
                dcc.Slider(id='sigma-slider', min=0.1, max=3, step=0.1, value=1.2,
                           marks={i: f"{i}" for i in range(1, 4)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={'marginBottom': '20px', 'marginRight': '1000px'}),

            html.Div([
            html.Label("Degrees of Freedom (df):", style={'fontWeight': 'bold', 'fontSize': '22px'}),
            dcc.Input(id='df-input', type='number', min=1, step=1, value=30,
                        style={'width': '100px', 'fontSize': '20px'}),
            ], style={'marginBottom': '20px', 'marginRight': '1000px'}),

            html.Button("Toggle Beta Region", id='toggle-beta', n_clicks=0,
                        style={'fontSize': '18px', 'marginRight': '15px'}),
            html.Button("Toggle Power Region", id='toggle-power', n_clicks=0, style={'fontSize': '18px'})
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.Img(src='/assets/confusion_matrix.png', style={'height': '300px', 'marginTop': '20px'})
        ], style={'width': '38%', 'display': 'inline-block', 'paddingLeft': '2%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    html.Br(), html.Br(),

    html.Div([
        html.Div([dcc.Graph(id='hypothesis-plot')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='subplots')], style={'width': '48%', 'display': 'inline-block', 'paddingLeft': '2%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])


# Callback
@app.callback(
    Output('crit-input', 'value'),
    Output('crit-slider', 'value'),
    Output('alpha-input', 'value'),
    Output('delta-input', 'value'),
    Output('delta-slider', 'value'),
    Output('sigma-input', 'value'),
    Output('sigma-slider', 'value'),
    Output('hypothesis-plot', 'figure'),
    Output('subplots', 'figure'),
    Input('crit-slider', 'value'),
    Input('crit-input', 'value'),
    Input('alpha-input', 'value'),
    Input('delta-slider', 'value'),
    Input('delta-input', 'value'),
    Input('sigma-slider', 'value'),
    Input('sigma-input', 'value'),
    Input('df-input', 'value'),
    Input('toggle-beta', 'n_clicks'),
    Input('toggle-power', 'n_clicks'),
    prevent_initial_call=True
)
def sync_and_plot(crit_slider, crit_input, alpha_input,
                  delta_slider, delta_input, sigma_slider, sigma_input, df_input,
                  toggle_beta, toggle_power):

    trigger = ctx.triggered_id
    show_beta = toggle_beta % 2 == 0
    show_power = toggle_power % 2 == 0

    sigma = sigma_input if trigger == 'sigma-input' and sigma_input is not None else sigma_slider
    delta = delta_input if trigger == 'delta-input' and delta_input is not None else delta_slider

    # if trigger == 'alpha-input' and alpha_input is not None:
    #     crit_val = norm.ppf(1 - alpha_input, loc=mu_null, scale=sigma)
    #     alpha_val = alpha_input
    # elif trigger == 'crit-input' and crit_input is not None:
    #     crit_val = crit_input
    #     alpha_val = 1 - norm.cdf(crit_val, loc=mu_null, scale=sigma)
    # else:
    #     crit_val = crit_slider
    #     alpha_val = 1 - norm.cdf(crit_val, loc=mu_null, scale=sigma)

    if trigger == 'alpha-input' and alpha_input is not None:
        crit_val = t.ppf(1 - alpha_input, df_input)
        alpha_val = alpha_input
    elif trigger == 'crit-input' and crit_input is not None:
        crit_val = crit_input
        alpha_val = 1 - t.cdf(crit_val, df_input)
    else:
        crit_val = crit_slider
        alpha_val = 1 - t.cdf(crit_val, df_input)

    fig = generate_figure(df_input, alpha_input, delta, sigma, show_beta, show_power)
    fig2 = generate_subplots(df_input, alpha_input, delta, sigma)
    return crit_val, crit_val, alpha_val, delta, delta, sigma, sigma, fig, fig2


# def update_all(crit, delta, sigma):
#     return generate_figure(crit, delta, sigma), generate_subplots(crit, delta, sigma)



if __name__ == '__main__':
    app.run(debug=True)