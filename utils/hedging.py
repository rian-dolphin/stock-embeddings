import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def get_vol(tick1, tick2, test_returns_df, plot=False):
    """-- Simulate a portfolio with two stocks and return realised volatility --"""
    # -- Equal weight
    combined_returns = (
        test_returns_df[tick1].values + test_returns_df[tick2].values
    ) / 2
    # -- Simulate accumulated value of portfolio
    portfolio_evolution = np.cumprod(1 + combined_returns)

    if plot:
        # -- Plot the portfolio evolution
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=test_returns_df.index, y=portfolio_evolution, name="Portfolio")
        )
        fig.add_trace(
            go.Scatter(
                x=test_returns_df.index,
                y=np.cumprod(1 + test_returns_df[tick1].values),
                name=tick1,
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_returns_df.index,
                y=np.cumprod(1 + test_returns_df[tick2].values),
                name=tick2,
                opacity=0.6,
            )
        )
        fig.update_layout(
            template="plotly_white",
            yaxis=dict(title="Cumulative Returns"),
            title=f"Hedged Portfolio Evolution | {tick1} - {tick2}",
        )
        fig.show()

    # -- Compute volatiltiy as standard deviation of log returns
    portfolio_log_returns = [
        np.log(
            1
            + (portfolio_evolution[i + 1] - portfolio_evolution[i])
            / portfolio_evolution[i]
        )
        for i in range(len(portfolio_evolution) - 1)
    ]
    portfolio_vol = np.std(portfolio_log_returns) * np.sqrt(252)
    return round(portfolio_vol, 3)


def get_vol_array(similarities, test_returns_df, tickers, n=50):
    # -- Get vol_list for a given similarity matrix

    vol_list = []

    # -- Ensure diagonal is nan so a stock will never hedge with itself
    np.fill_diagonal(similarities, np.nan)

    # -- Get the LOWEST similarity stock for each metric
    hedge_idxs = np.argsort(similarities, axis=0)
    # -- Choose hedge stock randomly from top-n least similar stocks
    # - Prevents the same stocks being chosen again and again
    hedge_idxs = [
        hedge_idxs[np.random.randint(0, n), i] for i in range(len(hedge_idxs))
    ]

    for i in range(len(tickers)):
        tick1 = tickers[i]
        tick2 = tickers[hedge_idxs[i]]

        temp_vol = get_vol(tick1, tick2, test_returns_df, plot=False)

        vol_list.append(temp_vol)

    return np.array(vol_list)


def repeat_vol_array(
    sims, test_returns_df, tickers, num_trials=50, top_k=50, verbose=True
):
    """Repeat the portfolio vol simulation multiple times and return the raw vols and means.
    The means can be used to do Tukey test and compare.

    Args:
        sims (_type_): _description_
        test_returns_df (_type_): _description_
        tickers (_type_): _description_
        n (int, optional): _description_. Defaults to 50.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    vols = []
    # - Also save the mean of each run so that we can do a tukey HSD
    mean_vols = []
    for _ in tqdm(range(num_trials), disable=not verbose):
        temp_vols = list(get_vol_array(sims, test_returns_df, tickers=tickers, n=top_k))
        vols += temp_vols
        mean_vols.append(np.mean(temp_vols))
    return vols, mean_vols


# PLOTTING FUNCTIONS

# Color scheme
OKABE_ITO_COLORS = [
    [230, 159, 0],  # orange
    [86, 180, 233],  # sky blue
    [0, 158, 115],  # bluish green
    [240, 228, 66],  # yellow
    [0, 114, 178],  # blue
    [213, 94, 0],  # vermilion
    [204, 121, 167],  # reddish purple
]


def get_kde_plot_values(vols):
    xvals = np.linspace(0.1, 0.5, 200)
    kde = gaussian_kde(vols)
    return xvals, kde(xvals)


def add_kde_trace(
    fig, x_vals, y_vals, name, line_dash, color_index, opacity, fill_pattern
):
    color = OKABE_ITO_COLORS[color_index]
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            fill="tozeroy",
            name=name,
            line=dict(dash=line_dash, color=f"rgb({color[0]},{color[1]},{color[2]})"),
            opacity=opacity,
            fillpattern_shape=fill_pattern,
            fillcolor=f"rgba({color[0]},{color[1]},{color[2]},{opacity})",
        )
    )


def plot_kdes(data_sets, names, line_styles, opacities, fill_patterns, color_indices):
    x_vals = np.linspace(
        min(min(data) for data in data_sets), max(max(data) for data in data_sets), 500
    )

    fig = go.Figure()

    for data, name, style, opacity, pattern, color_index in zip(
        data_sets, names, line_styles, opacities, fill_patterns, color_indices
    ):
        kde_vals = gaussian_kde(data)(x_vals)
        add_kde_trace(fig, x_vals, kde_vals, name, style, color_index, opacity, pattern)

    fig.update_layout(
        template="plotly_white",
        height=400,
        width=800,
        # font=dict(family="Times, serif", size=20, color="Black"),
        font=dict(size=17),
        xaxis_title="Volatility",
        yaxis_title="Probability Density",
        xaxis=dict(showgrid=False),
        xaxis_range=[min(x_vals), np.quantile(x_vals, 0.9)],
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(t=20),
    )
    fig.write_image("/Users/rian/Downloads/hedging_contrastive_results.pdf")
    fig.write_image("/Users/rian/Downloads/hedging_contrastive_results.pdf")
    fig.show()
