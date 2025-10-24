#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# 2CentsCapital Private Implementation
#

import re as _regex
from base64 import b64encode as _b64encode
from datetime import datetime as _dt
from io import StringIO
from math import ceil as _ceil
from math import sqrt as _sqrt
import tempfile
import numpy as _np
import pandas as _pd
from dateutil.relativedelta import relativedelta
from tabulate import tabulate as _tabulate
import os
from . import __version__
from . import plots as _plots
from . import stats as _stats
from . import utils as _utils

try:
    from IPython.display import HTML as iHTML
    from IPython.display import display as iDisplay
except ImportError:
    from IPython.core.display import HTML as iHTML
    from IPython.core.display import display as iDisplay


def _get_trading_periods(periods_per_year=365):
    """returns trading periods per year and half year"""
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _match_dates(returns, benchmark):
    """match dates of returns and benchmark"""
    if isinstance(returns, _pd.DataFrame):
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark


def html(
    returns,
    strategy_name: str = None,
    rm_name: str = None,
    tm_name: str = None,
    code_id: str = None,
    stock_name: str = None,
    timeframe: str = None,
    exchange: str = None,
    equity_df: _pd.DataFrame = None,
    benchmark: _pd.Series = None,
    rf: float = 0.0,
    grayscale: bool = False,
    title: str = "Strategy Tearsheet",
    output: str = None,
    compounded: bool = True,
    periods_per_year: int = 365,
    download_filename: str = "tearsheet.html",
    figfmt: str = "svg",
    template_path: str = None,
    match_dates: bool = True,
    parameters: dict = None,
    log_scale: bool = False,
    show_match_volatility: bool = True,
    _trade_start_bar: int = 0,
    trade_stats=None,  # Parameter for trade statistics
    trade_table=None,  # Parameter for trade table (DataFrame)
    backtest=None,     # Parameter for the Backtest object
    **kwargs,
):
    """
    Generates a full HTML tearsheet with performance metrics and plots

    Parameters
    ----------
    returns : pd.Series, pd.DataFrame
        Strategy returns
    benchmark : pd.Series, optional
        Benchmark returns
    rf : float, optional
        Risk-free rate, default is 0
    grayscale : bool, optional
        Plot in grayscale, default is False
    title : str, optional
        Title of the HTML report, default is "Strategy Tearsheet"
    output : str, optional
        Output file path
    compounded : bool, optional
        Whether to use compounded returns, default is True
    periods_per_year : int, optional
        Trading periods per year, default is 365
    download_filename : str, optional
        Download filename, default is "tearsheet.html"
    figfmt : str, optional
        Figure format, default is "svg"
    template_path : str, optional
        Custom template path
    match_dates : bool, optional
        Match dates of returns and benchmark, default is True
    parameters : dict, optional
        Strategy parameters
    trade_stats : pd.Series, optional
        Trade statistics
    trade_table : pd.DataFrame, optional
        Trade table
    backtest : Backtest, optional
        Backtest object for generating equity curve plot

    Returns
    -------
    None
    """

    if output is None and not _utils._in_notebook():
        raise ValueError("`output` must be specified")

    if match_dates:
        returns = returns.dropna()

    win_year, win_half_year = _get_trading_periods(periods_per_year)

    tpl = ""
    with open(template_path or __file__[:-4] + ".html") as f:
        tpl = f.read()
        f.close()

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    returns = _utils._prepare_returns(returns)

    strategy_title = kwargs.get("strategy_title", "Strategy")
    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
        if kwargs.get("benchmark_title") is None:
            if isinstance(benchmark, str):
                benchmark_title = benchmark
            elif isinstance(benchmark, _pd.Series):
                benchmark_title = benchmark.name
            elif isinstance(benchmark, _pd.DataFrame):
                benchmark_title = benchmark[benchmark.columns[0]].name

        tpl = tpl.replace(
            "{{benchmark_title}}", f"Benchmark is {benchmark_title.upper()} | "
        )
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        returns = returns.iloc[_trade_start_bar:]
        benchmark = benchmark.iloc[_trade_start_bar:]
        # if match_dates is True:
            # returns, benchmark = _match_dates(returns, benchmark)
    else:
        benchmark_title = None

    date_range = returns.index.strftime("%e %b, %Y")
    tpl = tpl.replace("{{date_range}}", date_range[0] + " - " + date_range[-1])
    tpl = tpl.replace("{{title}}", title)
    tpl = tpl.replace("{{v}}", __version__)
    tpl = tpl.replace("{{code_id}}", code_id or "UNKNOWN CODE ID")
    tpl = tpl.replace("{{stock_name}}", stock_name or "UNKNOWN STOCK")
    tpl = tpl.replace("{{timeframe}}", timeframe or "UNKNOWN TIMEFRAME")
    tpl = tpl.replace("{{exchange}}", exchange or "UNKNOWN EXCHANGE")
    tpl = tpl.replace("{{strategy_name}}", strategy_name or "UNKNOWN STRATEGY")
    tpl = tpl.replace("{{rm_name}}", rm_name or "UNKNOWN RM")
    tpl = tpl.replace("{{tm_name}}", tm_name or "UNKNOWN TM")

    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    mtrx = metrics(
        returns=returns,
        benchmark=benchmark,
        rf=rf,
        display=False,
        mode="full",
        sep=True,
        internal="True",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
    )[2:]

    mtrx.index.name = "Metric"
    tpl = tpl.replace("{{metrics}}", _html_table(mtrx))

    # Add all of the summary metrics

    # CAGR #
    # Get the value of the "Strategy" column where the "Metric" column is "CAGR% (Annual Return)"
    cagr = mtrx.loc["CAGR% (Annual Return)", strategy_title]
    # Add the CAGR to the template
    tpl = tpl.replace("{{cagr}}", cagr)

    # Total Return #
    # Get the value of the "Strategy" column where the "Metric" column is "Total Return"
    total_return = mtrx.loc["Total Return", strategy_title]
    # Add the total return to the template
    tpl = tpl.replace("{{total_return}}", total_return)

    # Max Drawdown #
    # Get the value of the "Strategy" column where the "Metric" column is "Max Drawdown"
    max_drawdown = mtrx.loc["Max Drawdown", strategy_title]
    # Add the max drawdown to the template
    tpl = tpl.replace("{{max_drawdown}}", max_drawdown)

    # RoMaD #
    # Get the value of the "Strategy" column where the "Metric" column is "RoMaD"
    romad = mtrx.loc["RoMaD", strategy_title]
    # Add the RoMaD to the template
    tpl = tpl.replace("{{romad}}", romad)

    # Longest Drawdown Duration #
    # Get the value of the "Strategy" column where the "Metric" column is "Longest Drawdown Duration"
    longest_dd_days = mtrx.loc["Longest DD Days", strategy_title]
    # Add the longest drawdown duration to the template
    tpl = tpl.replace("{{longest_dd_days}}", longest_dd_days)

    # Sharpe #
    # Get the value of the "Strategy" column where the "Metric" column is "Sharpe"
    sharpe = mtrx.loc["Sharpe", strategy_title]
    # Add the Sharpe to the template
    tpl = tpl.replace("{{sharpe}}", sharpe)

    # Sortino #
    # Get the value of the "Strategy" column where the "Metric" column is "Sortino"
    sortino = mtrx.loc["Sortino", strategy_title]
    # Add the Sortino to the template
    tpl = tpl.replace("{{sortino}}", sortino)

    # Prepare Trade Statistics table (if provided) with filtered stats
    trade_stats_html = ""
    trade_table_html = ""
    equity_curve_html = ""

    if trade_stats is not None:
        # Convert trade_stats (pandas Series) to a DataFrame for display
        trade_stats_df = _pd.DataFrame(trade_stats, columns=["Value"])
        trade_stats_df.index.name = "Statistic"
        
        
        # Filter the Trade Statistics to show only the most relevant ones
        selected_trade_stats = [
            "Start", "End", "Duration", "Exposure Time [%]", "Equity Final [$]",
            "Equity Peak [$]", "Return [%]", "Return (Ann.) [%]", "Volatility (Ann.) [%]",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max. Drawdown [%]",
            "# Trades", "Win Rate [%]", "Profit Factor", "Expectancy [%]","SQN ","Kelly Criterion"
        ]
        trade_stats_df = trade_stats_df[trade_stats_df.index.isin(selected_trade_stats)]
        print("Filtered trade_stats_df:", trade_stats_df)
        
        # Format specific columns for better readability
        for idx in trade_stats_df.index:
            if "Duration" in idx and isinstance(trade_stats_df.loc[idx, "Value"], _pd.Timedelta):
                trade_stats_df.loc[idx, "Value"] = str(trade_stats_df.loc[idx, "Value"])
            elif isinstance(trade_stats_df.loc[idx, "Value"], float):
                trade_stats_df.loc[idx, "Value"] = f"{trade_stats_df.loc[idx, 'Value']:.2f}"
        trade_stats_html = f"""
        <h3>Trade Statistics</h3>
        {_html_table(trade_stats_df)}
        """

    if trade_table is not None:
        # Format the trade table (DataFrame) for display
        trade_table = trade_table.copy()
        trade_table.set_index("Trade_ID", inplace=True)
        # Convert Duration to string if it's a timedelta
        if "Duration" in trade_table.columns:
            trade_table["Duration"] = trade_table["Duration"].astype(str)
        trade_table_html = f"""
        <h3>Trade Details</h3>
        {_html_table(trade_table)}
        """

    # Generate the equity curve plot using bt.plot
    if backtest is not None:
        try:
            # Verify that backtest and its results are valid
            if not hasattr(backtest, '_results') or backtest._results is None:
                print("Error: backtest._results is None. Ensure backtest.run() has been called.")
                equity_curve_html = "<p>Error: Could not generate equity curve. Backtest results are missing.</p>"
            else:
                print("Generating equity curve plot...")
                # Create a temporary file to store the plot
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                    temp_filename = temp_file.name
                    print(f"Temporary file created at: {temp_filename}")
                    # Call the plot function to generate the equity curve
                    backtest.plot(
                        results=backtest._results,  # Use the results from the Backtest object
                        filename=temp_filename,
                        plot_equity=True,  # Plot the equity curve
                        plot_return=False,
                        plot_pl=True,
                        plot_volume=False,
                        plot_drawdown=False,
                        plot_trades=True,
                        smooth_equity=False,
                        relative_equity=True,
                        superimpose=True,
                        resample=False,
                        reverse_indicators=False,
                        show_legend=True,
                        open_browser=False,
                        plot_allocation=False,
                        relative_allocation=True,
                        plot_indicator=True
                    )
                    # Read the HTML content of the plot
                    with open(temp_filename, "r", encoding="utf-8") as f:
                        equity_curve_html = f.read()
        except Exception as e:
            print(f"Error generating equity curve plot: {str(e)}")
            equity_curve_html = f"<p>Error generating equity curve: {str(e)}</p>"
    else:
        print("No backtest object provided. Skipping equity curve generation.")
        equity_curve_html = "<p>No backtest object provided to generate equity curve.</p>"

    # Equity_df below trade details
    if equity_df is not None:
        equity_df = equity_df.copy()
        equity_html = f"""
        <h3>Equity Curve</h3>
        {_html_table(equity_df)}
        """
    else:
        print("No equity_df provided")
        equity_html = "<p>No equity_df provided to generate equity curve.</p>"

    # Replace the placeholders for trade stats, trade details, and equity curve
    
    tpl = tpl.replace("{{trade_stats}}", trade_stats_html if trade_stats_html else "")
    tpl = tpl.replace("{{trade_details}}", trade_table_html if trade_table_html else "")
    tpl = tpl.replace("{{equity_curve}}", equity_curve_html if equity_curve_html else "<p>No equity curve generated.</p>")
    tpl = tpl.replace("{{equity_df}}", equity_html)

    if isinstance(returns, _pd.DataFrame):
        num_cols = len(returns.columns)
        for i in reversed(range(num_cols + 1, num_cols + 3)):
            str_td = "<td></td>" * i
            tpl = tpl.replace(
                f"<tr>{str_td}</tr>", '<tr><td colspan="{}"><hr></td></tr>'.format(i)
            )

    tpl = tpl.replace(
        "<tr><td></td><td></td><td></td></tr>", '<tr><td colspan="3"><hr></td></tr>'
    )
    tpl = tpl.replace(
        "<tr><td></td><td></td></tr>", '<tr><td colspan="2"><hr></td></tr>'
    )

    if parameters is not None:
        tpl = tpl.replace("{{parameters_section}}", parameters_section(parameters))

    if benchmark is not None:
        yoy = _stats.compare(
            returns, benchmark, "YE", compounded=compounded, prepare_returns=False
        )
        if isinstance(returns, _pd.Series):
            yoy.columns = [benchmark_title, strategy_title, "Multiplier", "Won"]
        elif isinstance(returns, _pd.DataFrame):
            yoy.columns = list(
                _pd.core.common.flatten([benchmark_title, strategy_title])
            )
        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns vs Benchmark</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))
    else:
        # pct multiplier
        yoy = _pd.DataFrame(_utils.group_returns(returns, returns.index.year) * 100)
        if isinstance(returns, _pd.Series):
            yoy.columns = ["Return"]
            yoy["Cumulative"] = _utils.group_returns(returns, returns.index.year, True)
            yoy["Return"] = yoy["Return"].round(2).astype(str) + "%"
            yoy["Cumulative"] = (yoy["Cumulative"] * 100).round(2).astype(str) + "%"
        elif isinstance(returns, _pd.DataFrame):
            # Don't show cumulative for multiple strategy portfolios
            # just show compounded like when we have a benchmark
            yoy.columns = list(_pd.core.common.flatten(strategy_title))

        yoy.index.name = "Year"
        tpl = tpl.replace("{{eoy_title}}", "<h3>EOY Returns</h3>")
        tpl = tpl.replace("{{eoy_table}}", _html_table(yoy))

    if isinstance(returns, _pd.Series):
        dd = _stats.to_drawdown_series(returns)
        dd_info = _stats.drawdown_details(dd).sort_values(
            by="max drawdown", ascending=True
        )[:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
        tpl = tpl.replace("{{dd_info}}", _html_table(dd_info, False))
    elif isinstance(returns, _pd.DataFrame):
        dd_info_list = []
        for col in returns.columns:
            dd = _stats.to_drawdown_series(returns[col])
            dd_info = _stats.drawdown_details(dd).sort_values(
                by="max drawdown", ascending=True
            )[:10]
            dd_info = dd_info[["start", "end", "max drawdown", "days"]]
            dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
            dd_info_list.append(_html_table(dd_info, False))

        dd_html_table = ""
        for html_str, col in zip(dd_info_list, returns.columns):
            dd_html_table = (
                dd_html_table + f"<h3>{col}</h3><br>" + StringIO(html_str).read()
            )
        tpl = tpl.replace("{{dd_info}}", dd_html_table)

    active = kwargs.get("active_returns", False)
    # plots
    plot_returns = _plots.log_returns if log_scale else _plots.returns
    placeholder_returns = "{{log_returns}}" if log_scale else "{{returns}}"

    figfile = _utils._file_stream()
    plot_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 5),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        cumulative=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace(placeholder_returns, _embed_figure(figfile, figfmt))

    if benchmark is not None and show_match_volatility:
        figfile = _utils._file_stream()
        plot_returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(8, 5),
            subtitle=False,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel=False,
            cumulative=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{vol_returns}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{eoy_returns}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(7, 4),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        compounded=compounded,
        prepare_returns=False,
    )
    tpl = tpl.replace("{{monthly_dist}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        prepare_returns=False,
        active=active,
    )
    tpl = tpl.replace("{{daily_returns}}", _embed_figure(figfile, figfmt))

    if benchmark is not None:
        figfile = _utils._file_stream()
        _plots.rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 3),
            subtitle=False,
            window1=win_half_year,
            window2=win_year,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel=False,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{rolling_beta}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_vol}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sharpe}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    _plots.rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
        period=win_half_year,
        periods_per_year=win_year,
    )
    tpl = tpl.replace("{{rolling_sortino}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    if isinstance(returns, _pd.Series):
        _plots.drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel=False,
            compounded=compounded,
            prepare_returns=False,
            log_scale=log_scale,
        )
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        embed = []
        for col in returns.columns:
            _plots.drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel=False,
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{dd_periods}}", _embed_figure(embed, figfmt))

    figfile = _utils._file_stream()
    _plots.drawdown(
        returns,
        grayscale=grayscale,
        figsize=(8, 3),
        subtitle=False,
        savefig={"fname": figfile, "format": figfmt},
        show=False,
        ylabel=False,
    )
    tpl = tpl.replace("{{dd_plot}}", _embed_figure(figfile, figfmt))

    figfile = _utils._file_stream()
    if isinstance(returns, _pd.Series):
        _plots.monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(8, 4),
            cbar=False,
            returns_label=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel=False,
            compounded=compounded,
            active=active,
        )
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        embed = []
        for col in returns.columns:
            _plots.monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(8, 4),
                cbar=False,
                returns_label=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel=False,
                compounded=compounded,
                active=active,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{monthly_heatmap}}", _embed_figure(embed, figfmt))

    figfile = _utils._file_stream()

    if isinstance(returns, _pd.Series):
        _plots.distribution(
            returns,
            grayscale=grayscale,
            figsize=(8, 4),
            subtitle=False,
            title=returns.name,
            savefig={"fname": figfile, "format": figfmt},
            show=False,
            ylabel=False,
            compounded=compounded,
            prepare_returns=False,
        )
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(figfile, figfmt))
    elif isinstance(returns, _pd.DataFrame):
        embed = []
        for col in returns.columns:
            _plots.distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(8, 4),
                subtitle=False,
                title=col,
                savefig={"fname": figfile, "format": figfmt},
                show=False,
                ylabel=False,
                compounded=compounded,
                prepare_returns=False,
            )
            embed.append(figfile)
        tpl = tpl.replace("{{returns_dist}}", _embed_figure(embed, figfmt))

    tpl = _regex.sub(r"\{\{(.*?)\}\}", "", tpl)
    tpl = tpl.replace("{{trade_stats}}", trade_stats_html if trade_stats_html else "")
    tpl = tpl.replace("{{trade_details}}", trade_table_html if trade_table_html else "")
    tpl = tpl.replace("{{equity_curve}}", equity_curve_html if equity_curve_html else "<p>No equity curve generated.</p>")

    if output is None:
        # _open_html(tpl)
        _download_html(tpl, download_filename)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(tpl)

    print(f"HTML report saved to: {output}")

    # Return the metrics
    return mtrx


def full(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=365,
    match_dates=True,
    **kwargs,
):
    """calculates and plots full performance metrics"""

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    returns = _utils._prepare_returns(returns)
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    if benchmark is not None:
        benchmark.name = benchmark_title
    if isinstance(returns, _pd.Series):
        returns.name = strategy_title
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_title

    dd = _stats.to_drawdown_series(returns)

    if isinstance(dd, _pd.Series):
        col = _stats.drawdown_details(dd).columns[4]
        dd_info = _stats.drawdown_details(dd).sort_values(by=col, ascending=True)[:5]
        if not dd_info.empty:
            dd_info.index = range(1, min(6, len(dd_info) + 1))
            dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
    elif isinstance(dd, _pd.DataFrame):
        col = _stats.drawdown_details(dd).columns.get_level_values(1)[4]
        dd_info_dict = {}
        for ptf in dd.columns:
            dd_info = _stats.drawdown_details(dd[ptf]).sort_values(
                by=col, ascending=True
            )[:5]
            if not dd_info.empty:
                dd_info.index = range(1, min(6, len(dd_info) + 1))
                dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)
            dd_info_dict[ptf] = dd_info

    if _utils._in_notebook():
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        iDisplay(
            metrics(
                returns=returns,
                benchmark=benchmark,
                rf=rf,
                display=display,
                mode="full",
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False,
                benchmark_title=benchmark_title,
                strategy_title=strategy_title,
            )
        )

        if isinstance(dd, _pd.Series):
            iDisplay(iHTML('<h4 style="margin-bottom:20px">Worst 5 Drawdowns</h4>'))
            if dd_info.empty:
                iDisplay(iHTML("<p>(no drawdowns)</p>"))
            else:
                iDisplay(dd_info)
        elif isinstance(dd, _pd.DataFrame):
            for ptf, dd_info in dd_info_dict.items():
                iDisplay(
                    iHTML(
                        '<h4 style="margin-bottom:20px">%s - Worst 5 Drawdowns</h4>'
                        % ptf
                    )
                )
                if dd_info.empty:
                    iDisplay(iHTML("<p>(no drawdowns)</p>"))
                else:
                    iDisplay(dd_info)

        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="full",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        print("\n\n")
        print("[Worst 5 Drawdowns]\n")
        if isinstance(dd, _pd.Series):
            if dd_info.empty:
                print("(no drawdowns)")
            else:
                print(
                    _tabulate(
                        dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                    )
                )
        elif isinstance(dd, _pd.DataFrame):
            for ptf, dd_info in dd_info_dict.items():
                if dd_info.empty:
                    print("(no drawdowns)")
                else:
                    print(f"{ptf}\n")
                    print(
                        _tabulate(
                            dd_info, headers="keys", tablefmt="simple", floatfmt=".2f"
                        )
                    )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="full",
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )

def basic(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=365,
    match_dates=True,
    **kwargs,
):
    """calculates and plots basic performance metrics"""

    # prepare timeseries
    if match_dates:
        returns = returns.dropna()
    returns = _utils._prepare_returns(returns)
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    benchmark_title = None
    if benchmark is not None:
        benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    strategy_title = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active_returns", False)

    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_title, str)
    ):
        strategy_title = list(returns.columns)

    if _utils._in_notebook():
        iDisplay(iHTML("<h4>Performance Metrics</h4>"))
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )
        iDisplay(iHTML("<h4>Strategy Visualization</h4>"))
    else:
        print("[Performance Metrics]\n")
        metrics(
            returns=returns,
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode="basic",
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=False,
            benchmark_title=benchmark_title,
            strategy_title=strategy_title,
        )

        print("\n\n")
        print("[Strategy Visualization]\nvia Matplotlib")

    plots(
        returns=returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="basic",
        periods_per_year=periods_per_year,
        prepare_returns=False,
        benchmark_title=benchmark_title,
        strategy_title=strategy_title,
        active=active,
    )

def parameters_section(parameters):
    """returns a formatted section for strategy parameters"""
    if parameters is None:
        return ""

    tpl = """
    <div id="params">
        <h3>Parameters Used</h3>
        <table style="width:100%; font-size: 12px">
    """

    # Add titles to the table
    tpl += "<thead><tr><th>Parameter</th><th>Value</th></tr></thead>"

    for key, value in parameters.items():
        # Make sure that the value is something that can be displayed
        if not isinstance(value, (int, float, str)):
            value = str(value)

        tpl += f"<tr><td>{key}</td><td>{value}</td></tr>"
    tpl += """
        </table>
    </div>
    """

    return tpl

def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=365,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """calculates and displays various performance metrics"""
    # if match_dates:
    #     returns = returns.dropna()
    returns.index = returns.index.tz_localize(None)
    win_year, _ = _get_trading_periods(periods_per_year)

    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")

    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    if prepare_returns:
        df = _utils._prepare_returns(returns)

    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame({"returns": returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        # if match_dates is True:
        #     returns, benchmark = _match_dates(returns, benchmark)
        df["benchmark"] = benchmark
        if isinstance(returns, _pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, _pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    if isinstance(returns, _pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, _pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    df = df.fillna(0)

    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate % "] = _pd.Series(s_rf) * 100
    metrics["Time in Market % "] = _stats.exposure(df, prepare_returns=False) * pct
    metrics["~"] = blank

    if compounded:
        metrics["Total Return"] = (_stats.comp(df) * pct).map("{:,.0f}%".format)  # No decimals for readability as it is a large number
    else:
        metrics["Total Return"] = (df.sum() * pct).map("{:,.2f}%".format)
    metrics["CAGR% (Annual Return) "] = _stats.cagr(df, rf, compounded, win_year) * pct
    metrics["~~~~~~~~~~~~~~"] = blank

    metrics["Sharpe"] = _stats.sharpe(df, rf, win_year, True)
    metrics["RoMaD"] = _stats.romad(df, win_year, True)

    if benchmark is not None:
        metrics["Corr to Benchmark "] = _stats.benchmark_correlation(df, benchmark, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _stats.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _stats.smart_sharpe(df, rf, win_year, True)

    metrics["Sortino"] = _stats.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        metrics["Smart Sortino"] = _stats.smart_sortino(df, rf, win_year, True)

    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
    metrics["Omega"] = _stats.omega(df, rf, 0.0, win_year)

    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Longest DD Days"] = blank

    if mode.lower() == "full":
        if isinstance(returns, _pd.Series):
            ret_vol = (
                _stats.volatility(df["returns"], win_year, True, prepare_returns=False)
                * pct
            )
        elif isinstance(returns, _pd.DataFrame):
            ret_vol = [
                _stats.volatility(
                    df[strategy_col], win_year, True, prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]
        if "benchmark" in df:
            bench_vol = (
                _stats.volatility(
                    df["benchmark"], win_year, True, prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(_pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            if isinstance(returns, _pd.Series):
                metrics["R^2"] = _stats.r_squared(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
                metrics["Information Ratio"] = _stats.information_ratio(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
            elif isinstance(returns, _pd.DataFrame):
                metrics["R^2"] = (
                    [
                        _stats.r_squared(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        _stats.information_ratio(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            if isinstance(returns, _pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, _pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        metrics["Calmar"] = _stats.calmar(df, prepare_returns=False, periods=win_year)
        metrics["Skew"] = _stats.skew(df, prepare_returns=False)
        metrics["Kurtosis"] = _stats.kurtosis(df, prepare_returns=False)

        metrics["~~~~~~~~~~"] = blank

        metrics["Expected Daily %%"] = (
            _stats.expected_return(df, compounded=compounded, prepare_returns=False)
            * pct
        )
        metrics["Expected Monthly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Expected Yearly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        metrics["Daily Value-at-Risk %"] = -abs(
            _stats.var(df, prepare_returns=False) * pct
        )
        metrics["Expected Shortfall (cVaR) %"] = -abs(
            _stats.cvar(df, prepare_returns=False) * pct
        )

    # returns
    metrics["~~"] = blank
    comp_func = _stats.comp if compounded else lambda x: _np.sum(x, axis=0)

    today = df.index[-1]  # _dt.today()
    metrics["MTD %"] = comp_func(df[df.index >= _dt(today.year, today.month, 1)]) * pct

    d = today - relativedelta(months=3)
    metrics["3M %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=6)
    metrics["6M %"] = comp_func(df[df.index >= d]) * pct

    metrics["YTD %"] = comp_func(df[df.index >= _dt(today.year, 1, 1)]) * pct

    d = today - relativedelta(years=1)
    metrics["1Y %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    metrics["All-time (ann.) %"] = _stats.cagr(df, 0.0, compounded, win_year) * pct

    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = (
            _stats.best(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Worst Day %"] = _stats.worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Month %"] = (
            _stats.worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Year %"] = (
            _stats.worst(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

    # dd
    metrics["~~~~"] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics["Recovery Factor"] = _stats.recovery_factor(df)
    metrics["Ulcer Index"] = _stats.ulcer_index(df)
    metrics["Serenity Index"] = _stats.serenity_index(df, rf)

    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            _stats.avg_win(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Avg. Down Month %"] = (
            _stats.avg_loss(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )

        # Get the win rate
        win_rate = _stats.win_rate(df, prepare_returns=False)

        # Number of win days in total
        metrics["Win Days"] = win_rate * len(df)

        # Number of loss days in total
        metrics["Loss Days"] = len(df) - metrics["Win Days"]

        metrics["Win Days %%"] = win_rate * pct
        metrics["Win Month %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Quarter %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="QE", prepare_returns=False
            )
            * pct
        )
        metrics["Win Year %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, _pd.Series):
                greeks = _stats.greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _stats.treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, _pd.DataFrame):
                greeks = [
                    _stats.greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _stats.treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    try:
        metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = _pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(_pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]

    if display:
        print(_tabulate(metrics, headers="keys", tablefmt="simple"))
        return None

    if not sep:
        metrics = metrics[metrics.index != ""]

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics


def plots(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(8, 5),
    mode="basic",
    compounded=True,
    periods_per_year=365,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """Plots for strategy performance"""

    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")
    active = kwargs.get("active", "False")

    if (
        isinstance(returns, _pd.DataFrame)
        and len(returns.columns) > 1
        and isinstance(strategy_colname, str)
    ):
        strategy_colname = list(returns.columns)

    win_year, win_half_year = _get_trading_periods(periods_per_year)

    if match_dates is True:
        returns = returns.dropna()

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        returns.columns = strategy_colname

    if mode.lower() != "full":
        _plots.snapshot(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0]),
            show=True,
            mode=("comp" if compounded else "sum"),
            benchmark_title=benchmark_colname,
            strategy_title=strategy_colname,
        )

        if isinstance(returns, _pd.Series):
            _plots.monthly_heatmap(
                returns,
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel=False,
                compounded=compounded,
                active=active,
            )
        elif isinstance(returns, _pd.DataFrame):
            for col in returns.columns:
                _plots.monthly_heatmap(
                    returns[col].dropna(),
                    benchmark,
                    grayscale=grayscale,
                    figsize=(figsize[0], figsize[0] * 0.5),
                    show=True,
                    ylabel=False,
                    returns_label=col,
                    compounded=compounded,
                    active=active,
                )

        return

    returns = _pd.DataFrame(returns)

    # prepare timeseries
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark.name = benchmark_colname
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)

    _plots.returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.6),
        show=True,
        ylabel=False,
        prepare_returns=False,
    )

    _plots.log_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel=False,
        prepare_returns=False,
    )

    if benchmark is not None:
        _plots.returns(
            returns,
            benchmark,
            match_volatility=True,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel=False,
            prepare_returns=False,
        )

    _plots.yearly_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel=False,
        prepare_returns=False,
    )

    _plots.histogram(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.5),
        show=True,
        ylabel=False,
        prepare_returns=False,
    )

    small_fig_size = (figsize[0], figsize[0] * 0.35)
    if len(returns.columns) > 1:
        small_fig_size = (
            figsize[0],
            figsize[0] * (0.33 * (len(returns.columns) * 0.66)),
        )

    _plots.daily_returns(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel=False,
        prepare_returns=False,
        active=active,
    )

    if benchmark is not None:
        _plots.rolling_beta(
            returns,
            benchmark,
            grayscale=grayscale,
            window1=win_half_year,
            window2=win_year,
            figsize=small_fig_size,
            show=True,
            ylabel=False,
            prepare_returns=False,
        )

    _plots.rolling_volatility(
        returns,
        benchmark,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel=False,
        period=win_half_year,
    )

    _plots.rolling_sharpe(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel=False,
        period=win_half_year,
    )

    _plots.rolling_sortino(
        returns,
        grayscale=grayscale,
        figsize=small_fig_size,
        show=True,
        ylabel=False,
        period=win_half_year,
    )

    if isinstance(returns, _pd.Series):
        _plots.drawdowns_periods(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            ylabel=False,
            prepare_returns=False,
        )
    elif isinstance(returns, _pd.DataFrame):
        for col in returns.columns:
            _plots.drawdowns_periods(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel=False,
                title=col,
                prepare_returns=False,
            )

    _plots.drawdown(
        returns,
        grayscale=grayscale,
        figsize=(figsize[0], figsize[0] * 0.4),
        show=True,
        ylabel=False,
    )

    if isinstance(returns, _pd.Series):
        _plots.monthly_heatmap(
            returns,
            benchmark,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            returns_label=returns.name,
            show=True,
            ylabel=False,
            active=active,
        )
    elif isinstance(returns, _pd.DataFrame):
        for col in returns.columns:
            _plots.monthly_heatmap(
                returns[col],
                benchmark,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                ylabel=False,
                returns_label=col,
                compounded=compounded,
                active=active,
            )

    if isinstance(returns, _pd.Series):
        _plots.distribution(
            returns,
            grayscale=grayscale,
            figsize=(figsize[0], figsize[0] * 0.5),
            show=True,
            title=returns.name,
            ylabel=False,
            prepare_returns=False,
        )
    elif isinstance(returns, _pd.DataFrame):
        for col in returns.columns:
            _plots.distribution(
                returns[col],
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0] * 0.5),
                show=True,
                title=col,
                ylabel=False,
                prepare_returns=False,
            )


def _calc_dd(df, display=True, as_pct=False):
    """Returns drawdown stats"""
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    if dd_info.empty:
        return _pd.DataFrame()

    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Longest DD Days": str(
                    _np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        dd_stats = {
            "returns": {
                "Max Drawdown %": ret_dd.sort_values(by="max drawdown", ascending=True)[
                    "max drawdown"
                ].values[0]
                / 100,
                "Longest DD Days": str(
                    _np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
            }
        }
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Longest DD Days": str(
                _np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(bench_dd["days"].mean())),
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T


def _html_table(obj, showindex="default"):
    """Returns HTML table"""
    obj = _tabulate(
        obj, headers="keys", tablefmt="html", floatfmt=".6f", showindex=showindex
    )
    obj = obj.replace(' style="text-align: right;"', "")
    obj = obj.replace(' style="text-align: left;"', "")
    obj = obj.replace(' style="text-align: center;"', "")
    obj = _regex.sub("<td> +", "<td>", obj)
    obj = _regex.sub(" +</td>", "</td>", obj)
    obj = _regex.sub("<th> +", "<th>", obj)
    obj = _regex.sub(" +</th>", "</th>", obj)
    return obj


def _download_html(html, filename="quantstats-tearsheet.html"):
    """Downloads HTML report"""
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var bl=new Blob(['{{html}}'],{type:"text/html"});
    var a=document.createElement("a");
    a.href=URL.createObjectURL(bl);
    a.download="{{filename}}";
    a.hidden=true;document.body.appendChild(a);
    a.innerHTML="download report";
    a.click();</script>""".replace("\n", ""),
    )
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))
    if _utils._in_notebook():
        iDisplay(iHTML(jscode.replace("{{filename}}", filename)))


def _open_html(html):
    """Opens HTML in a new tab"""
    jscode = _regex.sub(
        " +",
        " ",
        """<script>
    var win=window.open();win.document.body.innerHTML='{{html}}';
    </script>""".replace("\n", ""),
    )
    jscode = jscode.replace("{{html}}", _regex.sub(" +", " ", html.replace("\n", "")))
    if _utils._in_notebook():
        iDisplay(iHTML(jscode))


def _embed_figure(figfiles, figfmt):
    """Embeds the figure bytes in the html output"""
    if isinstance(figfiles, list):
        embed_string = "\n"
        for figfile in figfiles:
            figbytes = figfile.getvalue()
            if figfmt == "svg":
                return figbytes.decode()
            data_uri = _b64encode(figbytes).decode()
            embed_string.join(
                '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)
            )
    else:
        figbytes = figfiles.getvalue()
        if figfmt == "svg":
            return figbytes.decode()
        data_uri = _b64encode(figbytes).decode()
        embed_string = '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)
    return embed_string