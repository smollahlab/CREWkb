import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def create_cm(confusion_matrix, model_name=None, class_labels=None, save_to=None):
    """Generate a heatmap visualization of a confusion matrix.
    
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Square matrix where the `i`th row and `j`th column is the number of 
        class `i` samples that were classified as class `j`
    model_name : str, default=`None`
        Name of the model from which the confusion matrix. If `None` then the 
        heatmap will not have a title.
    class_labels : ndarray of shape (n_classes,), default=`None`
        Labels for each class. If `None` then the labels will either be ordinal categories 
        or the index/column information if `confusion_matrix` is a pandas.DataFrame.
    save_to : str, default=`None`
        Filepath to save heatmap image. If `None` then image will not be saved.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Heatmap of confusion matrix. 
    """
    plot_params = {
        'sns_style': 'white',
        'figsize': (4,4),
        'fontscale': 1.2,
        'class_fontsize': 14,
        'title_fontsize': 14,
        'cmap':'RdPu',
    }
    sns.set_theme(
        style=plot_params['sns_style'], 
        font_scale=plot_params['fontscale']
    )

    # create heatmap
    fig, ax = plt.subplots(figsize=plot_params['figsize'])
    ax = sns.heatmap(
        confusion_matrix,
        ax=ax,
        annot=True,
        square=True,
        cmap=plot_params['cmap'], 
        fmt="d",
        cbar_kws={"shrink": 0.85}
    )

    # format axes and spines
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    if class_labels:
        ax.set_xticklabels(class_labels, fontsize=plot_params['class_fontsize'])
        ax.set_yticklabels(class_labels, fontsize=plot_params['class_fontsize'])
    if model_name:
        ax.set_xlabel(
            model_name, 
            fontsize=plot_params['title_fontsize'], 
            labelpad=11
        )

    # save file
    if save_to:
        fig.savefig(
            save_to, 
            format="png", 
            bbox_inches='tight', 
            dpi=600
        )
    return ax


def create_cm_grid(df, class_labels=None, save_to=None):
    """Generate a 2x4 grid of heatmaps for multiple confusion matrices.
    
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_models, n_metrics)
        Matrix indexed by model name where the "Confusion Matrix" column of the 
        `i`th row is the confusion matrix of the `i`th model. 
    class_labels : ndarray of shape (n_classes,), default=`None`
        Labels for each class. If `None` then the labels will either be ordinal categories 
        or the index/column information if `confusion_matrix` is a pandas.DataFrame.
    save_to : str, default=`None`
        Filepath to save heatmap image. If `None` then image will not be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Grid of confusion matrix heatmaps. 
    """
    plot_params = {
        'sns_style': 'white',
        'figsize': (20, 8),
        'fontscale': 1.2,
        'class_fontsize': 14,
        'subtitle_fontsize': 14,
        'subtitle_weight': 'bold',
        'vertical_space': 0.5,
        'cmap':'RdPu',
    }

    sns.set_theme(
        style=plot_params['sns_style'], 
        font_scale=plot_params['fontscale']
    )
    fig, axes = plt.subplots(2, 4, figsize=plot_params['figsize'])
    axes = axes.flatten()
    for i in range(len(df)):
        cm = df.iloc[i]["Confusion Matrix"]
        ax = axes[i]
        model_name = df.index[i]

        # plot heatmap
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            square=True,
            cmap=plot_params['cmap'], 
            fmt="d",
            cbar_kws={"shrink": 0.85}
        )
        # format axes and spines
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        if class_labels:
            ax.set_xticklabels(class_labels, fontsize=plot_params['class_fontsize'])
            ax.set_yticklabels(class_labels, fontsize=plot_params['class_fontsize'])
        if model_name:
            ax.set_xlabel(
                model_name, 
                fontsize=plot_params['subtitle_fontsize'], 
                fontweight=plot_params['subtitle_weight'],
                labelpad=11
            )
    fig.subplots_adjust(hspace=plot_params['vertical_space'])

    # turn off unused axes if any
    for ax in axes[len(df):]:
        ax.axis('off')

    # save file
    if save_to:
        fig.savefig(
            save_to, 
            format="png", 
            bbox_inches='tight', 
            dpi=600
        )
    return fig


def _long_df(wide_df, var_name="Metric", in_place=False):
    """Unpivot `wide_df` such that values are indexed by model and metric."""

    if not in_place:
        wide_df = wide_df.copy()

    wide_df['Model Name'] = wide_df.index
    return wide_df.loc[:, wide_df.columns != "Confusion Matrix"].melt(
        id_vars="Model Name",
        var_name=var_name,
        value_name="Value"
    )


def all_metric_bar_chart(df, ymin=0, ymax=1, title=None, save_to=None):
    """Create a grouped bar chart of model metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-format DataFrame where each row corresponds to a model and
        columns correspond to evaluation metrics.
    ymin : float, optional
        Lower limit for the y-axis. Default is 0.
    ymax : float, optional
        Upper limit for the y-axis. Default is 1.
    title : str or None, optional
        Title for the plot. If None, no title is set.
    save_to : str or None, optional
        File path to save the figure as a PNG. If None, the figure is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib Axes object containing the bar plot.
    """
    plot_params = {
        'sns_style': 'white',
        'figsize': (10, 5),
        'subtitle_size': 14,
        'subtitle_weight': 'bold',
        'cmap': 'PuRd',
        'x_ticklabel_rotation': 40
    }
    
    with sns.axes_style(plot_params['sns_style']):
        # create barchart
        ldf = _long_df(df)
        fig, ax = plt.subplots(figsize=plot_params['figsize'])
        ax = sns.barplot(
            data=ldf, 
            x='Model Name', 
            y='Value', 
            hue='Metric', 
            palette=plot_params['cmap'])
        
        # format axes and spines
        sns.despine()
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.linspace(ymin, ymax, 9))
        ax.yaxis.grid(True)

        for label in ax.get_xticklabels():
            label.set_rotation(plot_params['x_ticklabel_rotation'])
            label.set_ha("right")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        
        # format legend and title
        ax.legend(
            loc='upper right',
            ncol=len(ax.get_legend_handles_labels()[1])
        )
        if title:
            ax.set_title(
                title, 
                fontsize=plot_params['subtitle_size'], 
                fontweight=plot_params['subtitle_weight'], 
                y=1.05
            )

    # save file
    if save_to:
        fig.savefig(
            save_to, 
            format="png", 
            bbox_inches='tight', 
            dpi=600
        )
    return ax


def metric_bar_chart_by_cv(
        loocv_df, 
        kfold_df, 
        metric, 
        ymin=0.8, 
        ymax=1, 
        title=None,
        save_to=None
    ):
    """
    Plot grouped bar charts of a selected metric across LOOCV and 5-Fold CV strategies.

    Parameters
    ----------
    loocv_df : pandas.DataFrame
        DataFrame containing Leave-One-Out cross-validation results.
        Must include model names as the index and metric columns.
    kfold_df : pandas.DataFrame
        DataFrame containing k-Fold cross-validation results with the
        same structure as `loocv_df`.
    metric : str
        Name of the metric column to compare between the two DataFrames.
    ymin : float, optional
        Lower limit of the y-axis. Default is 0.8.
    ymax : float, optional
        Upper limit of the y-axis. Default is 1.
    title : str or None, optional
        Title of the plot. If None, no title is added.
    save_to : str or None, optional
        File path to save the figure as a PNG. If None, the figure is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the generated bar plot.
    """
    plot_params = {
        'sns_style': 'white',
        'figsize': (10, 5),
        'title_size': 18,
        'subtitle_weight': 'bold',
        'cmap': 'PuRd',
        'x_ticklabel_rotation': 40
    }
    
    df = pd.DataFrame({
        "Leave One Out": loocv_df[metric],
        "5-Fold": kfold_df[metric]
    }, index=loocv_df.index)
    ldf = _long_df(df, var_name="CV Type")

    with sns.axes_style(plot_params['sns_style']):
        fig, ax = plt.subplots(figsize=plot_params['figsize'])
        ax = sns.barplot(
            data=ldf, 
            x='Model Name', 
            y='Value', 
            hue='CV Type', 
            palette=plot_params['cmap'])
        
        # format axes and spines
        sns.despine()
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.linspace(ymin, ymax, 9))
        ax.yaxis.grid(True)

        for label in ax.get_xticklabels():
            label.set_rotation(plot_params['x_ticklabel_rotation'])
            label.set_ha("right")
            label.set_fontweight(plot_params['subtitle_weight'])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        # format legend and title
        ax.legend(
            loc='upper right',
            ncol=len(ax.get_legend_handles_labels()[1])
        )
        if title:
            ax.set_title(
                title, 
                fontsize=plot_params['title_size'], 
                fontweight='bold', 
                y=1.05
            )

    # save file
    if save_to:
        fig.savefig(
            save_to, 
            format="png", 
            bbox_inches='tight', 
            dpi=600
        )
    return ax
