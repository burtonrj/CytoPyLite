from dim_reduction import dimensionality_reduction
from itertools import cycle
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

SEQ_COLOURS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
               'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
               'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
               'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
               'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
               'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
               'hot', 'afmhot', 'gist_heat', 'copper']


def _set_palette(discrete: bool,
                 palette: str):
    """
    Returns valid Matplotlib palette depending on type of plot

    Parameters
    ----------
    discrete: bool
        Is the label descrete?
    palette: str
        Requested palette

    Returns
    -------
    Str
        Valid palette
    """
    if not discrete:
        if palette not in SEQ_COLOURS:
            warn("Palette invalid for discrete labelling, defaulting to 'inferno'")
            return "inferno"
    return palette


def scatterplot(data: pd.DataFrame,
                method: str,
                label: str,
                discrete: bool,
                n_components: int = 2,
                size: str or None = None,
                scale_factor: int = 15,
                figsize: tuple = (10, 12),
                palette: str = "tab20",
                colourbar_kwargs: dict or None = None,
                legend_kwargs: dict or None = None,
                **kwargs):
    """
    Generates a standard scatterplot

    Parameters
    ----------
    data: Pandas.DataFrame
        Standard explorer summary dataframe as generated from CytoPy
    method: str
        Dim reduction method applied
    label: str
        Meta variable used to colour points
    discrete: bool
        Is the label discrete?
    n_components: int (default=2)
        Number of componenents, must be either 2 or 3
    size: str, optional
        Column that controls point size
    scale_factor: int (default=15)
        Scale factor (multiplied by the size variable)
    figsize: tuple (default=(10,12))
        Size of the scatterplot figure
    palette: str (default="tab20")
        Colour palette to use
    colourbar_kwargs: dict, optional
        Additional keyword arguments for colourbar
    legend_kwargs: dict, optional
        Additional keyword arguments for legend
    kwargs:
        Additional keyword arguments passed to Matplotlib.Axes.scatter call

    Returns
    -------
    Matplotlib.Axes
    """
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    dims = [data[f"{method}{i + 1}"].values for i in range(n_components)]
    colourbar_kwargs = colourbar_kwargs or {}
    legend_kwargs = legend_kwargs or {"bbox_to_anchor": (1.15, 1.)}
    palette = _set_palette(discrete=discrete, palette=palette)
    if not discrete:
        if size is not None:
            kwargs["s"] = data[size].values * scale_factor
        im = ax.scatter(*dims,
                        c=data[label],
                        cmap=palette,
                        **kwargs)
        ax.set_xlabel(f"{method}1")
        ax.set_ylabel(f"{method}2")
        fig.colorbar(im, ax=ax, **colourbar_kwargs)
        return ax
    colours = cycle(plt.get_cmap(palette).colors)
    for l, df in data.groupby(label):
        dims = [df[f"{method}{i + 1}"].values for i in range(n_components)]
        if size is not None:
            kwargs["s"] = df[size].values * scale_factor
        ax.scatter(*dims, color=next(colours), label=l, **kwargs)
    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    ax.legend(**legend_kwargs)
    return ax


def _scatterplot_defaults(**kwargs):
    """
    Generates a dictionary of default keyword arguments for scatterplot

    Parameters
    ----------
    kwargs:
        Optional keyword arguments to include/overwrite

    Returns
    -------
    dict
    """
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"edgecolor": "black",
                "alpha": 0.75,
                "linewidth": 2,
                "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def clusterplot(data: pd.DataFrame,
                label: str,
                features: list,
                discrete: bool,
                n_components: int = 2,
                method: str = "UMAP",
                dim_reduction_kwargs: dict or None = None,
                scale_factor: int = 15,
                figsize: tuple = (12, 8),
                palette: str = "tab20",
                colourbar_kwargs: dict or None = None,
                legend_kwargs: dict or None = None,
                **kwargs):
    """
    Generates a meta-clustering bubble plot

    Parameters
    ----------
    data: Pandas.DataFrame
        Standard explorer summary dataframe as generated from CytoPy
    features: list
        List of columns to use for dimensionality reduction task
    method: str
        Dim reduction method to apply
    label: str
        Meta variable used to colour points
    discrete: bool
        Is the label discrete?
    n_components: int (default=2)
        Number of componenents, must be either 2 or 3
    scale_factor: int (default=15)
        Scale factor (multiplied by the size variable)
    figsize: tuple (default=(10,12))
        Size of the scatterplot figure
    palette: str (default="tab20")
        Colour palette to use
    colourbar_kwargs: dict, optional
        Additional keyword arguments for colourbar
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments to pass to dimension reduction method
    legend_kwargs: dict, optional
        Additional keyword arguments for legend
    kwargs:
        Additional keyword arguments passed to Matplotlib.Axes.scatter call

    Returns
    -------
    Matplotlib.Axes
    """
    assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
    assert label in data.columns, f'{label} is not valid, must be an existing column in linked dataframe'
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    data = dimensionality_reduction(method=method,
                                    data=data,
                                    n_components=n_components,
                                    features=features,
                                    **dim_reduction_kwargs)
    kwargs = kwargs or {}
    kwargs = _scatterplot_defaults(**kwargs)
    return scatterplot(data=data,
                       method=method,
                       label=label,
                       n_components=n_components,
                       discrete=discrete,
                       figsize=figsize,
                       palette=palette,
                       colourbar_kwargs=colourbar_kwargs,
                       legend_kwargs=legend_kwargs,
                       size="meta_cluster_size",
                       scale_factor=scale_factor,
                       **kwargs)


def clustermap(data: pd.DataFrame,
               features: list,
               summary_method: str = "mean",
               **kwargs):
    """
    Generate a meta-clustering heatmap with the y-axis (index) being the
    name of meta-clusters. The axis are clustered using hierarchical clustering
    and can be controlled using additional keyword arguments (see Seaborn.clustermap)

    Parameters
    ----------
    data: Pandas.DataFrame
        Standard explorer summary dataframe as generated from CytoPy
    features: list
        List of columns to include in heatmap
    summary_method: str (default="mean")
    kwargs:
        Additional keyword arguments passed to Seaborn.clustermap

    Returns
    -------

    """
    kwgs = {"col_cluster": True,
            "figsize": (10, 15),
            "standard_scale": 1,
            "cmap": "vlag"}
    for k, v in kwargs.items():
        kwgs[k] = v
    data = data.copy()
    data[features] = data[features].apply(pd.to_numeric)
    assert "meta_label" in data.columns, "meta_label missing from data columns"
    if summary_method == "median":
        data = data.groupby("meta_label")[features].median()
    else:
        data = data.groupby("meta_label")[features].mean()
    return sns.clustermap(data, **kwgs)
