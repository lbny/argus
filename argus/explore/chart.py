import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import itertools

from argus.utils import to_array, to_ndarray

def plot_mirrorline(df, x_column, scale=False):
    if isinstance(x_column, str):
        x_points = df[x_column]
    else:
        x_points = to_array(x_column)
    assert df.shape[1] == 2, "Dataframe must have two columns"
    upper_line = df.columns[0]
    lower_line = df.columns[1]
    if scale:
        for col in [upper_line, lower_line]:
            df[col] = (df[col] - df[col].min())/ (df[col].max() - df[col].min())
    # Ensure lines are positive
    df[upper_line] = df[upper_line] - min(df[upper_line].min(), 0)
    df[lower_line] = - (df[lower_line] - min(df[lower_line].min(), 0))
    
    sns.lineplot(y=df[upper_line], x=x_points)
    sns.lineplot(y=df[lower_line], x=x_points)

    plt.legend()
    plt.show()


def plot_multiline(df: pd.DataFrame, x_column, labels=None, scale=False):
    if isinstance(x_column, str):
        x_points = df[x_column]
    else:
        x_points = to_array(x_column)
    
    if labels:
        labels = to_array(labels)
        assert df.shape[1] == labels.shape[0], "Data columns and labels must have same dimensionality"
    if isinstance(df, pd.Series):
        _df = pd.DataFrame()
        _df['_'] = df.values
        df = _df
    for col in df.columns:
        if scale:
            y = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            y = df[col]
        sns.lineplot(x=x_points, y=y)
    plt.legend()
    plt.show()

def plot_densities(data, labels=None):
    data = to_ndarray(data)
    if labels:
        labels = to_array(labels)
        assert data.shape[1] == labels.shape[0], "Data columsn and labels must have same dimensionality"
    
    for j in range(data.shape[1]):
        sns.distplot(data[:, j], label=labels[j])
    plt.legend()
    plt.show()

def plot_histo(x, bins=100, title=None):
    x = to_array(x)  

    fig, ax = plt.subplots()
    ax.set_title(title)
    plot.plt(x)
    plt.show()



def plot_grouped_bar(data, labels, series_names, width=0.35, y_label=None, title=None):
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    labels = to_array(labels)
    series_names = to_array(series_names)
    assert len(data.shape) == 2, "Input data must be a 2-dimensional array"
    assert data.shape[1] == labels.shape[0], "Labels and data columns must have same dimensionality"
    assert data.shape[0] == series_names.shape[0], "Series_names and data rows must have same dimensionality"
    x = np.arange(labels.shape[0])  # the label locations
    
    fig, ax = plt.subplots()
    rectangles = []
    for i in range(data.shape[0]):
        rectangles.append(ax.bar(x - i * width/data.shape[1], data[i, :], width/data.shape[1], label=series_names[i]))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
    
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
    #for rectangle in rectangles:
        #autolabel(rectangle)
        
    fig.tight_layout()

    plt.show()

def plot_multiscatter(df: pd.DataFrame, features_x, features_y, within_x=False, within_y=False):
    features_x = to_array(features_x)
    features_y = to_array(features_y)
    
    for x, y in itertools.product(features_x, features_y):
        plot_scatter(x=df[x], y=df[y])
    
    if within_x:
        plot_multiscatter(df, features_x, features_x)
        
    if within_y:
        plot_multiscatter(df, features_y, features_y)

def plot_scatter(x, y, z=None, title=None):
    ax = sns.scatterplot(x=x, y=y, hue=z)
    if title:
        ax.set_title(title)
    plt.show()
