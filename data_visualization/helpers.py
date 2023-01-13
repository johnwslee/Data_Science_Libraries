# ========== Packages ==========
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Functions ==========
def partition(df, target, test_size=.2, seed=8):
    """Partition data into train and test sets.
    Parameters
    ----------
    df: A pandas DataFrame to partition
    target: A string specifying the name of the target column
    test_size: (optional) An integer for number of test examples or a float
               for proportion of test examples
    seed: (optional) An integer to set seed for reproducibility
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=target)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=y)

def find_frequency(series):
    """Provide summary on frequency counts and proportions.
    Parameters
    ----------
    series: A pandas Series containing discrete values
    Returns
    -------
    A pandas DataFrame containing frequency counts and proportions for each
    category.
    """
    columns = ['p_frequency', 'n_frequency']
    frequency = pd.concat([series.value_counts(normalize=True),
                           series.value_counts()], keys=columns, axis=1)
    return frequency

def summarise(df):
    """Provide summary on missing values, unique values and data type.
    Parameters
    ----------
    df: A pandas DataFrame to summarise
    Returns
    -------
    A pandas DataFrame containing count and proportion of missing values,
    count of unique values and data type for each column.
    """
    columns = ['n_missing', 'p_missing', 'n_unique', 'dtype']
    summary = pd.concat([df.isnull().sum(),
                         df.isnull().mean(),
                         df.nunique(),
                         df.dtypes], keys=columns, axis=1)
    return summary.sort_values(by='n_missing', ascending=False)

def find_outlier(series, k=1.5):
    """Find outlier using first and third quartiles and interquartile range.
    Parameters
    ----------
    series: A pandas Series to find outlier in
    k: (optional) An integer indicating threshold of outlier in IQR from Q1/Q3
    Returns
    -------
    A pandas Series containing boolean values where True indicates an outlier.
    """
    q1 = series.quantile(.25)
    q3 = series.quantile(.75)
    iqr = q3-q1
    lower_bound = q1 - k*iqr
    upper_bound = q3 + k*iqr
    is_outlier = (series<lower_bound) | (series>upper_bound)
    return is_outlier

def describe_more(df, features, k=1.5):
    """Provide descriptive statistics and outlier summary for numerical features.
    Parameters
    ----------
    df: A pandas DataFrame to describe
    features: A list of numerical feature column names to use
    k: (optional) An integer indicating threshold of outlier in IQR from Q1/Q3
    Returns
    -------
    A pandas DataFrame containing descriptive statistics and outlier summary.
    """
    descriptives = df[features].describe()
    outliers = df[features].apply(find_outlier)
    descriptives.loc['n_outliers']= outliers.sum()
    descriptives.loc['p_outliers']= outliers.mean()
    return descriptives

def plot_discrete(df, feature, target, orientation='v', figsize=(14, 4)):
    """Plot target mean and counts for unique values in feature.
    Parameters
    ----------
    df: A pandas DataFrame to use
    feature: A string specifying the name of the feature column
    target: A string specifying the name of the target column
    orientation: (optional) 'h' for horizontal and 'v' for  orientation of bars
    figsize: (optional) A tuple specifying the shape of the plot
    Returns
    -------
    A plot containing 2 subplots. Left subplot shows counts of categories. Right
    subplot shows target mean value for each category.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    if orientation=='v':
        sns.countplot(data=df, x=feature, ax=ax[0])
        sns.barplot(data=df, x=feature, y=target, ax=ax[1])
        ax[1].set_ylim([0,1])

    elif orientation=='h':
        sns.countplot(data=df, y=feature, ax=ax[0])
        sns.barplot(data=df, x=target, y=feature, orient='h', ax=ax[1])
        ax[1].set_xlim([0,1])

    ax[0].set_title(f"Category counts in {feature}")
    ax[1].set_title(f"Mean target by category in {feature}")
    plt.tight_layout() # To ensure subplots don't overlay

def plot_continuous(df, feature, target, bins=30, figsize=(14, 5)):
    """Plot histogram, density plot, box plot and swarm plot for feature colour
    coded by target.
    Parameters
    ----------
    df: A pandas DataFrame to use
    feature: A string specifying the name of the feature column
    target: A string specifying the name of the target column
    bins: (optional) An integer for number of bins in histogram
    figsize: (optional) A tuple specifying the shape of the plot
    Returns
    -------
    A plot containing 4 subplots. Top left subplot shows number of histogram.
    Top right subplot shows density plot. Bottom left subplot shows box plot.
    Bottom right subplot shows swarm plot. Each contains overlaying graphs for
    each class in target.
    """
    fig, ax = plt.subplots(2, 2, figsize=(14,8))

    sns.histplot(data=df, x=feature, hue=target, bins=bins, ax=ax[0,0])
    ax[0,0].set_title(f'Histogram of {feature} by {target}')

    sns.kdeplot(data=df, x=feature, hue=target, common_norm=False, fill=True, 
                ax=ax[0,1])
    ax[0,1].set_title(f'Density plot of {feature} by {target}')

    sns.boxplot(data=df, y=feature, x=target, ax=ax[1,0])
    ax[1,0].set_title(f'Box plot of {feature} by {target}')

    sns.swarmplot(data=df.dropna(), y=feature, x=target, ax=ax[1,1])  
    ax[1,1].set_title(f'Swarm plot of {feature} by {target}')
    plt.tight_layout() # To ensure subplots don't overlay