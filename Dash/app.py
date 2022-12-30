import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State

# ********************* DATA PREPARATION *********************
# Load data
df = sns.load_dataset('titanic').drop(columns=['pclass', 'embarked', 'alive'])

# Format data for dashboard
df.columns = df.columns.str.capitalize().str.replace('_', ' ')
df.rename(columns={'Sex': 'Gender'}, inplace=True)
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.capitalize()

# Partition into train and test splits
TARGET = 'Survived'
y = df[TARGET]
X = df.drop(columns=TARGET)

numerical = X.select_dtypes(include=['number', 'boolean']).columns
categorical = X.select_dtypes(exclude=['number', 'boolean']).columns
X[categorical] = X[categorical].astype('object')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, 
                                                    stratify=y)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Build pipeline
ct = make_column_transformer(
    (
        make_pipeline(
            SimpleImputer(strategy='constant', fill_value='Missing'),
            OneHotEncoder(sparse=False)
        ),
        categorical
    ),
    (
        SimpleImputer(strategy='mean'),
        numerical
    )
)

pipeline = make_pipeline(
    ct, RandomForestClassifier(random_state=42)
)
pipeline.fit(X_train, y_train)

# Add predicted probabilities
test['Probability'] = pipeline.predict_proba(X_test)[:,1]
test['Target'] = test[TARGET]
test[TARGET] = test[TARGET].map({0: 'No', 1: 'Yes'})

labels = []
for i, x in enumerate(np.arange(0, 101, 10)):
    if i>0:
        labels.append(f"{previous_x}% to <{x}%")
    previous_x = x
test['Binned probability'] = pd.cut(test['Probability'], len(labels), labels=labels, 
                                    right=False)

# Helper functions for dropdowns and slider
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options
def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value
def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks

# Graphs
histogram = px.histogram(test, x='Probability', color=TARGET, 
                         marginal="box", nbins=30)
barplot = px.bar(test.groupby('Binned probability', 
                              as_index=False)['Target'].mean(), 
                 x='Binned probability', y='Target')
columns = ['Age', 'Gender', 'Class', 'Embark town', TARGET, 
           'Probability']
table = go.Figure(data=[go.Table(
    header=dict(values=columns),
    cells=dict(values=[test[c] for c in columns])
)])

# ********************* Dash app *********************
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div(
        [
            html.H1("Titanic predictions"),
            html.P("Summary of predicted probabilities for Titanic test dataset."),
            html.Img(src="assets/left_pane.png"),
            html.Label("Passenger class", className='dropdown-labels'), 
            dcc.Dropdown(
                id='class-dropdown', 
                className='dropdown',
                multi=True, 
                options=create_dropdown_options(test['Class']),
                value=create_dropdown_value(test['Class'])
                ),
            html.Label("Gender", className='dropdown-labels'), 
            dcc.Dropdown(
                id='gender-dropdown', 
                className='dropdown', 
                multi=True,
                options=create_dropdown_options(test['Gender']),
                value=create_dropdown_value(test['Gender'])
            ),
            html.Button(id='update-button', children="Update")
        ], id='left-container'
    ),
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="histogram", figure=histogram),
                    dcc.Graph(id="barplot", figure=barplot)
                ], id='visualisation'
            ),
            html.Div(
                [
                    dcc.Graph(id="table", figure=table),
                    html.Div(
                        [
                            html.Label("Survival status", className='other-labels'), 
                            daq.BooleanSwitch(id='target_toggle', className='toggle', on=True),
                            html.Label("Sort probability in an ascending order", className='other-labels'),
                            daq.BooleanSwitch(id='sort_toggle', className='toggle', on=True),
                            html.Label("Number of records", className='other-labels'), 
                            dcc.Slider(
                                id='n-slider',
                                min=5, max=20, step=1, value=10, 
                                marks=create_slider_marks([5, 10, 15, 20])
                            ),
                        ], id='table-side'
                    )
                ], id='data-extract'
            )
        ], id='right-container'
    )
], id='container')

if __name__ == '__main__':
    app.run_server(debug=True)