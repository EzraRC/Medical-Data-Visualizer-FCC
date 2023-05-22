import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import datafile
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where((df["weight"]/(df["height"]/100)**2) > 25,1,0)

# Normalize data by making 0 always good and 1 always bad. 
# If the value of 'cholesterol' or 'gluc' is 1, make the value 0. 
# If the value is more than 1, make the value 1.

#Cholesterol
df["cholesterol"] = np.where(df["cholesterol"] > 1, 1, 0)
#Glucose
df["gluc"] = np.where(df["gluc"] > 1, 1, 0)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    variables = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']

    #Reshape the DataFrame using 'melt' to create a long-form representation.
    df_cat = df.melt(id_vars = "cardio", value_vars = variables)

    # Group and reformat the data by 'cardio'

    # Group the data by 'cardio', 'variable', and 'value' columns
    df_cat = df_cat.groupby(["cardio", "variable", "value"])["value"]
    # Count the occurrences of each combination
    df_cat = df_cat.count()
    # Reset the index and rename the count column as 'total'
    df_cat = df_cat.reset_index(name = "total")

    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(x = "variable",
                        y = "total",
                        hue = "value",
                        col = "cardio",
                        kind = "bar",
                        data = df_cat
            )
    
    # Get the figure for the output
    fig = graph.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
        #Height Quantiles set
        (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) &
        #Weight Quantiles set
        (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    # Create a boolean mask for the upper triangular part of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype = bool))
    # Create an array of zeros with the same shape as the correlation matrix
    mask = np.zeros_like(corr)
    # Set the upper triangular elements of the mask to True
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (10, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr,
                annot = True,
                fmt = '.1f',
                linewidths = 0.5,
                square = True,
                mask = mask)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
