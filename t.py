# t_plot.py
import pandas as pd
from utils.analysis import compute_regression
from utils.plotting import make_scatterplot

# Sample data
df = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5],
    "Peak": [10, 20, 15, 25, 30]
})

# Compute regression
slope, intercept = compute_regression(df, "Rank", "Peak")

# Generate the scatterplot URI
uri = make_scatterplot(df, "Rank", "Peak", slope, intercept)

# Print the start of the URI so we can confirm it’s valid
print(uri[:100] + "...")
