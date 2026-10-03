import pandas as pd
import numpy as np
from scipy import stats
df = pd.DataFrame({
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
})
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)
print("\nMedian used for filling:", median_revenue)
print("\nAfter median fill:")
print(df.to_string(index=False))
revenue = df["Revenue"]
df["Revenue_MinMax"] = (
    (revenue - revenue.min()) /
    (revenue.max() - revenue.min())
)
df["Revenue_Zscore"] = stats.zscore(revenue)
q1 = revenue.quantile(0.25)
q3 = revenue.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df["Outlier"] = (
    (revenue < lower_bound) |
    (revenue > upper_bound)
)
print("\nIQR lower bound:", lower_bound)
print("IQR upper bound:", upper_bound)
print("\nFinal table:")
print(df.round(4).to_string(index=False))
# Show only the day numbers flagged as outliers.
outlier_days = df.loc[df["Outlier"], "Day"].tolist()
print("\nIQR outliers detected on day(s):", outlier_days)