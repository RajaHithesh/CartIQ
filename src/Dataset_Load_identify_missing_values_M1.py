import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the dataset (Make sure 'instacart_50k_Raw.csv' is in your project directory)
df = pd.read_csv("D:/ML/CartIQ/dataset/instacart_50k_Raw.csv")


# 2. Retrieve data in different ways




# View the first 5 rows
print("--- First 5 Rows ---")
print(df.head())


# Print specific columns
print("----print 6 columns----")
subset = df.iloc[:, 0:6]
print(subset)


# 2. Identify missing values per column
missing_counts = df.isnull().sum()
print("-----Missing Values Per Column:----------")


print(missing_counts)
print("-" * 40)


# 3. Detect duplicate rows
# Keeps the first occurrence and marks subsequent duplicates as True
duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)


# 4. Produce a missingness heatmap
plt.figure(figsize=(10, 6))
# cbar=False hides the color bar; yticklabels=False hides row numbers for cleaner visuals; :
# cmap="viridis" Uses a purple-to-yellow color scale for high contrast.
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")

# Save the heatmap so it stays reproducible, then display it
output_folder = "D:/ML/CartIQ/outputs/EDA_Analysis_outputs"
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, "Missing_Values_Heatmap_M1.png"),
            dpi=300, bbox_inches="tight")
plt.show()
