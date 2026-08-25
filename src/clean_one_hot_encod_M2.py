import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer




# ==========================================================
# Load CartIQ Grocery Basket Prediction Dataset
# Original dataset will NOT be modified
#
# NOTE (CartIQ adaptation):
# product_name has ~11,650 unique values. One-hot encoding it
# would create ~11,800 columns (several GB), so one-hot is
# applied ONLY to the low-cardinality categorical columns:
# eval_set, aisle and department.
# product_name is still handled by the label / ordinal /
# target / embedding encoding scripts.
# ==========================================================


df = pd.read_csv("D:/ML/CartIQ/dataset/instacart_50k_Raw.csv")


# Create a copy for processing
data = df.copy()


print("Original Dataset")
print("------------------------")
print(data.head())


print("Dataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(data.dtypes)






print("\nDuplicate Records:", df.duplicated().sum())






# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================


for col in data.select_dtypes(include="object").columns:
   data[col] = data[col].str.strip()




# ==========================================================
# 2. Identify Missing Values
# ==========================================================


print("Missing Values Before Cleaning:")
print(data.isnull().sum())




# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================


before_duplicates = data.shape[0]


data = data.drop_duplicates()


after_duplicates = data.shape[0]


print("\nDuplicate Records Removed:",
     before_duplicates - after_duplicates)




# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
   include=np.number
).columns.tolist()


cat_cols_all = data.select_dtypes(
   exclude=np.number
).columns.tolist()


# Exclude the very high-cardinality text column from ONE-HOT only
HIGH_CARDINALITY_EXCLUDE = ["product_name"]
cat_cols = [col for col in cat_cols_all if col not in HIGH_CARDINALITY_EXCLUDE]


print("\nCategorical columns used for One-Hot Encoding:")
print(cat_cols)
print("\nCategorical columns excluded from One-Hot (too many categories):")
print([col for col in cat_cols_all if col in HIGH_CARDINALITY_EXCLUDE])




# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================


if len(num_cols) > 0:


   num_imputer = SimpleImputer(
       strategy="mean"
   )


   data[num_cols] = num_imputer.fit_transform(
       data[num_cols]
   )





# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================


if len(cat_cols) > 0:


   cat_imputer = SimpleImputer(
       strategy="most_frequent"
   )


   data[cat_cols] = cat_imputer.fit_transform(
       data[cat_cols]
   )




# ==========================================================
# 6. One-Hot Encoding
# ==========================================================


if len(cat_cols) > 0:


   encoder = OneHotEncoder(
       sparse_output=False,
       handle_unknown="ignore"
   )


   encoded_values = encoder.fit_transform(
       data[cat_cols]
   )







   encoded_df = pd.DataFrame(
       encoded_values,
       columns=encoder.get_feature_names_out(cat_cols)
   )





   # Reset index for merging
   encoded_df.reset_index(
       drop=True,
       inplace=True
   )





   # Keep numerical columns
   numeric_df = data[num_cols].reset_index(
       drop=True
   )





   # Merge numerical + encoded columns
   final_output = pd.concat(
       [
           numeric_df,
           encoded_df
       ],
       axis=1
   )




else:


   final_output = data.copy()






# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================


print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())




# ==========================================================
# Save Final Result
# ==========================================================


final_output.to_csv(
   "D:/ML/CartIQ/dataset/clean_one_hot_encoding_M2.csv",
   index=False
)






print("\n======================================")
print("Original dataset is NOT modified.")
print("Cleaning and One-Hot Encoding completed.")
print("Output file:")
print("clean_one_hot_encoding_M2.csv")
print("======================================")
