import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set(style = "darkgrid")
from sklearn import neighbors, metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer





# This lil' guy right here will handle all of our preprocessing!
def cleanAndEncode(df):
    
    # Drops uneeded training data
    columns_to_drop = ['fire_number', 'fire_name', 'industry_identifier_desc', 'discovered_size', 'fire_id.1']
    df = colDropper(df, columns_to_drop)
    
    # Date columns to encode
    date_columns = ["fire_start_date", "discovered_date", "reported_date", "dispatch_date", "start_for_fire_date", "assessment_datetime", "ia_arrival_at_fire_date", "fire_fighting_start_date", "first_bucket_drop_date", "ex_fs_date"]
    df = dateEncoder(df, date_columns)
    
    # List of catagorical data to be encoded
    categorical_data = [
                "fire_origin", "general_cause_desc", "responsible_group_desc", "activity_class", "true_cause", "det_agent", "det_agent_type",
                "dispatched_resource", "assessment_resource", "fire_type", "fire_position_on_slope", "weather_conditions_over_fire", "wind_direction",
                "fuel_type", "initial_action_by", "ia_access", "bucketing_on_fire"
                ]
    df = catagoricalEncoding(df, categorical_data)

    
    
    
    # Simple Imputer: Predicts missing values based on other values in the same column
    # strategy has the following options: mean, median, mode, most_frequent, constant.    
    strategy = "most_frequent"  # Best for catagorical encoding
    missing_values = np.nan     # If strategy = "constant", change to desired constant value (like 0). Else, do not change!
    output_format = "pandas"    # return type of output
    simple_df = simpleImputing(df, strategy, missing_values, output_format)


    # Iterative Imputer: Okay so this imputes things... iterativly. Our pals at scikit-learn use better word talk:
                                # "A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion."
                                # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    max_iterations = 10         # Maximum number of iterations
    output_format = "pandas"    # return type of output
    iterative_df = iterativeImputing(df, max_iterations, output_format)
    
    # KNN Imputer: Fairly self explainatory as it imputes missing values based on n-nearest neighbours
    mass_X_gravity = "uniform"  # "Uniform" for equal WEIGHT among neighbours. "Distance" for weight = inverse distance. (Closer values are weighted higher)
    neighbours = 5              # Mighty neighbourly...
    output_format = "pandas"    # return type of output
    knn_df = knnImputing(df, mass_X_gravity, neighbours, output_format)
    
    

    
    # Wow, look at all these beautiful models
    models = [simple_df, iterative_df, knn_df]
    return models


# The death drop of columns
def colDropper(df, columns):
    df.drop(columns=columns, inplace=True)
    return df

# I'll take "fire_position_on_slope" for $500 please 
def catagoricalEncoding(df, categorical):
    
    # This type of encoding should be used for data that does not have an order
    ohe = OneHotEncoder(sparse_output=False)
    df = pd.get_dummies(df, columns=categorical, prefix=categorical)
    
    # This type of encoding can be used for data that can be ordered,
    # when initializing oe provide a list in order of columns of the order of the data
    oe = OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E']])
    oe.fit_transform(df[["size_class"]])[2] #here you list each of the columns in the same order you listed the lists of data above
    
    return df

# Real basic date encoder. Still need to add David's fancy version
def dateEncoder(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed').astype('int64') // 10**9
    return df


def datetime_to_sin(date: str, data) -> float:
  data[date] = pd.to_datetime(data[date])
  temp = data[date].dt.dayofyear
  data[date] = np.sin(2 * np.pi * temp / 365)
  
  
def dataTimeContinuum(df):
    
  # deal with date encoding
  dates = ['fire_start_date', 'discovered_date', 'reported_date', 'dispatch_date', 'start_for_fire_date', 'assessment_datetime', 'ia_arrival_at_fire_date', 'fire_fighting_start_date', 'first_bucket_drop_date', 'ex_fs_date']
  for date in dates:
    datetime_to_sin(date, df)

  return df


# It's simple, and it works. real good
def simpleImputing(df, gigaBrain, missing, output):
    imp = SimpleImputer(missing_values=missing, strategy=gigaBrain).set_output(transform=output)
    df = imp.fit_transform(df)
    return df


# This one doesn't skip leg day
def iterativeImputing(df, iterations, output):
    # Select only numeric columns for iterative imputation
    numeric_df = df.select_dtypes(include=[np.number])

    imputer = IterativeImputer(max_iter=iterations, random_state=42).set_output(transform=output)
    numeric_df = imputer.fit_transform(numeric_df)

    # Replace the imputed values back into the original DataFrame
    df[df.select_dtypes(include=[np.number]).columns] = numeric_df

    return df


# And this one went WEEE WEEE WEE WEEEEE, all the way... to their neighbour's?
def knnImputing(df, weighted, neighbours, output):
    numeric_df = df.select_dtypes(include=[np.number])
    imputer = KNNImputer(n_neighbors=neighbours, weights=weighted).set_output(transform=output)
    numeric_df = imputer.fit_transform(numeric_df)
    
    # Replace the imputed values back into the original DataFrame
    df[df.select_dtypes(include=[np.number]).columns] = numeric_df
    return df
