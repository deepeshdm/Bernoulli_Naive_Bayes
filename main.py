import pandas as pd
from helper_functions import prepare_data


# data preparation
df_train = pd.read_csv("train.csv", index_col="PassengerId")
df_test = pd.read_csv("test.csv", index_col="PassengerId")
test_labels = pd.read_csv("test_labels.csv", index_col="PassengerId", squeeze=True)

df_train = prepare_data(df_train)
df_test = prepare_data(df_test, train_set=False)

# handle missing values
embarked_mode = df_train.Embarked.mode()[0]
df_train["Embarked"].fillna(embarked_mode, inplace=True)

#-------------------------------------------------------------------------------------------------

# Takes dataframe & the name of label column.
def create_table(df, label_column):

    table = {}
    # determine values for the label
    counts = df[label_column].value_counts().sort_index()
    table["class_names"] = counts.index.to_numpy()
    table["class_counts"] = counts.values

    # determine probabilities for the features
    for feature in df.drop(label_column, axis=1).columns:
        table[feature] = {}
        counts = df.groupby(label_column)[feature].value_counts()
        df_counts = counts.unstack(label_column)

        # check for "problem of rare values"
        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace=True)
            df_counts += 1

        df_probabilities = df_counts / df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].to_numpy()
            table[feature][value] = probabilities
    return table


lookup_table = create_table(df_train, label_column="Survived")


# Takes single row dataframe & lookup table
def predict_example(row , lookup_table):

    class_estimates = lookup_table["class_counts"]
    for feature in row.index:

        try:
            value = row.loc[feature]
            probabilities = lookup_table[feature][value]
            class_estimates = class_estimates * probabilities

        except KeyError :
            continue

    index_max_class = class_estimates.argmax()
    prediction = lookup_table["class_names"][index_max_class]

    return prediction

#-------------------------------------------------------------------------------------------------

# Predicting test data

example_row = df_test.loc[904]  #passing 904th row
predicted_value = predict_example(example_row,lookup_table)
actual_value = test_labels.loc[904]
print(test_labels)
print("Predicted value : {} , Actual value : {}".format(predicted_value,actual_value))

