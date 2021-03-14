# Helper functions for Data preparation

def create_age_groups(age):
    if age <= 12:
        return "Child"
    if 12 < age <= 19:
        return "Teenager"
    if 19 < age:
        return "Adult"
    else:
        return "Unknown"
    

def prepare_data(df, train_set=True):
    
    # create new feature
    df["Age_Group"] = df.Age.apply(create_age_groups)
    
    # drop features that we are not going to use
    df.drop(["Name", "Age", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)
    
    # rename column "Parch" to "ParCh"
    df.rename({"Parch": "ParCh"}, axis=1, inplace=True)
    
    # rearange order of columns
    if train_set:
        df = df[["Sex", "Pclass", "Age_Group", "Embarked", "SibSp", "ParCh", "Survived"]]
    else:
        df = df[["Sex", "Pclass", "Age_Group", "Embarked", "SibSp", "ParCh"]]
    
    return df


def replace_strings(df):
    
    df.Age_Group.replace({"Adult": 0, "Unknown": 1, "Teenager": 2, "Child": 3}, inplace=True)
    df.Embarked.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
    df.Sex.replace({"male": 0, "female": 1}, inplace=True)

    return df