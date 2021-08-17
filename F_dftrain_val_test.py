from A_libraries import *
from B_FeatureImage import *
from C_PreData import *

df["category"] = df["category"].replace({0:'cat', 1:'dog'})
train_df, validate_df, = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15