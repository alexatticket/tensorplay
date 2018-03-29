import numpy as np
import pandas as pd
import tensorflow as tf

train = pd.read_csv('train.csv', ',')
test = pd.read_csv('test.csv', ',')
correct = pd.read_csv('gender_submission.csv', ',')

continuous_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['embarked_c', 'Pclass', 'sex_c']
label_column = 'Survived'

# fill missing data with median
train = train.fillna(train.median())
# fill missing data with mean
# train = train.fillna(train.mean())
# drop missing data
# train = train.dropna()


# todo fix cabin nr
# todo get titles from name
# todo predict age from other data
# todo calculate missing fare from Pclass

# shuffle rows
train = train.sample(frac=1)


def bucket_category(dataframe, name, new_name, no_buckets):
    hist = pd.value_counts(dataframe[name])
    hist_trunc = hist[0:no_buckets]
    dataframe[new_name] = dataframe.apply(lambda row: map_to_index(row, name, hist_trunc.index.tolist()), axis=1)


def map_to_index(row, name, categories):
    # if the category isn't in top n then we use n + 1
    index = len(categories) + 1
    try:
        index = categories.index(row[name])
    except:
        pass
    return index


bucket_category(train, 'Embarked', 'embarked_c', 4)
bucket_category(train, 'Sex', 'sex_c', 2)

bucket_category(test, 'Embarked', 'embarked_c', 4)
bucket_category(test, 'Sex', 'sex_c', 2)


# Converting DataFrame into Tensors
def input_fn(dataframe, training=True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(dataframe[k].values)
                       for k in continuous_features}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(dataframe[k].size)],
        values=dataframe[k].values,
        dense_shape=[dataframe[k].size, 1])
        for k in categorical_features}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(dataframe[label_column].values)

        # Returns the feature columns and the label.
        return feature_cols, label

    # Returns the feature columns
    return feature_cols


def train_input_fn():
    return input_fn(train)


def test_input_fn():
    return input_fn(test, False)


def get_feature_columns():
    feature_columns = []
    for column in continuous_features:
        feature_column = tf.feature_column.numeric_column(key=column)
        feature_columns.append(feature_column)
    return feature_columns


# create the model. type classifier.
model = tf.estimator.DNNClassifier(hidden_units=[100, 20], feature_columns=get_feature_columns(), n_classes=2)

# Train the model.
model.train(input_fn=train_input_fn, steps=1000)

# predict categories on the test set
predicted_output = model.predict(input_fn=test_input_fn)
sum = 0
correct['Name'] = test['Name']
for predict, actual in zip(predicted_output, correct.values):
    predict_survived = np.argmax(predict['probabilities'])
    sum += abs(predict_survived - actual[1])
    print("predict\t{}\t actual\t{} \t{}".format(predict_survived, actual[1], actual[2]))

totalno = len(correct['Survived'])
correct_ratio = 1 - float(sum) / len(correct['Survived'])
print("Total no errors {} of {} ratio {}".format(sum, totalno, correct_ratio))
