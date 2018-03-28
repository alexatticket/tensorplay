import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv('iris.csv', ',')

continuous_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
categorical_features = []

LABEL_COLUMN = 'species_c'
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


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


# change to numeric values so that we can input it later
bucket_category(df, 'species', 'species_c', 3)

# shuffle rows
df = df.sample(frac=1)

# select training set
msk = np.random.rand(len(df)) < 0.7
trainset = df[msk]

# select test and eval sets
test_evalset = df[~msk]
testset_size = len(test_evalset) / 2
testset, evalset = test_evalset.head(testset_size), test_evalset.tail(testset_size)


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
        label = tf.constant(dataframe[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label

    # Returns the feature columns
    return feature_cols


def train_input_fn():
    return input_fn(trainset)


def test_input_fn():
    return input_fn(testset, False)


def eval_input_fn():
    return input_fn(evalset)


def get_feature_columns():
    feature_columns = []
    for column in continuous_features:
        feature_column = tf.feature_column.numeric_column(key=column)
        feature_columns.append(feature_column)
    return feature_columns


# create the model. type classifier.
model = tf.estimator.DNNClassifier(hidden_units=[100], feature_columns=get_feature_columns(), n_classes=3)

# Train the model.
model.train(input_fn=train_input_fn, steps=100)

# Evaluate how the model performs on data it has not yet seen.
results = model.evaluate(input_fn=eval_input_fn, steps=10)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

# predict categories on the test set
#todo map back to names so that we can print pretty table
predicted_output = model.predict(input_fn=test_input_fn)
for predict, expec in zip(predicted_output, testset['species_c']):
    print("predict\t{}\t{}".format(np.argmax(predict['probabilities']), expec))
