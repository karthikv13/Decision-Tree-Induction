import pandas as pd
from collections import Counter
import math
from pprint import pprint


def entropy(probs):
    '''
    Takes a list of probabilities and calculates their entropy
    '''
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(a_list):
    '''
    Takes a list of items with discrete values (e.g., poisonous, edible)
    and returns the entropy for those items.
    '''

    # Tally Up:
    cnt = Counter(x for x in a_list)

    # Convert to Proportion
    num_instances = len(a_list) * 1.0
    probs = [x / num_instances for x in cnt.values()]

    # Calculate Entropy:
    return entropy(probs)


def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    '''
    Takes a DataFrame of attributes, and quantifies the entropy of a target
    attribute after performing a split along the values of another attribute.
    '''

    # Split Data by Possible Vals of Attribute:
    df_split = df.groupby(split_attribute_name)

    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x) / nobs]})[
        target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    if trace:  # helps understand what fxn is doing:
        print(df_agg_ent)

    # Calculate Information Gain:
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


def id3(df, target_attribute_name, attribute_names, default_class=None):
    # Tally target attribute:
    cnt = Counter(x for x in df[target_attribute_name])

    # First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return list(cnt.keys())[0]

    # Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class

    # Otherwise: This dataset is ready to be divvied up!
    else:
        # Get Default Value for next recursive call of this function:
        index_of_max = list(cnt.values()).index(max(cnt.values()))
        default_class = list(cnt.keys())[index_of_max]  # most common value of target attribute in dataset

        # Choose Best Attribute to split on:
        gains = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gains.index(max(gains))
        best_attr = attribute_names[index_of_max]

        # Create an empty tree, to be populated in a moment
        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                          target_attribute_name,
                          remaining_attribute_names,
                          default_class)
            tree[best_attr][attr_val] = subtree
        return tree


def classify(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):    # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result               # this is a label
    else:
        return default


df_shroom = pd.read_csv('Dataset/mushroom_data.csv')
# The initial entropy of the poisonous/not attribute for our dataset.
total_entropy = entropy_of_list(df_shroom['class'])

# print('Total Entropy: ' + str(total_entropy))
# print('\nInfo-gain for best attribute is ' + str(information_gain(df_shroom, 'odor', 'class')))

# Get Predictor Names (all but 'class')
attribute_names = list(df_shroom.columns)
attribute_names.remove('class')

# Run Algorithm:
# tree = id3(df_shroom, 'class', attribute_names)
# pprint(tree)
# df_shroom['predicted'] = df_shroom.apply(classify, axis=1, args=(tree, 'p'))

# classify func allows for a default arg: when tree doesn't have answer for a particular
# combitation of attribute-values, we can use 'poisonous' ('p') as the default guess (better safe than sorry!)

# print('Accuracy is ' + str(sum(df_shroom['class']==df_shroom['predicted']) / (1.0*len(df_shroom.index))))

total_rows = int(df_shroom.shape[0] * .8)
training_data = df_shroom.iloc[1:total_rows]  # 80% of data as training data
test_data = df_shroom.iloc[total_rows:]       # remaining as test data
train_tree = id3(training_data, 'class', attribute_names)
pprint(train_tree)
test_data['predicted2'] = test_data.apply(                                # <---- test_data source
                                          classify,
                                          axis=1,
                                          args=(train_tree, 'p'))  # <---- train_data tree

print('Accuracy is ' + str(sum(test_data['class']==test_data['predicted2']) / (1.0*len(test_data.index))))
