import ast
import copy
import sys
import arff
import math
import numpy as np
import random


# #################################################
# class for data
# #################################################
class Data:
    def __init__(self):
        self.examples = []
        self.classes = []
        self.attributes = []


# #################################################
# function for reading arff files
# #################################################
def read_arffdata(dataset, datafile):

    array = []
    for row in arff.load(datafile):
        array.append(list(row))

    dataset.examples = np.array(array)
    dataset.classes = dataset.examples[:, -1]
    dataset.examples = dataset.examples[:, 0:-1]
    dataset.attributes = range(len(dataset.examples[0]))


# #################################################
# tree node class that will make up the tree
# #################################################
class TreeNode:
    def __init__(self, parent):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None


# #################################################
# class for classcounts
# #################################################
class ClassCount:
    def __init__(self, classnames, classcount):
        self.classnames = classnames
        self.classcount = classcount


# #################################################
# count number of classes
# #################################################
def count_classes(classes):
    classnames = []
    classcount = []
    count = 0
    for i in range(0, len(classes)):
        if classes[i] not in classnames:
            classnames.append(classes[i])
    for i in range(0, len(classnames)):
        for j in range(0, len(classes)):
            if classnames[i] == classes[j]:
                count += 1
        classcount.append(count)
        count = 0

    map(float, classcount)
    return ClassCount(classnames, classcount)


# #################################################
# Calculate the entropy of the current dataset
# #################################################
def calc_dataset_entropy(dataset, class_counts):
    total_examples = float(len(dataset.examples))
    entropy = 0

    if len(class_counts.classnames) == 1:
        return 0
    else:
        for i in range(0, len(class_counts.classnames)):
            p = class_counts.classcount[i] / total_examples
            if p != 0:
                entropy += p * math.log(p, len(class_counts.classnames))

        entropy = -entropy
        return entropy


##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_gain(dataset, entropy, val, attr_index):
    attr_entropy = 0
    total_examples = float(len(dataset.examples))
    gain_upper_dataset = Data()
    gain_lower_dataset = Data()
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    for example in dataset.examples:
        if example[attr_index] >= val:
            gain_upper_dataset.examples.append(example)
        elif example[attr_index] < val:
            gain_lower_dataset.examples.append(example)

    if len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0:
        # Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
        return -1

    cl1 = count_classes(gain_upper_dataset.classes)
    cl2 = count_classes(gain_lower_dataset.classes)
    attr_entropy += calc_dataset_entropy(gain_upper_dataset, cl1) * len(gain_upper_dataset.examples) / total_examples
    attr_entropy += calc_dataset_entropy(gain_lower_dataset, cl2) * len(gain_lower_dataset.examples) / total_examples

    return entropy - attr_entropy


# #################################################
# Classify leaf
# #################################################
def classify_leaf(dataset):
    class_counts = count_classes(dataset.classes)

    max_count = 0
    max_class_name = ''
    for i in range(len(class_counts.classnames)):
        if max_count <= class_counts.classcount[i]:
            max_count = class_counts.classcount[i]
            max_class_name = class_counts.classnames[i]

    return max_class_name


# #################################################
# compute tree recursively
# #################################################

# initialize Tree
    # if dataset is pure (all one result) or there is other stopping criteria then stop
    # for all attributes a in dataset
        # compute information-theoretic criteria if we split on a
    # abest = best attribute according to above
    # tree = create a decision node that tests abest in the root
    # dv (v=1,2,3,...) = induced sub-datasets from D based on abest
    # for all dv
        # tree = compute_tree(dv)
        # attach tree to the corresponding branch of Tree
    # return tree

def compute_tree(dataset, parent_node, max_depth):
    node = TreeNode(parent_node)
    if parent_node is None:
        node.height = 0
    else:
        node.height = node.parent.height + 1

    class_counts = count_classes(dataset.classes)

    for i in range(len(class_counts.classnames)):
        if len(dataset.examples) == class_counts.classcount[i]:
            node.classification = class_counts.classnames[i]
            node.is_leaf = True
            return node
    else:
        node.is_leaf = False

    attr_to_split = None  # The index of the attribute we will split on
    ai_to_split = None
    max_gain = 0  # The gain given by the best attribute
    split_val = None
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset, class_counts)

    for ai in range(len(dataset.attributes)):
            attr_index = dataset.attributes[ai]

            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples]  # values we can split on
            attr_value_list = list(set(attr_value_list))  # remove duplicates from list of all attribute values
            if len(attr_value_list) > 100:
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total / 10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x * ten_percentile])
                attr_value_list = new_list

            for val in attr_value_list:

                # calculate the gain if we split on this value
                # if gain is greater than local_max_gain, save this gain and this value
                local_gain = calc_gain(dataset, dataset_entropy, val, attr_index)  # calc gain if we split on this value

                if local_gain > local_max_gain:
                    local_max_gain = local_gain
                    local_split_val = val

            if local_max_gain > max_gain:
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index
                ai_to_split = ai

    # attr_to_split is now the best attribute according to our gain metric
    if split_val is None or attr_to_split is None:
        if len(attr_value_list) == 1:
            node.classification = random.choice(class_counts.classnames)
            node.is_leaf = True
            return node
        print "Something went wrong. Couldn't find an attribute to split on or a split value."
    elif max_gain <= min_gain or node.height >= max_depth:

        node.is_leaf = True
        node.classification = classify_leaf(dataset)

        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[ai_to_split]
    node.attr_split_value = split_val
    # currently doing one split per node so only two datasets are created
    upper_dataset = Data()
    lower_dataset = Data()
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    ud = []
    ld = []

    for j in range(len(dataset.examples)):
        if attr_to_split is not None and dataset.examples[j, attr_to_split] >= split_val:
            ud.append(dataset.examples[j])
            upper_dataset.classes.append(dataset.classes[j])
        elif attr_to_split is not None:
            ld.append(dataset.examples[j])
            lower_dataset.classes.append(dataset.classes[j])

    upper_dataset.examples = np.array(ud)
    lower_dataset.examples = np.array(ld)
    node.upper_child = compute_tree(upper_dataset, node, max_depth)
    node.lower_child = compute_tree(lower_dataset, node, max_depth)

    return node


# #################################################
# classify single test instance
# #################################################
def classify_testinstance(example, node):
    if node.is_leaf:
        return node.classification
    else:
        if example[node.attr_split_index] >= node.attr_split_value:
            return classify_testinstance(example, node.upper_child)
        else:
            return classify_testinstance(example, node.lower_child)


# #################################################
# compute test instances for tree
# #################################################
def test_tree(testdata, root):
    results = []
    for i in range(len(testdata.examples)):
        results.append(classify_testinstance(testdata.examples[i], root))
    return results


# #################################################
# calculate testing accuracy
# #################################################
def calculate_accuracy(original, test):
    count_total = 0.0
    count_right = 0.0
    for i in range(len(original)):
        count_total += 1
        if original[i] == test[i]:
            count_right += 1
    return count_right / count_total


# #################################################
# computing forrest
# #################################################
def compute_randomforrest(dataset, n_estimators, max_depth, perc_examples, perc_attributes):
    forrest = []

    # check parameters
    if max_depth < 0:
        max_depth = 0

    if n_estimators < 1:
        n_estimators = 1

    if perc_examples > 1:
        perc_examples = 1
    if perc_examples < 0:
        perc_examples = 0.0000001

    if perc_attributes > 1:
        perc_attributes = 1
    if perc_attributes < 0:
        perc_attributes = 0.0000001

    # calc examples and attribures
    n_examples = round(len(dataset.examples) * perc_examples)
    if n_examples < 1:
        n_examples = 1
    n_attributes = round(len(dataset.attributes) * perc_attributes)
    if n_attributes < 1:
        n_attributes = 1

    # build forrest
    for i in range(n_estimators):
        selection_examples = copy.deepcopy(dataset.examples)
        selection_classes = copy.deepcopy(dataset.classes)
        selection_attributes = copy.deepcopy(dataset.attributes)
        part_dataset = Data()
        part_examples = []
        for j in range(int(n_examples)):
            random_example = random.randrange(len(selection_examples))
            part_dataset.classes.append(selection_classes[random_example])
            part_examples.append(selection_examples[random_example])
            selection_examples = np.delete(selection_examples, random_example, 0)
            selection_classes = np.delete(selection_classes, random_example)
        for k in range(int(n_attributes)):
            random_attribute = random.randrange(len(selection_attributes))
            part_dataset.attributes.append(selection_attributes[random_attribute])
            selection_attributes = np.delete(selection_attributes, random_attribute)
        part_dataset.examples = np.array(part_examples)
        forrest.append(compute_tree(part_dataset, None, max_depth))
    return forrest


# #################################################
# compute test instances for forrest
# #################################################
def test_forrest(testdata, forrest):
    results = []
    results_tree = []
    for i in range(len(testdata.examples)):
        for j in range(len(forrest)):
            results_tree = classify_testinstance(testdata.examples[i], forrest[j])
        cl = count_classes(results_tree)
        max_count = 0
        max_class_name = ''
        for i in range(len(cl.classnames)):
            if max_count <= cl.classcount[i]:
                max_count = cl.classcount[i]
                max_class_name = cl.classnames[i]
        results.append(max_class_name)
    return results


# #################################################
# classify test instances
# #################################################
def classify_testdata(testdata, forrest):
    results = []
    results_tree = []
    for i in range(len(testdata)):
        for j in range(len(forrest)):
            results_tree += [classify_testinstance(testdata[i], forrest[j])]
        cl = count_classes(results_tree)
        max_count = 0
        max_class_name = ''
        for i in range(len(cl.classnames)):
            if max_count <= cl.classcount[i]:
                max_count = cl.classcount[i]
                max_class_name = cl.classnames[i]
        results.append(max_class_name)
    return results


# #################################################
# functions for christian and pascal
# #################################################
def einfaches_klassifizieren_training(data, label, n_estimators, max_depth, perc_examples, perc_attributes):
    trainingdata = Data()
    trainingdata.examples = data
    trainingdata.classes = label
    trainingdata.attributes = range(len(data[0]))
    return compute_randomforrest(trainingdata, n_estimators, max_depth, perc_examples, perc_attributes)


def einfaches_klassifizieren(model, daten):
    label = classify_testdata(daten, model)
    return label


# #################################################
# function main for testing
# #################################################
def main():
    args = str(sys.argv)
    args = ast.literal_eval(args)
    if len(args) < 3:
        print 'To few parameters: trainingdatafile and testdatafile expected'
    else:
        # reading training data
        trainingdatafile = args[1]
        trainingdata = Data()
        read_arffdata(trainingdata, trainingdatafile)

        # reading test data
        testdatafile = args[2]
        testdata = Data()
        read_arffdata(testdata, testdatafile)

        # single tree
        # root = compute_tree(trainingdata, None, 20)
        # testresults1 = test_tree(testdata, root)
        # accuracy1 = calculate_accuracy(testdata.classes, testresults1)
        # print "single tree acc: " + str(accuracy1)

        # random forrest
        # compute_randomforrest(dataset, n_estimators, max_depth, perc_examples, perc_attributes)
        # forrest = compute_randomforrest(trainingdata, 20, 20, 1, 1)
        # testresults2 = test_forrest(testdata, forrest)
        # accuracy2 = calculate_accuracy(testdata.classes, testresults2)
        # print "ran forrest acc: " + str(accuracy2)

        # test christian und pascal functionen
        model = einfaches_klassifizieren_training(trainingdata.examples, trainingdata.classes, 20, 20, 1, 1)
        print einfaches_klassifizieren(model, testdata.examples)

if __name__ == "__main__":
    main()
