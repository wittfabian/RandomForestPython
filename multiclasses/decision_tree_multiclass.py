import ast
import sys
import arff
import math
import numpy as np


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
    for i in class_counts.classnames:
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

def compute_tree(dataset, parent_node):
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
    max_gain = 0  # The gain given by the best attribute
    split_val = None
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset, class_counts)

    for attr_index in range(len(dataset.examples[0])):

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

    # attr_to_split is now the best attribute according to our gain metric
    if split_val is None or attr_to_split is None:
        print "Something went wrong. Couldn't find an attribute to split on or a split value."
    elif max_gain <= min_gain or node.height > 20:

        node.is_leaf = True
        node.classification = classify_leaf(dataset)

        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
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
    node.upper_child = compute_tree(upper_dataset, node)
    node.lower_child = compute_tree(lower_dataset, node)

    return node


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

        # build tree
        root = compute_tree(trainingdata, None)

        # test tree
        # testresults = test_tree(testdata, root)

        # print tree
        # print_tree(root)


if __name__ == "__main__":
    main()
