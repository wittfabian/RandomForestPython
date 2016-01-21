# RandomForestPython

## Requirements
python [Download](https://www.python.org/downloads/)

## How to run
decision-tree.py accepts parameters passed via the command line. The possible paramters are:
* Filename for training (Required, must be the first argument after 'python decision_tree_multiclass.py')

* Datatype flag (-d) followed by datatype filename (Optional, defaults to 'datatypes.csv')
* Print flag (-s) (Optional, causes the dataset)
* Validate flag (-v) followed by validate filename (Optional, specifies file to use for validation)
* Test flag (-t) followed by test filename (Optional, specifies file to use for testing)

#### Examples
#####Example 1
```
python decision_tree_multiclass.py data/iris_training.arff -t data/iris_test.arff
```
This command runs decision_tree.py with iris_training.csv as the training set and iris_test.csv as the test set. The classifier is not specified so it defaults to the last column in the training set. Printing is not enabled.
#####Example 2
```
python decision_tree_multiclass.py data/iris_training.arff -t data/iris_test.arff -s
```
This command runs decision_tree.py with iris_training.csv as the training set and iris_test.csv as the test set. The classifier is not specified so it defaults to the last column in the training set. Printing is enabled.
