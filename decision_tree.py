# decision tree

import copy

class TreeNode:

    def __init__(self):
        self.cls_attr = None
        self.depth    = None

        self.cls      = None # only leaf node contains class label
        self.cluster  = None # only leaf node contains cluster

        self.pivot    = None # only internal node contains pivot
        self.attr     = None # only internal node contains attr for decision
        self.branches = None # only internal node contains branches to TreeNode


    def size(self):
        if self.cls is not None:
            return 1
        else:
            s = 0
            for b in self.branches.values():
                s += b.size()
            return s


    def majority(self):
        '''
        Interpret the class of self containted cluster, based on majority rule
        '''
        freq = {}

        for instance in self.cluster:
            cls = instance[self.cls_attr]
            if freq.has_key(cls):
                freq[cls] += 1
            else:
                freq[cls] = 1

        max_cls = None
        max_f   = .0
        for cls, f in freq.items():
            if f > max_f:
                max_cls = cls
                max_f   = f

        return max_cls


    def probe_deepest(self):
        if self.cls is not None:
            return self.depth
        else:
            deepest = 0
            for b in self.branches.values():
                deepest = max(deepest, b.probe_deepest())
            return deepest


    def merge_deepest(self, deepest):
        if self.depth + 1 == deepest:
            if self.cls is None:
                # do merge for "deepest" internal node
                self.cluster = []
                for b in self.branches.values():
                    self.cluster += b.cluster

                self.cls      = self.majority()
                self.pivot    = self.attr = None
                self.branches = None # gc branches
        else:
            if self.cls is None:
                for b in self.branches.values():
                    b.merge_deepest(deepest)


    def trim_last_lvl(self):
        deepest = self.probe_deepest()
        self.merge_deepest(deepest)


    def clone(self):
        c = TreeNode()
        c.cls_attr = self.cls_attr
        c.depth    = self.depth

        if self.cls is not None:
            c.cls = self.cls
            c.cluster = self.cluster[:]

            c.pivot = c.attr = c.branches = None
        else:
            c.pivot    = self.pivot
            c.attr     = self.attr
            c.branches = {}
            for attr, b in self.branches.items():
                c.branches[attr] = b.clone()

        return c
# end TreeNode


def build_tree(dataset, cls_attr, attr_strategy, measure=None, threshold=.0, quiet=True, _depth=0):
    '''
    Build a tree of decisions based on given the dataset to carry classification
    Each tree node is a function to partition the dataset
    Each leave node is a class
        - attr_strategy is a list of tuple: [(attr, strategy, sorting fn), ...]
    '''
    if not quiet:
        pad = ''
        for p in xrange(0, _depth):
            pad += '  '

    if threshold is None:
        threshold = .0

    if measure is None:
        import measure as m
        measure = m.entropy

    # if no more element for decision
    # return a leaf node for unclassified
    if len(dataset) == 0:
        leaf         = TreeNode()
        leaf.depth   = _depth
        leaf.cluster = []
        leaf.cls     = 'Un-classified'

        if not quiet:
            print '%sleaf - %s' % (pad, leaf.cls)
        return leaf

    # if no more attribute for decision
    # return a leaf node by majority
    if len(attr_strategy) == 0:
        leaf          = TreeNode()
        leaf.depth    = _depth
        leaf.cluster  = dataset
        leaf.cls_attr = cls_attr
        leaf.cls      = leaf.majority()

        if not quiet:
            print '%sleaf - %s by majority [no attr left]' % (pad, leaf.cls)
        return leaf

    # compute impurity for further processing
    impurity = measure(dataset, cls_attr)

    # if impurity of dataset is 0 ==> all instances belong to same class
    # return a leaf node as of that class
    if impurity == 0:
        leaf          = TreeNode()
        leaf.depth    = _depth
        leaf.cluster  = dataset
        leaf.cls_attr = cls_attr
        leaf.cls      = dataset[0][cls_attr]

        if not quiet:
            print '%sleaf - %s' % (pad, leaf.cls)
        return leaf

    # if impurity of dataset is below threshold
    # return a leaf node by majority
    elif impurity < threshold:
        leaf          = TreeNode()
        leaf.depth    = _depth
        leaf.cluster  = dataset
        leaf.cls_attr = cls_attr
        leaf.cls      = leaf.majority()

        if not quiet:
            print '%sleaf - %s by majority [threshold reach]' % (pad, leaf.cls)
        return leaf

    # pick a partition strategy by the best purity gain
    else:
        # get all gains
        attr_gain_map = {}
        for attr, strategy, _cmp in attr_strategy:
            attr_gain_map[attr] = strategy(
                dataset, attr, cls_attr, measure, impurity, _cmp)

        # retrieve best gain
        best_gain = .0
        best_attr = None
        pivot     = None
        clusters  = None
        for attr, result in attr_gain_map.items():
            p, g, c = result
            if g > best_gain:
                best_attr = attr
                pivot     = p
                best_gain = g
                clusters  = c

        if best_attr is None:
            # early return for gaining not much purity
            leaf          = TreeNode()
            leaf.depth    = _depth
            leaf.cluster  = dataset
            leaf.cls_attr = cls_attr
            leaf.cls      = leaf.majority()

            if not quiet:
                print '%sleaf - %s by majority [no further gain]' % (pad, leaf.cls)
            return leaf

        if not quiet:
            if isinstance(pivot, list):
                print '%simpurity: %s, gain: %s, attr: %s, decision: by %s' \
                    % (pad, impurity, best_gain, best_attr, pivot)
            else:
                print '%simpurity: %s, gain: %s, attr: %s, decision: val < %s' \
                    % (pad, impurity, best_gain, best_attr, pivot)


        # remove best attribute from further levels of decision
        attr_strategy = [(a, s, c) for a, s, c in attr_strategy if a != best_attr]

        tree = TreeNode()
        tree.cls_attr = cls_attr
        tree.depth    = _depth
        tree.pivot    = pivot
        tree.attr     = best_attr
        tree.branches = {}

        for val, c in clusters.items():
            tree.branches[val] = build_tree(
                c, cls_attr, attr_strategy, measure, threshold, quiet, _depth+1)

        return tree


def make_decision(tree, instance):
    '''
    Make classification based on the TreeNode structure
    '''
    node = tree
    while node.pivot is not None:
        pivot = node.pivot
        attr  = node.attr
        val   = instance[attr]

        if isinstance(pivot, list):
            # multi-way split
            if node.branches.has_key(val):
                node = node.branches[val]
            else:
                return None
        else:
            # binary split
            if val < pivot:
                if node.branches.has_key(1):
                    node = node.branches[1]
                else:
                    return None
            else:
                if node.branches.has_key(0):
                    node = node.branches[0]
                else:
                    return None
    return node.cls


def pruning_tree(tree, dataset, cls_attr, penalty=.5, quiet=True):
    '''
    Post-pruning a decision tree by err-estimate on data set
        pessimistic err-estimate =
            err-instances + leaf-count * size-penalty / data-set-size

        Continues pruning tree when trimmed-lvl-err-estimate < last-lvl-err-estimate
    '''
    tree = tree.clone()
    size = len(dataset)

    if not quiet:
        print 'post-pruning'

    o_estimate = None
    otree = None
    while True:
        err_count  = 0
        for instance in dataset:
            c = make_decision(tree, instance)
            if instance[cls_attr] != c:
                err_count += 1

        estimate = float(err_count + tree.size() * penalty) / size

        if not quiet:
            print 'leaf: %s, dataset: %s, err: %s [%s%%]' % \
                (tree.size(), size, err_count, 100.0 * err_count / size)
            print 'old err-estimate: %s, new err-estimate: %s' % \
                (o_estimate, estimate)
            print

        if o_estimate is not None and estimate >= o_estimate:
            # not (trimmed-lvl-err-estimate < last-lvl-err-estimate)
            break

        otree = tree.clone()
        o_estimate = estimate

        tree.trim_last_lvl()

    return otree


def __test__():
    import measure, strategy, datetime

    f = open('poker-hand-training.data')
    data = []
    for l in f.readlines():
        data.append(l.strip().split(','))
    f.close()


    tree = build_tree(data, 10, [
        (0, strategy.nominal,  None),
        (1, strategy.interval, None),
        (2, strategy.nominal,  None),
        (3, strategy.interval, None),
        (4, strategy.nominal,  None),
        (5, strategy.interval, None),
        (6, strategy.nominal,  None),
        (7, strategy.interval, None),
        (8, strategy.nominal,  None),
        (9, strategy.interval, None),
    ], quiet=True)
    print 'Tree size: %d' % tree.size()

    _tree = pruning_tree(tree, data, 10, quiet=False)
    print 'Tree size: %d' % _tree.size()

    _tree = pruning_tree(tree, data, 10, 1.0, quiet=False)
    print 'Tree size: %d' % _tree.size()


if __name__ == '__main__':
    __test__()
