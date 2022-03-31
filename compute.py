<<<<<<< HEAD
import numpy as np


###################################################################
'''
returns accuracy
< compute_acc >
input: predictions, target_labels
output: accuracy
'''
###################################################################


def compute_acc(predictions, target_labels):
=======
import numpy as np


###################################################################
'''
returns accuracy
< compute_acc >
input: predictions, target_labels
output: accuracy
'''
###################################################################


def compute_acc(predictions, target_labels):
>>>>>>> a276db7324688919f66b41a3801e2fc014889596
    return (np.array(predictions) == np.array(target_labels)).mean()