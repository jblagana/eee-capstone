import numpy as np

def concealment_module(class_list):
    """
    Detects concealment levels and counts instances for each class.

    Args:
        class_list: list of classes from detection results

    Returns:
        A numpy array with four elements representing the counts for each concealment class:
            [high_concealment_count, low_concealment_count, med_concealment_count, no_concealment_count].
    """
    
    concealment_counts = [0, 0, 0, 0]                               # Initialize all class counts to 0
    for class_id in class_list:
        #class_id = int(class_id)
        concealment_counts[int(class_id)] += 1
    return np.array(concealment_counts)
