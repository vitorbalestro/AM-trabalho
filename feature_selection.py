
import numpy as np
import math
import statistics

classes = {"Normal_Weight": 0, "Overweight_Level_I": 1, "Overweight_Level_II": 2, "Obesity_Type_I": 3, "Obesity_Type_II": 4, "Insufficient_Weight": 5, "Obesity_Type_III": 6}

def feature_selection(attributes,df):

    entropy_dict = {}

    for attr in attributes:

        values = df[attr].unique()
        entropy_mean = 0.0

        for value in values:
            distrib = [0 for i in range(7)]
            value_lines = df[df[attr] == value].values
            total = np.shape(value_lines)[0]
            for entry in value_lines:
                distrib[classes[entry[-1]]] += 1
        
            prob = [float(distrib[i]/total) for i in range(7)]
        
            entropy = 0.0
            for i in range(7):
                if prob[i] != 0:
                    entropy = entropy - prob[i]*math.log(prob[i])
            entropy_mean = entropy_mean + float(entropy / total)       

        entropy_dict[attr] = entropy_mean


    values_array = []
    for key in entropy_dict.keys():
        values_array.append(entropy_dict[key])

    """median_entropy = statistics.median(values_array)
    below_median_attributes = []

    for key in entropy_dict.keys():
        if entropy_dict[key] <= median_entropy:
            below_median_attributes.append(key)"""

    values_array.sort()

    upper_threshold = values_array[-1]

    below_threshold_attributes = []
    for key in entropy_dict.keys():
        if entropy_dict[key] < upper_threshold:
            below_threshold_attributes.append(key)

    return below_threshold_attributes