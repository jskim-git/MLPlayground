"""
Linear Tree Models
Combining the best of both worlds?
"""

import math

import numpy as np


class LinearModelTree:
    def __init__(self, min_node_size, node_model_function, min_split_gain=0):
        self.min_node_size = min_node_size
        self.node_model_function = node_model_function
        self.min_split_gain = min_split_gain

    def lm_prediction(self, X, y):
        lm = self.node_model_function(X, y)
        pred = lm.predict(X)
        if np.isnan(pred).any():
            print('Prediction value of NaN')
            print(pred)
            print(X)

        return pred, lm

    @staticmethod
    def convert_df_to_array(X):
        if type(X) == 'pandas.core.frame.DataFrame' or type(X) == 'pandas.core.series.Series':
            return X.values
        return X

    def build_tree(self, X, lm_X, y):
        X = self.convert_df_to_array(X)
        y = self.convert_df_to_array(y)
        self.root = Node.build_node_recursive(self, X, lm_X, y)

    def predict(self, X, lm_X):
        X = self.convert_df_to_array(X)
        data = []
        for i in range(X.shape[0]):
            data.append(self.predict_one(X[i, :], lm_X.iloc[[i]]))
        return np.array(data)

    def predict_one(self, X, lm_X):
        X = self.convert_df_to_array(X)
        return self.root.predict_one(X, lm_X)

    def predict_full(self, X, lm_X):
        X = self.convert_df_to_array(X)
        data = []
        for i in range(X.shape[0]):
            data.append(self.predict_full_one(X[i, :], lm_X.iloc[[i]]))

    def predict_full_one(self, X, lm_X):
        X = self.convert_df_to_array(X)
        return self.root.predict_full_one(X, lm_X)

    def node_count(self):
        return self.root.node_count()

    def serialize(self):
        return self.root.serialize()


class Node:
    def __init__(self, feature_index, pivot, lm):
        self.feature_index = feature_index
        self.pivot = pivot
        self.lm = lm
        self.row_count = 0
        self.left = None
        self.right = None

    def node_count(self):
        if self.feature_index is not None:
            return 1 + self.left.node_count() + self.right.node_count()
        else:
            return 1

    def predict_one(self, X, lm_X):
        local_value = self.lm.predict(lm_X)[0]
        if self.feature_index is not None:
            if X[self.feature_index] < self.pivot:
                child_value = self.left.predict_one(X, lm_X)
            else:
                child_value = self.right.predict_one(X, lm_X)

            return child_value + local_value
        else:
            return local_value

    def predict_full_one(self, X, lm_X, prefix='T'):
        local_value = self.lm.predict(lm_X)[0]
        if self.feature_index is not None:
            if X[self.feature_index] < self.pivot:
                result = self.left.predict_full_one(X, lm_X, prefix + 'L')
            else:
                result = self.right.predict_full_one(X, lm_X, prefix + 'R')
            result[1] += local_value
            return result
        else:
            return np.array([prefix, local_value + self.lm.predict(lm_X)[0]])

    def serialize(self, prefix='T'):
        if self.feature_index is not None:
            self_str = ',rc:%i,f:%i,v:%s'.format(
                self.row_count, self.feature_index, str(self.pivot))
            return "\n" + prefix + (self_str +
                                    self.left.serialize(prefix + 'L') +
                                    self.right.serialize(prefix + 'R'))
        else:
            self_str = ',rc:%i,f:_,v:_,int:%f,coef:%s'.format(
                self.row_count, self.lm.intercept_, str(self.lm.coef_))
            return "\n" + prefix + self_str

    @staticmethod
    def build_node_recursive(tree, X, lm_X, y):
        (feature_index, pivot, lm, res) = Node.find_best_split(tree, X, lm_X, y)
        node = Node(feature_index, pivot, lm)
        node.row_count = X.shape[0]

        if feature_index is not None:
            X_left, lm_X_left, left_res, X_right, lm_X_right, right_res = Node.split_on_pivot(
                X, lm_X, res, feature_index, pivot)
            node.left = Node.build_node_recursive(tree, X_left, lm_X_left, left_res)
            node.right = Node.build_node_recursive(tree, X_right, lm_X_right, right_res)

        return node

    @staticmethod
    def find_best_split(tree, X, lm_X, y):
        pred, lm = tree.lm_prediction(lm_X, y)
        res = y - pred
        row_counts = X.shape[0]
        sse = (res**2).sum()
        res_sum = res.sum()
        sse_best = sse
        feature_best = None
        pivot_best = None

        for i in range(X.shape[1]):
            ind = X[:, i].argsort()
            X_sort = X[ind]
            res_sort = res[ind]
            sum_left = 0
            sum_right = res_sum
            ss_left = 0
            ss_right = sse
            left_counts = 0
            right_counts = row_counts
            pivot_index = 0

            while right_counts >= tree.min_node_size:
                row_y = res_sort[pivot_index]
                sum_left += row_y
                sum_right -= row_y
                ss_left += row_y * row_y
                ss_right -= row_y * row_y
                left_counts += 1
                right_counts -= 1
                pivot_index += 1

                if left_counts >= tree.min_node_size and right_counts >= tree.min_node_size:
                    rmse_left = math.sqrt((left_counts * ss_left) - (sum_left * sum_left)) / left_counts
                    sse_left = rmse_left * rmse_left * left_counts
                    rmse_right = math.sqrt((right_counts * ss_right) - (sum_right * sum_right)) / right_counts
                    sse_right = rmse_right * rmse_right * right_counts
                    sse_split = sse_left + sse_right

                    if (sse_split < sse_best and sse - sse_split > tree.min_split_gain and
                            (left_counts <= 1 or X_sort[pivot_index, i] != X_sort[pivot_index - 1, i])):
                        sse_best = sse_split
                        feature_best = i
                        pivot_best = X_sort[pivot_index, i]
        return feature_best, pivot_best, lm, res

    @staticmethod
    def split_on_pivot(X, lm_X, y, feature_index, pivot):
        sorting_index = X[:, feature_index].argsort()
        X_sort = X[sorting_index]
        pivot_index = np.argmax(X_sort[:, feature_index] >= pivot)
        lm_X_sort = lm_X[sorting_index]
        y_sort = y[sorting_index]

        return (X_sort[:pivot_index, :],
                lm_X_sort.iloc[:pivot_index, :],
                y_sort[:pivot_index],
                X_sort[pivot_index:, :],
                lm_X_sort.iloc[pivot_index:, :],
                y_sort[pivot_index:])

