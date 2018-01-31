#! /usr/bin/python
# /usr/bin/python2.6 cluster_adjust.py XXOO XXYY

from __future__ import division
import numpy as np
from time import clock
from datetime import datetime
from itertools import groupby
import flask


# List[:-1] eliminates the last element, List[:-2] eliminates the last two elements. List[-1:] yields nothing, List[-2:] yields the last two elements.


def test_two_groups():
	vals = [1, 2, 3, 8, 5]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
	weights = [0.65, 0.35]
	adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
	answer = [-1.81999, -1.16999, -1.33666, 3.66333, 0.66333]
	for ans, res in zip(answer, adj_vals):
		assert abs(ans - res) < 1e-5

def test_three_groups():
	vals = [1, 2, 3, 8, 5]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
	grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
	weights = [.15, .35, .5]
	adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
	answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
	for ans, res in zip(answer, adj_vals):
		assert abs(ans - res) < 1e-5

def test_missing_vals():
	vals = [1, np.NaN, 3, 5, 8, 7]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
	weights = [.65, .35]
	adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
	answer = [-2.47, np.NaN, -1.170, -0.4533333, 2.54666666, 1.54666666]
	for ans, res in zip(answer, adj_vals):
		if ans is None:
			assert res is None
		elif np.isnan(ans):
			assert np.isnan(res)
		else:
			assert abs(ans - res) < 1e-5


def test_weights_len_equals_group_len():
	# Need to have 1 weight for each group
	vals = [1, np.NaN, 3, 5, 8, 7]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
	weights = [.65]
	with pytest.raises(ValueError):
		group_adjust(vals, [grps_1, grps_2], weights)

def test_group_len_equals_vals_len():
	# The groups need to be same shape as vals
	vals = [1, np.NaN, 3, 5, 8, 7]
	grps_1 = ['USA']
	grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
	weights = [.65]
	with pytest.raises(ValueError):
		group_adjust(vals, [grps_1, grps_2], weights)

def test_missing_weights():
	vals = [1, 2, 3, 8, 5]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
	grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
	weights = [.15, .35]
	adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
	answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
	for ans, res in zip(answer, adj_vals):
		assert abs(ans - res) < 1e-5

def test_missing_groups():
	vals = [1, 2, 3, 8, 5]
	grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
	grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
	grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON']
	weights = [.15, .35, .5]
	adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
	answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
	for ans, res in zip(answer, adj_vals):
		assert abs(ans - res) < 1e-5


def test_performance():
	vals = 100000 * [1, np.NaN, 3, 5, 8, 7]
	grps_1 = 100000 * [1, 1, 1, 1, 1, 1]
	grps_2 = 100000 * [1, 1, 1, 1, 2, 2]
	grps_3 = 100000 * [1, 2, 2, 3, 4, 5]
	weights = [.20, .30, .50]
	start = datetime.now()
	start_time = clock()
	group_adjust(vals, [grps_1, grps_2, grps_3], weights)
	end = datetime.now()
	end_time = clock()
	diff = end - start
	# print "Total performance test time: {}", format(diff.total_seconds())
	print "The time cost is : ", str(end_time - start_time)


	########## ##########  This is the divider bar  ##########  ##########
	########## ##########  This is the divider bar  ##########  ##########
	########## ##########  This is the divider bar  ##########  ##########


def group_adjust(vals, groups, weights):
	if len(weights) < len(groups):
		raise ValueError

	weighted_means = [0]*len(vals) # Create a list whose elements are the weighted averages.
	for order, group in enumerate(groups):
		if len(group) != max([len(l) for l in groups]):
			raise ValueError

		val_converted = np.array(vals)
		where_are_nan = np.isnan(val_converted)
		val_converted[where_are_nan] = 0 # Zero all the np.NaN elements.

		pair_sorted = sorted(zip(group, val_converted)) # Sort the pairs according to the country/state names.
		dictionary = dict()
		for key, item in groupby(pair_sorted, key=lambda x: x[0]): # Group the pairs into clusters according to the country/state names.
			listing = [value for key, value in item if value!=0] # Exclude all the np.NaN elements.
			dictionary[key] = sum(listing)/len(listing)

		list_mean = []
		for i in xrange(0, len(vals)):
			if where_are_nan[i] == True:
				list_mean.append(np.NaN) # Recover all the np.NaN elements.
			else:
				list_mean.append(dictionary[group[i]]) # Recover all the other elements.
		weighted_means = map(lambda (a,b):a+b, zip(weighted_means, [x*weights[order] for x in list_mean])) # Multiply the means with the weights.
	demeaned = map(lambda (a,b):a-b, zip(val_converted, weighted_means)) # Subtract the weighted average group means from each original value.
	# print "The demeaned result is:\n", demeaned, "\n"
	return demeaned


if "__main__" == __name__:
	print "\n=== Ladies and Gentlmen, Now the Run Begins! ==="
	vals = [10, 15, 1, 20, 99]
	grps_1 = ['USA', 'USA', 'Russia', 'USA', 'China']
	grps_2 = ['A', 'A', 'B', 'A', 'B']
	weights = [0.8, 0.2]
	sample = group_adjust(vals, [grps_1, grps_2], weights)
	print sample

	vals = [1, 2, 3]
	ctry_grp = ['USA', 'USA', 'USA']
	state_grp = ['MA', 'MA', 'CT']
	weights = [0.35, 0.65]
	sample = group_adjust(vals, [ctry_grp, state_grp], weights)
	print sample

	test_two_groups()
	test_three_groups()
	test_missing_vals()
	test_missing_weights()
	test_missing_groups()
	test_weights_len_equals_group_len()
	test_group_len_equals_vals_len()
	print "=== Ladies and Gentlmen, Now the Run Finishes! ===\n"
	test_performance()

# END
