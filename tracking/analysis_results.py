import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'dtb70'
"""CAT"""
trackers.extend(trackerlist(name='CATod', parameter_name='cat_od_384', dataset_name='dtb70', run_ids=2, display_name='CATod'))


dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
