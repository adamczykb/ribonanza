from matplotlib import pyplot as pt
import numpy as np

from data_class import SequenceFile



def map_reduce_length(new_d: list[SequenceFile], new_a: list[SequenceFile]):
    sizes = dict({})
    for i in new_d:
        for j in i.sequences:
            if not len(j.sequence) in sizes:
                sizes[len(j.sequence)] = 1
            else:
                sizes[len(j.sequence)] += 1
    new_d_stats = np.asarray([[key, value] for key, value in sizes.items()])
    sizes = dict({})
    for i in new_a:
        for j in i.sequences:
            if not len(j.sequence) in sizes:
                sizes[len(j.sequence)] = 1
            else:
                sizes[len(j.sequence)] += 1
    new_a_stats = np.asarray([[key, value] for key, value in sizes.items()])
    return new_d_stats, new_a_stats


def get_longest_sequence_size(new_d: list[SequenceFile], new_a: list[SequenceFile]):
    new_d_stats, new_a_stats = map_reduce_length(new_d, new_a)

    return max(max(new_a_stats[:, 0]), max(new_d_stats[:, 0]))


def sequence_cardinality(new_d: list[SequenceFile], new_a: list[SequenceFile]):
    new_d_stats, new_a_stats = map_reduce_length(new_d, new_a)
    pt.bar(new_a_stats[:, 0], new_a_stats[:, 1])
    pt.bar(new_d_stats[:, 0], new_d_stats[:, 1])
