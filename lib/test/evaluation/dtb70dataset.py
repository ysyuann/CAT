import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import pandas
import json

class DTB70Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb70_dir
        self.sequence_list = self._get_sequence_list()
        # self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = None
        return Sequence(sequence_name, frames_list, 'dtb70', ground_truth_rect,
                        object_class=target_class, target_visible=None, language=None)


    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        ltr_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(ltr_path, 'dtb70.txt')

        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()

        return sequence_list
