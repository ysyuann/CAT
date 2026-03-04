import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import pandas


class UAV112Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav112_dir
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
        anno_path = '{}/anno/{}.txt'.format(self.base_path, sequence_name)
        f = open(anno_path)  # 返回一个文件对象
        file = f.readlines()
        bbox = []
        for ii in range(len(file)):
            line = file[ii].strip('\n').split(',')
            if ('nan' not in line) and ('NaN NaN NaN NaN' not in line) and ('NaN' not in line) and ('NaNs' not in line):
                # print(ii, sequence_name)
                line[0] = np.float64(line[0])
                line[1] = np.float64(line[1])
                line[2] = np.float64(line[2])
                line[3] = np.float64(line[3])
            else:
                line=[0, 0, 0, 0]
                line[0] = np.float64(0.000)
                line[1] = np.float64(0.000)
                line[2] = np.float64(0.000)
                line[3] = np.float64(0.000)
            bbox.append(line)
        ground_truth_rect = np.array(bbox)

        frames_path = '{}/data_seq/{}'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = None
        return Sequence(sequence_name, frames_list, 'uav112', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=None)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        ltr_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(ltr_path, 'uav112.txt')

        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()

        return sequence_list
