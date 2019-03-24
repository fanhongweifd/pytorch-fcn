#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import torch
import xlrd
from torch.utils import data
from torch import nn
from collections import defaultdict
import pickle


def standardization(data, mean, var):
    """
    对数据进行标准化，用mean和var
    :param mean:
    """
    assert data[0].shape[0] == mean.shape[0], 'data dim = %s not equal to mean dimension = %s'%(data.shape[1], mean.shape[0])
    assert data[0].shape[0] == var.shape[0], 'data dim = %s not equal to var dimension = %s'%(data.shape[1], var.shape[0])
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    var = var ** 0.5
    var = var.unsqueeze(-1).unsqueeze(-1)
    for i in range(len(data)):
        data[i] = data[i] - mean
        data[i] = data[i] / (var + 1e-10)
    return data


def get_list_index(input):
    """
    :param input: 可迭代对象, 返回元素及其索引
    """
    result = defaultdict(list)
    for index, value in enumerate(input):
        result[value].append(index)
    return result


def transfer_feature(head_name, feature_values):
    season = {
        'spring': 0,
        'summer': 1,
        'autumn': 2,
        'winter': 3
    }
    
    holiday = {
        'Weekday': 0,
        'Weekend': 1
    }
    
    week = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    weather = {
        'Sunny/clear': 0,
        'Haze': 1,
        'Rain': 2
    }
    
    feature_values[head_name['日期']] = list(map(lambda x: xlrd.xldate_as_tuple(x, 0), feature_values[head_name['日期']]))
    feature_values[head_name['季节']] = list(map(lambda x: season[x], feature_values[head_name['季节']]))
    feature_values[head_name['星期']] = list(map(lambda x: week[x], feature_values[head_name['星期']]))
    feature_values[head_name['节假日']] = list(map(lambda x: holiday[x], feature_values[head_name['节假日']]))
    feature_values[head_name['天气']] = list(map(lambda x: weather[x], feature_values[head_name['天气']]))
    feature_values[head_name['网格号列']] = list(map(lambda x: int(x), feature_values[head_name['网格号列']]))
    feature_values[head_name['网格号行']] = list(map(lambda x: int(x), feature_values[head_name['网格号行']]))
    
    return feature_values


def parse_xlsx_data(file_path, sheet_index=0):
    """
    解析excel的数据
    :param file_path: excel path
    :param sheet_index: 表单索引，默认为0
    :return:
    """
    feature_book = xlrd.open_workbook(file_path)
    feature_sheet = feature_book.sheet_by_index(sheet_index)
    head_name = {value: idx for idx, value in enumerate(feature_sheet.row_values(0))}
    feature_values = [feature_sheet.col_values(id, 1) for id in range(feature_sheet.ncols)]
    feature_values = transfer_feature(head_name, feature_values)
    return head_name, feature_values


def read_test_xlsx(data_file, feature_dim=87):
    """
    输入测试的excel，输出n×55×55的feature
    :param data_file: 55*55 data file
    """
    data_head, data_values = parse_xlsx_data(data_file)
    data_time = list(zip(data_values[data_head['日期']], data_values[data_head['时刻']]))
    data_time_index = get_list_index(data_time)
    data_feature = np.array(data_values[5:])
    if feature_dim not in data_feature.shape:
        print('feature_dim is %s,but data_feature.shape is %s' % (feature_dim, data_feature.shape))
        raise TypeError
    if data_feature.shape[0] == feature_dim:
        data_feature = np.transpose(data_feature)
    data = []
    date_time = []
    mask = []
    for id, one_date_time in enumerate(data_time_index):
        if id % 100 == 0:
            print('load data %s/%s' % (id, len(data_time_index.keys())))
        feature_index = data_time_index[one_date_time]
        if len(feature_index) == 0:
            continue
        feature_mat = torch.zeros(feature_dim, 55, 55)
        mask_mat = torch.zeros(1, 55, 55)
        # if not len(feature_index) == 55*55:
        #     print('len(feature_index) = %s, not equal to 55*55 = 3025'%len(feature_index))
        #     raise TypeError
        
        feature_index_set = set()
        for index in feature_index:
            col_index = data_values[data_head['网格号列']][index] - 1
            row_index = data_values[data_head['网格号行']][index] - 1
            try:
                feature_mat[:, row_index, col_index] = torch.from_numpy(data_feature[index].astype(float))
                feature_index_set.add((row_index, col_index))
                mask_mat[0, row_index, col_index] = 1
            except Exception as e:
                print(e)
                print('col_index = %s, row_index = %s' % (col_index, row_index))
                print('data_feature = %s' % (data_feature[index]))
        
        data.append(feature_mat)
        date_time.append(one_date_time)
        mask.append(mask_mat)
    
    return data, date_time, mask


def read_transport_xlsx(data_file, label_file, feature_dim=87):
    """
    输入训练和label的excel，输出n×55×55的训练feature和55×55的label
    :param data_file: 55*55 data file
    :param label_file:  pm2.5 label file
    """
    data_head, data_values = parse_xlsx_data(data_file)
    data_time = list(zip(data_values[data_head['日期']], data_values[data_head['时刻']]))
    data_time_index = get_list_index(data_time)
    data_feature = np.array(data_values[5:])
    if feature_dim not in data_feature.shape:
        print('feature_dim is %s,but data_feature.shape is %s' % (feature_dim, data_feature.shape))
        raise TypeError
    if data_feature.shape[0] == feature_dim:
        data_feature = np.transpose(data_feature)
    
    label_head, label_values = parse_xlsx_data(label_file)
    label_time = list(zip(label_values[label_head['日期']], label_values[label_head['时刻']]))
    # label_feature = np.array(label_values[5:-1])
    # if feature_dim not in label_feature.shape:
    #     print('feature_dim is %s,but label_feature.shape is %s' % (feature_dim, label_feature.shape))
    #     raise TypeError
    # if label_feature.shape[0] == feature_dim:
    #     label_feature = np.transpose(label_feature)
    label_pm25 = label_values[label_head['Calibrated PM2.5']]
    label_time_index = get_list_index(label_time)
    
    data = []
    label = []
    for id, one_date_time in enumerate(label_time_index):
        if id % 100 == 0:
            print('load data %s/%s' % (id, len(label_time_index.keys())))
        label_index = label_time_index[one_date_time]
        feature_index = data_time_index[one_date_time]
        if len(feature_index) == 0:
            continue
        feature_mat = torch.zeros(feature_dim, 55, 55)
        # if not len(feature_index) == 55*55:
        #     print('len(feature_index) = %s, not equal to 55*55 = 3025'%len(feature_index))
        #     raise TypeError
        
        feature_index_set = set()
        for index in feature_index:
            col_index = data_values[data_head['网格号列']][index] - 1
            row_index = data_values[data_head['网格号行']][index] - 1
            try:
                feature_mat[:, row_index, col_index] = torch.from_numpy(data_feature[index].astype(float))
                feature_index_set.add((row_index, col_index))
            except Exception as e:
                print(e)
                print('col_index = %s, row_index = %s' % (col_index, row_index))
                print('data_feature = %s' % (data_feature[index]))
        
        label_mat = torch.zeros(1, 55, 55)
        for index in label_index:
            col_index = label_values[label_head['网格号列']][index] - 1
            row_index = label_values[label_head['网格号行']][index] - 1
            if (row_index, col_index) in feature_index_set:
                # 只有label没有feature的，不给对应的label赋值
                pm25 = label_pm25[index]
                label_mat[0, row_index, col_index] = pm25
        
        data.append(feature_mat)
        label.append(label_mat)
    
    return data, label


class TransportData(data.Dataset):
    def __init__(self, data_file, label_file, file_type='xlsx', feature_dim=87, standardization=True):
        if file_type == 'xlsx':
            self.data, self.label = read_transport_xlsx(data_file, label_file, feature_dim=feature_dim)
        elif file_type == 'pickle':
            with open(data_file, 'rb') as f:
                load_data = pickle.load(f)
                self.data = load_data['data']
            with open(label_file, 'rb') as f:
                load_data = pickle.load(f)
                self.label = load_data['label']
        else:
            raise ValueError('data file_type must be either xlsx or pickle')
        
        if standardization:
            normal = nn.BatchNorm2d(feature_dim, affine=False, momentum=1)
            data = self.data
            num = len(data)
            data_tensor = torch.zeros([len(data)] + list(data[0].size()))
            for i, one_data in enumerate(data):
                data_tensor[i, :] = data[i]
            data_standard = normal(data_tensor)
            self.data = [data_standard[i, :] for i in range(num)]
            self.data_mean = normal.running_mean
            self.data_var = normal.running_var
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label


if __name__ == '__main__':
    # data,label = read_transport_xlsx('../../transport_data/55X55训练数据/R1_R55_Feature_Data/inference_data_all.xlsx',\
    #                                  '../../transport_data/55X55训练数据/train_label.xlsx')
    # save_data = {'data':data}
    # save_label = {'label':label}
    # with open('train_data.pickle', 'wb') as f:
    #     pickle.dump(save_data, f)
    # with open('train_label.pickle', 'wb') as f:
    #     pickle.dump(save_label, f)
    
    # data, label = read_transport_xlsx('../../transport_data/55X55训练数据/R1_R55_Feature_Data/inference_data_all.xlsx',
    #                                   '../../transport_data/55X55训练数据/test_label.xlsx')
    # save_data = {'data': data}
    # save_label = {'label': label}
    # with open('test_data.pickle', 'wb') as f:
    #     pickle.dump(save_data, f)
    # with open('test_label.pickle', 'wb') as f:
    #     pickle.dump(save_label, f)
    
    # data, date_time, mask = read_test_xlsx('../../transport_data/55X55训练数据/test.xlsx')
    # print(date_time)
    
    # with open('train.pickle', 'rb') as f:
    #     load_data = pickle.load(f)
    #     print(load_data)

    transfer = TransportData('../../transport_data/data/pickle/train_data.pickle', '../../transport_data/data/pickle/train_label.pickle', \
                  file_type='pickle', standardization = False)
    ori_data = transfer.data
    
    with open('../../transport_data/data/pickle/train_data.pickle', 'rb') as f:
        load_data = pickle.load(f)['data']
        
    transfer = TransportData('../../transport_data/data/pickle/train_data.pickle', '../../transport_data/data/pickle/train_label.pickle', \
                  file_type='pickle', standardization = True)
    transfer_data = transfer.data
    
    stand_data = standardization(ori_data, transfer.data_mean, transfer.data_var)
