
import argparse
import datetime
import os
import os.path as osp
import torch.optim.lr_scheduler as lr_scheduler
import torch
import yaml
import sys
sys.path.insert(0, '../../')
import torchfcn
here = osp.dirname(osp.abspath(__file__))

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume_model', type=str, required=True, help='checkpoint path')
    parser.add_argument('--test_data_file', type=str, help='test_data_file_excel')
    parser.add_argument('--standardization', default=True, type=bool, help='standardization')
    parser.add_argument('--inner_layer', type=str, required=True, help=' inner layer you want')
    parser.add_argument('--output', type=str, default='result.csv', help='standardization')
    args = parser.parse_args()
    model = torch.load(args.resume_model)['model']
    model_inner = getattr(model, args.inner_layer)
    model_inner.register_forward_hook(get_activation(args.inner_layer))
    data, date_time, mask = torchfcn.datasets.read_test_xlsx(args.test_data_file)
    
    if args.standardization:
        mean_and_var = torch.load(args.resume_model)['mean_and_var']
        data = torchfcn.datasets.standardization(data, mean=mean_and_var['mean'], var=mean_and_var['var'])
        
    output_file = open(args.output, 'w')
    output_file.write('日期,时刻,网格号行,网格号列,输出特征\n')
        
    with torch.no_grad():
        for i, data_feature in enumerate(data):
            data_feature = data_feature.unsqueeze(0)
            output = model(data_feature).squeeze(0)
            inner_layer_output = activation[args.inner_layer].squeeze(0)
            data_mask = mask[i].squeeze()
            rows, cols = data_mask.shape
            for row in range(rows):
                for col in range(cols):
                    if data_mask[row, col]>0:
                        date = '/'.join([str(x) for x in date_time[i][0][0:3]])
                        time = str(int(date_time[i][1]))
                        output_feature = ','.join([str(x) for x in inner_layer_output[:, row, col].tolist()])
                        output_list = [date, time, str(row+1), str(col+1), output_feature]
                        output_str = ','.join(output_list)
                        output_file.write(output_str + '\n')
    output_file.close()
            
    
if __name__ == '__main__':
    main()
