# load pickle
python train_fcn_pm25.py -g 0  --lr 0.0000001 --max-epoch 500 --train_data_file /data/tag/pytorch/pytorch-fcn/transport_data/pickle/train_data.pickle  --train_label_file /data/tag/pytorch/pytorch-fcn/transport_data/pickle/train_label.pickle --valid_data_file /data/tag/pytorch/pytorch-fcn/transport_data/pickle/test_data.pickle --valid_label_file /data/tag/pytorch/pytorch-fcn/transport_data/pickle/test_label.pickle  --file_type pickle 

# load xlsx
# python train_fcn_pm25.py -g 0  --lr 0.0000001 --max-epoch 500 --train_data_file /data/tag/pytorch/pytorch-fcn/transport_data/data/xlsx/inference_data_all.xlsx  --train_label_file /data/tag/pytorch/pytorch-fcn/transport_data/data/xlsx/train_label.xlsx --valid_data_file /data/tag/pytorch/pytorch-fcn/transport_data/data/xlsx/inference_data_all.xlsx --valid_label_file /data/tag/pytorch/pytorch-fcn/transport_data/data/xlsx/test_label.xlsx  --file_type pickle 

