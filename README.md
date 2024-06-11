# BTextCAN

## 执行步骤

1 下载bert预训练语言模型，放到bert_pretrain目录下，通常包含以下文件。下载地址为：https://github.com/ymcui

  pytorch_model.bin 
  
  bert_config.json 
  
  vocab.txt

2 新建按如下格式新建目录

  Data:
    dataset
    logs
    save_dict

3 将数据集放在Data/dataset目录下，数据下载地址：https://github.com/laishanyan/Consumer-fraud-detection-dataset

4 python main.py

## Article
  title={BTextCAN: Consumer fraud detection via group perception},  
  author={Lai, Shanyan and Wu, Junfang and Ma, Zhiwei and Ye, Chunyang},  
  journal={Information Processing \& Management},  
  volume={60},  
  number={3},  
  pages={103307},  
  year={2023},  
  publisher={Elsevier}  
  
