import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import torch
from torch.utils.data import DataLoader,TensorDataset
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from arise_config import cfg

# blur_attack_dict = {'botnet':['mirai_kit_abnormal','aidra','mirai_router', 'bashlite'],
#                     'ddos':['tcp_ddos', 'udp_ddos', 'http_ddos', 'tcp_ddos_router'],
#                     'scan':['os_scan', 'service_scan', 'os_scan_router', 'port_scan_router'],
#                     'theft':['data_theft', 'keylogging']}

# blur_attack_dict = {'botnet':['mirai_kit_abnormal','aidra','mirai_router_filter', 'bashlite'],
#                     'ddos':['tcp_ddos', 'udp_ddos', 'http_ddos'],
#                     'scan':['os_scan', 'service_scan'],
#                     'theft':['data_theft', 'keylogging']}

blur_attack_dict = cfg('blur_attack_dict', {'botnet':['Mirai_Botnet'],
                    'ddos':['SSDP_Flood', 'SYN_DoS', 'SSL_Renegotiation'],
                    'scan':['OS_Scan', 'Fuzzing'],
                    'MitM':['Video_Injection', 'ARP_MitM', 'Active_Wiretap']})

# blur_attack_dict = {'botnet':['botnet'],
#                     'ddos':['ddos'],
#                     'scan':['scan'],
#                     'brute':['brute']}

# blur_attack_dict = {'DDoS': ['DDoS-ACK_Fragmentation', 'DDoS-HTTP_Flood-', 'DDoS-ICMP_Flood',
#                              'DDoS-ICMP_Fragmentation', 'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood',
#                              'DDoS-SYN_Flood', 'DDoS-SlowLoris', 'DDoS-SynonymousIP_Flood',
#                              'DDoS-TCP_Flood', 'DDoS-UDP_Flood', 'DDoS-UDP_Fragmentation'],
#                     'Mirai': ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain'],
#                     'Prob': ['Recon-HostDiscovery', 'Recon-PingSweep',
#                              'Recon-PortScan'],
#                     # 'Brute': ['']
#                     }

def file_name_walk(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".csv":
                file_list.append("{}/{}".format(root, file))
    return file_list


def dir_name_walk(file_dir):
    dir_list = []
    for root, dirs, files in os.walk(file_dir):
        for dir in dirs:
            dir_list.append("{}/{}".format(root, dir))
    return dir_list


def load_seq_data(dataset_path,type_list=[],label=0,begin=0,end=20):
    df_normal = pd.DataFrame()
    attack_list = [item for sublist in blur_attack_dict.values() for item in sublist]

    if len(type_list) == 0:
        type_list = os.listdir(dataset_path)
    for type_index, type_name in enumerate(type_list):
        if '.csv' in type_name:
            file_list = [dataset_path + '{:}'.format(type_name)]
        else:
            file_list = file_name_walk(
                dataset_path + '{:}'.format(type_name))
        # if label == 1 and type_name not in attack_list:
        #     continue
        file_list.sort()
        df_normal_type = pd.DataFrame()
        begin_num = begin
        end_num = end
        for i, file_path in enumerate(file_list[begin_num:end_num]):
            # print('Loading file: ', file_path)
            try:
                # tmp_df = pd.read_csv(file_path, header=None).iloc[1:,:]
                tmp_df = pd.read_csv(file_path, header=None)
                df_normal_type = df_normal_type.append(tmp_df, ignore_index=True)
            except:
                print(file_path)
        df_normal = df_normal.append(df_normal_type, ignore_index=True)
    df_normal['class'] = label

    df_normal.fillna(0,inplace=True)
    return df_normal


def reshape_channel(X):
    # Pad input_size to 105, reshape the input into 5 channels by the time-window.
    
    # Padding
    X = np.pad(X, ((0,0),(3,0)), 'constant')
    
    # Generate index
    index = np.empty((0,0))
    Port_index = np.arange(4,-1,-1).reshape(5,-1)
    MIstat_index = np.arange(5,20).reshape(5,-1)
    HHstat_index = np.arange(20,55).reshape(5,-1)
    HHstat_jit_index = np.arange(55,70).reshape(5,-1)
    HpHpstat_index = np.arange(70,105).reshape(5,-1)
    for i in range(5):
        index = np.append(index, Port_index[i])
        index = np.append(index, MIstat_index[i])
        index = np.append(index, HHstat_index[i])
        index = np.append(index, HHstat_jit_index[i])
        index = np.append(index, HpHpstat_index[i])
    
    # Reshape
    X = X[:, index.astype(np.int).tolist()].reshape(-1, 5, 21)
    return X


def preprocess_train_ul(df_normal_train, df_attack, params_dir):
     # data preparing / transfer data format
    scaler = MinMaxScaler()

    df_normal_train, df_normal_valid = train_test_split(df_normal_train, test_size=0.2, random_state=20)
    df_attack_test, df_attack_valid = train_test_split(df_attack, test_size=0.2, random_state=20)

    df_train = df_normal_train
    df_valid = pd.concat([df_normal_valid,df_attack_valid],axis=0)
    
    
    X_train, y_train, X_valid, y_valid = df_train.drop(columns=['class']).values, df_train['class'].values, df_valid.drop(columns=['class']).values, df_valid['class'].values
    

    X_train = scaler.fit_transform(X_train)

    # if not os.path.exists('./params_kit/'):
    #         os.makedirs('./params_kit/')

    scaler_path = params_dir+'scaler.pkl'
    pickle.dump(scaler, open(scaler_path, 'wb'))

    X_valid = scaler.transform(X_valid)

    # X_train = reshape_channel(X_train)
    # X_valid = reshape_channel(X_valid)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)

    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)

    return X_train, y_train, X_valid, y_valid


def preprocess_train_contra(df_normal_train, df_attack, df_anchor, params_dir):
    # data preparing / transfer data format
    scaler = MinMaxScaler()

    df_normal_train, df_normal_valid = train_test_split(df_normal_train, test_size=0.2, random_state=20)
    df_attack_test, df_attack_valid = train_test_split(df_attack, test_size=0.2, random_state=20)
    
    # df_anchor = df_attack_valid.sample(n=674, random_state=20)
    # df_attack_valid = df_attack_valid.drop(df_anchor.index)

    df_train = df_normal_train
    df_valid = pd.concat([df_normal_valid,df_attack_valid],axis=0)
    
    print(f'anchors shape: {df_anchor.shape}')
    
    
    # condition = df_anchor[df_anchor.columns[-4]].isin(blur_attack_dict['botnet'])
    # X_anchor = df_anchor[condition].drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values
    X_anchor = df_anchor.drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values
    
    print(f'anchors shape: {X_anchor.shape}')

    # X_train, y_train, X_valid, y_valid = df_train.drop(columns=[0,'class']).values, df_train['class'].values, df_valid.drop(columns=[0,'class']).values, df_valid['class'].values
    X_train, y_train, X_valid, y_valid = df_train.drop(columns=['class']).values, df_train['class'].values, df_valid.drop(columns=['class']).values, df_valid['class'].values
    # X_train, y_train, X_valid, y_valid, X_anchor = df_train.drop(columns=[0,'class']).values, df_train['class'].values, df_valid.drop(columns=[0,'class']).values, df_valid['class'].values, df_anchor.drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values
    # X_train, y_train, X_valid, y_valid, X_anchor = df_train.drop(columns=[0,'class']).values, df_train['class'].values, df_valid.drop(columns=[0,'class']).values, df_valid['class'].values, df_anchor.drop(columns=[0,'class']).values
    

    X_train = scaler.fit_transform(X_train)

    # if not os.path.exists('./params/'):
    #         os.makedirs('./params/')

    scaler_path = params_dir+'scaler.pkl'
    pickle.dump(scaler, open(scaler_path, 'wb'))

    X_valid = scaler.transform(X_valid)
    X_anchor = scaler.transform(X_anchor)

    # X_train = reshape_channel(X_train)
    # X_valid = reshape_channel(X_valid)
    # X_anchor = reshape_channel(X_anchor)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_anchor = torch.tensor(X_anchor, dtype=torch.float32)

    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)

    return X_train, y_train, X_valid, y_valid, X_anchor


def preprocess_train_embedding(df_attack, df_anchor):
    # data preparing / transfer data format
    scaler_path = './params_cic23/scaler_attack.pkl'
    scaler = pickle.load(open(scaler_path, 'rb'))
    # scaler = MinMaxScaler()
    # scaler.fit_transform(df_attack.drop(columns=[0,'class']).values)

    df_attack_train, df_attack_test = train_test_split(df_attack, test_size=0.3, random_state=20)
    df_attack_train, df_attack_valid = train_test_split(df_attack_train, test_size=0.2, random_state=20)
    
    X_train, y_train = df_attack_train.drop(columns=[0,'class']).values, df_attack_train['class'].values
    print('training dataset: ', Counter(y_train))
    # oversampler = SMOTE(sampling_strategy={3: int(Counter(y_train)[1]/4), 0: Counter(y_train)[1]})
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_train)[2])})
    # X_train, y_train = oversampler.fit_resample(X_train, y_train)
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_train)[3])})
    oversampler = SMOTE(sampling_strategy={3: int(Counter(y_train)[2])}) # CIC-IoT-2023
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    print('oversampled training dataset: ', Counter(y_train))
    
    
    X_valid, y_valid = df_attack_valid.drop(columns=[0,'class']).values, df_attack_valid['class'].values
    print('valid dataset: ', Counter(y_valid))
    # oversampler = SMOTE(sampling_strategy={3: int(Counter(y_valid)[1]/2), 0: Counter(y_valid)[1]})
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_valid)[2])})
    # X_valid, y_valid = oversampler.fit_resample(X_valid, y_valid)
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_valid)[3])})
    oversampler = SMOTE(sampling_strategy={3: int(Counter(y_valid)[2])}) # CIC-IoT-2023
    X_valid, y_valid = oversampler.fit_resample(X_valid, y_valid)
    print('oversampled valid dataset: ', Counter(y_valid))
    
    X_test, y_test = df_attack_test.drop(columns=[0,'class']).values, df_attack_test['class'].values
    print('test dataset: ', Counter(y_test))
    # oversampler = SMOTE(sampling_strategy={3: int(Counter(y_test)[1]/2), 0: Counter(y_test)[1]})
    # X_test, y_test = oversampler.fit_resample(X_test, y_test)
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_test)[2])})
    # X_test, y_test = oversampler.fit_resample(X_test, y_test)
    # oversampler = SMOTE(sampling_strategy={0: int(Counter(y_test)[3])})
    oversampler = SMOTE(sampling_strategy={3: int(Counter(y_test)[2])}) # CIC-IoT-2023
    X_test, y_test = oversampler.fit_resample(X_test, y_test)
    print('oversampled test dataset: ', Counter(y_test))
    
    # X_anchor, y_anchor = df_anchor.drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values, df_anchor['class'].values
    
    X_anchor_dict = {}
    X_anchor = torch.tensor([])
    y_anchor = torch.tensor([], dtype=torch.int8)

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # X_train = reshape_channel(X_train)
    # X_valid = reshape_channel(X_valid)
    # X_test = reshape_channel(X_test)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.int8)
    y_valid = torch.tensor(y_valid, dtype=torch.int8)
    y_test = torch.tensor(y_test, dtype=torch.int8)
    
    for index, label in enumerate(blur_attack_dict.keys()):
        X_anchor_tmp = df_anchor[df_anchor['class'] == label].drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values
        X_anchor_tmp = scaler.transform(X_anchor_tmp)
        X_anchor_tmp = torch.tensor(X_anchor_tmp, dtype=torch.float32)
        X_anchor = torch.cat((X_anchor, X_anchor_tmp))
        y_anchor = torch.cat((y_anchor, torch.tensor([index]*len(X_anchor_tmp), dtype=torch.int8)))

    print(y_anchor.shape)
    print('anchor: ', Counter(y_anchor.numpy()))
    oversampler = RandomOverSampler(random_state=20) # CIC-IoT-2023
    X_anchor, y_anchor = oversampler.fit_resample(X_anchor.numpy(), y_anchor.numpy())
    X_anchor, y_anchor = torch.tensor(X_anchor), torch.tensor(y_anchor)
    print('oversampled anchor: ', Counter(y_anchor.numpy()))
    
    for i in range(len(blur_attack_dict.keys())):
        X_anchor_dict[i] = X_anchor[y_anchor == i]
    
    # for index, label in enumerate(blur_attack_dict.keys()):
    #     X_anchor = df_anchor[df_anchor['class'] == label].drop(df_anchor.columns[[-4, -3, -2, -1]], axis=1).values
    #     X_anchor = scaler.transform(X_anchor)
    #     # X_anchor = reshape_channel(X_anchor)
    #     X_anchor = torch.tensor(X_anchor, dtype=torch.float32)
    #     X_anchor_dict[index] = X_anchor

    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_anchor_dict




def preprocess_test(df_normal_test, df_attack, params_dir):
    # data preparing / transfer data format

    df_attack_test, df_attack_valid = train_test_split(df_attack, test_size=0.2, random_state=20)
    df_test = pd.concat([df_normal_test,df_attack_test],axis=0)
    # X_test, y_test = df_test.drop(columns=[0,'class']).values, df_test['class'].values
    X_test, y_test = df_test.drop(columns=['class']).values, df_test['class'].values

    scaler_path = params_dir+'scaler.pkl'
    scaler = pickle.load(open(scaler_path, 'rb'))

    X_test = scaler.transform(X_test)
    # X_test = reshape_channel(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test)

    return X_test, y_test


def get_train_dataloaders_ul(normal_dir, attack_dir, batch_size, params_dir):
    print('*'*10+'Loading Training Datasets'+'*'*10)
    
    # df_normal = load_seq_data(dataset_path=normal_dir, label=0).drop(columns=[0])
    # df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    # df_normal_train = load_seq_data(dataset_path=normal_dir, label=0, begin=0, end=4).drop(columns=[0])
    # df_normal_train = load_seq_data(dataset_path=normal_dir, label=0, begin=0, end=6)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    # df_attack = load_seq_data(dataset_path=attack_dir, label=1).drop(columns=[0])
    df_attack = load_seq_data(dataset_path=attack_dir, label=1)
    # df_normal_train = load_seq_data(dataset_path=normal_dir+'train/', label=0)
    df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    
    X_train, y_train, X_valid, y_valid = preprocess_train_ul(df_normal_train, df_attack, params_dir)
    
    print(X_train.shape)
    print(X_valid.shape)
    
    train_datasets = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_datasets = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader, valid_loader


def get_test_dataloaders(normal_dir, attack_dir, batch_size, params_dir):
    print('*'*10+'Loading Testing Datasets'+'*'*10)

    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    df_normal_test = load_seq_data(dataset_path=normal_dir+'test/', label=0)
    # df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    # df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=2, end=3)
    df_attack = load_seq_data(dataset_path=attack_dir, label=1)
    
    X_test, y_test = preprocess_test(df_normal_test,df_attack, params_dir)
    
    print("test X shape: ", X_test.shape)
    
    test_datasets = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return test_loader


def get_test_dataloaders_per_blur_attack(normal_dir, attack_dir, batch_size, params_dir):
    print('*'*10+'Loading Testing Datasets Per-attack'+'*'*10)
    
    test_loader_dict = {}

    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=6, end=9)
    # df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    # df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=2, end=3)
    df_normal_test = load_seq_data(dataset_path=normal_dir + 'test/', label=0)
    # df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    # df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0)
    
    attack_list = os.listdir(attack_dir)
            
    for blur_attack in blur_attack_dict.keys():
        print(blur_attack)
        df_attack_types = []  # 存储每个攻击类型的数据
        blur_attack_list = blur_attack_dict[blur_attack]
        
        # 收集该blur attack下所有攻击类型的数据
        for attack_type in attack_list:
            if attack_type not in blur_attack_list:
                continue
            tmp = load_seq_data(dataset_path=attack_dir, type_list=[attack_type], label=1)
            print(f'{attack_type} shape: {tmp.shape}')
            df_attack_types.append(tmp)
        
        # 如果有多个攻击类型，进行类别平衡
        if len(df_attack_types) > 1:
            # 计算平均样本数量
            sample_counts = [len(df) for df in df_attack_types]
            median_samples = int(np.median(sample_counts))
            # total_samples = sum([len(df) for df in df_attack_types])
            # avg_samples = int(total_samples / len(df_attack_types))
            print(f'Median samples for {blur_attack}: {median_samples}')
            
            # 对每个攻击类型进行采样到平均样本数
            df_attack_balanced = []
            for i, df in enumerate(df_attack_types):
                if len(df) > median_samples:
                    # 如果样本数大于平均值，进行下采样
                    df_sampled = df.sample(n=median_samples, random_state=20)
                elif len(df) < median_samples:
                    # 如果样本数小于平均值，进行上采样（有放回抽样）
                    df_sampled = df.sample(n=median_samples, replace=True, random_state=20)
                else:
                    # 如果样本数等于平均值，直接使用
                    df_sampled = df
                df_attack_balanced.append(df_sampled)
                print(f'  Attack type {i+1}: {len(df)} -> {len(df_sampled)} samples')
            
            # 合并平衡后的数据
            df_attack = pd.concat(df_attack_balanced, ignore_index=True)
        elif len(df_attack_types) == 1:
            df_attack = df_attack_types[0]
        else:
            print(f"No attack data found for {blur_attack}")
            continue
            
        print(f'{blur_attack} balanced shape: {df_attack.shape}')
        
        # 根据攻击数据量确定正常数据量（5倍关系）
        attack_samples = len(df_attack)
        normal_samples_needed = attack_samples * 2
        # if blur_attack == 'brute':
        #     normal_samples_needed = attack_samples * 10
        print(f'Attack samples: {attack_samples}, Normal samples needed: {normal_samples_needed}')
        
        # 从正常数据中采样所需数量
        if len(df_normal_test) >= normal_samples_needed:
            df_normal_test = df_normal_test.sample(n=normal_samples_needed, random_state=20)
        else:
            # 如果正常数据不够，进行有放回抽样
            df_normal_test = df_normal_test.sample(n=normal_samples_needed, replace=True, random_state=20)
        print(f'Normal samples used: {len(df_normal_test)}')
        
        X_test, y_test = preprocess_test(df_normal_test, df_attack, params_dir)
        print("test X shape: ", X_test.shape)
        test_datasets = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader_dict[blur_attack] = test_loader
    
    return test_loader_dict


def get_train_dataloaders_contra(normal_dir, attack_dir, anchor_dir, batch_size, params_dir):
    print('*'*10+'Loading Training Datasets'+'*'*10)
    
    # df_normal_train = load_seq_data(dataset_path=normal_dir, label=0, begin=0, end=4)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    # df_normal_train = load_seq_data(dataset_path=normal_dir, label=0, begin=0, end=2)
    df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    df_attack = load_seq_data(dataset_path=attack_dir, label=1)
    df_anchor = load_seq_data(dataset_path=anchor_dir, label=1)
    # df_anchor = pd.DataFrame()
    
    X_train, y_train, X_valid, y_valid, X_anchor = preprocess_train_contra(df_normal_train, df_attack, df_anchor, params_dir)
    
    train_datasets = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    valid_datasets = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    return train_loader, valid_loader, X_anchor


def get_dataloaders_embedding(attack_dir, anchor_dir, batch_size, train=True):
    # if train:
    #     print('*'*10+'Loading Training Datasets'+'*'*10)
    # else:
    #     print('*'*10+'Loading Testing Datasets'+'*'*10)
    df_attack = pd.DataFrame()
    attack_list = os.listdir(attack_dir)
    for attack_type in attack_list:
        # if attack_type not in ['mirai_kit_cut','aidra','mirai_router_filter', 'bashlite',\
        #              'tcp_ddos', 'udp_ddos', 'http_ddos', 'tcp_ddos_router', 'udp_ddos_router',\
        #              'os_scan', 'service_scan', 'os_scan_router', 'port_scan_router',\
        #              'data_theft', 'keylogging']:
        #     continue
        # print(attack_type)
        df_attack_tmp = load_seq_data(dataset_path=attack_dir, type_list=[attack_type], label=attack_type)
        df_attack = df_attack.append(df_attack_tmp, ignore_index=True)
        
    for index, blur_attack in enumerate(blur_attack_dict.keys()):
        attack_list = blur_attack_dict[blur_attack]
        for attack_type in attack_list:
            df_attack['class'].replace(attack_type,index,inplace=True)
    
    df_anchor = pd.DataFrame()
    anchor_list = os.listdir(anchor_dir)
    for anchor_type in anchor_list:
        df_anchor_tmp = load_seq_data(dataset_path=anchor_dir, type_list=[anchor_type], label=anchor_type.replace('.csv',''))
        df_anchor = df_anchor.append(df_anchor_tmp, ignore_index=True)

    X_train, y_train, X_valid, y_valid, X_test, y_test, X_anchor_dict = preprocess_train_embedding(df_attack, df_anchor)
    
    if train:
        train_datasets = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_datasets = TensorDataset(X_valid, y_valid)
        valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        
        return train_loader, valid_loader, X_anchor_dict
    
    else:
        test_datasets = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        
        return test_loader, X_anchor_dict

def get_test_dataloaders_per_attack_zeroday(normal_dir, attack_dir, batch_size, params_dir):
    print('*'*10+'Loading Testing Datasets Per-attack'+'*'*10)
    
    test_loader_dict = {}

    df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    _, df_normal_test = train_test_split(df_normal_test, test_size=0.1)
    
    attack_list = os.listdir(attack_dir)
    for attack_type in attack_list:
        print(attack_type)
        df_attack = load_seq_data(dataset_path=attack_dir, type_list=[attack_type], label=1)
        X_test, y_test = preprocess_test(df_normal_test,df_attack,params_dir=params_dir)
        print("test X shape: ", X_test.shape)
        test_datasets = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader_dict[attack_type] = test_loader
    
    return test_loader_dict


def get_test_dataloaders_per_attack(normal_dir, attack_dir, batch_size, params_dir):
    print('*'*10+'Loading Testing Datasets Per-attack'+'*'*10)
    
    test_loader_dict = {}

    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=4, end=6)
    # df_normal = load_seq_data(dataset_path=normal_dir, label=0)
    # df_normal_train, df_normal_test = train_test_split(df_normal, test_size=0.2, random_state=20)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0, begin=2, end=3)
    # df_normal_test = load_seq_data(dataset_path=normal_dir, label=0)
    df_normal_test = load_seq_data(dataset_path=normal_dir+'test/', label=0)
    
    attack_list = os.listdir(attack_dir)
            
    for attack_type in attack_list:
        print(attack_type)
        
        # 加载单个攻击类型的数据
        df_attack = load_seq_data(dataset_path=attack_dir, type_list=[attack_type], label=1)
        print(f'{attack_type} shape: {df_attack.shape}')
        
        # 根据攻击数据量确定正常数据量（2倍关系）
        attack_samples = len(df_attack)
        normal_samples_needed = attack_samples * 2
        print(f'Attack samples: {attack_samples}, Normal samples needed: {normal_samples_needed}')
        
        # 从正常数据中采样所需数量
        if len(df_normal_test) >= normal_samples_needed:
            df_normal_sampled = df_normal_test.sample(n=normal_samples_needed, random_state=20)
        else:
            # 如果正常数据不够，进行有放回抽样
            df_normal_sampled = df_normal_test.sample(n=normal_samples_needed, replace=True, random_state=20)
        print(f'Normal samples used: {len(df_normal_sampled)}')
        
        X_test, y_test = preprocess_test(df_normal_sampled, df_attack, params_dir)
        print("test X shape: ", X_test.shape)
        test_datasets = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader_dict[attack_type] = test_loader
    
    return test_loader_dict