import albumentations

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils import data
import random
## Simple augumentation to improtve the data generalibility




class Img_DataLoader(data.Dataset):
    def __init__(self, img_list='', in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_external=False, df_features = None):
        super(Img_DataLoader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external
        self.df_features = df_features
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        # prepare image
        orig_img = cv2.imread(img_path)
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[:,:, 0] = gray
        image[:,:, 1] = gray
        image[:,:, 2] = gray
        '''
        ###################################

        ###################################
        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                print(image)

        label = img_path.split('/')[-2]
        # print(img.shape)
        #if self.if_external:
        if img.shape[0]!=64:
            img = img[16:80,16:80,:]
        #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # img = img.reshape(3,96,96)

        img = np.einsum('ijk->kij', img)

        high_level_name = label
        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            if img_path.split('/')[2] == 'BM_cytomorphology_data':
                label = img_path.split('/')[-3]
                high_level_name = mapping_dic[label]

            mask = self.df[self.df['Cell_Types'] == high_level_name].iloc[:, 2:].to_numpy()


            sample["label"] = torch.from_numpy(mask).reshape(self.df.shape[0],).float()  # one hot encoder #1,


        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        return sample

class Img_DataLoader_pair(data.Dataset):
    def __init__(self, img_list='', in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_external=False, df_features = None):
        super(Img_DataLoader_pair, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external
        self.df_features = df_features
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        # prepare image
        orig_img = cv2.imread(img_path)
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[:,:, 0] = gray
        image[:,:, 1] = gray
        image[:,:, 2] = gray
        '''
        ###################################
        
        ###################################
        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                print(image)

        label = img_path.split('/')[-2]
        match = np.array((1,0))
        
        # permutate
        permute = np.random.choice([0, 1])
        lists = self.df['Cell_Types'].tolist()
        if permute == 1:
            lists.remove(label)

            label = random.choice(lists)
            match = np.array((0,1))
        else:
            pass
        # print(img.shape)
        #if self.if_external:
        if img.shape[0]!=64:
            img = img[16:80,16:80,:]
        #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # img = img.reshape(3,96,96)
        high_level_name = label
        img = np.einsum('ijk->kij', img)

        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            if img_path.split('/')[2] == 'BM_cytomorphology_data':
                label = img_path.split('/')[-3]
                high_level_name = mapping_dic[label]

            mask = self.df[self.df['Cell_Types'] == high_level_name].iloc[:, 2:].to_numpy()
            mask = mask.reshape(23,)
            sample["label"] = torch.from_numpy(mask).float()  # one hot encoder


        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        sample["agree"] = torch.from_numpy(np.array(match)).long()
        return sample


class Img_DataLoader_HF(data.Dataset):
    def __init__(self, img_list='', in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_external=False, df_features = None):
        super(Img_DataLoader_HF, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external
        self.df_features = df_features
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        # prepare image
        orig_img = cv2.imread(img_path)
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[:,:, 0] = gray
        image[:,:, 1] = gray
        image[:,:, 2] = gray
        '''
        ###################################

        ###################################
        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                print(image)

        label = img_path.split('/')[-3]


        # permutate

        abnormal = img_path.split('/')[-2]
        lists = self.df['Cell_Types'].tolist()
        if abnormal == 'abnormal':

            match = np.array((0,1))
        else:
            assert abnormal =='normal', 'something is off'
            match = np.array((1,0))
            pass
        # print(img.shape)
        #if self.if_external:
        if img.shape[0]!=64:
            img = img[16:80,16:80,:]
        #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # img = img.reshape(3,96,96)
        high_level_name = label
        img = np.einsum('ijk->kij', img)

        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            if img_path.split('/')[2] == 'BM_cytomorphology_data':
                label = img_path.split('/')[-3]
                high_level_name = mapping_dic[label]

            mask = self.df[self.df['Cell_Types'] == high_level_name].iloc[:, 2:].to_numpy()
            mask = mask.reshape(23,)
            sample["label"] = torch.from_numpy(mask).float()  # one hot encoder


        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        sample["agree"] = torch.from_numpy(np.array(match)).float()
        return sample