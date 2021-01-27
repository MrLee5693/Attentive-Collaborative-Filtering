import torch
from torch.utils.data import DataLoader, Dataset 

class Train_Dataset(Dataset):
    def __init__(self, user_tensor, positive_tensor,negative_tensor,feature_tensor):
        self.user_tensor = user_tensor
        self.positive_tensor = positive_tensor
        self.negative_tensor = negative_tensor
        self.feature_tensor = feature_tensor
        
    def __getitem__(self, index): 
        return self.user_tensor[index], self.positive_tensor[index], self.negative_tensor[index], self.feature_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)
    
class Train_Generator(object):
    def __init__(self, user, positive_item, negative_item,item_feature):
        self.user = user
        self.positive_item = positive_item 
        self.negative_item = negative_item
        self.item_feature = item_feature
        
    def instance_a_train_loader(self, batch_size):
        dataset = Train_Dataset(user_tensor=torch.LongTensor(self.user),
                            positive_tensor=torch.LongTensor(self.positive_item),
                            negative_tensor=torch.LongTensor(self.negative_item),
                            feature_tensor=torch.FloatTensor(self.item_feature))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 1)
    
class Test_Dataset(Dataset):
    def __init__(self, user_tensor, positive_tensor,negative_tensor,feature_tensor):
        self.user_tensor = user_tensor
        self.positive_tensor = positive_tensor
        self.negative_tensor = negative_tensor
        self.feature_tensor = feature_tensor
        
    def __getitem__(self, index): 
        return self.user_tensor[index], self.positive_tensor[index], self.negative_tensor[index], self.feature_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)
    
class Test_Generator(object):
    def __init__(self, user, positive_item, negative_item,item_feature):
        self.user = user
        self.positive_item = positive_item 
        self.negative_item = negative_item
        self.item_feature = item_feature
        
    def instance_a_test_loader(self, batch_size):
        dataset = Train_Dataset(user_tensor=torch.LongTensor(self.user),
                            positive_tensor=torch.LongTensor(self.positive_item),
                            negative_tensor=torch.LongTensor(self.negative_item),
                            feature_tensor=torch.FloatTensor(self.item_feature))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 1)

    