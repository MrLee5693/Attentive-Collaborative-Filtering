import bson
import pandas as pd
import numpy as np
from collections import Counter
import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from feature_extract import *
import torch
import torch.autograd as A
import torchvision.models as M
import random
random.seed(42)
def preprocess():
    filtering = 80 # 15
    with open('./data/subset_iccv_board_pins.bson','rb') as f:
        user = bson.decode_all(f.read())

    with open('./data/subset_iccv_pin_im.bson','rb') as f:
        img = bson.decode_all(f.read())

    user = pd.DataFrame(user)
    img = pd.DataFrame(img)

    pin_img = dict(zip(img["pin_id"],img["im_name"]))
    img_pin = dict(zip(img["im_name"],img["pin_id"]))
    img_url = dict(zip(img["im_name"],img["im_url"]))

    counter = Counter(img["im_name"])
    img_dict = {x: counter[x] for x in counter}
    img_dict = dict(filter(lambda x : x[1]>=filtering, img_dict.items()))
    img_idx = {}
    idx_img = {}
    for idx,kv in enumerate(img_dict):
        img_idx[kv] = idx
        idx_img[idx] = kv
    user['train_positive'] = user['pins'].map(lambda x: [img_idx.get(pin_img[pin]) for pin in x if img_idx.get(pin_img[pin]) != None])
    user["num_img"] = user["train_positive"].map(len)
    user = user[user["num_img"] > 1]
    user["id"] = range(len(user))
    user["test_positive"] = user["train_positive"].apply(lambda x : random.sample(x,1))
    user.apply(lambda x : x["train_positive"].remove(x["test_positive"][0]), axis = 1)
    user["negative"] = user.apply(lambda x : list(set(range(len(img_dict))) - set(x["train_positive"]) - set(x["test_positive"])), axis = 1)
    user["num_train"] = user["train_positive"].map(len)
    user["negative_sample"] = user.apply(lambda x : random.sample(x["negative"],x["num_train"]),axis=1)
    user["pred_sample"] = user.apply(lambda x : random.sample(x["negative"],99),axis=1)
    
    user_set = np.array(np.repeat(user["id"], user["num_train"]))
    positive_item = np.array([item for items in user['train_positive'] for item in items])
    negative_item = np.array([item for items in user['negative_sample'] for item in items])
    
    
    print(user[["id","train_positive","test_positive","negative_sample"]])
    print("Num of User : {}".format(len(user)))
    print("Num of Item : {}".format(len(img_dict)))
    
    
    test_positive_user_set = np.array(np.repeat(user["id"], 1))
    test_positive_pos = np.array([item for items in user['test_positive'] for item in items])
    #test_positive_neg = np.array([item for items in user[''] for item in items])
    
    test_nagative_user_set = np.array(np.repeat(user["id"], 99))
    test_negative_pos = np.array([item for items in user['pred_sample'] for item in items])
    
    
    
    def feature_extraction(x):
        IMG_URL = img_url[idx_img[x]]
        resnet152 = M.resnet152(pretrained=True)
        for param in resnet152.parameters():
            param.requires_grad = False
        response = requests.get(IMG_URL)
        try:
            img_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            
            preprocess = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            img_pil = preprocess(img_pil)
            img_pil = img_pil.view(1,3,224,224)
                        
            resnet152_fe = ResNetFeatureExtractor(resnet152)
            h152 = resnet152_fe(img_pil)
            h152 = h152.view(-1,7,7)
        except:
            pass
  
        return h152
    img_feature = np.array([feature_extraction(item).numpy() for items in user['train_positive'] for item in items])
    print(img_feature.shape)    
        
    return user_set,positive_item,negative_item,img_feature,test_positive_user_set,test_nagative_user_set,test_positive_pos,test_negative_pos