import argparse
import os
import Make_Dataset
from model import ACF
from data_loader import *
from utils import optimizer

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")



def main():
    #wandb.init(project="Attentive Collaborative Filtering")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="pinterest",
                help='dataset')
    parser.add_argument('--optim',
                type=str,
                default='adam',
                help='optimizer')
    parser.add_argument('--epochs',
                type=int,
                default=400,
                help='epoch')
    parser.add_argument('--batch_size',
                type=int,
                default=256,
                help='batch size')
    parser.add_argument('--dim',
                type=int,
                default=128,
                help='dimension')    
    parser.add_argument('--lr',
                type=float,
                default=0.01,
                help='learning rate')    
    parser.add_argument('--reg',
                type=float,
                default=0.01,
                help='l2_regularization') 
    parser.add_argument('--hold_out',
                type=int,
                default=1,
                help='cv')
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')

    args = parser.parse_args()
    #wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Train data + Test data
    user,positive,negative,image_feature,test_positive_user,test_negative_user,test_positive_item,test_negative_item = Make_Dataset.preprocess()
    
    
    # Model
    acf = ACF(num_user=1610,num_item=133,embd_dim=5)
    acf = nn.DataParallel(acf)
    acf = acf.cuda()
    
    print(acf)
    optim = optimizer(optim=args.optim, lr=args.lr, model=acf)
    
    # Loss
    def my_loss(pos, neg):
        cus_loss = - torch.sum(torch.log(torch.sigmoid(pos - neg) + 1e-10))
        return cus_loss
    
    # Train
    for epoch in range(args.epochs):
        print('Epoch {} starts !'.format(epoch+1))
        print('-' * 80)
        
        acf.train()
        total_loss = 0

        sample = Train_Generator(user=user,positive_item=positive,negative_item=negative,item_feature=image_feature)
        train_loader = sample.instance_a_train_loader(args.batch_size)
        
        
        print("Train Loader 생성 완료")
        for batch_id, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                optim.zero_grad()
                users,positives,negatives,img_fe = batch[0], batch[1], batch[2], batch[3]
                users,positives,negatives,img_fe = users.cuda(), positives.cuda(), negatives.cuda(), img_fe.cuda()
                pos,neg = acf(users, positives, negatives,img_fe)
        
                loss = my_loss(pos,neg)
                loss.backward()
                optim.step()
                loss = loss.item()
                #wandb.log({'Batch Loss': loss})
                total_loss += loss

        acf.eval()
        if torch.no_grad():
            engine = Engine()
            test_sample = Test_Generator(user=test_positive_user,positive_item=test_positive_item,negative_item=test_negative_item,item_feature=image_feature)
            test_loader = test_sample.instance_a_test_loader(args.batch_size)
            
            test_neg_sample = Test_Generator(user=test_negative_user,positive_item=test_positive_item,negative_item=test_negative_item,item_feature=image_feature)
            test_neg_loader = test_neg_sample.instance_a_test_loader(args.batch_size)
            hit_ratio, ndcg = engine.evaluate(acf, test_loader, test_neg_loader, epoch_id=epoch)




if __name__ == '__main__':
    main()
        