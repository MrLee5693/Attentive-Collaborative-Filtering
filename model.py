import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACF(nn.Module):
    def __init__(self,num_user,num_item,embd_dim):
        super(ACF,self).__init__()
        
        self.dim = embd_dim
        self.user_embedding = nn.Embedding(num_user,embd_dim)
        self.item_i_embedding = nn.Embedding(num_item,embd_dim)
        self.item_j_embedding = nn.Embedding(num_item,embd_dim)
                
        # Attention Feature
        self.conv1 = nn.Conv2d(in_channels=2048,out_channels=embd_dim,kernel_size=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels=embd_dim,out_channels=embd_dim,kernel_size=1,stride=1)
        self.lin1 = nn.Linear(embd_dim,embd_dim)
        self.lin2 = nn.Linear(embd_dim,embd_dim)
        self.lin3 = nn.Linear(embd_dim,embd_dim)
    
    def forward(self,user_indices,item_i_indices,item_j_indices,feature):    
        user = self.user_embedding(user_indices)
        
        item_i = self.item_i_embedding(item_i_indices)
        item_j = self.item_j_embedding(item_j_indices)
        
        # Feature Level
        conv_feature1 = self.conv1(feature)
        #print(conv_feature1.shape)
        user_feature = self.lin1(user)
        #print(user_feature.shape)
        user_feature = user_feature.view(-1,self.dim,1,1)
        conv_feature = torch.add(conv_feature1,user_feature) # UW + Xlm W
        #print(conv_feature.shape)
        conv_feature = F.relu(conv_feature)
        conv_feature2 = self.conv2(conv_feature) # [batch, dim, 7, 7]
        
        conv_feature1 = conv_feature1.view(-1,self.dim,49)
        conv_feature2 = conv_feature2.view(-1,self.dim,49)
        atten_feature = F.softmax(conv_feature2,dim=-1)
        
        #print(atten_feature.shape)
        feature = torch.mul(atten_feature,conv_feature1)
        #print(feature.shape)
        feature = torch.sum(feature,dim=-1) # [batch, dim]
        #print(feature.shape)

        # Item Level
        user_item = self.lin2(user)
        positive_item = self.lin3(item_i)
        atten_item = torch.add(user_item,positive_item)
        atten_item = F.softmax(atten_item)
        #print(atten_item.shape)
        user_update = torch.sum(torch.mul(atten_item,feature)) # [batch, dim]
        
        # User update
        user = torch.add(user,user_update)       
        # Score
        score_i = torch.mul(user,item_i)
        score_j = torch.mul(user,item_j)
        return score_i, score_j
        
"""





class Model(object):
    def __init__(self,trainset,testset,num_user,num_item,dim,reg,lr,prefix):
        self.trainset = trainset
        self.testset = testset
        self.reg = np.float32(reg)
        self.lr = np.float32(lr)
        self.num_item = num_item
        #self.video_features = theano.shared(value=self.trainset.video_features, name='video_features', borrow=True)

        #T.config.compute_test_value = 'warn'

        u = T.ivector('u') #[num_sample,]
        iv = T.ivector('iv') #[num_sample,]
        jv = T.ivector('jv') #[num_sample,]
        mask_frame = T.itensor3('mask_frame')  #[num_sample, num_video, num_frame]
        mask = T.imatrix('mask') #[num_sample, num_video]
        feat_idx = T.imatrix('feat_idx')  #[num_sample, num_video]


        u.tag.test_value = np.asarray([0,1,2],dtype='int32')
        iv.tag.test_value = np.asarray([4,5,2],dtype='int32')
        jv.tag.test_value = np.asarray([1,3,0],dtype='int32')
        mask.tag.test_value = np.asarray([[1,1,0],[1,0,0],[1,1,1]],dtype='int32')
        feat_idx.tag.test_value = np.asarray([[3,4,-1],[5,-1,-1],[6,2,4]],dtype='int32')
        mask_frame.tag.test_value = self.trainset.frame_mask.take(feat_idx.tag.test_value,axis=0)

        rng = np.random
        layers = []

        Uemb = UsrEmblayer(rng,num_user,dim,'usremblayer',prefix)
        Vemb = VidEmblayer(rng,num_item,dim,'videmblayer',prefix)
        feat = self.video_features.take(feat_idx,axis=0) #[num_sample, num_video,dim_feat]
        layers.append(Uemb)
        layers.append(Vemb)
        uemb_vec = GetuEmbLayer(u,Uemb.output,'uemb',prefix)
        iemb_vec = GetvEmbLayer(iv,Vemb.output,'v1emb',prefix)
        jemb_vec = GetvEmbLayer(jv,Vemb.output,'v2emb',prefix)

        layers.append(AttentionLayer_Feat(rng, 2048, uemb_vec.output, feat, dim, dim, mask_frame, 'attentionlayer_feat',prefix))

        layers.append(AttentionLayer_Item(rng, uemb_vec.output, layers[-1].output,dim,dim,mask,'attentionlayer_item',prefix))

        u_vec = uemb_vec.output + layers[-1].output # layers[-1].output은 Attention weight 적용된 것
        self.layers = layers
        y_ui = T.dot(u_vec, iemb_vec.output.T).diagonal()
        y_uj = T.dot(u_vec, jemb_vec.output.T).diagonal()
        self.params = []
        loss = - T.sum(T.log(T.nnet.sigmoid(y_ui - y_uj)))
        for layer in layers:.
            self.params += layer.params #[U,V,W_Tran,Wu,Wv,b,c]


        regularizer = self.reg * ((uemb_vec.output ** 2).sum() + (iemb_vec.output ** 2).sum() + (jemb_vec.output ** 2).sum() )

        for param in self.params[2:]:
            regularizer += self.reg * (param ** 2).sum()

        loss = regularizer + loss

        updates = [(param, param-self.lr*T.grad(loss,param)) for param in self.params]

        self.train_model = theano.function(
            inputs = [u,iv,jv,mask_frame,mask,feat_idx],
            outputs = loss,
            updates=updates
        )

        self.test_model = theano.function(
            inputs = [u,mask_frame,mask,feat_idx],
            outputs= [u_vec,Vemb.output],
        )
        """