import torch
import time
from metrics import MetronAtK


class Engine(object):
    def __init__(self):
        self._metron = MetronAtK(top_k=10)
        
    def evaluate(self, model, test_loader, test_neg_loader, epoch_id, **kwargs):
        #Evaluate model
        a=time.time()
        model.eval()
 
        t_test_users=[]
        t_negative_users=[]
        t_test_items=[]
        t_negative_items=[]
        test_score=[]
        negative_score=[]
        dataloader_iterator = iter(test_loader)
        
        for i , data1 in enumerate(test_neg_loader): 

            try :
                data2 = next(dataloader_iterator)
                with torch.no_grad():    

                    test_positives_users, test_positives_pos,test_positives_neg,test_positives_image = data2
                    test_negatives_users, test_negatives_pos,test_negatives_neg,test_negatives_image = data1
                
                    test_scores = model(test_positives_users, test_positives_pos,test_positives_neg,test_positives_image)[1]
                    negative_scores = model(test_negatives_users, test_negatives_pos,test_negatives_neg,test_negatives_image)[1]
                

                    test_scores = test_scores.cpu()
                    negative_scores = negative_scores.cpu()

                    t_test_users.extend(test_positives_users.detach().numpy())
                    t_test_items.extend(test_positives_pos.detach().numpy())
                    t_negative_users.extend(test_negatives_users.detach().numpy())
                    t_negative_items.extend(test_negatives_pos.detach().numpy())
                    test_score.extend(test_scores.detach().numpy())
                    negative_score.extend(negative_scores.detach().numpy())

            except StopIteration: 
                with torch.no_grad():    
                    negative_users, negative_items = data1
                    
                    negative_scores = model(negative_users, negative_items)                    

                    
                    
                    
                    negative_scores = negative_scores.cpu()
                    t_negative_users.extend(negative_users.detach().numpy())
                    t_negative_items.extend(negative_items.detach().numpy())
                    negative_score.extend(negative_scores.detach().numpy())
    
        
        
        self._metron.subjects = [t_test_users,
                            t_test_items,
                            test_score,
                            t_negative_users,
                            t_negative_items,
                            negative_score]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        print('[Evluating Epoch {}] HR = {:.4f}, HR2 = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, ndcg))
        
        b=time.time()
        print("evaluate time:",b-a)  
        return hit_ratio, ndcg