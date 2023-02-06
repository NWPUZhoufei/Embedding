import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.encoder = Encoder()
        self.metanet = MetaNetwork()
    def forward(self, support_set, query_set):
        task_context = Task_concate(support_set) #（way, way-1,1024）
        context1 = self.encoder(task_context) #(way,640)
        predict = self.metanet(support_set, query_set, context1)
        return predict

class MetaNetwork(nn.Module):
    def __init__(self):
        super(MetaNetwork, self).__init__()
        self.distanceNetwork = DistanceNetwork(metric = 'euclidean')

    def forward(self, support_set, query_set, context1):
        way, fc = support_set.size()
        support_set = support_set.unsqueeze(1) # (way,1,640)
        _, query, _ = query_set.size()
        query_set = query_set.view(way*query, fc)
        context1 = context1.unsqueeze(1) # (way,1,640)

        predict = []
        for i in range(way):
            current_support_set = support_set[i] # (1,640)
            current_context1 = context1[i] #(1,640)          
            current_support_feature = current_support_set * current_context1
            current_query_feature = query_set * current_context1
    
            current_support_feature = current_support_feature * current_context1
            current_query_feature = current_query_feature * current_context1
            
            # classifier
            current_predict = self.distanceNetwork(current_support_feature, current_query_feature) #(way*query)
            predict.append(current_predict)
        predict = torch.stack(predict, 0) #(way,way*query)
        predict = predict.t() #(way*query,way)

        return predict # (way*query,way)



def Task_concate(task_support_set):
    [n_way,c] =  task_support_set.size() 
    task_data_pair = [] 
    for i in range(0,n_way):
        current_class_data_pair = []
        for j in range(0,n_way):
            if j!=i :
                pair_tempt = torch.cat((task_support_set[i], task_support_set[j]),0) #(fc*2)
                current_class_data_pair.append(pair_tempt)  
        current_class_data_pair = torch.stack(current_class_data_pair, 0) #(way-1,fc*2)
        task_data_pair.append(current_class_data_pair)  
    task_data_pair = torch.stack(task_data_pair, 0) #(way,way-1,fc*2)
    return task_data_pair

 


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 640)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.9)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.7)

    def forward(self, x):
        
        way, shot, _ = x.size()  # (way,way-1,1280)
        x = x.view(way*shot, -1) # (way*way-1,1280)
        x = self.fc1(x)
        x = self.leakyRelu1(x)  
        x = self.dropout1(x)
        

        x = self.fc2(x)
        x = self.leakyRelu2(x)  
        x = self.dropout2(x)
        y2 = x.view(way, shot, -1)
        y2 = torch.mean(y2,1) # （5，640）

    
        return y2

 

 

# support (1,640) query (75,640)
class DistanceNetwork(nn.Module):

    def __init__(self, metric = 'euclidean'):
        super(DistanceNetwork, self).__init__()
        self.metric = metric
    
    def forward(self, support, query):
        if self.metric == 'cosine':
            norm_s = F.normalize(support, p=2, dim=1)  
            norm_i = F.normalize(query, p=2, dim=1)  
            similarities = norm_s.mm(norm_i.t()) 
            similarities = similarities.t() #(way*query,1)
            similarities = similarities.squeeze() # (75)

        elif self.metric  == 'euclidean':
            euclidean_distance = torch.sqrt(torch.sum((support-query)**2,dim=1))
            similarities = - euclidean_distance
        
        return similarities #(way*query)


if __name__ == '__main__':
    support_set = torch.rand(2,640).cuda()
    query_set = torch.rand(2,2,640).cuda()
    generatorNet = GeneratorNet().cuda()
    predict = generatorNet(support_set, query_set)
    print(predict.size())








