import argparse

import torch
from torch.utils.data import DataLoader

from model import ModelClass
from utils import RecommendationDataset
import pandas as pd
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import torch.optim as optim




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='train loader batch size')

    args = parser.parse_args()
    
    def create_model(user_num, item_num, factor_num):
        model = ModelClass(user_num,item_num,20,3,0.0,)
        
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        return model, loss_function, optimizer

    model, criterion, optimizer = create_model(610, 193609, 20)
    model.cuda()

    train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    train_data, validation_data = train_test_split(train_data,test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    

    for epoch in range(5):
        cost = 0;
        for users, items, ratings in train_loader:
            users = users.cuda()
            items = items.cuda()
            ratings = ratings.float().cuda()
            
            optimizer.zero_grad()
            ratings_pred = model(users,items)
            loss = criterion(ratings_pred, ratings)
            loss.backward()
            optimizer.step()
            cost += loss.item()*len(ratings)            
        cost/= 81676
        
        print(f"Epoch: {epoch}")
        print("train cost: {: .6f}".format(cost))
        
        cost_validation = 0;
        for users, items, ratings in validation_loader:
            users = users.cuda()
            items = items.cuda()
            ratings = ratings.float().cuda()
            
            ratings_pred = model(users,items)
            loss = criterion(ratings_pred, ratings)
            cost_validation += loss.item()*len(ratings)   
        cost_validation/= 9076
        print("validation cost: {: .6f}".format(cost_validation))        
    


    torch.save(model.state_dict(), args.save_model)
