import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from pytorch_forecasting.metrics.point import MAPE
from pytorch_forecasting.metrics import MAPE
import sys
from torch.optim import Adam


df = pd.read_csv("data/df_venues_processed.csv",sep=";")



#X = np.zeros((50,3,5))
X = np.zeros((50,4,5))
for i in range(0,250,5):
    X[i//5,0,:] = df.iloc[i:(i+5),1] # venues
    X[i//5,1,:] = df.iloc[i:(i+5),13] / 25 # temp
    X[i//5,3,:] = df.iloc[i:(i+5),14] / 15  # precip.
    X[i//5,2,:] = df.iloc[i:(i+5),19] # resa

def audessus100(X):
    n = X.shape[0]
    ligne_a_enlever=[]
    for i in range(n):
        if True in [X[i,0,j]<100 for j in range(5)]:
            ligne_a_enlever.append(i)
    k = len(ligne_a_enlever)
    Xaudessus100 = np.delete(X,ligne_a_enlever,axis=0)
    return (n-k,Xaudessus100)

n,X = audessus100(X)



print('taille finale: ',n)

Xapp = np.zeros((n-2,5,4))
# Xapp = np.zeros((n-2,5,2))
#Xapp = np.zeros((n-2,5,3))
Yapp = np.zeros((n-2,5))
for i in range(n-2):
    Xapp[i,:,0] = X[i+1,0,:] # lagged of order 1. #venues
    # Xapp[i,:,0] = X[i,0,:] # lagged of order 2. #venues
    #Xapp[i,:,1] = X[i+1,1,:]  # temp
    Xapp[i,:,1] = X[i,1,:]  #temp
    Xapp[i,:,2] = X[i,3,:]   #precip.
    # Xapp[i,:,3] = X[i,1,:]
    #Xapp[i,:,1] = X[i+1,2,:] #lagged of order 1. #resa
    Xapp[i,:,3] = X[i,2,:] #resa
    Yapp[i,:] = X[i+2,0,:]

mean_Xapp = np.zeros()
for i in range(n-2):
    mean_Xapp[i] = Xapp[i,:,0] 




for nb_semaine in range(1):
    X = Xapp[0:(n-2-nb_semaine),:,:]
    Y = Yapp[0:(n-2-nb_semaine),:]




    class LayerNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            # d is the normalization dimension
            self.d = d
            self.eps = eps
            self.alpha = nn.Parameter(torch.randn(d))
            self.beta = nn.Parameter(torch.randn(d))
        def forward(self, x):
            # x is a torch.Tensor
            # avg is the mean value of a layer
            avg = x.mean(dim=-1, keepdim=True)
            # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
            std = x.std(dim=-1, keepdim=True) + self.eps
            return (x - avg) / std * self.alpha + self.beta
            

    class FeedForward(nn.Module):
        def __init__(self, in_d=1, out_d=7,hidden=[4,4,4], dropout=0.1, activation=F.relu):
            # in_d      : input dimension, integer
            # hidden    : hidden layer dimension, array of integers
            # dropout   : dropout probability, a float between 0.0 and 1.0
            # activation: activation function at each layer
            super().__init__()
            self.sigma = activation
            dim = [in_d] + hidden + [out_d]
            self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
            self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
            self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])
        def forward(self, t):
            for i in range(len(self.layers)-1):
                t = self.layers[i](t)
                # skipping connection
                t = t + self.ln[i](t)
                t = self.sigma(t)
                # apply dropout
                t = self.dp[i](t)
            # linear activation at the last layer
            return self.layers[-1](t)


    def _inner_product(f1, f2, h):
        prod = f1 * f2 # (B, J = len(h) + 1)
        return torch.matmul((prod[:, :-1] + prod[:, 1:]), torch.unsqueeze(h,dim=-1))/2

    def _l1(f, h):
    # f dimension : ( B bases, J )
        B, J = f.size()
        return _inner_product(torch.abs(f), torch.ones((B, J)).to(device), h)

    def _l2(f, h):
            # f dimension : ( B bases, J )
            # output dimension - ( B bases, 1 )
        return torch.sqrt(_inner_product(f, f, h)) 
            

        

    class AdaFNN(nn.Module):
        def __init__(self, n_base=4, base_hidden=[64, 64, 64], grid=(0, 1),
                    dropout=0.1, lambda1=0.0, lambda2=0.0,
                    device=None):
            """
            n_base      : number of basis nodes, integer
            base_hidden : hidden layers used in each basis node, array of integers
            grid        : observation time grid, array of sorted floats including 0.0 and 1.0
            sub_hidden  : hidden layers in the subsequent network, array of integers
            dropout     : dropout probability
            lambda1     : penalty of L1 regularization, a positive real number
            lambda2     : penalty of L2 regularization, a positive real number
            device      : device for the training
            """
            super().__init__()
            self.n_base = n_base
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.device = device
            # grid should include both end points
            grid = np.array(grid)
            # send the time grid tensor to device
            self.t = torch.tensor(grid).to(device).float()
            self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()
            # instantiate each basis node in the basis layer
            self.BL = nn.ModuleList([FeedForward(1,1, hidden=base_hidden, dropout=dropout, activation=F.selu)
                                    for _ in range(n_base)])
        def forward(self, x):
            B, J = x.size()
            assert J == self.h.size()[0] + 1
            T = self.t.unsqueeze(dim=-1)
            # evaluate the current basis nodes at time grid
            self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]
            """
            compute each basis node's L2 norm
            normalize basis nodes
            """
            l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
            self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
            # compute each score <basis_i, f> 
            score = torch.cat([_inner_product(b.repeat((B, 1)), x, self.h) # (B, 1)
                            for b in self.bases], dim=-1) # score dim = (B, n_base)
            return score
        def R1(self, l1_k):
            """
            L1 regularization
            l1_k : number of basis nodes to regularize, integer        
            """
            if self.lambda1 == 0: return torch.zeros(1).to(self.device)
            # sample l1_k basis nodes to regularize
            selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
            selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
            return self.lambda1 * torch.mean(_l1(selected_bases, self.h))
        def R2(self, l2_pairs):
            """
            L2 regularization
            l2_pairs : number of pairs to regularize, integer  
            """
            if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
            k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
            f1, f2 = [None] * k, [None] * k
            for i in range(k):
                a, b = np.random.choice(self.n_base, 2, replace=False)
                f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
            return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                    torch.cat(f2, dim=0),
                                                                    self.h)))



    class SmoothFNN(nn.Module):
        def __init__(self, n_base=5, base_hidden=[64, 64, 64], grid=(0, 1),
                    dropout=0.1, alpha= 1.0, device=None):
            """
            n_base      : number of basis nodes, integer
            base_hidden : hidden layers used in each basis node, array of integers
            grid        : observation time grid, array of sorted floats including 0.0 and 1.0
            sub_hidden  : hidden layers in the subsequent network, array of integers
            dropout     : dropout probability
            lambda1     : penalty of L1 regularization, a positive real number
            lambda2     : penalty of L2 regularization, a positive real number
            device      : device for the training
            """
            super().__init__()
            self.n_base = n_base
            self.alpha = alpha
            self.device = device
            # grid should include both end points
            grid = np.array(grid)
            # send the time grid tensor to device
            self.t = torch.tensor(grid).to(device).float()
            self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()
            # instantiate each basis node in the basis layer
            self.BL = nn.ModuleList([FeedForward(1,1, hidden=base_hidden, dropout=dropout, activation=F.selu)
                                    for _ in range(n_base)])
        def forward(self, x):
            B, J = x.size()
            # assert J == self.h.size()[0] + 1
            T = self.t.unsqueeze(dim=-1)
            # evaluate the current basis nodes at time grid
            self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]
            """
            compute each basis node's L2 norm
            normalize basis nodes
            """
            l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
            self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
            # compute each score <basis_i, f> 
            score = torch.cat([_inner_product(b.repeat((B, 1)), x, self.h) # (B, 1)
                            for b in self.bases], dim=-1) # score dim = (B, n_base)
            return score
        def Smooth(self):
            """
            L2 regularization
            alpha : smoothing parameter        
            """
            acc = [torch.diff(bases,n=2) for bases in self.bases]
            return self.alpha * torch.mean(torch.Tensor([_l2(accel,self.h[1:-1]) for accel in acc]))

    class ClassificationHead(nn.Module):
        """Head for classification task."""
        def __init__(self, hidden_size,final_dropout,num_labels):
            super().__init__()
            self.dense1 = nn.Linear(hidden_size,hidden_size)
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.norm = LayerNorm(hidden_size)
            self.dropout = nn.Dropout(final_dropout)
            self.out_proj = nn.Linear(hidden_size, num_labels)
            self.norm2 = LayerNorm(num_labels)
        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense1(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            # x = self.dense2(x)
            # x = torch.tanh(x)
            # x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class MultiAda(nn.Module):
        def __init__(self, n_base=4, base_hidden=[80, 80, 80], grid1=(0, 1),
                    dropout=0.1, lambda1=[], lambda2=[], alpha=1.0, dim=64, num_layers=128,
                    device=torch.device("cuda") ):
            """
            n_base      : number of basis nodes, integer
            base_hidden : hidden layers used in each basis node, array of integers
            grid        : observation time grid, array of sorted floats including 0.0 and 1.0
            sub_hidden  : hidden layers in the subsequent network, array of integers
            dropout     : dropout probability
            lambda1     : penalty of L1 regularization, a positive real number
            lambda2     : penalty of L2 regularization, a positive real number
            device      : device for the training
            """
            super().__init__()
            self.n_base = n_base
            self.device = device
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.sigma = F.relu
            # self.ln1 = LayerNorm(self.n_base*80*2) 
            # self.ln2 = LayerNorm(500)
            # self.ln3 = LayerNorm(4) 
            # instantiate each basis node in the basis layer
            self.Ada = nn.ModuleList([AdaFNN(n_base=n_base, base_hidden=base_hidden, grid=grid1,
                    dropout=dropout, lambda1=i, lambda2=j,
                    device=device) for i,j in zip(self.lambda1,self.lambda2)])
            self.dp =nn.Dropout(dropout)
            # instantiate the initial network
            # self.encoder_layers = nn.TransformerEncoderLayer(d_model=24, nhead=4, dim_feedforward=512, batch_first=True)
            # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, 2)
            # self.flat = nn.Flatten()       
            self.encoder_layers = nn.TransformerEncoderLayer(d_model=4, nhead=4, dim_feedforward=dim, batch_first=True)
            # self.norm = LayerNorm(3)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)#, norm=self.norm)
            # self.encoder_layers3 = nn.TransformerEncoderLayer(d_model=20, nhead=4, dim_feedforward=1024, batch_first=True)
            # self.norm3 = LayerNorm(20)
            # self.transformer_encoder3 = nn.TransformerEncoder(self.encoder_layers3, 6, norm=self.norm3, enable_nested_tensor=False)
            # self.encoder_layers2 = nn.TransformerEncoderLayer(d_model=self.n_base*24, nhead=16, dim_feedforward=246, batch_first=True)
            # self.norm2 = LayerNorm(self.n_base*24)
            # self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layers2, 1, norm=self.norm2)
            self.classif = ClassificationHead(4*self.n_base,dropout,5)
            # self.smooth = SmoothFNN(grid=grid1, alpha=alpha, device=device)
        def forward(self, x, mask, src_mask): 
            #x, hidden = self.lstm(x) 
            #out1 = self.network(torch.unsqueeze(xmel,1))
            # src_mask = nn.Transformer.generate_square_subsequent_mask(20, device=device)
            # B,s,e = x.size()
            # mask = self.make_mask(B,s,e)
            x = self.transformer_encoder(x,mask,src_mask)# mask=mask, src_key_padding_mask=src_mask)
            out = []
            for i in range(4):
                out.append(self.Ada[i](x[:,:,i]))
            out = torch.cat(out,dim=-1)
            final = self.classif(out)
            # smooth_final = self.smooth(final)
            return final  
        def L1(self, l1_k):
            return torch.sum(torch.stack([self.Ada[i].R1(l1_k) for i in range(4)]))
        def L2(self, l2_pairs):
            return torch.sum(torch.stack([self.Ada[i].R2(l2_pairs) for i in range(4)]))
        def Lsmooth(self):
            return self.smooth.Smooth()




            









    print(X.shape)
    print(Y.shape)



    class DataLoader:
        def __init__(self, batch_size, n_augmented,X,Y,T, split=(8, 1, 1), random_seed=1234):        
            """
            batch_size : batch size, integer
            X - (n, m, J) :  observed multivariate functional data, n - subject number, J - number of time points, m - number of covariate
            Y - (n, 7) :  response with 7 classes 
            split      : train/valid/test split
            random_seed: random seed for training data re-shuffle
            """         
            self.n, _,_ = X.shape
            self.t = T.iloc[0, :].to_numpy()
            # X, Y = X.values, Y.values
            # train/valid/test split
            self.batch_size = batch_size
            #train_n = self.n // sum(split) * split[0]
            train_n = 41-nb_semaine 
            #train_n = 40
            #valid_n = self.n // sum(split) * split[1]
            #valid_n = 2
            valid_n = 2 
            #test_n = self.n - train_n - valid_n
            test_n = 1
            self.train_B = train_n // batch_size
            self.valid_B = valid_n // batch_size
            self.test_B = test_n // batch_size
            self.test_n = test_n 
            # Store the indices of the test set 
            self.test_indices = np.arange((self.train_B + self.valid_B) * self.batch_size, self.n)
        
            self.valid_X = X[:(self.valid_B * self.batch_size),:, :]
            # self.train_Xmel = Xmel[:(self.train_B * self.batch_size), :, :]
            self.valid_Y = Y[:(self.valid_B * self.batch_size), :]
            np.random.seed(random_seed)
            _order = np.arange(self.valid_B*batch_size*(n_augmented+1))
            np.random.shuffle(_order)
            self.valid_X = self.valid_X[_order, :, :]
            self.valid_Y = self.valid_Y[_order, :]
            self.train_X = X[(self.valid_B * self.batch_size):((self.train_B + self.valid_B) * self.batch_size),:, :]
            # self.valid_Xmel = Xmel[(self.train_B * self.batch_size):((self.train_B + self.valid_B) * self.batch_size), :, :]
            self.train_Y = Y[(self.valid_B * self.batch_size):((self.train_B + self.valid_B) * self.batch_size), :]        
            _order2 = np.arange(self.train_B*batch_size)
            np.random.shuffle(_order2)
            self.train_X = self.train_X[_order2,:,:]
            self.train_Y = self.train_Y[_order2, :]

            self.test_X = X[((self.train_B + self.valid_B) * self.batch_size):, :, :]
            # self.test_Xmel = Xmel[((self.train_B + self.valid_B) * self.batch_size):, :, :]
            self.test_Y = Y[((self.train_B + self.valid_B) * self.batch_size):, :]




        def get_test_indices(self):
            return self.test_indices
        def get_test_data(self):
            return self.test_X, self.test_Y, self.test_n
        def get_mean_var(self):
            return self.mean, self.var
        def shuffle(self):
            # re-shuffle the training dataset
            train_size = self.train_X.shape[0]
            new_order = np.arange(train_size)
            np.random.shuffle(new_order)
            self.train_X = self.train_X[new_order, :, :]
            # self.train_Xmel = self.train_Xmel[new_order,:,:]
            self.train_Y = self.train_Y[new_order]
        def _batch_generator(self, X, Y, N):
            def generator_func():
                for i in range(N):
                    x = X[((i ) * self.batch_size):((i+1) * self.batch_size), :, :]
                    # xmel = Xmel[((i - 1) * self.batch_size):((i) * self.batch_size), :, :]
                    y = Y[((i ) * self.batch_size):((i+1) * self.batch_size)]
                    yield torch.Tensor(x), torch.Tensor(y)
            return generator_func()
        def get_train_batch(self):
            return self._batch_generator(self.train_X, self.train_Y, self.train_B)
        def get_valid_batch(self):
            return self._batch_generator(self.valid_X, self.valid_Y, self.valid_B)
        def get_test_batch(self):
            return self._batch_generator(self.test_X, self.test_Y, self.test_B) 
        



    n_augmented = 0
    batch_size = 1
    split = (47, 1, 2)

    T1 = pd.DataFrame([np.linspace(0,1,5)])
    grid1 = T1.iloc[0, :].to_list()
    # T2 = pd.DataFrame([np.linspace(0,1,160)])
    # grid2 = T2.iloc[0, :].to_list()

    dataLoader = DataLoader(batch_size, n_augmented, X, Y, T1, split)

    # Obtenir les indices des données de test 

    test_indices = dataLoader.get_test_indices()

    # Obtenir les données de test 

    test_X, test_Y, test_n  = dataLoader.get_test_data()

    # Utiliser ces indices pour récupérer les données de test 

    test_X_subset = test_X[:test_n]
    test_Y_subset = test_Y[:test_n]

    # # moyenne, variance
    # mean, var = dataLoader.get_mean_var()
    # print('moyenne:',mean)

    # print('variance:',var)
    # device = torch.device('cuda') 
    device = torch.device('cpu') 
    # model configuration
    """
    You can use a different model by modifing base_hidden, n_base.
    """


    base_hidden = [64, 64, 64]
    n_base = 50
    alpha =  0.5
    lambda1 = np.zeros(4)
    # lambda1 = np.ones(25)/2
    l1_k = 20
    # lambda2 = (np.flip(np.arange(20))/60)[:-1]
    lambda2 = np.zeros(4)
    # lambda2 = np.ones(25)
    l2_pairs = 30
    dropout = 0.2
    save_model_every = 1000
    dim = 2
    num_layers = 4
    model = MultiAda(n_base=n_base,
                base_hidden=base_hidden,
                grid1=grid1,
                dropout=dropout,
                lambda1=lambda1,
                lambda2=lambda2,
                alpha= alpha,
                dim=dim,
                num_layers=num_layers,
                device=device)
    # send model to CPU/GPU
    _ = model.to(device)


    # training configuration
    epoch = 10
    pred_loss_train_history = []
    total_loss_train_history = []
    loss_valid_history = []
    # instantiate an optimizer
    optimizer = Adam(model.parameters(), lr=3e-4)


    
    compute_loss = torch.nn.L1Loss(reduction='mean')
    # compute_loss = MAPE()
    # compute_loss = nn.BCEWithLogitsLoss()
    # compute_loss = nn.NLLLoss()
    min_valid_loss = sys.maxsize

    folder = "train_venues"+str(nb_semaine)+"/"
    Path(folder).mkdir(parents=True, exist_ok=True)

    def save_model(folder, k, n_base, base_hidden, grid1, dropout, lambda1, lambda2, alpha, dim, num_layers, model, optimizer):
        checkpoint = {'n_base': n_base,
                    'base_hidden': base_hidden,
                    'grid1': grid1,
                    'dropout': dropout,
                    'lambda1' : lambda1,
                    'lambda2' : lambda2,
                    'alpha' : alpha,
                    'dim' : dim,
                    'num_layers' : num_layers,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


    def load_model(file_path, device):
        checkpoint = torch.load(file_path)
        model = MultiAda(n_base=checkpoint['n_base'],
                    base_hidden=checkpoint['base_hidden'],
                    grid1=checkpoint['grid1'],
                    dropout=checkpoint['dropout'],
                    lambda1=checkpoint['lambda1'],
                    lambda2=checkpoint['lambda2'],
                    alpha=checkpoint['alpha'],
                    dim=checkpoint['dim'],
                    num_layers=checkpoint['num_layers'], 
                    device=device)
        model.load_state_dict(checkpoint['state_dict'])
        _ = model.to(device)
        return model, checkpoint['grid1']
        



    for k in range(epoch):
        if k and k % save_model_every == 0:
            save_model(folder, k, n_base, base_hidden, grid1, dropout, lambda1, lambda2, alpha, dim, num_layers, model, optimizer)
        pred_loss_train = []
        total_loss_train = []
        loss_valid = []
        dataLoader.shuffle()
        # set model training state
        model.train()
        for i, (x, y) in enumerate(dataLoader.get_train_batch()):
            x, y = x.to(device), y.to(device)
            B,s,_ = x.size()
            out = model.forward(x,None,None)#,make_mask(B,s,device),src_make_mask(B,s,device))
            # y_tensor = y.clone().detach().type(torch.long)
            pred_loss = compute_loss(out, y.type(torch.long))
            # loss = pred_loss+ model.L1(l1_k)+model.L2(l2_pairs)+model.Lsmooth()
            loss = pred_loss
            # record training loss history
            total_loss_train.append(loss.item())
            pred_loss_train.append(pred_loss.item())
            # update parameters using backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss_train_history.append(np.mean(total_loss_train))
        pred_loss_train_history.append(np.mean(pred_loss_train))
        # model evaluation mode
        with torch.no_grad():
            model.eval()
            for x, y in dataLoader.get_valid_batch():
                x, y = x.to(device), y.to(device)
                valid_y = model.forward(x,None,None)
                # y_tensor = y.clone().detach().type(torch.long)
                # valid_loss = compute_loss(valid_y, y.type(torch.long))
                valid_loss = compute_loss(valid_y,y.type(torch.long))
                # print("valid - check out: ", check_tensor([valid_loss]))
                loss_valid.append(valid_loss.item()) 
        if np.mean(loss_valid) < min_valid_loss:
            save_model(folder, "best", n_base, base_hidden, grid1, dropout, lambda1, lambda2, alpha, dim, num_layers, model, optimizer)
            min_valid_loss = np.mean(loss_valid)
        loss_valid_history.append(np.mean(loss_valid))   
        # if (k+1) % 10 == 0:
        print("epoch:", k+1, "\n",
                "prediction training loss = ", pred_loss_train_history[-1],
                "validation loss = ", loss_valid_history[-1])


    plt.plot(list(range(1, epoch+1)), pred_loss_train_history, label='train_total')
    plt.plot(list(range(1, epoch+1)), loss_valid_history, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('Cross entropy')
    plt.legend()
    plt.savefig('loss_plot_venues.png')
    plt.close()




    # print("test")


    # folder = "train051223V28/"
    # Path(folder).mkdir(parents=True, exist_ok=True)
    # checkpoint = torch.load(folder + str("best") + '_' + 'checkpoint.pth')

    # model = MultiAda(n_base=checkpoint['n_base'],
    #                    base_hidden=checkpoint['base_hidden'],
    #                    grid1=checkpoint['grid1'],
    #                    dropout=checkpoint['dropout'],
    #                    lambda1=checkpoint['lambda1'],
    #                    lambda2=checkpoint['lambda2'],
    #                    device=device)
    # model.load_state_dict(checkpoint['state_dict'])
    # _ = model.to(device)
    ck = folder + "best_checkpoint.pth"
    # load the best model
    model, t = load_model(ck, device)
    T = torch.tensor(t).to(device)
    t = np.array(t)


    ### initialiser test_y pour stocker les résultats ### 
    #test_y = []
    loss_test = []
    with torch.no_grad():
            model.eval()
            for x, y in dataLoader.get_test_batch():
                x, y = x.to(device), y.to(device)
                test_y = model.forward(x,None,None)
                # # y_tensor = y.clone().detach().type(torch.long)
                # # valid_loss = compute_loss(valid_y, y.type(torch.long))
                test_loss = compute_loss(test_y,y.type(torch.long))
                # # print("valid - check out: ", check_tensor([valid_loss]))
                #loss_test.append(test_loss.item())
                #print(test_loss)

    print('test loss:', test_loss.item())
    print('reel:',y.tolist())
    print('resultat:',test_y.tolist())
    # print(np.mean(np.array(loss_test)))


