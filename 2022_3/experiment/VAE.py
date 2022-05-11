# ref https://github.com/pytorch/examples/blob/main/vae/main.py
# ref https://avandekleut.github.io/vae/
from __future__ import print_function
from sklearn.model_selection import train_test_split
import glob, os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

from dataloader import *
from utils import *
from args import *
from models import *

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
#
args = parser.parse_args()

def test(epoch):
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, args.num_features).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":

    '''
    ####################
    #### Simulation  ###
    ####################
    embedding_dims = 2
    batch_size = 10
    epochs = 2
    num_features = 8
    test_size = 0.2

    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # PATH = "2022_3/experiment/sort/" # for python console
    PATH = "sort/simulation_v3/"
    submit_filename = "submission_trVAE.csv"
    datatype = 'Simulation'
    # if os.path.exists(PATH + "all.txt"):
    #     os.remove(PATH + "all.txt")
    # else:
    #     print("The file does not exist")
    # 
    # filenames = glob.glob(PATH + '*txt')
    # # print(filenames)
    # with open(PATH + 'all.txt', 'w') as outfile:
    #     for filename in sorted(filenames):
    #         with open(filename) as file:
    #             for line in file:
    #                 outfile.write(line)
    '''
    
    ####################
    ####  Network    ###
    ####################
    embedding_dims = 2
    batch_size = 100
    epochs = 1  # 50
    num_features = 8
    test_size = 0.2
    
    PATH = "sort/network_v2/"
    submit_filename = "submission_trVAE.csv"
    datatype = 'Network'

    if os.path.exists('{}{}_{}'.format(PATH, datatype, "all.txt")):
        os.remove('{}{}_{}'.format(PATH, datatype, "all.txt"))
    else:
        print("The file does not exist")

    filenames = glob.glob(PATH + '*txt')
    # print(filenames)
    with open('{}{}_{}'.format(PATH, datatype, "all.txt"), 'w') as outfile:
        for filename in sorted(filenames):
            with open(filename) as file:
                for line in file:
                    outfile.write(line)


    ####################
    ####   Common    ###
    ####################

    torch.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.get_device_name()

    df = pd.read_csv('{}{}_{}'.format(PATH, datatype, "all.txt"), sep=" ", header=None)
    df = df.sample(frac=1).reset_index(drop=True)


    train_df, test_df = train_test_split(df, test_size=test_size)
    train_ds = Simulation(train_df.reset_index(), train=True,
                          transform=transforms.ToTensor())  # .Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)  # 4)

    test_ds = Simulation(test_df.reset_index(), train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)  # 4)

    model = VAE().to(device).double()
    model = torch.jit.script(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.jit.script(TripletLoss())
    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, _, _, anchor_label, _) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            optimizer.zero_grad()
            # anchor_img = torch.from_numpy(anchor_img).float()
            # recon_batch, mu, logvar = model(anchor_img.float())
            recon_batch, mu, logvar = model(anchor_img.double())
            loss = loss_function(recon_batch, anchor_img, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
                }, '{}{}_{}'.format(PATH, datatype,  "trained_model_VAE.pth"))

    train_results_mu = []
    train_results_var = []
    labels = []
    labels_u = []

    model.eval()
    with torch.no_grad():
        for img, _, _, label, label_u in tqdm(train_loader):
            recon_batch, mu, logvar = model(img.double())
            # print(mu, logvar)
            train_results_mu.append(mu.detach().numpy())
            train_results_var.append(logvar.detach().numpy())
            labels.append(label)
            labels_u.append(label_u)

    train_results_mu = np.concatenate(train_results_mu)
    train_results_var = np.concatenate(train_results_var)

    labels = np.concatenate(labels)
    labels_u = np.concatenate(labels_u)

    # print(train_results_mu.shape, train_results_var.shape)

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results_var[labels == label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

    submit = pd.DataFrame(columns=['Label','Label_u', 'mu1', 'mu2', 'var1', 'var2'])
    # submit = pd.DataFrame(columns=['Label','Label_u', 'mu1', 'mu2', 'mu3', 'var1', 'var2', 'var3'])

    submit.mu1 = train_results_mu[:, 0]
    submit.mu2 = train_results_mu[:, 1]
    # submit.mu3 = train_results_mu[:, 2]

    submit.var1 = train_results_var[:, 0]
    submit.var2 = train_results_var[:, 1]
    # submit.var3 = train_results_var[:, 2]

    submit.Label = labels
    submit.Label_u = labels_u

    submit.head()
    submit.to_csv('{}{}_{}'.format(PATH, datatype, submit_filename), index=False)

