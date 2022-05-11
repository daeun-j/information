from dataloader import *
from models import *
from utils import *
from args import *
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torch.utils.data import DataLoader
import glob, os

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    ####################
    #### Simulation  ###
    ####################

    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # PATH = "2022_3/experiment/sort/" # for python console
    PATH = "sort/simulation_v2/"
    submit_filename = "submission.csv"
    datatype = 'Simulation'
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

    '''
    ####################
    ####  Network    ###
    ####################
    
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # PATH = "2022_3/experiment/sort/network/" # for python console
    PATH = "sort/network_v2/"
    submit_filename = "submission.csv"
    datatype = 'Network_{}'.format(args.epochs)

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

    '''
    ####################
    ####   Common    ###
    ####################

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.get_device_name()


    df = pd.read_csv('{}{}_{}'.format(PATH, datatype, "all.txt"), sep=" ", header=None)  # , names=["label_s", "label_y"])
    df = df.sample(frac=1).reset_index(drop=True)
    # train_df = pd.read_csv(PATH+"train.csv")
    # test_df = pd.read_csv(PATH+"test.csv")
    test_size = 0.2
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size)
    train_ds = Simulation(train_df.reset_index(), num_features = args.num_features, train=True,
                          transform=transforms.ToTensor())  # .Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)  # 4)

    test_ds = Simulation(test_df.reset_index(), num_features = args.num_features,
                         train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)  # 4)

#    model = Network(args.num_features, args.embedding_dims).double()

    model = Network2(args.num_features, args.embedding_dims).double()


    model.apply(init_weights)
    model = torch.jit.script(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.jit.script(TripletLoss())
    model.train()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label, _) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            optimizer.zero_grad()
            anchor_out = model(anchor_img.double())
            positive_out = model(positive_img.double())
            negative_out = model(negative_img.double())

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Loss: {:.4f}".format(loss))

    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.epochs, np.mean(running_loss)))

    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
                }, '{}{}_{}'.format(PATH, datatype, "trained_model.pth"))

    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, _, _, label, _ in tqdm(train_loader):
            train_results.append(model(img.to(device)).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    print(train_results.shape)

    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels == label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

    plt.legend()
    plt.show()

    tree = XGBClassifier(seed=args.seed, use_label_encoder=True)
    tree.fit(train_results, labels+1) # xtrain, ytrain
    test_results = []
    test_labels = []
    labels = labels-1
    model.eval()
    with torch.no_grad():
        #for img in tqdm(test_loader):
        for img, test_label, _ in tqdm(test_loader):
            test_results.append(model(img.to(device)).cpu().numpy())
            test_labels.append(test_label.numpy().item())

    test_results = np.concatenate(test_results)

    plt.figure(figsize=(15, 10), facecolor="azure")
    plt.scatter(test_results[:, 0], test_results[:, 1], label=label)
    print(test_results.shape)
    plt.legend()
    plt.show()
    #submit = pd.read_csv(PATH + "sample_submission.csv")
    submit = pd.DataFrame(columns=['Label', 'Results'])
    submit.Label = test_labels
    submit.Results = tree.predict(test_results)-1

    submit.head()

    # submit.to_csv(PATH + "submission.csv", index=False)
    submit.to_csv('{}{}_{}'.format(PATH, datatype, submit_filename))