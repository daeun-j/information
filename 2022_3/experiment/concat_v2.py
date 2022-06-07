from dataloader import *
from models import *
from utils import *
from args import *
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob, os
import itertools
import numba
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from sklearn.model_selection import train_test_split



if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))

    ####################
    #### Simulation  ###
    ####################

    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)
    # PATH = "2022_3/experiment/sort/" # for python console
    PATH = "sort/simulation_v3/"
    submit_filename = "submission.csv"
    datatype = 'simulation_{}'.format(args.epochs)
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
    # datatype = '{}_{}'.format(args.epochs, str(args.test_size)[-1])
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

    df = pd.read_csv('{}{}_{}'.format(PATH, datatype, "all.txt"), sep=" ", header=None)  # , names=["label_s", "label_y"])
    df = df.sample(frac=1).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=args.test_size)
    val_df, test_df = train_test_split(val_df, test_size=args.test_size)

    train_ds = Simulation(train_df.reset_index(), num_features = args.num_features, train=True, transform=transforms.ToTensor())  # .Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)  # 4)

    val_ds = Simulation(val_df.reset_index(), num_features=args.num_features, train=False, transform=transforms.ToTensor())
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)  # 4)

    test_ds = Simulation(test_df.reset_index(), num_features=args.num_features, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=True)  # 4)

    model = concat_v2(args.num_features, args.embedding_dims, args.batch).to(device).double()
    model.apply(init_weights)
    # model = torch.jit.script(model).to(device)
    # model = torch.jit.script(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.jit.script(TripletLoss_hz())
    # criterion = TripletLoss()
    # for epoch in tqdm(range(args.epochs), desc="Epochs"):
    # with tqdm(train_loader, desc="Training") as tepoch: #, leave=False
    model.train()
    # for step, (anchor, positive, negative, anchor_label, anchor_label_u) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
    anc_names = ['r_{}'.format(i) for i in range(args.num_features)] + ['h_{}'.format(i) for i in range(args.num_features)] + ['z_{}'.format(i) for i in range(args.num_features*2)] +['anchor_label', 'anchor_label_u']
    loss_names = ['loss']
    anc_data, label_data, loss_data = torch.zeros(1, args.num_features*4), torch.zeros(1, 2), torch.zeros(1, 1)

    for epoch in range(args.epochs):

        # for step, (anchor, positive, negative, anchor_label, anchor_label_u) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        for step, (anchor, positive, negative, anchor_label, anchor_label_u) in enumerate(train_loader):

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            hanc, zanc= model(anchor.double())
            hpos, zpos = model(positive.double())
            hneg, zneg = model(negative.double())
            loss= criterion(hanc, hpos, hneg, zanc, zpos, zneg)
            _loss = loss.cpu().detach().numpy()

            loss.backward()
            optimizer.step()

            _anc_data = torch.cat((anchor.view(args.batch, -1), hanc.view(args.batch, -1), zanc.view(args.batch, -1)), 1)
            _label_data = torch.cat((anchor_label.view(args.batch, -1), anchor_label_u.view(args.batch, -1)), 1)
            _loss_data = torch.Tensor(_loss).resize(1, 1)
            # print(loss_data, _loss_data)
            loss_data = torch.cat((loss_data, _loss_data), 0)
            anc_data = torch.cat((anc_data, _anc_data.view(args.batch, -1)), 0)
            label_data = torch.cat((label_data, _label_data), 0)

        ## model save, output save ##
        if epoch % 10 == 0:

            anc_df = pd.DataFrame(np.hstack([anc_data.cpu().detach().numpy(),label_data.cpu().detach().numpy()]), columns=anc_names)
            loss_df = pd.DataFrame(loss_data.cpu().detach().numpy(), columns=loss_names)

            anc_df.to_csv('{}{}_{}_{}'.format(PATH+str("result_0607/"), datatype, "epo"+str (epoch), "hz.csv"), index=False)
            loss_df.to_csv('{}{}_{}_{}'.format(PATH+str("result_0607/"), datatype, "epo"+str (epoch), "hz_loss.csv"), index=False)
            print("epoch {} batch {} loss {}".format(epoch, step, loss.data))

        ## validation ##
        if epoch%5 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for step, (anchor, positive, negative, anchor_label, anchor_label_u) in enumerate(val_loader):
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)

                    hanc, zanc= model(anchor.double())
                    hpos, zpos = model(positive.double())
                    hneg, zneg = model(negative.double())
                    v_loss= criterion(hanc, hpos, hneg, zanc, zpos, zneg)
                    val_loss += v_loss
                print("validation loss {}".format(val_loss))


    # print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.epochs, np.mean(loss)))
    torch.save({"model_state_dict": model.state_dict(), "optimzier_state_dict": optimizer.state_dict()}, '{}{}_{}_{}'.format(PATH+str("result_0607/"), datatype, epoch, "trained_model.pth"))


    # FIXME 하나의 하일로 ml 까지 수행하는 쉘파일 만들기
    # TODO epoch, data ratio, data type 별로 확인하기

    #### train ###

    with torch.no_grad():
        test_loss = 0.0
        for step, (anchor, positive, negative, anchor_label, anchor_label_u) in enumerate(test_loader):

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            hanc, zanc= model(anchor.double())
            hpos, zpos = model(positive.double())
            hneg, zneg = model(negative.double())
            t_loss= criterion(hanc, hpos, hneg, zanc, zpos, zneg)
            test_loss += t_loss
        print("Test loss for batch {} : {}".format(step, test_loss))
