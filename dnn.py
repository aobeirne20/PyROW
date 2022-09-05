import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(37, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.fc3 = torch.nn.Linear(1000, 200)
        self.fc4 = torch.nn.Linear(200, 1)
        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU()
        self.lsm = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return self.lsm(x)


class NetWrapper:
    def __init__(self):
        self.net = Net().to("cuda:0")

    def learn(self, train_loaded, batc_s, learning_rate, momentum, n_epochs):
        optimizer = torch.optim.SGD(self.net.parameters(), learning_rate, momentum)
        criterion = torch.nn.BCELoss()
        print(f'\nNeural networking training with {n_epochs} epochs:')
        for epoch in range(n_epochs):
            s_time = time.time()
            loss_total = 0
            for batch_idx, (data, target) in enumerate(train_loaded):
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                optimizer.zero_grad()

                net_out = self.net(data.to("cuda:0"))
                loss = criterion(torch.squeeze(net_out), torch.squeeze(target.to("cuda:0")))

                loss.backward()
                optimizer.step()
                loss_total += loss
                leng = len(train_loaded)*batc_s
                n_blocks = int(batch_idx // (leng/batc_s/20))
                space = " "
                bar = u'\u2588'
                if epoch < 9:
                    print(f'\rEpoch {epoch+1}  |{bar*n_blocks}{space*(20-n_blocks)}| {batch_idx*batc_s}/{leng}', end='')
                else:
                    print(f'\rEpoch {epoch+1} |{bar*n_blocks}{space*(20-n_blocks)}| {batch_idx*batc_s}/{leng}', end='')
            if epoch < 9:
                print(f'\rEpoch {epoch + 1}  |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(loss_total / (leng/batc_s)):.4f}')
            else:
                print(f'\rEpoch {epoch + 1} |{bar * 20}| {leng}/{leng}', end='')
                print(f'   {(time.time() - s_time):.2f}s  Avg Loss: {(loss_total / (leng/batc_s)):.4f}')
        return loss_total / (leng/batc_s)

    def test(self, test_loaded):
        n_correct = 0
        n_incorrect = 0
        for batch_idx, (data, target) in enumerate(test_loaded):
            if torch.cuda.is_available():
                net_out = self.net(data.to('cuda:0'))
            else:
                net_out = self.net(data)
            if np.argmax(net_out.cpu().detach().numpy()) == target[0]:
                n_correct += 1
            else:
                n_incorrect += 1
            percent_correct = n_correct/(n_incorrect+n_correct)*100
            print(f'\r{batch_idx+1}/{len(test_loaded)} tests complete, {percent_correct:.2f}% correct', end='')
        print(f'')
        return percent_correct
