import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SMOGOGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta1=0.9, beta2=0.9):
        super(SMOGOGCN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

        # self.fc1 = nn.Linear(input_size, output_size)
        # self.bn1 = nn.BatchNorm1d(output_size)
        # self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # weight init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        spk_rec = []
        mem_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

            # spk_rec.append(spk1)
            # mem_rec.append(mem1)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class SMOGOVCDN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta1=0.9, beta2=0.9):
        super(SMOGOVCDN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        spk_rec = []
        mem_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)
