import torch

# import torch.nn as nn
import torch.optim as optim

# import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR

# from snntorch import spikegen
# from sklearn.metrics import f1_score

from smogoutils import get_all_data_adj, compute_spiking_node_representation as csnr
from smogomodels import SMOGOGCN, SMOGOVCDN
from smogotrain import pretrain, smogotrain, smogotrain_v2, smogotest, smogotest_v2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_device("cpu")


############# Data
print("\nDATA PREPROCESSING")
data_folder = "SMOGONET/BRCA"
view_list = [1, 2, 3]
num_view = 3
num_class = 5

(
    data_tr_list,
    data_trte_list,
    trte_idx,
    labels_trte,
    labels_tr_tensor,
    onehot_labels_tr_tensor,
    adj_tr_list,
    adj_te_list,
    dim_list,
) = get_all_data_adj(
    data_folder,
    view_list,
    num_class,
)


K = 3
num_steps = 100

enc_function = "rate"
# enc_function = 'latency'

H_tr_v1 = csnr(data_tr_list[0], adj_tr_list[0], K, num_steps, enc_function).to(device)
H_tr_v2 = csnr(data_tr_list[1], adj_tr_list[1], K, num_steps, enc_function).to(device)
H_tr_v3 = csnr(data_tr_list[2], adj_tr_list[2], K, num_steps, enc_function).to(device)

H_trte_v1 = csnr(data_trte_list[0], adj_te_list[0], K, num_steps, enc_function).to(
    device
)
H_trte_v2 = csnr(data_trte_list[1], adj_te_list[1], K, num_steps, enc_function).to(
    device
)
H_trte_v3 = csnr(data_trte_list[2], adj_te_list[2], K, num_steps, enc_function).to(
    device
)

H_te_v1 = H_trte_v1  # only te part is extracted later
H_te_v2 = H_trte_v2
H_te_v3 = H_trte_v3

labels_tr = labels_tr_tensor.to(device)
labels_trte_tensor = torch.tensor(labels_trte).to(device)
labels_te = labels_trte_tensor[trte_idx["te"]].to(device)

# print("Data shapes after preprocessing")
# print(" Training: ", H_tr_v1.shape, H_tr_v2.shape, H_tr_v3.shape)
# print("Traing + Test: ", H_trte_v1.shape, H_trte_v2.shape, H_trte_v3.shape)
# print("Labels: ", labels_tr.shape, labels_trte_tensor.shape, labels_te.shape)


############# Models
print("\nMODEL INITIALIZATION")
output_size = num_class

hidden_size1 = 50
hidden_size2 = 50
hidden_size3 = 50

beta1a = 0.9
beta2a = 0.9
beta3a = 0.9

beta1b = 0.9
beta2b = 0.9
beta3b = 0.9

sgcn1 = SMOGOGCN(
    H_tr_v1.shape[2], hidden_size1, output_size, beta1=beta1a, beta2=beta1b
).to(device)
sgcn2 = SMOGOGCN(
    H_tr_v2.shape[2], hidden_size2, output_size, beta1=beta2a, beta2=beta2b
).to(device)
sgcn3 = SMOGOGCN(
    H_tr_v3.shape[2], hidden_size3, output_size, beta1=beta3a, beta2=beta3b
).to(device)

# sgcns output torch.stack(spk_rec), torch.stack(mem_rec)
# output spikes of shape for each: torch.Size([100, 875, 5])
# output probs of shape for each: torch.Size([875, 5])

# operations pre vcdn
# after outer product torch.Size([875, 125, 1])
# torch.Size([875, 125])
# torch.Size([100, 875, 125])

vcdn_input_size = 125
vcdn_hidden_size = 50
vcdn_beta1 = 0.9
vcdn_beta2 = 0.9

svcdn = SMOGOVCDN(
    vcdn_input_size, vcdn_hidden_size, output_size, beta1=vcdn_beta1, beta2=vcdn_beta2
).to(device)


############# Pretrain sgcns
print("\nPRETRAINING")

num_epoch_pretrain1 = 100
num_epoch_pretrain2 = 100
num_epoch_pretrain3 = 100

lr1 = 0.1
lr2 = 0.1
lr3 = 0.1
print("SGCN1")
losses1 = pretrain(sgcn1, H_tr_v1, labels_tr, num_epoch_pretrain1, lr=lr1)
print("SGCN2")
losses2 = pretrain(sgcn3, H_tr_v3, labels_tr, num_epoch_pretrain2, lr=lr2)
print("SGCN3")
losses3 = pretrain(sgcn2, H_tr_v2, labels_tr, num_epoch_pretrain3, lr=lr3)


# plt.figure(figsize=(15, 5))
# plt.suptitle("Pretrain Losses")
# plt.subplot(1, 3, 1)
# plt.plot(losses1)
# plt.title("SGCN View 1")
# plt.subplot(1, 3, 2)
# plt.plot(losses2)
# plt.title("SGCN View 2")
# plt.subplot(1, 3, 3)
# plt.plot(losses3)
# plt.title("SGCN View 3")
# plt.show()


############# Train Jointly
print("\nTRAINING")

num_epochs = 500
snns = [sgcn1, sgcn2, sgcn3, svcdn]
H_trs = [H_tr_v1, H_tr_v2, H_tr_v3]

lr1 = 0.01
lr2 = 0.01
lr3 = 0.01
lr4 = 0.1  # .1 better start than .01

step_size1 = 10
step_size2 = 10
step_size3 = 10
step_size4 = 10

gamma1 = 0.5
gamma2 = 0.5
gamma3 = 0.5
gamma4 = 0.5

optimizer1 = optim.Adam(sgcn1.parameters(), lr=lr1)
optimizer2 = optim.Adam(sgcn2.parameters(), lr=lr2)
optimizer3 = optim.Adam(sgcn3.parameters(), lr=lr3)
optimizer4 = optim.Adam(svcdn.parameters(), lr=lr4)

scheduler1 = StepLR(optimizer1, step_size=step_size1, gamma=gamma1)
scheduler2 = StepLR(optimizer2, step_size=step_size2, gamma=gamma2)
scheduler3 = StepLR(optimizer3, step_size=step_size3, gamma=gamma3)
scheduler4 = StepLR(optimizer4, step_size=step_size4, gamma=gamma4)

opt_list = [optimizer1, optimizer2, optimizer3, optimizer4]
sched_list = [scheduler1, scheduler2, scheduler3, scheduler4]

losses = smogotrain_v2(
    snns,
    H_trs,
    labels_tr,
    opt_list=opt_list,
    sched_list=sched_list,
    num_epochs=num_epochs,
    l2_lambda=0.0001,  # set to 0 to not use regularization
)

# plt.title("Training Loss")
# plt.plot(losses)


############# Test
print("\nTESTING")

load = True
load = False
if load:
    sgcn1 = SMOGOGCN(H_te_v1.shape[2], hidden_size1, output_size)
    sgcn2 = SMOGOGCN(H_te_v2.shape[2], hidden_size2, output_size)
    sgcn3 = SMOGOGCN(H_te_v3.shape[2], hidden_size3, output_size)
    svcdn = SMOGOVCDN(vcdn_input_size, vcdn_hidden_size, output_size)

    sgcn1.load_state_dict(torch.load("SMOGONET/models/sgcn1.pth"))
    sgcn2.load_state_dict(torch.load("SMOGONET/models/sgcn2.pth"))
    sgcn3.load_state_dict(torch.load("SMOGONET/models/sgcn3.pth"))
    svcdn.load_state_dict(torch.load("SMOGONET/models/svcdn.pth"))


idx_te = trte_idx["te"]
snn_list = [sgcn1, sgcn2, sgcn3, svcdn]
H_te_list = [H_te_v1, H_te_v2, H_te_v3]
# a = smogotest(snn_list, H_te_list, labels_trte_tensor, labels_te, idx_te)
a = smogotest_v2(snn_list, H_te_list, labels_trte_tensor, labels_te, idx_te)


save = True
save = False
if save:
    torch.save(sgcn1.state_dict(), "SMOGONET/models/sgcn1.pth")
    torch.save(sgcn2.state_dict(), "SMOGONET/models/sgcn2.pth")
    torch.save(sgcn3.state_dict(), "SMOGONET/models/sgcn3.pth")
    torch.save(svcdn.state_dict(), "SMOGONET/models/svcdn.pth")
