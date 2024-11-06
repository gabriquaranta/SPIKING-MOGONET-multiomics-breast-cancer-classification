import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import StepLR
from snntorch import spikegen
from sklearn.metrics import f1_score

from smogoutils import get_all_data_adj, compute_spiking_node_representation as csnr
from smogomodels import SMOGOGCN, SMOGOVCDN
from smogotrain import (
    pretrain,
    smogotrain,
    smogotest,
    smogotrain_v2,
    smogotest_v2,
    check_convergence,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_device("cpu")


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

H_te_v1 = H_trte_v1
H_te_v2 = H_trte_v2
H_te_v3 = H_trte_v3

labels_tr = labels_tr_tensor.to(device)
labels_trte_tensor = torch.tensor(labels_trte).to(device)
labels_te = labels_trte_tensor[trte_idx["te"]].to(device)

# print(H_tr_v1.shape, H_tr_v2.shape, H_tr_v3.shape)
# print(H_trte_v1.shape, H_trte_v2.shape, H_trte_v3.shape)
# print(labels_tr.shape, labels_trte_tensor.shape, labels_te.shape)


dim_list = [H_tr_v1.shape, H_tr_v2.shape, H_tr_v3.shape]

# # RANDOM GENERATION


# betas= [0.4,0.5,0.7,0.9]
# hidden_sizes= [30,40,50,60,70]
# starting_lrs= [0.1,0.01,0.001]
# step_sizes= [10,15,20,25,30]
# gammas= [0.2,0.4,0.5,0.6,0.8]


betas = np.linspace(0.2, 0.99, 10)
hidden_sizes = [int(i) for i in np.linspace(30, 80, 10)]
starting_lrs = np.linspace(0.1, 0.001, 10)
step_sizes = [int(i) for i in np.linspace(10, 30, 10)]
gammas = np.linspace(0.2, 0.8, 10)


def random_individual():
    ind = {}
    ind["sgcn1"] = [
        random.choice(betas),
        random.choice(betas),
        random.choice(hidden_sizes),
        random.choice(starting_lrs),
        random.choice(step_sizes),
        random.choice(gammas),
    ]
    ind["sgcn2"] = [
        random.choice(betas),
        random.choice(betas),
        random.choice(hidden_sizes),
        random.choice(starting_lrs),
        random.choice(step_sizes),
        random.choice(gammas),
    ]
    ind["sgcn3"] = [
        random.choice(betas),
        random.choice(betas),
        random.choice(hidden_sizes),
        random.choice(starting_lrs),
        random.choice(step_sizes),
        random.choice(gammas),
    ]
    ind["vcdn"] = [
        random.choice(betas),
        random.choice(betas),
        random.choice(hidden_sizes),
        random.choice(starting_lrs),
        random.choice(step_sizes),
        random.choice(gammas),
    ]

    return ind


def models_from_ind_dict(ind_dict):
    sgcn1 = SMOGOGCN(
        input_size=dim_list[0][2],
        hidden_size=50,
        output_size=num_class,
        beta1=ind_dict["sgcn1"][0],
        beta2=ind_dict["sgcn1"][1],
    ).to(device)
    sgcn2 = SMOGOGCN(
        input_size=dim_list[1][2],
        hidden_size=50,
        output_size=num_class,
        beta1=ind_dict["sgcn2"][0],
        beta2=ind_dict["sgcn2"][1],
    ).to(device)
    sgcn3 = SMOGOGCN(
        input_size=dim_list[2][2],
        hidden_size=50,
        output_size=num_class,
        beta1=ind_dict["sgcn3"][0],
        beta2=ind_dict["sgcn3"][1],
    ).to(device)
    vcdn = SMOGOVCDN(
        input_size=125,
        hidden_size=ind_dict["vcdn"][2],
        output_size=num_class,
        beta1=ind_dict["vcdn"][0],
        beta2=ind_dict["vcdn"][1],
    ).to(device)

    return [sgcn1, sgcn2, sgcn3, vcdn]


def optim_scheduler_from_ind_dict(ind_dict, models_list):
    optim1 = optim.Adam(models_list[0].parameters(), lr=ind_dict["sgcn1"][3])
    optim2 = optim.Adam(models_list[1].parameters(), lr=ind_dict["sgcn2"][3])
    optim3 = optim.Adam(models_list[2].parameters(), lr=ind_dict["sgcn3"][3])
    optim4 = optim.Adam(models_list[3].parameters(), lr=ind_dict["vcdn"][3])

    scheduler1 = StepLR(
        optim1, step_size=ind_dict["sgcn1"][4], gamma=ind_dict["sgcn1"][5]
    )
    scheduler2 = StepLR(
        optim2, step_size=ind_dict["sgcn2"][4], gamma=ind_dict["sgcn2"][5]
    )
    scheduler3 = StepLR(
        optim3, step_size=ind_dict["sgcn3"][4], gamma=ind_dict["sgcn3"][5]
    )
    scheduler4 = StepLR(
        optim4, step_size=ind_dict["vcdn"][4], gamma=ind_dict["vcdn"][5]
    )

    return [optim1, optim2, optim3, optim4], [
        scheduler1,
        scheduler2,
        scheduler3,
        scheduler4,
    ]


# # FITNESS


def fitness(ind_dict, epoch_pre=100, epoch_tr=500, printing=False, saving=False):
    models = models_from_ind_dict(ind_dict)
    optims, schedulers = optim_scheduler_from_ind_dict(ind_dict, models)

    idx_te = trte_idx["te"]
    H_trs = [H_tr_v1, H_tr_v2, H_tr_v3]
    H_te_list = [H_te_v1, H_te_v2, H_te_v3]

    if printing:
        print("\npretraining")
    l1 = pretrain(
        models[0],
        H_tr_v1,
        labels_tr,
        epoch_pre,
        ind_dict["sgcn1"][1],
        printing=printing,
    )
    l2 = pretrain(
        models[1],
        H_tr_v2,
        labels_tr,
        epoch_pre,
        ind_dict["sgcn2"][1],
        printing=printing,
    )
    l3 = pretrain(
        models[2],
        H_tr_v3,
        labels_tr,
        epoch_pre,
        ind_dict["sgcn3"][1],
        printing=printing,
    )

    if printing:
        print("\ntraining")
    losses = smogotrain_v2(
        models,
        H_trs,
        labels_tr,
        opt_list=optims,
        sched_list=schedulers,
        num_epochs=epoch_tr,
        printing=printing,
        l2_lambda=0.0,
    )

    if printing:
        print("\ntesting")
    a = smogotest_v2(
        models, H_te_list, labels_trte_tensor, labels_te, idx_te, printing=printing
    )

    if saving:
        torch.save(models[0].state_dict(), f"SMOGONET/models/ea_sgcn1.pth")
        torch.save(models[1].state_dict(), f"SMOGONET/models/ea_sgcn2.pth")
        torch.save(models[2].state_dict(), f"SMOGONET/models/ea_sgcn3.pth")
        torch.save(models[3].state_dict(), f"SMOGONET/models/ea_svcdn.pth")

    return a


# pop_size=3
# pop=[random_individual() for i in range(pop_size)]

# print([fitness(p) for p in pop])
# # 20 mins cpu

# # MUTATION


def mutate(ind_dict_original):
    """
    possible mutations:
    - sgcns x3: beta, lr pre, lr tr, step_size, gamma
    - vcdn: beta1, beta2, hidden_size, lr, step_size, gamma
    """
    models_to_mutate = ["sgcn1", "sgcn2", "sgcn3", "vcdn"]
    model_to_mutate = random.choice(models_to_mutate)

    ind_dict = ind_dict_original.copy()

    if model_to_mutate == "vcdn":
        # mutate one of the fields
        field_to_mutate = random.choice([0, 1, 2, 3, 4, 5])
        if field_to_mutate == 0:
            ind_dict["vcdn"][0] = random.choice(betas)
        elif field_to_mutate == 1:
            ind_dict["vcdn"][1] = random.choice(betas)
        elif field_to_mutate == 2:
            ind_dict["vcdn"][2] = random.choice(hidden_sizes)
        elif field_to_mutate == 3:
            ind_dict["vcdn"][3] = random.choice(starting_lrs)
        elif field_to_mutate == 4:
            ind_dict["vcdn"][4] = random.choice(step_sizes)
        elif field_to_mutate == 5:
            ind_dict["vcdn"][5] = random.choice(gammas)
    else:
        # mutate one of the fields
        field_to_mutate = random.choice([0, 1, 2, 3, 4])
        if field_to_mutate == 0:
            ind_dict[model_to_mutate][0] = random.choice(betas)
        elif field_to_mutate == 1:
            ind_dict[model_to_mutate][1] = random.choice(starting_lrs)
        elif field_to_mutate == 2:
            ind_dict[model_to_mutate][2] = random.choice(starting_lrs)
        elif field_to_mutate == 3:
            ind_dict[model_to_mutate][3] = random.choice(step_sizes)
        elif field_to_mutate == 4:
            ind_dict[model_to_mutate][4] = random.choice(gammas)

    return ind_dict


# # CROSSOVER


def xover(ind_dict_1, ind_dict_2):
    xover_point = random.choice([0, 1, 2])
    ind1 = ind_dict_1.copy()
    ind2 = ind_dict_2.copy()

    if xover_point == 0:
        ind1["sgcn1"] = ind_dict_2["sgcn1"]
        ind2["sgcn1"] = ind_dict_1["sgcn1"]
    elif xover_point == 1:
        ind1["sgcn1"] = ind_dict_2["sgcn1"]
        ind2["sgcn1"] = ind_dict_1["sgcn1"]
        ind1["sgcn2"] = ind_dict_2["sgcn2"]
        ind2["sgcn2"] = ind_dict_1["sgcn2"]
    elif xover_point == 2:
        ind1["vcdn"] = ind_dict_2["vcdn"]
        ind2["vcdn"] = ind_dict_1["vcdn"]

    return ind1, ind2


# # EA


POP_SIZE = 10
NUM_GENS = 15
NUM_PARENTS_BEST = 2
EPOCH_PRE = 2
EPOCH_TR = 5
MUTATION_RATE = 0.2

# POP_SIZE=20
# NUM_GENS=100
# NUM_PARENTS_BEST=5
# EPOCH_PRE = 20
# EPOCH_TR = 80
# MUTATION_RATE=0.2

all_fitnesses = []
best = None
best_fitness = 0

pop = [random_individual() for i in range(POP_SIZE)]

for gen in range(NUM_GENS):
    print("\nGeneration: ", gen)

    # fitnesses=[fitness(p) for p in pop]
    fitnesses = [
        fitness(p, epoch_pre=EPOCH_PRE, epoch_tr=EPOCH_TR) for p in pop
    ]  # for testing algo
    print(" Fitnesses: ", fitnesses)
    print(" Best fitness: ", max(fitnesses))
    print(" Average fitness: ", sum(fitnesses) / len(fitnesses))
    all_fitnesses.append(fitnesses)

    if gen == NUM_GENS - 1:
        break

    sortedp = sorted(zip(pop, fitnesses), key=lambda x: x[1], reverse=True)[
        :NUM_PARENTS_BEST
    ]

    if sortedp[0][1] > best_fitness:
        best_fitness = sortedp[0][1]
        best = sortedp[0][0]

    # xover + mutate
    new_pop = [p[0] for p in sortedp]  # keep some daddies
    while len(new_pop) < POP_SIZE:

        f1, f2 = xover(random.choice(sortedp)[0], random.choice(sortedp)[0])

        if random.random() < MUTATION_RATE:
            f1 = mutate(f1)
        if random.random() < MUTATION_RATE:
            f1 = mutate(f2)

        new_pop.extend([f1, f2])
    pop = new_pop

    if (
        check_convergence(
            [np.mean(f) for f in all_fitnesses], window_size=3, min_gradient=0.01
        )
        and MUTATION_RATE < 0.9
    ):
        # TODO add counter for how many times this has happened to stop at some point
        MUTATION_RATE += 0.025
        print("Convergence detected. Increasing mutation rate to: ", MUTATION_RATE)


best_final = pop[fitnesses.index(max(fitnesses))]
print("\n\nBest individual final generation: ", best_final)
print("Best fitness final generation: ", max(fitnesses))
print("\nBest individual overall: ", best)
print("Best fitness overall: ", best_fitness)


# plt.figure()
# plt.title("Fitness Over Generations")
# plt.plot([sum(f) / len(f) for f in all_fitnesses], label="Average fitness")
# plt.plot([max(f) for f in all_fitnesses], label="Best fitness")
# plt.plot([min(f) for f in all_fitnesses], label="Worst fitness")
# plt.legend()
# plt.show()


saving = True
saving = False
fitness(best, epoch_pre=100, epoch_tr=500, printing=True, saving=saving)
