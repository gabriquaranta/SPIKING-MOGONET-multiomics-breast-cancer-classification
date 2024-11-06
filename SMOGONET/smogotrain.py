import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from snntorch import spikegen
from sklearn.metrics import f1_score


def pretrain(snn, inputs, targets, num_epoch_pretrain=10, lr=0.1, printing=True):
    optimizer = optim.Adam(snn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epoch_pretrain):
        optimizer.zero_grad()
        outputs, _ = snn(inputs)
        outputs = outputs.mean(dim=0)  # average over time steps

        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 and printing:
            print(f" Epoch {epoch + 1}/{num_epoch_pretrain}, Loss: {loss.item()}")
    return losses


def check_convergence(values, window_size=5, min_gradient=0.001):
    """
    the min_gradient of 0.001 can be interpreted as follows:

    - Meaning: The gradient represents the rate of change of the values over the
    specified window. A gradient of 0.001 means that, on average, the value is
    changing by 0.001 units per step within the window.

    - Direction: In the context of loss values in machine learning, a negative
    gradient is typically expected (as loss should decrease). A gradient greater
    than -0.001 (i.e., closer to zero or positive) suggests the loss is no
    longer significantly decreasing.
    """
    if len(values) < window_size + 1:
        return False

    recent_values = values[-window_size:]
    x = np.arange(window_size)
    y = np.array(recent_values)
    gradient, _ = np.polyfit(x, y, 1)

    return abs(gradient) < min_gradient


def smogotrain(
    snn_list,
    inputs_list,
    targets,
    opt_list,
    sched_list,
    num_epochs=100,
    printing=True,
    l2_lambda=0.01,
):
    """
    Train the SMOGOMODEL

    Args:
    - snn_list: list of SNNs in order: [sgcn1, sgcn2, sgcn3, svcdn]
    - inputs_list: list of input tensors in order: [H_tr_v1, H_tr_v2, H_tr_v3]
    - targets: target tensor ie labels_tr
    - lrs: list of learning rates in order: [lr1, lr2, lr3, lr4]
    - step_sizes: list of step sizes in order: [step_size1, step_size2, step_size3, step_size4]
    - gammas: list of gammas for steplr in order: [gamma1, gamma2, gamma3, gamma4]
    - num_epochs: number of epochs to train

    """

    num_view = 3
    num_class = 5

    sgcn1 = snn_list[0]
    sgcn2 = snn_list[1]
    sgcn3 = snn_list[2]
    svcdn = snn_list[3]

    H_tr_v1 = inputs_list[0]
    H_tr_v2 = inputs_list[1]
    H_tr_v3 = inputs_list[2]

    labels_tr = targets

    optimizer1 = opt_list[0]
    optimizer2 = opt_list[1]
    optimizer3 = opt_list[2]
    optimizer4 = opt_list[3]

    scheduler1 = sched_list[0]
    scheduler2 = sched_list[1]
    scheduler3 = sched_list[2]
    scheduler4 = sched_list[3]

    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        outputs1, _ = sgcn1(H_tr_v1)
        outputs2, _ = sgcn2(H_tr_v2)
        outputs3, _ = sgcn3(H_tr_v3)

        outputs1 = outputs1.mean(dim=0)
        outputs2 = outputs2.mean(dim=0)
        outputs3 = outputs3.mean(dim=0)

        in_list = [outputs1, outputs2, outputs3]

        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])

        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
            (-1, pow(num_class, 2), 1),
        )
        x = torch.reshape(
            torch.matmul(x, in_list[2].unsqueeze(1)),
            (-1, pow(num_class, 3), 1),
        )

        # x = torch.einsum("bi,bj,bk->bijk", in_list[0], in_list[1], in_list[2])
        # x = x.reshape(-1, num_class**num_view)

        vcdn_feat = torch.reshape(x, (-1, pow(num_class, num_view)))

        vcdn_input_spike_encoding = spikegen.rate(vcdn_feat, 100)

        ##### EINSUM NO BACK N FORTH CONVERSION -> WORST PERFORMANCE

        # vcdn_input_spike_encoding = torch.einsum(
        #     "tnc,tnd,tne->tncde", outputs1, outputs2, outputs3
        # )  # (T, N, C, C, C)

        # T, N, C, _, _ = vcdn_input_spike_encoding.shape

        # vcdn_input_spike_encoding = vcdn_input_spike_encoding.reshape(T, N, C * C * C)

        # if epoch == 0:
        #     print(vcdn_input_spike_encoding.shape)
        #####

        ##### NO BACK N FORTHCONVERSION NO EINSUM -> STIll WORSE THAN BACK AND FORTH
        # T, N, C = outputs1.shape

        # # Reshape tensors to prepare for broadcasting
        # t1 = outputs1.view(T, N, C, 1, 1)
        # t2 = outputs2.view(T, N, 1, C, 1)
        # t3 = outputs3.view(T, N, 1, 1, C)

        # vcdn_input_spike_encoding = t1 * t2 * t3

        # vcdn_input_spike_encoding = vcdn_input_spike_encoding.view(T, N, C * C * C)
        #####

        outputs4, _ = svcdn(vcdn_input_spike_encoding)

        outputs4 = outputs4.mean(dim=0)

        loss = loss_fn(outputs4, labels_tr)

        # l2 reg
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for model in snn_list:
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)

        loss = loss + l2_lambda * l2_reg

        losses.append(loss.item())
        loss.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

        if (epoch) % 10 == 0 and printing:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        if check_convergence(losses, window_size=5, min_gradient=0.0001):
            print(f"Converged at epoch {epoch}, Loss = {loss.item()}")
            break

    return losses


def smogotrain_v2(
    snn_list,
    inputs_list,
    targets,
    opt_list,
    sched_list,
    num_epochs=100,
    printing=True,
    l2_lambda=0.01,
):
    """
    Train the SMOGOMODEL without the back and fort conversoin from/to spikes for vcdn

    Args:
    - snn_list: list of SNNs in order: [sgcn1, sgcn2, sgcn3, svcdn]
    - inputs_list: list of input tensors in order: [H_tr_v1, H_tr_v2, H_tr_v3]
    - targets: target tensor ie labels_tr
    - lrs: list of learning rates in order: [lr1, lr2, lr3, lr4]
    - step_sizes: list of step sizes in order: [step_size1, step_size2, step_size3, step_size4]
    - gammas: list of gammas for steplr in order: [gamma1, gamma2, gamma3, gamma4]
    - num_epochs: number of epochs to train

    """

    num_view = 3
    num_class = 5

    sgcn1 = snn_list[0]
    sgcn2 = snn_list[1]
    sgcn3 = snn_list[2]
    svcdn = snn_list[3]

    H_tr_v1 = inputs_list[0]
    H_tr_v2 = inputs_list[1]
    H_tr_v3 = inputs_list[2]

    labels_tr = targets

    optimizer1 = opt_list[0]
    optimizer2 = opt_list[1]
    optimizer3 = opt_list[2]
    optimizer4 = opt_list[3]

    scheduler1 = sched_list[0]
    scheduler2 = sched_list[1]
    scheduler3 = sched_list[2]
    scheduler4 = sched_list[3]

    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        outputs1, _ = sgcn1(H_tr_v1)
        outputs2, _ = sgcn2(H_tr_v2)
        outputs3, _ = sgcn3(H_tr_v3)

        # outputs1 = outputs1.mean(dim=0)
        # outputs2 = outputs2.mean(dim=0)
        # outputs3 = outputs3.mean(dim=0)

        #### NO BACK N FORTHCONVERSION NO EINSUM
        T, N, C = outputs1.shape

        # Reshape tensors to prepare for broadcasting
        t1 = outputs1.view(T, N, C, 1, 1)
        t2 = outputs2.view(T, N, 1, C, 1)
        t3 = outputs3.view(T, N, 1, 1, C)

        vcdn_input_spike_encoding = t1 * t2 * t3

        vcdn_input_spike_encoding = vcdn_input_spike_encoding.view(T, N, C * C * C)
        #####

        outputs4, _ = svcdn(vcdn_input_spike_encoding)

        outputs4 = outputs4.mean(dim=0)

        loss = loss_fn(outputs4, labels_tr)

        # l2 reg
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for model in snn_list:
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)

        loss = loss + l2_lambda * l2_reg

        losses.append(loss.item())
        loss.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

        if (epoch) % 10 == 0 and printing:
            print(f" Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        if (
            check_convergence(losses, window_size=5, min_gradient=0.0001)
            and epoch >= num_epochs // 10
        ):
            print(f" Converged at epoch {epoch}, Loss = {loss.item()}")
            break

    return losses


def smogotest(
    snn_list, inputs_list, labels_trte_tensor, labels_te, idx_te, printing=True
):

    num_view = 3
    num_class = 5

    sgcn1 = snn_list[0]
    sgcn2 = snn_list[1]
    sgcn3 = snn_list[2]
    svcdn = snn_list[3]

    H_te_v1 = inputs_list[0]
    H_te_v2 = inputs_list[1]
    H_te_v3 = inputs_list[2]

    # labels_te = targets[idx_te]
    # labels_trte_tensor = targets

    outputs1, _ = sgcn1(H_te_v1)
    outputs2, _ = sgcn2(H_te_v2)
    outputs3, _ = sgcn3(H_te_v3)

    outputs1 = outputs1.mean(dim=0)
    outputs2 = outputs2.mean(dim=0)
    outputs3 = outputs3.mean(dim=0)

    in_list = [outputs1, outputs2, outputs3]

    accuracies_trte = []
    accuracies_te = []
    f1s_te = []
    f1s_te_macro = []
    recalls_te = []

    num_tests = 10

    for i in range(num_view):
        in_list[i] = torch.sigmoid(in_list[i])

    x = torch.reshape(
        torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
        (-1, pow(num_class, 2), 1),
    )
    x = torch.reshape(
        torch.matmul(x, in_list[2].unsqueeze(1)),
        (-1, pow(num_class, 3)),
    )

    vcdn_input_spike_encoding = spikegen.rate(x, 100)

    for i in range(num_tests):

        outputs4, _ = svcdn(vcdn_input_spike_encoding)

        accuracy = (
            torch.argmax(outputs4.mean(dim=0), dim=1) == labels_trte_tensor
        ).sum().item() / labels_trte_tensor.size(0)
        accuracies_trte.append(accuracy)

        accuracy_te = (
            torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1) == labels_te
        ).sum().item() / labels_te.size(0)
        accuracies_te.append(accuracy_te)

        # f1 scores expects cpu numpy arrays
        labels_te_np = labels_te.cpu().numpy()
        predictions_te_np = (
            torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1).cpu().numpy()
        )

        f1_te = f1_score(
            labels_te_np,
            predictions_te_np,
            average="weighted",
        )
        f1s_te.append(f1_te)

        f1_te_macro = f1_score(
            labels_te_np,
            predictions_te_np,
            average="macro",
        )
        f1s_te_macro.append(f1_te_macro)

        # f1_te = f1_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="weighted",
        # )
        # f1s_te.append(f1_te)

        # f1_te = f1_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="macro",
        # )
        # f1s_te_macro.append(f1_te)

        # recall = recall_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="weighted",
        # )
        # recalls_te.append(recall)

    # 1. F1 Score: It is a measure that combines precision (how many of the
    #    positive predictions were correct) and recall (how many of the
    #    actual positives were correctly identified) into a single score. It
    #    is the harmonic mean of precision and recall.

    # 2. Recall Score: It measures the proportion of actual positive
    #    instances that were correctly identified as positive. It helps to
    #    evaluate how well a model can identify all the relevant instances.
    if printing:
        print(f"Over 10 test runs:")
        print(f"- Accuracy on train+test set  : {sum(accuracies_trte) / num_tests}")
        print(f"- Accuracy on test set        : {sum(accuracies_te) / num_tests}")
        print(f"- F1 on test set              : {sum(f1s_te) / num_tests}")
        print(f"- F1 macro on test set        : {sum(f1s_te_macro) / num_tests}")
        # print(f"- Recall on test set          : {sum(recalls_te) / num_tests}")

    return sum(accuracies_te) / num_tests


def smogotest_v2(
    snn_list, inputs_list, labels_trte_tensor, labels_te, idx_te, printing=True
):

    num_view = 3
    num_class = 5

    sgcn1 = snn_list[0]
    sgcn2 = snn_list[1]
    sgcn3 = snn_list[2]
    svcdn = snn_list[3]

    H_te_v1 = inputs_list[0]
    H_te_v2 = inputs_list[1]
    H_te_v3 = inputs_list[2]

    # labels_te = targets[idx_te]
    # labels_trte_tensor = targets

    outputs1, _ = sgcn1(H_te_v1)
    outputs2, _ = sgcn2(H_te_v2)
    outputs3, _ = sgcn3(H_te_v3)

    # outputs1 = outputs1.mean(dim=0)
    # outputs2 = outputs2.mean(dim=0)
    # outputs3 = outputs3.mean(dim=0)

    # in_list = [outputs1, outputs2, outputs3]

    accuracies_trte = []
    accuracies_te = []
    f1s_te = []
    f1s_te_macro = []
    recalls_te = []

    num_tests = 10

    #### NO BACK N FORTHCONVERSION NO EINSUM
    T, N, C = outputs1.shape

    # Reshape tensors to prepare for broadcasting
    t1 = outputs1.view(T, N, C, 1, 1)
    t2 = outputs2.view(T, N, 1, C, 1)
    t3 = outputs3.view(T, N, 1, 1, C)

    vcdn_input_spike_encoding = t1 * t2 * t3

    vcdn_input_spike_encoding = vcdn_input_spike_encoding.view(T, N, C * C * C)
    #####

    for i in range(num_tests):

        outputs4, _ = svcdn(vcdn_input_spike_encoding)

        accuracy = (
            torch.argmax(outputs4.mean(dim=0), dim=1) == labels_trte_tensor
        ).sum().item() / labels_trte_tensor.size(0)
        accuracies_trte.append(accuracy)

        accuracy_te = (
            torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1) == labels_te
        ).sum().item() / labels_te.size(0)
        accuracies_te.append(accuracy_te)

        # f1 scores expects cpu numpy arrays
        labels_te_np = labels_te.cpu().numpy()
        predictions_te_np = (
            torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1).cpu().numpy()
        )

        f1_te = f1_score(
            labels_te_np,
            predictions_te_np,
            average="weighted",
        )
        f1s_te.append(f1_te)

        f1_te_macro = f1_score(
            labels_te_np,
            predictions_te_np,
            average="macro",
        )
        f1s_te_macro.append(f1_te_macro)

        # f1_te = f1_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="weighted",
        # )
        # f1s_te.append(f1_te)

        # f1_te = f1_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="macro",
        # )
        # f1s_te_macro.append(f1_te)

        # recall = recall_score(
        #     labels_te,
        #     torch.argmax(outputs4.mean(dim=0)[idx_te], dim=1),
        #     average="weighted",
        # )
        # recalls_te.append(recall)

    # 1. F1 Score: It is a measure that combines precision (how many of the
    #    positive predictions were correct) and recall (how many of the
    #    actual positives were correctly identified) into a single score. It
    #    is the harmonic mean of precision and recall.

    # 2. Recall Score: It measures the proportion of actual positive
    #    instances that were correctly identified as positive. It helps to
    #    evaluate how well a model can identify all the relevant instances.
    if printing:
        print(f"Over 10 test runs:")
        print(f"- Accuracy on train+test set  : {sum(accuracies_trte) / num_tests}")
        print(f"- Accuracy on test set        : {sum(accuracies_te) / num_tests}")
        print(f"- F1 on test set              : {sum(f1s_te) / num_tests}")
        print(f"- F1 macro on test set        : {sum(f1s_te_macro) / num_tests}")
        # print(f"- Recall on test set          : {sum(recalls_te) / num_tests}")

    return sum(accuracies_te) / num_tests


if __name__ == "__main__":
    pass
