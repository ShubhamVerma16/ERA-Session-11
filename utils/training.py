from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    L1=False,
    l1_lambda=0.01,
):
    model.train()
    pbar = tqdm(train_loader)

    train_losses = []
    train_acc = []
    lrs = []

    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        if L1:
            l1_loss = 0
            for p in model.parameters():
                l1_loss = l1_loss + p.abs().sum()
            loss = loss + l1_lambda * l1_loss
        else:
            loss = loss

        train_loss += loss.item()
        train_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item():0.2f} Accuracy={100*correct/processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)
        lrs.append(scheduler.get_last_lr())

    return train_losses, train_acc, lrs


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    test_acc = 100.0 * correct / len(test_loader.dataset)

    return test_loss, test_acc
