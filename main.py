import numpy as np
import toml
import torch
import torch.nn as nn


from models.resnet import ResNet18
from utils.data import train_dataset, test_dataset
from utils.training import train, test

from utils.config import config
from utils.common import (
    find_lr,
    one_cycle_lr,
    show_model_summary,
    show_img_grid,
    show_random_images,
    lossacc_plots,
    lr_plots,
    get_misclassified,
    plot_misclassified,
)
from utils.gradcam import generate_gradcam, visualize_gradcam

trainloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=config.data.batch_size,
    shuffle=config.data.shuffle,
    num_workers=config.data.num_workers,
)

testloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=config.data.batch_size,
    shuffle=config.data.shuffle,
    num_workers=config.data.num_workers,
)

images, labels = next(iter(trainloader))

print(f"Sample Image Shape: {images[0].shape, labels[0]}")
print(f"Target Classes: {train_dataset.classes}")
show_random_images(data_loader=trainloader)
show_img_grid(images[25:30])


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ResNet18().to(device)
show_model_summary(model, config.data.batch_size)


optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
)
criterion = nn.CrossEntropyLoss()


find_lr(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    trainloader=trainloader,
    numiter=config.training.lrfinder.numiter,
    endlr=config.training.lrfinder.endlr,
    startlr=config.training.lrfinder.startlr,
)

epochs = config.training.epochs
maxlr = 1.33e-02
scheduler = one_cycle_lr(
    optimizer=optimizer, maxlr=maxlr, steps=len(trainloader), epochs=epochs
)

results = dict(trainloss=[], trainacc=[], testloss=[], testacc=[], epoch=[], lr=[])

for epoch in range(1, epochs + 1):
    print(f"Epoch: {epoch}")

    batch_trainloss, batch_trainacc, lrs = train(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=trainloader,
        scheduler=scheduler,
    )

    results["trainloss"].append(np.mean(batch_trainloss))
    results["trainacc"].append(np.mean(batch_trainacc))

    testloss, testacc = test(
        model=model, device=device, criterion=criterion, test_loader=testloader
    )
    results["testloss"].append(testloss)
    results["testacc"].append(testacc)

    results["lr"].extend(lrs)
    results["epoch"].append(epoch)


lossacc_plots(results)

lr_plots(results, length=len(trainloader) * epochs)

misimgs, mistgts, mispreds = get_misclassified(model, testloader, device, mis_count=10)

plot_misclassified(misimgs, mistgts, mispreds, train_dataset.classes)
# Actual | Predicted

target_layers = [model.layer3[-1]]  # 8x8
rgb_imgs = [(img / img.max()).permute(1, 2, 0).cpu().numpy() for img in misimgs]
cam_images = generate_gradcam(
    model=model,
    target_layers=target_layers,
    images=misimgs,
    labels=mistgts,
    rgb_imgs=rgb_imgs,
)

visualize_gradcam(cam_images, mistgts, mispreds, train_dataset.classes)
# Actual | Predicted
