from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt


def generate_gradcam(model, target_layers, images, labels, rgb_imgs):
    results = []
    cam = GradCAM(model=model, target_layers=target_layers)

    for image, label, np_image in zip(images, labels, rgb_imgs):
        targets = [ClassifierOutputTarget(label.item())]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(
            input_tensor=image.unsqueeze(0), targets=targets, aug_smooth=True
        )

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            np_image / np_image.max(), grayscale_cam, use_rgb=True
        )
        results.append(visualization)
    return results


def visualize_gradcam(misimgs, mistgts, mispreds, classes):
    fig, axes = plt.subplots(len(misimgs) // 2, 2)
    fig.tight_layout()
    for ax, img, tgt, pred in zip(axes.ravel(), misimgs, mistgts, mispreds):
        ax.imshow(img)
        ax.set_title(f"{classes[tgt]} | {classes[pred]}")
        ax.grid(False)
        ax.set_axis_off()
    plt.show()
