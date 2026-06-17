import torch
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

from cv_training import BoundingBoxDataset, get_resnet50_model, apply_nms
from cv_plotting import generate_label_colormap, generate_label_text_colormap


# TODO: Factor out plotting code, and eliminate duplicate code from within cv_plotting
def single_image_classify(dataset, idx: int = 0, plot_transformed=False):
    # NOTE: Index parameter is left for convenience if wanting to look at specific images in a large
    # dataset, but the intended use of this function is single image datasetsm hence the default.

    model = get_resnet50_model(dataset)
    state_dict = torch.load("models/finetuned_wellplate_model.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    img, target_true = dataset[idx]
    img_tensor = T.ToPureTensor()(img)
    img_batch = img_tensor.unsqueeze(0)
    device = "cpu"
    model.to(device)
    img_batch = img_batch.to(device)

    with torch.no_grad():
        target_pred = model(img_batch)[0]
    
    boxes = target_pred['boxes']   # (M,4)
    scores = target_pred['scores']
    labels = target_pred['labels']
    score_thresh = 0.35  # 0.3-0.5 recommended
    # NOTE: Overlap filter only applies to bounding boxes of the same class unless batched=False
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh)
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh,
                                      batched=False)

    if not plot_transformed:
        img_raw, _ = dataset.get_non_transformed_item(idx)
        boxes_rescaled = recover_original_image_dimensions(boxes, F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        target_true["boxes"] = recover_original_image_dimensions(target_true["boxes"], F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        img = img_raw
        boxes = boxes_rescaled

    label_cmap = generate_label_colormap(dataset)
    label_text_cmap = generate_label_text_colormap(label_cmap)

    output_image = img
    # class_names_true = [dataset.classes[i-1] for i in target_true["labels"]]
    # output_image = draw_bounding_boxes(
    #     img, target_true["boxes"],
    #     colors="blue", width=1, font="Arial", font_size=12)
    class_names_pred = [dataset.classes[i-1] for i in labels]
    class_names_w_score = [f"{class_pred}: {score:.3f}" for (class_pred, score) in zip(class_names_pred, scores)]
    output_image = draw_bounding_boxes(
        output_image, boxes, labels=class_names_w_score, 
        colors=[label_cmap[label] for label in class_names_pred], width=3, font="Arial", font_size=20,
        fill_labels=True, label_colors=[label_text_cmap[label] for label in class_names_pred])

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


