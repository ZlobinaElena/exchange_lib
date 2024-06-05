from pathlib import Path
from tqdm import tqdm
import torch
from transformers import TableTransformerForObjectDetection
from safetensors.torch import load_file as safetensors_load
import matplotlib.pyplot as plt
from torchvision import transforms
from segmenters.yolov8_segmenter import YOLOv8Segmenter
import pandas as pd
YOLO_WEIGHTS = "./dog_seg_model/yolo8n_doclay_full_best.pt"
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tabel_transformer_model_dir = "table_transformer"


def bbox_intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        return x1, y1, x2, y2
    else:
        return None


def object_to_crops(img, bbox, padding: int = 10):
    """
    Process the bounding boxes produced by the table detection model
    """
    bbox = [bbox.coordinates[0] - padding,
            bbox.coordinates[1] - padding,
            bbox.coordinates[2] + padding,
            bbox.coordinates[3] + padding]
    cropped_img = img.crop(bbox)
    return cropped_img.convert("RGB")


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def predict_table_cells(cropped_table, structure_model, structure_transform):
    pixel_values = structure_transform(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(DEVICE)
    # forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # update id2label to include "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
    return cells


def find_cell_boxes(cells: list[dict[str, str | float]], page_coordinates: tuple[float, float, float, float],
                    padding: int = 10):
    columns = [item['bbox'] for item in cells if item['label'] == 'table column']
    rows = [item['bbox'] for item in cells if item['label'] == 'table row']

    rows_sorted = sorted(rows, key=lambda bbox: bbox[1])

    # Sort columns by x-coordinate (left edge)
    columns_sorted = sorted(columns, key=lambda bbox: bbox[0])

    # Find cell bboxes
    split_cells = []
    for row_bbox in rows_sorted:
        row_cells = []
        for col_bbox in columns_sorted:
            cell_bbox = bbox_intersection(col_bbox, row_bbox)
            cell_bbox1 = (cell_bbox[0] + page_coordinates[0] - padding, cell_bbox[1] + page_coordinates[1] - padding,
                          cell_bbox[2] + page_coordinates[0] - padding, cell_bbox[3] + page_coordinates[1] - padding)
            row_cells.append(cell_bbox1)
        split_cells.append(row_cells)
    return split_cells


def normalizing_word_coords(page) -> list[dict]:
    normalized_words = []
    for word in page.words:
        normalized_coords = ((word.get("x0", 0) - page.shift_x) / page.scale[0],
                             (word.get("top", 0) - page.shift_y) / page.scale[1],
                             (word.get("x1", 0) - page.shift_x) / page.scale[0],
                             (word.get("bottom", 0) - page.shift_y) / page.scale[1])
        normalized_words.append({"coords": normalized_coords, "text": word["text"]})
    return normalized_words


def mapping_words_to_cells(page, cells):
    normalized_words = normalizing_word_coords(page)
    # TODO refactor this function
    words_cells = []
    eps = 0.7
    for row in cells:
        row_words = []
        for cell in row:
            cell_words = []
            for tmp in normalized_words:
                word_coords = tmp.get("coords")
                intersection_coords = bbox_intersection(cell, word_coords)
                if intersection_coords is not None:
                    area = (((intersection_coords[2] - intersection_coords[0]) *
                            (intersection_coords[3] - intersection_coords[1])) /
                            ((word_coords[2] - word_coords[0]) * (word_coords[3] - word_coords[1])))
                    if area > eps:
                        cell_words.append(tmp.get("text"))
            row_words.append(" ".join(cell_words))
        words_cells.append(row_words)
    return words_cells


def plot_cells(row_cells,
               words_cells,
               image,
               output_path: Path | str = "extracted_tables",
               file_name: str = "table_cells.png"):
    plt.figure(figsize=(32,28))
    plt.imshow(image)
    ax = plt.gca()
    for row in row_cells:
        for cell in row:
            xmin, ymin, xmax, ymax = cell[0], cell[1], cell[2], cell[3]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=1))
            plt.axis('off')

    ax1 = plt.gca()
    for w in words_cells:
        w = w['coords']
        xmin, ymin, xmax, ymax = w[0], w[1], w[2], w[3]
        ax1.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="green", linewidth=1))
        plt.axis('off')
    output_path = Path(output_path)
    plt.savefig(output_path / file_name)


def extraction_pipeline(pdf_file: Path | str, output_path: Path | str = "extracted_tables"):
    segmenter = YOLOv8Segmenter(YOLO_WEIGHTS)
    state_dict = safetensors_load(f"{tabel_transformer_model_dir}/model.safetensors")

    pages = segmenter.segment_pdf(pdf_file, (300, 301, 302, 303, 304, 308), save_images=True)
    structure_model = TableTransformerForObjectDetection.from_pretrained(tabel_transformer_model_dir,
                                                                         state_dict=state_dict)
    structure_model.to(DEVICE)
    structure_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor()
    ])
    for current_page in tqdm(pages, desc="Processing page"):
        num_of_table = 0
        for bbox in current_page.bboxes:
            if bbox.type == "Table":
                cropped_table = object_to_crops(current_page.image, bbox=bbox)
                cells = predict_table_cells(cropped_table, structure_model, structure_transform)
                cells_by_rows = find_cell_boxes(cells, bbox.coordinates)
                words_cells = mapping_words_to_cells(current_page, cells_by_rows)

                # TODO temporary solution delete after refactoring
                normalized_words = normalizing_word_coords(current_page)
                plot_cells(cells_by_rows, normalized_words, current_page.image,
                           output_path,
                           file_name=f"table_{current_page.number}_{num_of_table}.png")

                try:
                    df = pd.DataFrame(words_cells)
                    df.to_csv(output_path / f"table_{current_page.number}_{num_of_table}.csv", index=False)
                except ValueError:
                    print("Error creating DataFrame. Table is not correctly detected")
                    continue
                num_of_table += 1


if __name__ == "__main__":
    output_folder = Path("extracted_tables")
    output_folder.mkdir(exist_ok=True, parents=True)
    extraction_pipeline("annualreport-2023.pdf", output_path=output_folder)
