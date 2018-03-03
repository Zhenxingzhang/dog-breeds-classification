import os
import xml.etree.ElementTree
from src.common import paths
from src.data_preparation import dataset
import numpy as np


def parse_annotation(path_):
    xml_root = xml.etree.ElementTree.parse(path_).getroot()
    object_ = xml_root.findall('object')[0]
    name = object_.findall('name')[0].text.lower()
    bound_box = object_.findall('bndbox')[0]
    size_ = xml_root.findall('size')[0]
    width = float(size_.find('width').text)
    height = float(size_.find('height').text)

    ymin = float(bound_box.findall('ymin')[0].text)
    xmin = float(bound_box.findall('xmin')[0].text)
    ymax = float(bound_box.findall('ymax')[0].text) / (height+1)
    xmax = float(bound_box.findall('xmax')[0].text) / (width+1)

    ymin = ymin / height if ymin > 0 else 0.0
    xmin = xmin / width if xmin > 0 else 0.0
    ymax = ymax / height if ymax < height else 1.0
    xmax = xmax / width if xmax < width else 1.0

    assert ymin < 1.0
    assert xmin < 1.0
    assert ymax < 1.0
    assert xmax < 1.0

    return {
        'breed': name,
        'bndbox_xmin': ymin,
        'bndbox_ymin': xmin,
        'bndbox_xmax': ymax,
        'bndbox_ymax': xmax
    }


def get_image_path(breed_dir_, filename):
    path_ = os.path.join(images_root_dir, breed_dir_, filename + '.jpg')
    return path_


if __name__ == "__main__":

    images_root_dir = os.path.join(paths.STANFORD_DATA_DIR, 'Images')
    annotations_root_dir = os.path.join(paths.STANFORD_DATA_DIR, 'Annotation')

    sparse_encoder, _, _ = dataset.sparse_label_coder()

    stanford_csv = paths.STANFORD_CSV_FILE
    print('Creating csv:{}'.format(stanford_csv))

    # paths_gen = glob.glob((os.path.join(DATA_DIR, 'train'))+'/*/images/*JPEG')
    with open(stanford_csv, 'w') as writer:
        for breed_dir in [d for d in os.listdir(annotations_root_dir) if not d.startswith('.')]:
            print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
                sparse_label = sparse_encoder([annotation['breed']])[0]
                bbox = [annotation['bndbox_ymin'],
                        annotation['bndbox_xmin'],
                        annotation['bndbox_ymax'],
                        annotation['bndbox_xmax']]
                path = get_image_path(breed_dir, annotation_file)
                writer.write('{},{},{}\n'.format(os.path.abspath(path), " ".join(str(x) for x in bbox), sparse_label))

    print("Finish writing csv file: {}".format(stanford_csv))

    train_csv = paths.TRAIN_CSV_FILE
    val_csv = paths.VAL_CSV_FILE

    np.random.seed(0)

    with open(stanford_csv) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

        print(len(lines))
        with open(train_csv, 'w') as writer:
            writer.writelines(lines[:-1200])
        with open(val_csv, 'w') as writer:
            writer.writelines(lines[-1200:])
