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

    return {
        'breed': name,
        'bndbox_xmin': bound_box.findall('xmin')[0].text,
        'bndbox_ymin': bound_box.findall('ymin')[0].text,
        'bndbox_xmax': bound_box.findall('xmax')[0].text,
        'bndbox_ymax': bound_box.findall('ymax')[0].text
    }


def get_image_path(breed_dir, filename):
    path_ = os.path.join(images_root_dir, breed_dir, filename + '.jpg')
    return path_


if __name__ == "__main__":

    images_root_dir = os.path.join(paths.DATA_ROOT, 'Images')
    annotations_root_dir = os.path.join(paths.DATA_ROOT, 'Annotation')

    sparse_encoder, _, _ = dataset.sparse_label_coder()

    csv_file = paths.CSV_FILE
    print('Creating csv:{}'.format(csv_file))

    # paths_gen = glob.glob((os.path.join(DATA_DIR, 'train'))+'/*/images/*JPEG')
    with open(csv_file, 'w') as writer:
        for breed_dir in [d for d in os.listdir(annotations_root_dir) if not d.startswith('.')]:
            print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
                sparse_label = sparse_encoder([annotation['breed']])[0]
                bbox = [annotation['bndbox_xmin'],
                        annotation['bndbox_ymin'],
                        annotation['bndbox_xmax'],
                        annotation['bndbox_ymax']]
                path = get_image_path(breed_dir, annotation_file)
                writer.write('{},{},{}\n'.format(os.path.abspath(path), " ".join(str(x) for x in bbox), sparse_label))

    print("Write csv file finished!")

    # train
    train_csv = paths.TRAIN_CSV_FILE
    val_csv = paths.VAL_CSV_FILE

    np.random.seed(0)

    with open(csv_file) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

        print(len(lines))
        with open(train_csv, 'w') as writer:
            writer.writelines(lines[:-1200])
        with open(val_csv, 'w') as writer:
            writer.writelines(lines[-1200:])
