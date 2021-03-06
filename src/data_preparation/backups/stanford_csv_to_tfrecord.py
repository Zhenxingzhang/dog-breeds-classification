import os
import xml.etree.ElementTree
from src.common import consts
from src.data_preparation import dataset
from src.common import paths
from src.data_preparation.backups.tf_record_utils import *


def parse_annotation(path):
    xml_root = xml.etree.ElementTree.parse(path).getroot()
    object = xml_root.findall('object')[0]
    name = object.findall('name')[0].text.lower()
    bound_box = object.findall('bndbox')[0]

    return {
        'breed': name,
        'bndbox_xmin': bound_box.findall('xmin')[0].text,
        'bndbox_ymin': bound_box.findall('ymin')[0].text,
        'bndbox_xmax': bound_box.findall('xmax')[0].text,
        'bndbox_ymax': bound_box.findall('ymax')[0].text
    }


def parse_image(breed_dir_, filename):
    path = os.path.join(images_root_dir, breed_dir_, filename + '.jpg')
    img_raw = open(path, 'r').read()

    return img_raw


def build_stanford_example(img_raw, inception_output, one_hot_label, annotation):
    example_ = tf.train.Example(features=tf.train.Features(feature={
        'label': bytes_feature(annotation['breed'].encode()),
        consts.IMAGE_RAW_FIELD: bytes_feature(img_raw),
        consts.LABEL_ONE_HOT_FIELD: float_feature(one_hot_label),
        consts.INCEPTION_OUTPUT_FIELD: float_feature(inception_output)}))

    return example_


def build_stanford_images(img_raw, label):
    example_ = tf.train.Example(features=tf.train.Features(feature={
        consts.IMAGE_RAW_FIELD: bytes_feature(img_raw),
        consts.LABEL_ONE_HOT_FIELD: int64_feature(label)
    }))

    return example_


if __name__ == '__main__':
    images_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Images')
    annotations_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Annotation')

    # one_hot_encoder, _ = dataset.one_hot_label_encoder()
    #
    # with tf.Graph().as_default(), \
    #         tf.Session().as_default() as sess, \
    #         tf.python_io.TFRecordWriter(paths.STANFORD_DS_TF_RECORDS,
    #                                     tf.python_io.TFRecordCompressionType.NONE) as writer:
    #
    #     incept_model = inception.inception_model()
    #
    #     def get_inception_ouput(img):
    #         inception_output = incept_model(sess, img).reshape(-1).tolist()
    #         return inception_output
    #
    #
    #     for breed_dir in [d for d in os.listdir(annotations_root_dir) if not d.startswith('.')]:
    #         print(breed_dir)
    #         for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
    #             # print(annotation_file)
    #             annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
    #             one_hot_label = one_hot_encoder([annotation['breed']]).reshape(-1).tolist()
    #             image = parse_image(breed_dir, annotation_file)
    #             example = build_stanford_example(image, get_inception_ouput(image), one_hot_label, annotation)
    #
    #             writer.write(example.SerializeToString())
    #
    #     writer.flush()
    #     writer.close()
    #
    #     print('Finished')

    sparse_encoder, _ = dataset.sparse_label_coder()
    with tf.Graph().as_default(), \
        tf.Session().as_default() as sess, \
        tf.python_io.TFRecordWriter(paths.STANFORD_DS_TF_RECORDS,
                                    tf.python_io.TFRecordCompressionType.NONE) as writer:

        for breed_dir in [d for d in os.listdir(annotations_root_dir) if not d.startswith('.')]:
            print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                # print(annotation_file)
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
                one_hot_label = sparse_encoder([annotation['breed']])[0]
                image = parse_image(breed_dir, annotation_file)
                example = build_stanford_images(image, one_hot_label)

                writer.write(example.SerializeToString())

        writer.flush()
        writer.close()

        print('Finished')
