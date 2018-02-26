import os
from src.common import paths
from src.data_preparation import dataset
from PIL import Image
import csv


if __name__ == "__main__":

    sparse_encoder, _, _ = dataset.sparse_label_coder()

    with open(paths.KG_TRAIN_CSV, 'w') as writer, open(paths.KG_CSV) as reader:
        csv_reader = csv.reader(reader)
        next(csv_reader)
        for row in csv_reader:
            sparse_label = sparse_encoder([row[-1]])[0]

            image_path = os.path.join(paths.KG_DATA_DIR, "train", row[0] + ".jpg")
            # [width, height]
            shape = Image.open(image_path).size
            if os.path.isfile(image_path):
                # [xmin, ymin, xmax, ymax], [width_min, height_min, width_max, height_max]
                bbox = [0, 0, shape[0], shape[1]]
                writer.write('{},{},{}\n'.format(image_path, " ".join(str(x) for x in bbox), sparse_label))

    with open(paths.TEST_CSV_RECORDS, 'w') as writer:
        for image_file in [f for f in os.listdir(os.path.join(paths.KAGGLE_DATA_DIR, "test"))]:
            full_path = os.path.join(paths.KAGGLE_DATA_DIR, "test", image_file)
            writer.write('{},{}\n'.format(os.path.abspath(full_path), image_file.split(".")[0]))

    print("Finish: {}".format(paths.KG_TRAIN_CSV))

