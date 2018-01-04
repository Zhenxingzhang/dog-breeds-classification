import os
from src.common import paths


if __name__ == "__main__":

    with open(paths.TEST_CSV_RECORDS, 'w') as writer:
        for image_file in [f for f in os.listdir(os.path.join(paths.KAGGLE_DATA_DIR, "test"))]:
            full_path = os.path.join(paths.KAGGLE_DATA_DIR, "test", image_file)
            writer.write('{},{}\n'.format(os.path.abspath(full_path), image_file.split(".")[0]))

    print("Finish test csv file.")

