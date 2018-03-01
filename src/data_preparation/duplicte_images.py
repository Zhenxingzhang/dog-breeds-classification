import imagehash
from PIL import Image


def hash_file(file_):
    try:
        img = Image.open(file_)

        # file_size = get_file_size(file)
        # image_size = get_image_size(img)
        # capture_time = get_capture_time(img)

        # 0 degree hash
        hashes_ = str(imagehash.phash(img))

        print("\tHashed {}".format(file_))
        return hashes_
    except OSError:
        print("\tUnable to open {}".format(file_))
        return None


def csv_to_hash(csv_input):
    with open(csv_input) as f:
        lines = f.readlines()

    hashes_ = {}
    for line in lines:
        img_path_ = str(line.split(',')[0])
        img_hash = hash_file(img_path_)
        if img_hash is not None:
            hashes_[img_hash] = img_path_
    return hashes_


if __name__ == '__main__':
    stanford_csv = "/data/dog_breeds/stanford_ds/stanford.csv"
    kaggle_csv = "/data/dog_breeds/kaggle/train.csv"

    kaggle_non_dup_csv = "/data/dog_breeds/kaggle/train_no_dups.csv"

    stanford_hashes = csv_to_hash(stanford_csv)

    dup_count = 0

    with open(kaggle_csv) as f:
        lines = f.readlines()

    with open('data/duplicates.csv', 'w') as csvfile, open(kaggle_non_dup_csv, 'w') as no_dup_csv:
        for line in lines:
            img_path = str(line.split(',')[0])
            img_hash = hash_file(img_path)

            if img_hash in stanford_hashes:
                csvfile.write("{},{}\n".format(img_hash, stanford_hashes[img_hash]))
                dup_count += 1
            else:
                no_dup_csv.write(line)

    print(dup_count)
