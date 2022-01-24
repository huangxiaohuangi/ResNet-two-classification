import os


def create_dataset_txt():
    for dir in os.listdir(files_path):
        # print(dir.split('_')[-1])
        for image in os.listdir(os.path.join(files_path, dir)):
            with open(txt_path, 'a+') as f:
                f.write(image + ',' + dir.split('_')[-1] + '\n')


if __name__ == '__main__':
    files_path = 'G:/water/water_data/plate51/pink//'
    txt_path = 'G:/water/water_data/plate51/pink.txt'
    create_dataset_txt()
