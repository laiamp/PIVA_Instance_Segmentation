import yaml
from mask_r2 import MaskRCNN2


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = read_yaml('config.yaml')
    if config['model'] == 'mask_r2':
        model = MaskRCNN2()



if __name__ == "__main__":
    main()