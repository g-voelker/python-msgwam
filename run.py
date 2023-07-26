import argparse

from lib.config import load_config
from lib.integrate import integrate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    parser.add_argument('output_path', type=str, help='where to save output')

    args = parser.parse_args()
    load_config(args.config_path)
    integrate().to_netcdf(args.output_path)
