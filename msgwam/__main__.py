import argparse
import time

from .config import load_config
from .integrate import SBDF2Integrator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    parser.add_argument('output_path', type=str, help='where to save output')

    args = parser.parse_args()
    load_config(args.config_path)

    start = time.time()
    SBDF2Integrator().integrate().to_netcdf(args.output_path)
    print(f'Integration took {(time.time() - start):.2f} s')