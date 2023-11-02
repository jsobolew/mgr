import argparse
import sys


def main(config_name):
    f = open(f"configs/experiments/{config_name}.yaml")
    lines = f.readlines()
    f.close()

    params_lines = []
    for line in lines:
        if "#" not in line:
            line = line.strip()
            line = line.replace(": ", "=")
            equals_sign = line.find("=") + 1
            line = line[:equals_sign] + '"' + line[equals_sign:] + '"'
            params_lines.append(line)

    params = " ".join(params_lines)
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name")
    args = parser.parse_args()

    params = main(args.config_name)
    print(params)
    # sys.stdout.write(params)
    # sys.exit(0)