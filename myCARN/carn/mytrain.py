import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from mysolver import Solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="carn")  # 此处增加了default=‘carn'
    parser.add_argument("--ckpt_name", type=str, default='carn')  # 此次增加了default=’carn‘

    parser.add_argument("--print_interval", type=int, default=1000)  # 1000被改为了10
    parser.add_argument("--train_data_path", type=str,
                        default="/home/mist/ASR dataset/Visdrone_train_.h5")  # 该处被置换为绝对路径
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint_MYASR")
    parser.add_argument("--sample_dir", type=str,
                        default="sample/")

    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=0)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--group", type=int, default=1)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)  # batchsize源代码中是default=64
    parser.add_argument("--max_steps", type=int, default=140000)  # 200000被改为1000
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    parser.add_argument("--loss_fn", type=str,
                        choices=["MSE", "L1", "SmoothL1"], default="L1")

    return parser.parse_args()


def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    solver = Solver(net, cfg)
    solver.fit()


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
