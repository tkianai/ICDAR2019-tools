"""This is helper function for visualizing
====
Features:
> training loss
> validation mAP
"""

import os
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training loss!")
    parser.add_argument('--log-file', default=None)
    parser.add_argument('--save-name',  default=None)
    return parser.parse_args()


def parse_log(args):

    with open(args.log_file) as r_obj:
        epochs = []
        iters = []
        losses = []
        segm_map = []
        for line in r_obj:
            line = line.strip()
            if line and 'eta' in line:
                iters_str = line.split(' ')[6].split(']')
                epoch = int(iters_str[0][1:])
                max_iters = int(iters_str[1].split('/')[-1])
                itr = (epoch - 1) * max_iters +  int(iters_str[1].split('/')[0][1:])
                iters.append(itr)
                losses.append(float(line.split(' ')[-1]))
            if line and 'segm_mAP' in line:
                iters_str = line.split(' ')[6].split(']')
                epoch = int(iters_str[0][1:])
                epochs.append(epoch)
                segm_map.append(float(line.split(' ')[-1]))

    save_dir = os.path.dirname(args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.clf()
    plt.plot(iters, losses)
    plt.xlim(0, max(iters))
    plt.ylim(0, max(losses))
    plt.title('Training Losses')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(args.save_name + '.loss.png', dpi=400)

    plt.clf()
    plt.plot(epochs, segm_map)
    plt.xlim(0, max(epochs))
    plt.ylim(0, max(segm_map))
    plt.title('Validation mAP')
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('mAP')
    plt.savefig(args.save_name + '.mAP.png', dpi=400)


if __name__ == "__main__":

    args = parse_args()
    parse_log(args)


"""Notes about validation / rects
Epoch |  mAP(bbox)  |  mAP(segm)
---   |  ---        |  ---
1     |     0.554    |  0.536
2     |     0.596    |  0.572
3     |     0.605    |  0.586
4     |     0.610    |  0.592
5     |     0.620    |  0.601
6     |     0.612    |  0.588
7     |     0.624    |  0.599
8     |     0.616    |  0.596
9     |     0.619    |  0.597
"""
