# ICDAR2019-tools

This repo develops the tools for ICDAR2019 competitions. You can find details [here](http://rrc.cvc.uab.es/?ch=12).

> Competitions

- [x] LSVT

[Large-scale Street View Text with Partial Labeling](http://rrc.cvc.uab.es/?ch=16)

- [x] ReCTS

[Reading Chinese Text on Signboard](http://rrc.cvc.uab.es/?ch=12)

## Usage

- Transfer the detection format from coco-style to submission style

```sh
# for rects
python format_transfer.py --dt-file task3-results/results.pkl.json --mode rects --save data/rects_task3.txt
```

- [x] rects format test passed
- [x] lsvt format test passed

- Evaluating performace of the model

```sh
# for lsvt
python eval.py --gt-file data/lsvt_val_v2.json --dt-file data/lsvt_val_v2_det.json
```

- [x] Too Slow for evaluation

