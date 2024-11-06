# SMOGONET

We present SMOGONET (Spiking Multi-Omics Graph cOnvolutional NETwork), an
adaptation of MOGONET that incorporates principles from spiking neural networks
as a spiking framework for multi-omics data integration and classification, with
a specific focus on breast cancer diagnosis.

## REPOSITORY STRUCTURE:

- `SMOGONET/`: main folder containing the code for the SMOGONET model.
- `requirements.txt`: file containing the dependencies needed to run the code.
- `reference/`: folder containing referecne papers.
- `dumpdrawer/`: folder containing initial testing codes and files to get accustomed to the material.
- `MOGONET/`: folder containing original MOGONET implementation by Wang et al.
- `SpikingGCN/`: folder containing code from the Spiking GCN by Zhu et al.

## USAGE:

The code has been developed using python 3.96, no other versions have been tested.

Dependencies are listed in the requirements.txt file. To install them, run:

```bash
pip install -r requirements.txt
```

#### Training SMOGONET:

A full training run (until convergence) can be performed running `SMOGONET/run.py` (_`-W ignore`_ to suppress some warnings from orginal implementation of MOGONET).

Average accuracy, F1 and F1-macro scores over 10 run are printed at the end of the training.

```bash
python -W ignore SMOGONET/run.py
```

#### Training SMOGONET with EA initialization:

A full run of the EA for model parameters initialization can be performed running `SMOGONET/run_ea.py`.

A full training run of the best model found by the EA is then performed and the average accuracy, F1 and F1-macro scores over 10 run are printed at the end of the training.

```bash
python -W ignore SMOGONET/run_ea.py
```

#### Notebooks:

- **`SMOGONET/test_joined.ipynb`**: notebook to train the joined architecture, 100 epochs pretraining for the SGCNs and 500 epochs for the SVCDN. The best model is saved in the `/models` folder and can be loaded to just perform testing run.
- **`SMOGONET/test_ea.ipynb`**: notebook to perform EA for model parameters initialization, the best model the goes through a full training run like in `test_joined`.

## AUTHORS:

- [Gabriele Quaranta](https://github.com/gabriquaranta)
- [Giulio Maselli](https://github.com/giuliomsl)
- [Mattia Pecce](https://github.com/MattiaPecce)
- [Alperen Bitirgen](https://github.com/bitirgenalperen)

## REFERENCES:

[Wang et al, "MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification", 2021.](https://doi.org/10.1038/s41467-021-23774-w)

- https://github.com/txWang/MOGONET

[Zulun Zhu et al, "Spiking Graph Convolutional Networks," 2022.](https://arxiv.org/abs/2205.02767)

- https://github.com/ZulunZhu/SpikingGCN

# TODOS:

**Documentation:**

- [x] Update readme
- [x] Project report
- [ ] Presentation

**Utils:**

- [x] Modify 'compute_spiking_node_representation' to have the encoding function as parameter
- [x] Clean imported utils from original mogonet

**Models Architecture:**

- [x] Use the SGCN spiking ouput as they are for SVCDN instead of dec->ccdt->enc

**Training / Testing:**

- [x] Define train/test functions in own .py not ipynb for single model
- ~~[ ] Rn all the data is passed at once, add batch processing~~ (original MOGONET uses full batch)
- [x] Save/load single models
- [x] Save best ea model checkpoints
- [x] Make .py to run joined from terminal
- [x] Make .py to run ea from terminal

**Possible Experiments:**

- [x] EA for model params ?
