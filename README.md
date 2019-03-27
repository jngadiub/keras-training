# keras-training

## Installation
Install `miniconda2` by sourcing `install_miniconda.sh` in your home directory. Log out and log back in after this.
```bash
cp install_miniconda.sh ~/
cd ~
source install_miniconda.sh
```

Install the rest of the dependencies:
```bash
cd ~/keras-training
source install.sh
```

Each time you log in set things up:
```bash
source setup.sh
```

## Conversion of data
All of the data ntuple files are available here: https://cernbox.cern.ch/index.php/s/AgzB93y3ac0yuId

To add the truth values and flatten the trees (you can skip this step)
```bash
cd ~/keras-training/convert
python addTruth.py -t t_allpar \
../data/processed-pythia82-lhc13-*-pt1-50k-r1_h022_e0175_t220_nonu.root
```

To `hadd` these files and convert from `TTree` to `numpy array` with
random shuffling (you can skip this step)
```bash
hadd -f \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.root \
../data/processed-pythia82-lhc13-*-pt1-50k-r1_h022_e0175_t220_nonu_truth.root
python convert.py -t t_allpar_new \
../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.root
```

You can also copy this file directly from: https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v

## Training and evaluation
To run a simple training with L1 regularization (lamba = 1e-4):
```bash
cd ~/keras-training/train
python train.py -t t_allpar_new \
	-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
	-c train_config_threelayer.yml \
	-o train_simple_l10p0001/
```

and evaluate the training:
```bash
python eval.py -t t_allpar_new \
	-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
	-m train_simple/KERAS_check_best_model.h5 \
	-c train_config_threelayer.yml \
	-o eval_simple_l10p0001/
```

## Pruning and retraining
To prune the trained model by removing weights below a certain
percentile (relative weight percentile < 32.7%):
```bash
mkdir -p prune_simple_l10p0001_33perc
python prune.py -m train_simple_l10p0001/KERAS_check_best_model.h5 \
	-o prune_simple_l10p0001_33perc/trained_model_33perc.h5 \
	--relative-weight-percentile 32.7
```

and evaluate the pruned model:
```bash
python eval.py -t t_allpar_new \
	-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
	-m prune_simple_l10p0001_33perc/pruned_model.h5 \
	-c train_config_threelayer.yml \
	-o eval_simple_l10p0001_33perc/
```

To retrain the pruned model (keeping the pruned weights fixed to 0)
with L1 regularization (lamba = 1e-4):
```bash
python retrain.py -t t_allpar_new \
	-i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
	-o retrain_simple_l10p0001_33perc \
	-m prune_simple_l10p0001_33perc/pruned_model.h5 \
	-c train_config_threelayer.yml \
	-d prune_simple_l10p0001_33perc/pruned_model_drop_weights.h5
```

and prune again (this time 48 percent of the weights):
```bash
mkdir -p prune_simple_l10p0001_48perc
python prune.py -m retrain_simple_l10p0001_33perc/KERAS_check_best_model.h5 \
       --relative-weight-percentile 47.5 \
       -o prune_simple_l10p0001_48perc/pruned_model.h5
```
	   
and evalute the pruned model (2nd iteration):
```bash
python eval.py -t t_allpar_new \
       -i ../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z \
       -m prune_simple_l10p0001_48perc/pruned_model.h5 \
       -c train_config_threelayer.yml \
       -o eval_simple_l10p0001_48perc/
```

This procedure can be repeated as done in [train/train_prune_eval_retrain.sh](train/train_prune_eval_retrain.sh).
