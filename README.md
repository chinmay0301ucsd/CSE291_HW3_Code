# NeRF implmementation for CSE 291 

Refer to the config files in the config folder for all the hyperparameters, data paths etc to specify. 

To run without hierarchical sampling - 

```
python3 train_hw3.py --config_path configs/hw3.yaml 
```

I have integrated tensorboard in this implementation. Start it in a separate implementation. 

```
tensorboard --logir=logs/
```

To run with hierarchical sampling - 

```
python3 train_hw3_hierarchical.py --config_path configs/hw3_hier.yaml 
```