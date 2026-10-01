# Goal
The goal is to see how much Tiny Recursive Model (TRM) can adapt to solve problems that are non deterministic such as playing the Go game.

# Setup
If you use pip :
```bash
pip install requirements.txt
```
if you use uv:
```bash
uv sync 
```



# Implementation
First the model will be trained to play the simplified version of Go which is 9x9 instead of the classical 19x19. 

# Dataset
https://homepages.cwi.nl/~aeb/go/games/

# Changes Made:

## Added Repository 
``` bash
TRM/9x9_go
```
This folder inclueds all the datasets (both row and preprocessed) as well as the SGF original dataset alongside the file required to extract the 9x9 games from such folder


## Added Files
### TRM
``` bash
TRM/go_dataset.py
```
This file TODO

``` bash
TRM/go_preprocessing.py
```
This file TODO

### TRM/models
``` bash
TRM/models/go_encoder.py
```
This file TODO

``` bash
TRM/models/go_policy_head.py
```
This file TODO

``` bash
TRM/micro_train.py
```
This file TODO

``` bash
TRM/train.py
```
This file TODO

## Modified files 
``` bash
TRM/models/go_trm.py
```
Actually this is a new file, but it took the trm.py architecture for the trm model and just changed only what was needed in order to make it work with the new data format


# Pipeline
1. Extract data from original .sgf
2. preprocess them running the go_preprocessing.py file 
3. 