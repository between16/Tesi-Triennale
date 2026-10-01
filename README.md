# Goal
The goal is to see how much Tiny Recursive Model (TRM) can adapt to solve problems that are deterministic such as playing the Go game keeping intact the core recursive architecture.

The traing is supervised learning and is made with the simplified version 9x9.

# Setup
If you use pip :
```bash
pip install requirements.txt
```
if you use uv:
```bash
uv sync 
```

# Dataset
The source of the dataset is: https://homepages.cwi.nl/~aeb/go/games/

# Changes Made:

## Added Repository 
``` bash
TRM/9x9_go
```
This folder inclueds all the datasets (both row and preprocessed) as well as the SGF original dataset alongside the file required to extract the 9x9 games from such folder


## Added Files
### TRM

``` bash
TRM/go_preprocessing.py
```
This file takes the data from the .sgf files and we previously got and filtered in order to save only the 9x9 games and after convertining into a more suited format it changes the shape into a format that suits our problem.

 In this case we are creating two tensor of shape X = [SIZE, CHANNELS, 9, 9] and y = [SIZE] with SIZE being the actual number of elements, CHANNELS the number a matrix's layers (7) that carry information about the game and then the size of the board columns and rows.

Therefore in output we are going to have 3 dataset (.npz) files: one for training, one for validation and test.

``` bash
TRM/go_dataset.py
```
This file is dadicated to the shrinking of the full single dataset into batches as well as to performe the shuflling, random indexing and loading onto the GPU. 

The output are still two diffrent numpy vectors that have the following shape: X = [BATCH, CHANNELS, 9, 9] and y = [BATCH]

### TRM/models
``` bash
TRM/models/go_encoder.py
```
This file performe the necessary encoding of the dataset in order to feed it to the neural networ that being a Transformer like requires a specific input data format. 

In oder to change the shape while keeping the necessary informations about the whole board condition and not loosing the space informations we perform two importat preliminary step: one of them being convolution (3x3 + padding to ensure no information is lost even on the edges) and the second one positional embedding (make make the network learn that in which area of the board is playing is important)

Output is in following shape: [Batch, Sequency_Lenght, Hidden_size], ready to be fed into the trm core.

``` bash
TRM/models/go_policy_head.py
```
This file is necessary in order to create the logits that map moves, besically what it does is $\text{Forma:} \quad [\text{Batch}, 81, \text{hidden\_size}] \longrightarrow [\text{Batch}, 81, 1] \xrightarrow{\text{squeeze}} [\text{Batch}, 81]$ with the addition of a second tensor for the PASS move. 

So we have a tensor of shape [BATCH, 82] (81 moves + PASS move)

``` bash
TRM/micro_train.py
```
This file is just short training in order to see if everything is working fine and it's properly connected before launching the real training that will take longer. 

``` bash
TRM/train.py
```
This file brings together all the parts necessary for the training and launches is it. 

## Modified files 
``` bash
TRM/models/go_trm.py
```
Actually this is a new file, but it took the trm.py architecture for the trm model and just changed only what was needed in order to make it work with the new data format


# Pipeline
1. Extract data from original .sgf (offline)
2. preprocess them running the go_preprocessing.py file (offline)
3. Train the model with train.py 
4. inference on the saved model