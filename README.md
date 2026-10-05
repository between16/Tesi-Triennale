# Goal
The goal is to test if Tiny Recursive Model (TRM) is suitable for solving deterministic problems like playing Go, changing only what is necessary and keeping intact the core recursive architecture.

The training is supervised and is made on the 9x9 version of the game, which can also be 19x19.

# Setup
In order to replicate the project on your machine you need to install all the necessary libraries and you can do that in two ways:

1. If you use pip :
```bash
pip install -r requirements.txt
```
2. if you use uv:
```bash
uv sync 
```

# Dataset
The dataset source is: https://homepages.cwi.nl/~aeb/go/games/

# Changes Made:
Below you can find the list of all the changes made to the original TRM:
## Added Repository 
``` bash
TRM/9x9_go
```
This is a compleatly new folder that cointains the raw dataset (in sgf format) from which i selected only the 9x9 games with the `extract_9x9.py` file. Also the folder stores the final datasets (training, val adn test) as numpy arrays are achived through the `preprocessing.py`.  

## Added Files
### TRM

``` bash
TRM/go_preprocessing.py
```
The goal of this module is to create the final datasets as numpy arrays already split into training, validation and test.

In this case we are creating two tensor of shape X = [SIZE, CHANNELS, 9, 9] and y = [SIZE] with SIZE being the actual number of elements, CHANNELS the number a matrix's layers (7) that carry information about the game and then the size of the board columns and rows.

Therefore in output we are going to have 3 dataset (.npz) files: one for training, one for validation and test stored in TRM/9x9_go/go_dataset.

``` bash
TRM/go_dataset.py
```
This file is dedicated to shrink each individual dataset into smaller batches as well as to perform the shuffling, random indexing and the loading onto the GPU.

The output are still two diffrent numpy arrays that have the following shape: X = [BATCH, CHANNELS, 9, 9] and y = [BATCH]

``` bash
TRM/micro_train.py
```
This file is just short training in order to see if everything is working fine, if everything properly connected and there is no overfitting before launching the real training that will take longer. 

``` bash
TRM/train.py
```
This file brings together all the parts necessary for the training and launches is it. 

``` bash
TRM/game_env.py
```
This file implements a class that handles the following:
- Storing the board state and the history of the last 3 turns (required by the `GoEncoder`).
- Checking for legal moves to generate the notorious Legal Move Mask.
- Managing turn transitions and "Pass" moves.
- Calculating the Win Condition (Score) using Tromp-Taylor rules (Area Scoring) when the game ends.
It knows the rules of Go, calculates legal moves, and declares the winner at the end of the game.

``` bash
TRM/agents.py
```
This file implements two classes: one for the player (HumanAgent) and one for the model (TRMagent) that interface with the arbiter. This way, the system makes no distinction regarding who is playing; it simply requests a move from the agent whose turn it is.

- TRMAgent (agents.py): Knows the neural network, reads the board state, and decides on the best move while filtering out illegal ones.
- HumanAgent (agents.py): Receives user clicks from the interface.

``` bash
TRM/app.py
```
This file is responsible for the GUI using the pygame library. I implemented few game modes: Player (eather white or black) vs Model or Model vs Model.
You can choose at which speed make the model play and even stop it (only Model vs Model match). 


### TRM/models
``` bash
TRM/models/go_encoder.py
```
This file performes the necessary encoding of the dataset in order to feed it to the neural networ that being a Transformer like requires a specific input data format. 

In oder to change the shape while keeping the necessary informations about the whole board condition and not loosing the space informations we perform two importat preliminary step: one of them being convolution (3x3 + padding, to capture the relationship between near cells as well as ensure no information is lost even on the edges) and the second one positional embedding (make make the network learn that in which area of the board is playing is important)

Output is in following shape: [Batch, Sequency_Lenght, Hidden_size], ready to be fed into the trm core.

``` bash
TRM/models/go_policy_head.py
```
This file is necessary in order to create the logits that map moves. It's job is to take the output of the TRM core and translate it into a valid position on the board.

So the output is a tensor of shape [BATCH, 82] (81 moves + PASS move)

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