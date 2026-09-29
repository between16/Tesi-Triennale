import os
import shutil
from sgfmill import sgf

# Path for games folder
Source = "./games" 

# 2. Destination folder for 9x9 games
destination = "./9x9_games"

os.makedirs(destination, exist_ok=True)

count_found = 0
count_copied = 0

print(f"Source: {Source} ...")

# os.walk() to traverse the directory tree
for root, dirs, files in os.walk(Source):
    for file in files:
        if file.endswith(".sgf") or file.endswith(".SGF"):
            filepath = os.path.join(root, file)
            count_found += 1
            
            try:
                with open(filepath, "rb") as f:
                    game = sgf.Sgf_game.from_bytes(f.read())
                    
                # check if the game is 9x9
                if game.get_size() == 9:
                    count_copied += 1
                    # Create a new name for the copied file
                    new_name = f"game_9x9_{count_copied:05d}.sgf"
                    dest_path = os.path.join(destination, new_name)
                    
                    shutil.copy2(filepath, dest_path)
            except Exception:
                # any error in reading or processing the SGF file will be ignored
                continue

print(f"File SGF analized: {count_found}")
print(f"9x9 Games in '{destination}': {count_copied}")