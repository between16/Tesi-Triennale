import numpy as np
import torch
from sgfmill import boards

class GoGameEnv:
    def __init__(self, board_size=9, komi=7.5):
        self.board_size = board_size
        self.komi = komi
        self.reset()

    def reset(self):
        """Reinizializza la scacchiera e lo storico per una nuova partita."""
        self.board = boards.Board(self.board_size)
        # Teniamo traccia di T-2, T-1 e T per l'input a 7 canali
        self.history = [None, None, self.board.copy()]
        self.current_player = 'b'
        self.consecutive_passes = 0
        self.game_over = False

    def get_legal_moves_mask(self):
        """
        Genera la Legal Move Mask [82]. 
        Indice 0-80: Caselle. Indice 81: Pass.
        Ritorna un tensore PyTorch booleano (True = mossa legale).
        """
        mask = torch.zeros(self.board_size * self.board_size + 1, dtype=torch.bool)
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board.get(r, c) is None:
                    # sgfmill controlla per noi Suicidio e Regola del Ko
                    test_board = self.board.copy()
                    try:
                        test_board.play(r, c, self.current_player)
                        mask[r * self.board_size + c] = True
                    except ValueError:
                        pass # Mossa illegale
        
        mask[81] = True # Il PASS è sempre un'azione legale
        return mask

    def step(self, move_idx):
        """
        Esegue la mossa (0-80 per caselle, 81 per PASS).
        Ritorna lo stato di game_over.
        """
        if self.game_over:
            return True

        if move_idx == 81: # PASS
            self.consecutive_passes += 1
        else:
            r = move_idx // self.board_size
            c = move_idx % self.board_size
            try:
                self.board.play(r, c, self.current_player)
                self.consecutive_passes = 0
            except ValueError as e:
                raise ValueError(f"Tentativo di mossa illegale {r},{c}: {e}")

        # Aggiorniamo lo storico scorrendo i frame all'indietro
        self.history[0] = self.history[1]
        self.history[1] = self.history[2]
        self.history[2] = self.board.copy()

        # Win Condition: due pass consecutivi
        if self.consecutive_passes >= 2:
            self.game_over = True
        else:
            # Cambio turno
            self.current_player = 'w' if self.current_player == 'b' else 'b'

        return self.game_over

    def calculate_score(self):
        """
        Algoritmo di Area Scoring (Tromp-Taylor).
        Valuta i territori a fine partita per determinare il vincitore.
        """
        black_score = 0.0
        white_score = self.komi
        visited = set()

        # Aiuto per navigare adiacenze (su, giù, sx, dx)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in visited:
                    continue

                stone = self.board.get(r, c)
                
                # Se c'è una pietra, dà direttamente 1 punto al proprietario
                if stone == 'b':
                    black_score += 1
                    visited.add((r, c))
                elif stone == 'w':
                    white_score += 1
                    visited.add((r, c))
                else:
                    # Trovata una casella vuota: eseguiamo BFS per trovare l'intero territorio vuoto
                    empty_cluster = []
                    queue = [(r, c)]
                    visited.add((r, c))
                    surrounding_colors = set()

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        empty_cluster.append((curr_r, curr_c))

                        for dr, dc in directions:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                adj_stone = self.board.get(nr, nc)
                                if adj_stone is None and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                                elif adj_stone is not None:
                                    surrounding_colors.add(adj_stone)

                    # Assegna il territorio al colore che lo circonda interamente
                    if len(surrounding_colors) == 1:
                        color = list(surrounding_colors)[0]
                        if color == 'b':
                            black_score += len(empty_cluster)
                        else:
                            white_score += len(empty_cluster)

        winner = 'Nero' if black_score > white_score else 'Bianco'
        margin = abs(black_score - white_score)
        
        return {
            "black": black_score,
            "white": white_score,
            "winner": winner,
            "margin": margin
        }