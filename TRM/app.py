import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

from agents import HumanAgent, TRMAgent
from game_env import GoGameEnv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 760
FPS = 60
BOARD_SIZE = 9
KOMI = 7.5

BOARD_MARGIN = 54
DEFAULT_BOARD_TOP = 72
DEFAULT_BOARD_LEFT = 34
DEFAULT_BOARD_PIXEL_SIZE = 600
PANEL_GAP = 34
MIN_WINDOW_WIDTH = 1100
MIN_WINDOW_HEIGHT = 720

# Current Go environment uses 0..80 for board points and 81 for PASS on 9x9.
PASS_INDEX = BOARD_SIZE * BOARD_SIZE

AI_DELAYS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

# Colors
BG = (31, 34, 39)
PANEL = (42, 46, 53)
PANEL_2 = (51, 56, 64)
TEXT = (238, 240, 243)
TEXT_MUTED = (170, 176, 186)
ACCENT = (84, 145, 255)
ACCENT_HOVER = (108, 163, 255)
SUCCESS = (86, 190, 125)
WARNING = (235, 183, 76)
ERROR = (224, 92, 92)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
BOARD_WOOD = (220, 181, 116)
BOARD_LINE = (66, 48, 28)
STAR = (64, 47, 28)


@dataclass
class ModeConfig:
    name: str
    black: str
    white: str
    pause_available: bool


MODE_HUMAN_BLACK = ModeConfig(
    "Human (Black) vs TRM (White)",
    "Human",
    "TRM",
    False,
)
MODE_TRM_BLACK = ModeConfig(
    "TRM (Black) vs Human (White)",
    "TRM",
    "Human",
    False,
)
MODE_AI_VS_AI = ModeConfig(
    "TRM (Black) vs TRM (White)",
    "TRM",
    "TRM",
    True,
)

MODES = [MODE_HUMAN_BLACK, MODE_TRM_BLACK, MODE_AI_VS_AI]


class Button:
    def __init__(self, rect, text, callback=None, enabled=True):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.enabled = enabled
        self.hovered = False

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        if not self.enabled or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if self.rect.collidepoint(event.pos):
            if self.callback:
                self.callback()
            return True
        return False

    def draw(self, surface, font):
        if not self.enabled:
            bg = (62, 65, 71)
            fg = (118, 122, 130)
        elif self.hovered:
            bg = ACCENT_HOVER
            fg = TEXT
        else:
            bg = PANEL_2
            fg = TEXT

        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, (79, 84, 94), self.rect, width=1, border_radius=8)

        text_surface = font.render(self.text, True, fg)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class Slider:
    def __init__(self, rect, values, initial_index=4):
        self.rect = pygame.Rect(rect)
        self.values = values
        self.index = max(0, min(initial_index, len(values) - 1))
        self.dragging = False

    @property
    def value(self):
        return self.values[self.index]

    def _index_from_x(self, x):
        x0 = self.rect.left
        x1 = self.rect.right
        if x1 <= x0:
            return 0
        ratio = (x - x0) / float(x1 - x0)
        ratio = max(0.0, min(1.0, ratio))
        return round(ratio * (len(self.values) - 1))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            knob_center = self.knob_center()
            if pygame.Vector2(event.pos).distance_to(knob_center) <= 14 or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.index = self._index_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.index = self._index_from_x(event.pos[0])
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return False

    def knob_center(self):
        x0 = self.rect.left
        x1 = self.rect.right
        if len(self.values) == 1:
            x = x0
        else:
            x = x0 + (x1 - x0) * (self.index / (len(self.values) - 1))
        return pygame.Vector2(x, self.rect.centery)

    def draw(self, surface):
        pygame.draw.line(surface, (105, 110, 120), (self.rect.left, self.rect.centery), (self.rect.right, self.rect.centery), 6)
        knob = self.knob_center()
        pygame.draw.circle(surface, ACCENT, knob, 10)
        pygame.draw.circle(surface, (221, 227, 236), knob, 10, width=2)


class TinyRecursiveGo:
    """Pygame front-end for GoGameEnv + Go agents.

    The environment remains the authority for Go rules and move legality.
    This class owns UI state, timing, rendering, and human input only.
    """

    def __init__(self, checkpoint_path):
        pygame.init()
        pygame.display.set_caption("Tiny Recursive Go")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.board_left = DEFAULT_BOARD_LEFT
        self.board_top = DEFAULT_BOARD_TOP
        self.board_pixel_size = DEFAULT_BOARD_PIXEL_SIZE
        self.side_left = 700
        self.side_width = WINDOW_WIDTH - self.side_left - 28
        self.clock = pygame.time.Clock()

        # Fonts are rebuilt only when the window scale changes. This keeps the
        # UI readable when the window is resized while avoiding unnecessary
        # font creation on every frame.
        self.font_scale = None
        self.update_fonts()

        self.checkpoint_path = Path(checkpoint_path)
        self.mode = MODE_HUMAN_BLACK

        self.env = None
        self.black_agent = None
        self.white_agent = None

        self.paused = False
        self.game_started_at = None
        self.last_move_index = None
        self.error_message = None
        self.error_until = 0.0
        self.info_message = None
        self.info_until = 0.0

        self.ai_waiting_until = 0.0
        self.ai_thinking = False

        self.running = True

        self.mode_button_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_button = Button((0, 0, 0, 0), "New Game", self.reset_game)
        self.pause_button = Button((0, 0, 0, 0), "Pause", self.toggle_pause)
        self.pass_button = Button((0, 0, 0, 0), "Pass", self.pass_move)
        self.slider = Slider((0, 0, 0, 10), AI_DELAYS, initial_index=4)

        self.update_layout()
        self.reset_game()

    # ------------------------------------------------------------------
    # Responsive layout
    # ------------------------------------------------------------------

    def update_fonts(self):
        """Scale all UI fonts with the current window size."""
        width, height = self.screen.get_size()
        scale = min(width / WINDOW_WIDTH, height / WINDOW_HEIGHT)
        scale = max(0.80, min(scale, 2.0))

        # Avoid recreating fonts when the size has not materially changed.
        scale_key = round(scale, 2)
        if self.font_scale == scale_key:
            return
        self.font_scale = scale_key

        def size(base):
            return max(12, int(round(base * scale_key)))

        self.font_title = pygame.font.SysFont(None, size(34), bold=True)
        self.font_subtitle = pygame.font.SysFont(None, size(24), bold=True)
        self.font_body = pygame.font.SysFont(None, size(21))
        self.font_small = pygame.font.SysFont(None, size(18))
        self.font_tiny = pygame.font.SysFont(None, size(16))

        # Board labels should grow a little less aggressively than the panel
        # text, otherwise they become disproportionately large on big screens.
        board_font_size = max(13, int(round(18 * scale_key)))
        self.font_board = pygame.font.SysFont(None, board_font_size, bold=True)

    def update_layout(self):
        """Recalculate responsive geometry for the board and side panel."""
        width, height = self.screen.get_size()
        self.update_fonts()

        outer = max(24, int(28 * self.font_scale))
        panel_width = min(470, max(360, int(width * 0.30)))
        board_available_width = width - outer - panel_width - PANEL_GAP - outer
        board_available_height = height - max(110, int(118 * self.font_scale))

        board_size = int(min(board_available_width, board_available_height))
        board_size = max(420, board_size)

        if board_size + panel_width + PANEL_GAP + 2 * outer > width:
            board_size = max(360, width - panel_width - PANEL_GAP - 2 * outer)

        self.board_left = outer
        self.board_top = max(48, (height - board_size) // 2)
        self.board_pixel_size = int(board_size)
        self.side_left = self.board_left + self.board_pixel_size + PANEL_GAP
        self.side_width = max(320, width - self.side_left - outer)

        # Build the panel as a vertical flow. These values are reused by draw_panel
        # so text, buttons and cards stay separated when fonts grow on large windows.
        gap = max(8, int(10 * self.font_scale))
        small_gap = max(5, int(6 * self.font_scale))
        x = self.side_left
        w = self.side_width
        y = max(28, int(30 * self.font_scale))

        title_h = self.font_title.get_height()
        subtitle_h = self.font_small.get_height()
        body_h = self.font_body.get_height()
        small_h = self.font_small.get_height()
        tiny_h = self.font_tiny.get_height()

        self.panel_title_rect = pygame.Rect(x, y, w, title_h)
        y += title_h + 2
        self.panel_subtitle_rect = pygame.Rect(x, y, w, subtitle_h)
        y += subtitle_h + gap

        self.mode_label_rect = pygame.Rect(x, y, w, tiny_h)
        y += tiny_h + small_gap
        mode_h = max(48, body_h + 18)
        self.mode_button_rect = pygame.Rect(x, y, w, mode_h)
        y += mode_h + gap

        turn_h = max(94, self.font_subtitle.get_height() + self.font_small.get_linesize() + 34)
        self.turn_rect = pygame.Rect(x, y, w, turn_h)
        y += turn_h + gap

        rules_h = max(68, tiny_h + small_h + 26)
        self.score_rect = pygame.Rect(x, y, w, rules_h)
        y += rules_h + gap

        self.speed_label_rect = pygame.Rect(x, y, w, tiny_h)
        y += tiny_h + small_gap
        self.slider.rect = pygame.Rect(x + 10, y, max(180, w - 20), max(10, int(12 * self.font_scale)))
        y += max(18, int(14 * self.font_scale)) + small_gap
        self.speed_value_rect = pygame.Rect(x, y, w, body_h)
        y += body_h + small_gap
        self.speed_labels_rect = pygame.Rect(x, y, w, tiny_h)
        y += tiny_h + gap

        message_h = max(104, tiny_h + self.font_small.get_linesize() * 3 + 34)
        self.message_rect = pygame.Rect(x, y, w, message_h)
        y += message_h + gap

        # Bottom controls are anchored to the bottom of the panel, while the
        # informational content above uses the dynamic flow.
        footer_h = tiny_h
        footer_y = height - max(30, int(34 * self.font_scale))
        button_h = max(46, body_h + 18)
        button_y = footer_y - gap - button_h

        reset_width = max(170, int(w * 0.42))
        pause_x = x + reset_width + 12
        self.reset_button.rect = pygame.Rect(x, button_y, reset_width, button_h)
        self.pause_button.rect = pygame.Rect(
            pause_x, button_y, max(120, w - reset_width - 12), button_h
        )

        pass_y = button_y - gap - button_h
        self.pass_button.rect = pygame.Rect(x, pass_y, w, button_h)

        self.footer_rect = pygame.Rect(x, footer_y, w, footer_h)

    # ------------------------------------------------------------------
    # Agent / game setup
    # ------------------------------------------------------------------

    def make_trm_agent(self):
        return TRMAgent(checkpoint_path=str(self.checkpoint_path))

    def build_agents_for_mode(self):
        if self.mode is MODE_HUMAN_BLACK:
            black = HumanAgent()
            white = self.make_trm_agent()
        elif self.mode is MODE_TRM_BLACK:
            black = self.make_trm_agent()
            white = HumanAgent()
        elif self.mode is MODE_AI_VS_AI:
            black = self.make_trm_agent()
            white = self.make_trm_agent()
        else:
            raise RuntimeError(f"Unsupported mode: {self.mode.name}")
        return black, white

    def reset_game(self):
        self.env = GoGameEnv(board_size=BOARD_SIZE, komi=KOMI)
        self.black_agent, self.white_agent = self.build_agents_for_mode()
        self.paused = False
        self.game_started_at = time.monotonic()
        self.last_move_index = None
        self.error_message = None
        self.info_message = "New game started. Black to move."
        self.info_until = time.monotonic() + 2.0
        self.ai_waiting_until = time.monotonic()
        self.ai_thinking = False

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def ai_delay(self):
        return self.slider.value

    @property
    def current_agent(self):
        return self.black_agent if self.env.current_player == "b" else self.white_agent

    @property
    def current_color_name(self):
        return "Black" if self.env.current_player == "b" else "White"

    @property
    def current_stone_color(self):
        return BLACK if self.env.current_player == "b" else WHITE

    @property
    def human_turn(self):
        return isinstance(self.current_agent, HumanAgent)

    @property
    def ai_turn(self):
        return not self.human_turn and not self.env.game_over

    def set_error(self, message):
        self.error_message = message
        self.error_until = time.monotonic() + 4.0

    def set_info(self, message, duration=2.5):
        self.info_message = message
        self.info_until = time.monotonic() + duration

    def clear_expired_messages(self):
        now = time.monotonic()
        if self.error_message and now >= self.error_until:
            self.error_message = None
        if self.info_message and now >= self.info_until:
            self.info_message = None

    # ------------------------------------------------------------------
    # Game actions
    # ------------------------------------------------------------------

    def try_play_move(self, move_index):
        """Apply a move through GoGameEnv and keep turn on illegal move."""
        if self.env.game_over:
            return False

        try:
            game_over = self.env.step(move_index)
        except ValueError as exc:
            self.set_error(f"Illegal move: {exc}")
            return False

        self.last_move_index = move_index
        self.error_message = None
        self.ai_thinking = False
        self.ai_waiting_until = time.monotonic() + self.ai_delay

        if game_over:
            self.paused = False
            self.set_info("Game over. Press New Game to start another match.", duration=60.0)
        else:
            self.set_info(f"{self.current_color_name} to move.", duration=1.2)
        return True

    def pass_move(self):
        if self.env.game_over or not self.human_turn:
            return
        self.try_play_move(PASS_INDEX)

    def toggle_pause(self):
        if self.mode is not MODE_AI_VS_AI or self.env.game_over:
            return
        self.paused = not self.paused
        if self.paused:
            self.ai_thinking = False
            self.set_info("Match paused.", duration=60.0)
        else:
            self.ai_waiting_until = time.monotonic()
            self.set_info("Match resumed.", duration=1.5)

    def cycle_mode(self):
        if self.ai_thinking:
            return
        current = MODES.index(self.mode)
        self.mode = MODES[(current + 1) % len(MODES)]
        self.reset_game()

    # ------------------------------------------------------------------
    # Board geometry / input
    # ------------------------------------------------------------------

    def board_rect(self):
        return pygame.Rect(self.board_left, self.board_top, self.board_pixel_size, self.board_pixel_size)

    def board_points_rect(self):
        return self.board_rect().inflate(-BOARD_MARGIN, -BOARD_MARGIN)

    def point_spacing(self):
        return self.board_points_rect().width / (self.env.board_size - 1)

    def point_to_screen(self, row, col):
        inner = self.board_points_rect()
        spacing = self.point_spacing()
        x = inner.left + col * spacing
        y = inner.top + row * spacing
        return int(round(x)), int(round(y))

    def screen_to_point(self, pos):
        if self.env.game_over:
            return None

        x, y = pos
        inner = self.board_points_rect()
        spacing = self.point_spacing()
        col_float = (x - inner.left) / spacing
        row_float = (y - inner.top) / spacing
        col = int(round(col_float))
        row = int(round(row_float))

        if not (0 <= row < self.env.board_size and 0 <= col < self.env.board_size):
            return None

        point_x, point_y = self.point_to_screen(row, col)
        distance = pygame.Vector2(x, y).distance_to((point_x, point_y))
        if distance > min(22.0, spacing * 0.34):
            return None

        return row, col

    def hovered_move_index(self, mouse_pos):
        point = self.screen_to_point(mouse_pos)
        if point is None:
            return None
        row, col = point
        if self.env.board.get(row, col) is not None:
            return None
        return row * self.env.board_size + col

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        self.reset_button.update(mouse_pos)
        self.pause_button.update(mouse_pos)
        self.pass_button.update(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                width = max(MIN_WINDOW_WIDTH, event.w)
                height = max(MIN_WINDOW_HEIGHT, event.h)
                if (width, height) != (event.w, event.h):
                    self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                self.update_layout()
                continue

            if event.type == pygame.QUIT:
                self.running = False
                continue

            if self.slider.handle_event(event):
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    continue
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                    self.update_layout()
                    continue
                if event.key == pygame.K_r:
                    self.reset_game()
                    continue
                if event.key == pygame.K_SPACE and self.mode is MODE_AI_VS_AI:
                    self.toggle_pause()
                    continue
                if event.key == pygame.K_p and self.human_turn:
                    self.pass_move()
                    continue

            if self.reset_button.handle_event(event):
                continue
            if self.pause_button.handle_event(event):
                continue
            if self.pass_button.handle_event(event):
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.mode_button_rect.collidepoint(event.pos):
                    self.cycle_mode()
                    continue

                if self.human_turn and not self.env.game_over and not self.ai_thinking:
                    move_index = self.hovered_move_index(event.pos)
                    if move_index is not None:
                        self.try_play_move(move_index)
                    else:
                        point = self.screen_to_point(event.pos)
                        if point is not None:
                            row, col = point
                            if self.env.board.get(row, col) is not None:
                                self.set_error("That intersection is already occupied.")

    # ------------------------------------------------------------------
    # AI update
    # ------------------------------------------------------------------

    def update_ai(self):
        if not self.ai_turn or self.paused or self.ai_thinking:
            return

        now = time.monotonic()
        if now < self.ai_waiting_until:
            return

        # The inference itself is synchronous. Pygame remains responsive between
        # moves; the only non-interactive period is the model's inference call.
        self.ai_thinking = True
        self.error_message = None

        try:
            move_index = self.current_agent.get_move(self.env)
        except Exception as exc:  # Keep GUI alive and expose the real error.
            self.ai_thinking = False
            self.set_error(f"AI error: {exc}")
            if self.mode is MODE_AI_VS_AI:
                self.paused = True
            return

        legal_mask = self.env.get_legal_moves_mask()
        if not bool(legal_mask[move_index].item()):
            self.ai_thinking = False
            self.set_error(f"TRM returned an illegal move: index {move_index}.")
            if self.mode is MODE_AI_VS_AI:
                self.paused = True
            return

        self.try_play_move(move_index)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw_text(self, text, pos, font, color=TEXT):
        self.screen.blit(font.render(text, True, color), pos)

    def draw_centered_text(self, text, rect, font, color=TEXT):
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def draw_board(self):
        rect = self.board_rect()
        pygame.draw.rect(self.screen, BOARD_WOOD, rect, border_radius=10)

        inner = rect.inflate(-BOARD_MARGIN, -BOARD_MARGIN)
        spacing = inner.width / (self.env.board_size - 1)

        # Grid
        for i in range(self.env.board_size):
            x = int(round(inner.left + i * spacing))
            y = int(round(inner.top + i * spacing))
            pygame.draw.line(self.screen, BOARD_LINE, (x, inner.top), (x, inner.bottom), 2)
            pygame.draw.line(self.screen, BOARD_LINE, (inner.left, y), (inner.right, y), 2)

        # Star points for 9x9
        star_indices = [2, 6]
        for r in star_indices:
            for c in star_indices:
                x = int(round(inner.left + c * spacing))
                y = int(round(inner.top + r * spacing))
                pygame.draw.circle(self.screen, STAR, (x, y), 5)

        # Last move marker
        if self.last_move_index is not None and self.last_move_index < PASS_INDEX:
            r = self.last_move_index // self.env.board_size
            c = self.last_move_index % self.env.board_size
            x = int(round(inner.left + c * spacing))
            y = int(round(inner.top + r * spacing))
            pygame.draw.circle(self.screen, WARNING, (x, y), 6, width=2)

        # Stones
        for r in range(self.env.board_size):
            for c in range(self.env.board_size):
                stone = self.env.board.get(r, c)
                if stone is None:
                    continue
                x = int(round(inner.left + c * spacing))
                y = int(round(inner.top + r * spacing))
                radius = max(17, int(spacing * 0.39))

                if stone == "b":
                    fill = (18, 18, 18)
                    edge = (78, 78, 78)
                else:
                    fill = (242, 242, 242)
                    edge = (125, 125, 125)

                pygame.draw.circle(self.screen, fill, (x, y), radius)
                pygame.draw.circle(self.screen, edge, (x, y), radius, width=2)

                if self.last_move_index == r * self.env.board_size + c:
                    marker = WARNING if stone == "w" else (235, 235, 235)
                    pygame.draw.circle(self.screen, marker, (x, y), 5)

        # Coordinates around the board
        for i in range(self.env.board_size):
            label = chr(ord("A") + i)
            x = int(round(inner.left + i * spacing))
            self.draw_centered_text(label, pygame.Rect(x - 12, rect.bottom - 36, 24, 20), self.font_board, BOARD_LINE)

            row_number = str(self.env.board_size - i)
            self.draw_centered_text(row_number, pygame.Rect(rect.left + 7, int(round(inner.top + i * spacing)) - 10, 24, 20), self.font_board, BOARD_LINE)

        # Hover preview for human turns
        if self.human_turn and not self.env.game_over:
            move_idx = self.hovered_move_index(pygame.mouse.get_pos())
            if move_idx is not None:
                r = move_idx // self.env.board_size
                c = move_idx % self.env.board_size
                x = int(round(inner.left + c * spacing))
                y = int(round(inner.top + r * spacing))
                pygame.draw.circle(self.screen, self.current_stone_color, (x, y), max(14, int(spacing * 0.29)), width=2)

    def draw_panel(self):
        height = self.screen.get_height()
        panel_rect = pygame.Rect(self.side_left - 18, 22, self.side_width + 18, height - 44)
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=12)

        self.draw_text(
            "Tiny Recursive Go",
            self.panel_title_rect.topleft,
            self.font_title,
        )
        self.draw_text(
            "TRM  •  9×9",
            self.panel_subtitle_rect.topleft,
            self.font_small,
            TEXT_MUTED,
        )

        self.draw_text(
            "Game mode",
            self.mode_label_rect.topleft,
            self.font_tiny,
            TEXT_MUTED,
        )
        mode_rect = self.mode_button_rect
        pygame.draw.rect(self.screen, PANEL_2, mode_rect, border_radius=8)
        pygame.draw.rect(self.screen, (88, 94, 105), mode_rect, width=1, border_radius=8)

        # Keep the mode selector clean: the whole button is clickable, and the
        # mode name is centered without adding a second text label that could
        # collide with it on larger font scales.
        mode_text = self.mode.name
        mode_surface = self.font_body.render(mode_text, True, TEXT)
        mode_text_rect = mode_surface.get_rect(center=mode_rect.center)
        self.screen.blit(mode_surface, mode_text_rect)

        # Turn card
        turn_rect = self.turn_rect
        pygame.draw.rect(self.screen, (47, 52, 60), turn_rect, border_radius=10)

        if self.env.game_over:
            turn_title = "Game Over"
            turn_subtitle = "Press New Game to play again"
            turn_color = SUCCESS
        elif self.paused:
            turn_title = "Paused"
            turn_subtitle = "Press Resume or Space"
            turn_color = WARNING
        elif self.ai_thinking:
            turn_title = f"{self.current_color_name} • TRM thinking"
            turn_subtitle = "Calculating next move..."
            turn_color = ACCENT
        else:
            turn_title = f"{self.current_color_name} to move"
            turn_subtitle = "Human input" if self.human_turn else "TRM will play automatically"
            turn_color = TEXT

        self.draw_text(
            turn_title,
            (turn_rect.x + 16, turn_rect.y + max(10, int(12 * self.font_scale))),
            self.font_subtitle,
            turn_color,
        )
        self.draw_text(
            turn_subtitle,
            (turn_rect.x + 16, turn_rect.y + turn_rect.height - self.font_small.get_height() - max(10, int(12 * self.font_scale))),
            self.font_small,
            TEXT_MUTED,
        )

        # Rules / scoring
        score_rect = self.score_rect
        pygame.draw.rect(self.screen, (47, 52, 60), score_rect, border_radius=10)
        self.draw_text("Rules", (score_rect.x + 14, score_rect.y + 10), self.font_tiny, TEXT_MUTED)
        rules_text = "Tromp–Taylor area scoring"
        self.draw_text(rules_text, (score_rect.x + 14, score_rect.y + 30), self.font_small)
        komi_text = f"Komi {self.env.komi:g}"
        komi_surface = self.font_small.render(komi_text, True, TEXT_MUTED)
        komi_rect = komi_surface.get_rect(midright=(score_rect.right - 14, score_rect.centery + 8))
        self.screen.blit(komi_surface, komi_rect)

        # Speed
        self.draw_text("AI move delay", self.speed_label_rect.topleft, self.font_tiny, TEXT_MUTED)
        self.slider.draw(self.screen)

        speed_value = f"{self.slider.value:.2f} s"
        if self.slider.value == 0:
            speed_value = "Instant"
        self.draw_text(speed_value, self.speed_value_rect.topleft, self.font_body)
        self.draw_text("Slower", self.speed_labels_rect.topleft, self.font_tiny, TEXT_MUTED)
        faster_surface = self.font_tiny.render("Faster", True, TEXT_MUTED)
        faster_rect = faster_surface.get_rect(topright=self.speed_labels_rect.topright)
        self.screen.blit(faster_surface, faster_rect)

        # Message card
        message_rect = self.message_rect
        pygame.draw.rect(self.screen, (47, 52, 60), message_rect, border_radius=10)
        self.draw_text("Message", (message_rect.x + 14, message_rect.y + 10), self.font_tiny, TEXT_MUTED)

        text_area = pygame.Rect(
            message_rect.x + 14,
            message_rect.y + 33,
            message_rect.width - 28,
            message_rect.height - 43,
        )
        if self.error_message:
            self.draw_wrapped_text(self.error_message, text_area, self.font_small, ERROR)
        elif self.info_message:
            self.draw_wrapped_text(
                self.info_message,
                text_area,
                self.font_small,
                SUCCESS if self.env.game_over else TEXT,
            )
        else:
            self.draw_text("Ready.", (text_area.x, text_area.y), self.font_small, TEXT_MUTED)

        # Human controls
        self.pass_button.enabled = self.human_turn and not self.env.game_over and not self.ai_thinking
        self.pass_button.text = "Pass  (P)"
        self.pass_button.update(pygame.mouse.get_pos())
        self.pass_button.draw(self.screen, self.font_body)

        # Bottom buttons
        self.reset_button.update(pygame.mouse.get_pos())
        self.reset_button.draw(self.screen, self.font_body)

        self.pause_button.enabled = self.mode is MODE_AI_VS_AI and not self.env.game_over
        self.pause_button.text = "Resume" if self.paused else "Pause"
        self.pause_button.update(pygame.mouse.get_pos())
        self.pause_button.draw(self.screen, self.font_body)

        # Footer help
        footer_text = "R reset  •  Space pause  •  P pass  •  Esc quit"
        self.draw_text(footer_text, self.footer_rect.topleft, self.font_tiny, TEXT_MUTED)

    def draw_wrapped_text(self, text, rect, font, color):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= rect.width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        line_height = font.get_linesize()
        y = rect.y
        for line in lines[:3]:
            self.draw_text(line, (rect.x, y), font, color)
            y += line_height

    def draw_game_over_overlay(self):
        if not self.env.game_over:
            return

        score = self.env.calculate_score()
        board = self.board_rect()
        overlay_width = min(int(board.width * 0.82), max(430, int(430 * self.font_scale)))
        overlay_height = min(int(board.height * 0.42), max(250, int(250 * self.font_scale)))
        overlay = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
        overlay.fill((20, 22, 26, 232))
        rect = overlay.get_rect(center=board.center)
        self.screen.blit(overlay, rect)

        title = self.font_title.render("Game Over", True, SUCCESS)
        self.screen.blit(title, title.get_rect(center=(rect.centerx, rect.y + 42)))

        self.draw_centered_text(
            f"Black  {score['black']:.1f}",
            pygame.Rect(rect.x + 35, rect.y + 85, 170, 30),
            self.font_body,
        )
        self.draw_centered_text(
            f"White  {score['white']:.1f}",
            pygame.Rect(rect.x + 225, rect.y + 85, 170, 30),
            self.font_body,
        )
        self.draw_centered_text(
            f"Winner: {score['winner']}  •  margin {score['margin']:.1f}",
            pygame.Rect(rect.x + 25, rect.y + 125, rect.width - 50, 30),
            self.font_subtitle,
            WHITE,
        )
        self.draw_centered_text(
            "Press New Game / R to start another match",
            pygame.Rect(rect.x + 20, rect.y + 170, rect.width - 40, 35),
            self.font_small,
            TEXT_MUTED,
        )

    def draw(self):
        self.screen.fill(BG)
        self.draw_board()
        self.draw_panel()
        self.draw_game_over_overlay()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.update_layout()
            self.clear_expired_messages()
            self.handle_events()
            self.update_ai()
            self.draw()

        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Play Tiny Recursive Go with Pygame.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to the TRM checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        app = TinyRecursiveGo(args.checkpoint)
        app.run()
    except FileNotFoundError as exc:
        pygame.quit()
        print(f"Checkpoint/file not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        pygame.quit()
        print(f"Application error: {exc}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
