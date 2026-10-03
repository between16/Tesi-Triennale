import streamlit as st
import time

from game_env import GoGameEnv
from agents import TRMAgent, HumanAgent


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tiny Recursive Go",
    page_icon="⚪",
    layout="centered"
)


# ---------------------------------------------------------
# Game initialization
# ---------------------------------------------------------

def create_agents(mode):
    """Create agents according to the selected game mode."""

    if mode == "Human (B) vs AI (W)":
        agent_b = HumanAgent()
        agent_w = TRMAgent(checkpoint_path="checkpoints/best_model.pt")

    else:  # AI (B) vs AI (W)
        agent_b = TRMAgent(checkpoint_path="checkpoints/best_model.pt")
        agent_w = TRMAgent(checkpoint_path="checkpoints/best_model.pt")

    return agent_b, agent_w


def init_game(mode=None):
    """
    Initialize/reset the game state.

    If mode is omitted, use the currently selected game mode.
    """

    if mode is None:
        mode = st.session_state.get(
            "game_mode",
            "Human (B) vs AI (W)"
        )

    st.session_state.env = GoGameEnv(
        board_size=9,
        komi=7.5
    )

    st.session_state.agent_b, st.session_state.agent_w = create_agents(mode)

    st.session_state.game_message = "Game Started. Black's turn."

    # Used only to force new widget keys after reset
    st.session_state.game_id = st.session_state.get("game_id", 0) + 1


# ---------------------------------------------------------
# Move handling
# ---------------------------------------------------------

def apply_move(move_idx):
    """Attempt to apply a move (or pass)."""

    env = st.session_state.env

    try:
        game_over = env.step(move_idx)

        if game_over:
            st.session_state.game_message = (
                "Game Over! Calculating score..."
            )
            return True

        # Update turn message
        next_color = (
            "Black"
            if env.current_player == "b"
            else "White"
        )

        st.session_state.game_message = (
            f"{next_color}'s turn."
        )

        return False

    except ValueError as e:
        st.session_state.game_message = f"Invalid move: {e}"
        return False


# ---------------------------------------------------------
# Board rendering
# ---------------------------------------------------------

def render_board(board_container, interactive=False):
    """
    Render the current Go board.

    interactive=False:
        Board is only displayed.

    interactive=True:
        Empty intersections are clickable.
    """

    env = st.session_state.env

    # Replace the previous content of the placeholder
    with board_container.container():

        for r in range(env.board_size):

            cols = st.columns(env.board_size)

            for c in range(env.board_size):

                stone = env.board.get(r, c)

                if stone == "b":
                    symbol = "⚫"
                elif stone == "w":
                    symbol = "⚪"
                else:
                    symbol = "➕"

                flat_idx = r * env.board_size + c

                # -------------------------------------------------
                # Interactive board: human can click intersections
                # -------------------------------------------------

                if interactive:

                    if cols[c].button(
                        symbol,
                        key=f"btn_{st.session_state.game_id}_{r}_{c}"
                    ):

                        current_agent = (
                            st.session_state.agent_b
                            if env.current_player == "b"
                            else st.session_state.agent_w
                        )

                        try:
                            valid_move = current_agent.get_move(
                                env,
                                move_idx=flat_idx
                            )

                            if valid_move is not None:
                                apply_move(valid_move)
                                st.rerun()

                        except ValueError:
                            st.session_state.game_message = (
                                "That move is illegal "
                                "(Suicide or Ko)!"
                            )
                            st.rerun()

                # -------------------------------------------------
                # Non-interactive board
                # -------------------------------------------------

                else:
                    cols[c].markdown(
                        f"<div style='text-align:center; "
                        f"font-size:26px;'>{symbol}</div>",
                        unsafe_allow_html=True
                    )


# ---------------------------------------------------------
# Initial state
# ---------------------------------------------------------

if "env" not in st.session_state:

    st.session_state.game_mode = "Human (B) vs AI (W)"

    st.session_state.ai_delay = 1.0

    init_game(st.session_state.game_mode)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("⚙️ Game Settings")


mode = st.sidebar.selectbox(
    "Game Mode",
    [
        "Human (B) vs AI (W)",
        "AI (B) vs AI (W)"
    ],
    key="game_mode"
)


# ---------------------------------------------------------
# Detect mode changes
# ---------------------------------------------------------

if mode == "Human (B) vs AI (W)":

    if not isinstance(
        st.session_state.agent_b,
        HumanAgent
    ):
        st.session_state.agent_b = HumanAgent()

    if not isinstance(
        st.session_state.agent_w,
        TRMAgent
    ):
        st.session_state.agent_w = TRMAgent(
            checkpoint_path="checkpoints/best_model.pt"
        )


else:  # AI vs AI

    if not isinstance(
        st.session_state.agent_b,
        TRMAgent
    ):
        st.session_state.agent_b = TRMAgent(
            checkpoint_path="checkpoints/best_model.pt"
        )

    if not isinstance(
        st.session_state.agent_w,
        TRMAgent
    ):
        st.session_state.agent_w = TRMAgent(
            checkpoint_path="checkpoints/best_model.pt"
        )


# ---------------------------------------------------------
# AI delay slider
# ---------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 AI Speed")

st.session_state.ai_delay = st.sidebar.slider(
    "Delay after AI move (seconds)",
    min_value=0.0,
    max_value=3.0,
    value=st.session_state.ai_delay,
    step=0.25,
    help=(
        "Controls how long the board remains visible "
        "after each AI move."
    )
)


# ---------------------------------------------------------
# Reset button
# ---------------------------------------------------------

if st.sidebar.button(
    "🔄 Reset Game",
    use_container_width=True
):

    init_game(st.session_state.game_mode)

    st.rerun()


# ---------------------------------------------------------
# Rules
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.write("**Rules:** Tromp-Taylor (Area Scoring)")
st.sidebar.write(
    f"**Komi:** {st.session_state.env.komi}"
)


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

env = st.session_state.env

st.title("Tiny Recursive Go - 9x9")

st.markdown(
    f"**Status:** {st.session_state.game_message}"
)

current_color_str = (
    "Black (⚫)"
    if env.current_player == "b"
    else "White (⚪)"
)

st.subheader(f"Turn: {current_color_str}")


# ---------------------------------------------------------
# Game-over state
# ---------------------------------------------------------

if env.game_over:

    st.success(
        "Game finished! Two consecutive passes."
    )

    score = env.calculate_score()

    st.write("### Final Score")

    st.write(
        f"⚫ **Black:** {score['black']}"
    )

    st.write(
        f"⚪ **White:** {score['white']} "
        f"(includes {env.komi} komi)"
    )

    st.write(
        f"🏆 **Winner:** {score['winner']} "
        f"by {score['margin']} points"
    )

    # Always show final board
    board_slot = st.empty()
    render_board(
        board_slot,
        interactive=False
    )


# ---------------------------------------------------------
# Game in progress
# ---------------------------------------------------------

else:

    current_agent = (
        st.session_state.agent_b
        if env.current_player == "b"
        else st.session_state.agent_w
    )

    # -----------------------------------------------------
    # BOARD
    # -----------------------------------------------------
    # IMPORTANT:
    # The board is now rendered for BOTH human and AI turns.
    # -----------------------------------------------------

    board_slot = st.empty()

    render_board(
        board_slot,
        interactive=isinstance(
            current_agent,
            HumanAgent
        )
    )


    # -----------------------------------------------------
    # HUMAN TURN
    # -----------------------------------------------------

    if isinstance(current_agent, HumanAgent):

        st.markdown("---")

        if st.button(
            "Pass Turn",
            use_container_width=True,
            key=f"pass_{st.session_state.game_id}"
        ):

            apply_move(81)

            st.rerun()


    # -----------------------------------------------------
    # AI TURN
    # -----------------------------------------------------

    else:

        status_placeholder = st.empty()

        status_placeholder.info(
            f"🤖 {current_color_str} AI is thinking..."
        )

        # Ask the model for its move
        move_idx = current_agent.get_move(env)

        # Apply the move
        game_over = apply_move(move_idx)

        # Immediately show the newly played stone
        render_board(
            board_slot,
            interactive=False
        )

        status_placeholder.empty()

        # Keep the new board visible before the next AI move
        if not game_over:
            time.sleep(
                st.session_state.ai_delay
            )

            st.rerun()