
import torch
from game import SnakeGameAI
from agent import Agent


def play():
    agent = Agent()
    game = SnakeGameAI()

    # Load the saved trained model
    model_path = "model/model.pth"

    agent.model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    agent.model.eval()

    print("Loaded trained Snake AI!")
    print("Playing...")

    while True:
        state = agent.get_state(game)

        with torch.no_grad():
            prediction = agent.model(
                torch.tensor(state, dtype=torch.float)
            )

        move_index = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move_index] = 1

        _, done, score = game.play_step(final_move)

        if done:
            print("Game Over!")
            print("Score:", score)
            game.reset()


if __name__ == "__main__":
    play()
