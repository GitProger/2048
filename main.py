from rewards import *
from board import Board, Pygame
from agent import Agent
from model import *
from train import train
import matplotlib.pyplot as plt

def play_human():
    board = Board(4)
    ui = Pygame(board)
    ui.run()

def play_agent(model_filename_prefix):
    model = load_model(model_filename_prefix)
    board = Board(4)
    ui = Pygame(board)
    agent = Agent(model)
    ui.run(agent)

def graph(metrics):
    rewards   = [m[0] for m in metrics]
    max_tiles = [m[1] for m in metrics]
    steps     = [m[2] for m in metrics]
    episodes = list(range(len(metrics)))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards)
    plt.title("Total reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(3, 1, 2)
    plt.plot(episodes, max_tiles)
    plt.title("Max tile per episode")
    plt.xlabel("Episode")
    plt.ylabel("Tile")

    plt.subplot(3, 1, 3)
    plt.plot(episodes, steps)
    plt.title("Steps per episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.tight_layout()
    plt.show()

def main():
    print("Choose mode:")
    print("1) play by yourself")
    print("2) play with an AI")
    print("3) train new model")
    k = input("> ").strip()

    if k == "1":
        play_human()
    elif k == "2":
        filename = input("Model's filename prefix: ")
        play_agent(filename)
    elif k == "3":
        s1 = int(input("Size of layer 1 (better power of 2): "))
        s2 = int(input("Size of layer 2 (better 1/2 of the previous): "))
        fn = int(input("Reward function, 1 - simple, 2 - balanced, 3 - advanced: "))
        ep = int(input("Training episodes: "))
        function = {1: reward_simple, 2: reward_medium, 3: reward_advanced}[fn]
        prefix = input("Filename prefix: ")
        model, metrics = train(s1, s2, ep, function)
        save_model(model, prefix, s1, s2)
        graph(metrics)
    else:
        print("Unknown command")

if __name__ == '__main__':
    # prefix = "large2"
    # #s1, s2, ep, function = 16, 8, 128, reward_simple
    # s1, s2, ep, function = 128, 64, 1024, reward_advanced
    # model, m = train(s1, s2, ep, function)
    # save_model(model, prefix, s1, s2)
    # graph(m)
    main()