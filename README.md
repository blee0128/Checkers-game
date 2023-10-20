# Checkers game

## Game history
English draughts, also called American checkers, are a popular game played by two opponents on opposite sides of an 8X8 game board. It is a game on which AI witnessed one of its earliest successes, as best evidenced from a self-learning checkers program written in 1959 by Arthur Samuel, a pioneer in computer gaming who also popularized the term “machine learning”.

## Project summary
For this project, we will write a game playing agent capable of playing Checkers. we will construct a general alpha-beta game playing agent which takes the current state as input and returns a move/action to be made by the agent. We will use the algorithm from the AIMA book at https://github.com/aimacode, specifically we will use the MinMax Search and alpha beta search algorithm.

## Game process
- Take as input a move from the user.
- Update the board with the user's move.
- Output the agent's move from the alpha-beta search. o Update the board with the agent's move.
- Repeat the steps until the end of the game (you don’t need to check whether a state is a draw).