import numpy as np
from agents import SignalingModule
from languages import State

class SignalTree(SignalingModule):
    """Signaling network for learning functions from lists of states (e.g., truth values of propositions) to states (e.g., the truth value of a complex proposition).
    
    A SignalTree is a signaling network with one agent as the final output classifier. 
    
    For the task of predicting the truth values of complex propositional logic sentences from a list of their atoms' truth values, a natural network topology when agents are limited to binary input is the sentence's (binary) syntactic tree representation.
    """
    def __init__(self, input_size: int) -> None:
        # initialize layers
        self.tree = build_tree(input_size)

        self.input_layer = None
        self.hidden_layers = None
        self.output_layer = None

    def forward(self, x: list[State]) -> State:
        return self.output_layer(self.hidden_layer(self.input_layer(x)))

    def reward(self, amount: float) -> None:
        [layer.reward() for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]]

def build_tree(input_size: int = 2) -> dict[str, list[SignalingModule]]:
    """Construct a signaling network as a binary tree with the input layer of agents as leaves and the output agent as the root.
    
    This may create 'redundant' nodes in hidden layers that would not exist in the minimal syntactic tree representation of a sentence with `input_size` atoms, when `input_size` is an odd number. 

    Args:
        input_size: the size of the input layer of the network, e.g., the number of propositional atoms of the complex sentence to predict the truth value of.

    Returns:

        layers: a dict containing the levels of the binary tree, corresponding to layers of the signaling network. Of the form 
            {
                "input": (list of agents),
                "hidden": (list of lists of agents),
                "output": (one agent),
            } 
    """
    
    # recursively build tree starting at root node, 
    # and prune the input layer if `input_size` is odd.

