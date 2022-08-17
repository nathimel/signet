"""A Generalized self-ASSembling (GLASS) agent can learn boolean functions of arity (input length) 0 to 2 boolean truth-values, and always ouputs one boolean truth-value. We focus on the case of 2-arity functions.

Variant on the signaling tree network problem for boolean games, with the following adjustments:

    - Restrict to three possible boolean inputs
    - Restrict to two possible agents
    - One agent must come in a "later" layer than the previous
    - Each agent is restricted to arity-2 functions.


So, we have agent A, and agent B. Agent A can take two inputs from the list of [input_p, input_q, input_r]. Agent B can take two inputs from the list of [input_p, input_q, input_r, signal_b], where signal_b is the output of Agent B's decision. 
"""

from agents.module import Sequential
from languages import State, Signal, StateSpace
from game.boolean.signaltree import InputSender, OutputReceiver, HiddenSignaler
from agents.module import SignalingModule
from game.boolean.functional import get_quaternary_sender
from agents.basic import Compressor

INPUT_SIZE = 3 # restrict to input list of 3 atoms
BOOLEAN_SPACE = StateSpace(states=[State("0"), State("1")])

def state_to_signal(x: State) -> Signal:
    """Converts a state to the analogous signal. Kind of arbitrary, but so is the distinction."""
    return Signal(form=x.name, meaning=BOOLEAN_SPACE.referents)

class GlassTree(Sequential):
    """For now, we abstract away from signaling where states need to be mapped (learned) to signals in order to be sent across agents, and just let the objects passed around be states."""
    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        self.input_size = input_size
        self.agent_a = GlassInputSender(input_size=self.input_size) # input list
        self.agent_b = OutputReceiver(input_size=self.input_size+1) # input list plus result of agent_a
        super().__init__(layers=[self.agent_a, self.agent_b])
    
    def forward(self, x: list[State]) -> State:
        # override forward because the second layer needs to see input layer in addition to previous layer.

        if len(x) != self.input_size:
            raise Exception(f"length of input passed ({len(x)}) does not match network input size ({self.input_size}).")

        signal = self.agent_a(x) # 
        state_b = self.agent_b([state_to_signal(state) for state in x] + [signal])
        return state_b

class GlassInputSender(Sequential):
    """Like an InputSender but with two attention layers."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.compressor = Compressor(input_size)
        self.sender = get_quaternary_sender()
        super().__init__(layers=[self.compressor, self.sender])

    def forward(self, x: list[State]) -> Signal:
        # print("hidden signaler forward called")
        return super().forward(x)