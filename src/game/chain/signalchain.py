from agents.module import SignalingModule, Sequential, Layer
from agents.basic import ReceiverModule, SenderModule


class SignalChain(Sequential):
    """A signaling chain performs operations on bitstrings by iteratively transforming its input via bit flipping."""

    def __init__(self, input_size: int) -> None:

        layers = Layer(agents=[FlipAgent() for _ in range(input_size)])
        super().__init__(layers)


class FlipAgent(SignalingModule):
    """A Flipper is an agent that flips a single bit of its input before returning."""

    def __init__(self, input_size: int) -> None:
        pass
