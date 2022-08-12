"""File for creating the default agents, languages, and other data structures used in for predicting the truth values of boolean sentences with signaling networks."""

from .languages import Signal, SignalMeaning, SignalingLanguage, State, StateSpace
from .agents import AttentionAgent, AttentionSignaler, Compressor, Receiver, ReceiverModule, ReceiverSender, Sender, SenderModule


def get_language() -> SignalingLanguage:
    """Get a SignalingLanguage instance initialized for boolean games."""

    states = [State(name="0"), State(name="1")]
    universe = StateSpace(states)

    dummy_meaning = SignalMeaning(
        states=states,
        universe=universe,
    )
    signals = [Signal(form="0", meaning=dummy_meaning), Signal(form="1", meaning=dummy_meaning)]

    return SignalingLanguage(
        signals=signals,
    )

def get_sender() -> SenderModule:
    """Get a SenderModule instance initialized for boolean games."""
    return SenderModule(sender=Sender(language=get_language()))

def get_receiver() -> ReceiverModule:
    """Get a ReceiverModule instance initialized for boolean games."""    
    return ReceiverModule(receiver=Receiver(language=get_language()))

def get_attention() -> AttentionAgent:
    """Get an AttentionAgent instance initialized for boolean games."""

def get_receiver_sender() -> ReceiverSender:
    """Get a ReceiverSender instance initialized for boolean games."""    

def get_compressor() -> Compressor:
    """Get a Compressor instance initialized for boolean games."""
    attention_1 = get_attention()
    attention_2 = get_attention()

    receiver_sender = get_receiver_sender()

    return Compressor(
        attention_1=attention_1,
        attention_2=attention_2,
        receiver_sender=receiver_sender,
    )

def get_attention_sender() -> AttentionSignaler:
    """Get an AttentionSignaler (Sender) instance initialized for boolean games."""

def get_attention_receiver() -> AttentionSignaler:
    """Get an AttentionSignaler (Receiver) instance initialized for boolean games."""
