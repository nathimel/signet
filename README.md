# SIGNET: Pilot experiments with deep discrete signaling networks

## Intro

"Signaling networks of different kinds are the locus of information transmission and processing at all levels of biological and social organization. The study of information processing in signaling networks is a new direction for naturalistic epistemology." -- (Skyrms, 2010: p. 47)

## Summary

This repo contains code for running various pilot experiments for learning with signaling networks.

Signaling games can be used to model the evolution of language, and information processing more generally. Some steps have been taken to model how very simple signaling networks (two-Sender, one-Receiver) RL agents can learn boolean concepts (Barrett, 2007; LaCroix, 2020). A natural question is: how much can signaling networks learn? This repo attempts to address this question empirically.

### Networks

A signaling network, in this context, is a graph of with agents as nodes and lines of communication as edges. We consider the simple  topology of a *binary tree*, where two or more inputs are passed to an 'input layer' of Senders, who then send signals to intermediate agents, who in turn send signals to zero or more intermediate layers, until a single Receiver takes an appropriate action on behalf of the entire signaling tree. The signals are discrete objects, and agents have very simple parametric form (e.g., Roth-Erev agents as in typical Lewis-Skyrms signaling games) and are typically trained under very simple reinforcement learning.

### Boolean concept learning

The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root. We further assume that each agent, at each layer, must solve an _attention_ sub-problem: it must learn which signals outputted from the previous layer to take as input and combine. (see [src/game/boolean/signaltree.py](src/game/boolean/signaltree.py) for more information.)

### Preliminary results

Perhaps somewhat unsurprisingly, large signaling trees cannot predict the truth values of complex functions with accuracy above chance. For modest sizes, such as boolean functions of three atoms, trees can learn (under simple reinforcement learning,  win-stay-lose-randomize, with inertia, etc.).

Intuitively, learning the correct logic 'circuit' is just too improbable: as long as the network is rewarded for an output produced by a suboptimal structure, it will always get pushed away from correctly learning the desired boolean function. This highlights a fundamental tension between constraining the possible search space of network structures (to make learning the correct semantic trees easier) and allowing the general pressures for reward maximization to operate (that is, to allow diverse solutions to boolean function approximation).

## References

> Skyrms, Brian. Signals: Evolution, learning, and information. OUP Oxford, 2010.

> Barrett, Jeffrey A. "Dynamic partitioning and the conventionality of kinds." Philosophy of Science 74.4 (2007): 527-546.

> LaCroix, Travis. "Using logic to evolve more logic: Composing logical operators via self-assembly." The British Journal for the Philosophy of Science (2020).
