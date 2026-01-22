#ACCEPTANCE-GUIDED ADAPTIVE SPECULATIVE DECODING
source code for the paper 'ACCEPTANCE-GUIDED ADAPTIVE SPECULATIVE DECODING FOR EFFICIENT LARGE LANGUAGE MODEL INFERENCE'

This repository contains the official implementation of Acceptance-Guided Adaptive Speculative Decoding, a lightweight and practical framework for accelerating large language model inference.

As Large Language Models grow, autoregressive decoding leads to severe inference latency. Speculative decoding mitigates this issue by allowing a lightweight draft model to generate multiple candidate tokens, which are then fed into the target model for parallel verification. A key hyperparameter is the draft length K: a larger K may waste computation due to early rejection, while a smaller K underutilizes the draft model’s capacity. Prior work fixes K, which is suboptimal across contexts. We propose Acceptance-Guided Adaptive Speculative Decoding, which retains an upper bound but dynamically selects an effective length via an acceptance head attached to the draft model. The head predicts per-token acceptance probabilities and triggers early exit when rejection risk exceeds a principled threshold derived from a runtime cost model. This reduces wasted verification and exploits confident drafts. Experiments on four tasks under two temperature settings, evaluated on two series of LLMs, show reduced latency

#Code Structure
