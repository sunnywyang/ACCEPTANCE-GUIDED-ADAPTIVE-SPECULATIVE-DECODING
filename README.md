
##ACCEPTANCE-GUIDED ADAPTIVE SPECULATIVE DECODING source code for the paper 'ACCEPTANCE-GUIDED ADAPTIVE SPECULATIVE DECODING FOR EFFICIENT LARGE LANGUAGE MODEL INFERENCE'
This repository contains the official implementation of Acceptance-Guided Adaptive Speculative Decoding, a lightweight and practical framework for accelerating large language model inference.

As Large Language Models grow, autoregressive decoding leads to severe inference latency. Speculative decoding mitigates this issue by allowing a lightweight draft model to generate multiple candidate tokens, which are then fed into the target model for parallel verification. A key hyperparameter is the draft length K: a larger K may waste computation due to early rejection, while a smaller K underutilizes the draft model’s capacity. Prior work fixes K, which is suboptimal across contexts. We propose Acceptance-Guided Adaptive Speculative Decoding, which retains an upper bound but dynamically selects an effective length via an acceptance head attached to the draft model. The head predicts per-token acceptance probabilities and triggers early exit when rejection risk exceeds a principled threshold derived from a runtime cost model. This reduces wasted verification and exploits confident drafts. 


##Code Structure

* ACC  
Contains all components related to the Acceptance Head, including its architecture, training pipeline, and data preparation.
    *  acc_head  
      Implementation for loading, training, and inference of the Acceptance Head.
    *  train data   
       Scripts for generating training data required by the Acceptance Head.
    
* eagle  
Implementation of the draft model and speculative decoding evaluation
    


