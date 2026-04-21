# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 5, 6, 5

## Abstract
Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their hidden state. Inspired by prior work, we present a practical framework to instantiate sequence modeling layers with linear complexity and expressive hidden states. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, our layers are called Test-Time Training (TTT) layers. We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model and a two-layer MLP respectively. We evaluate our instantiations at the scale of 125M to 1.3B parameters, comparing with a strong Transformer and Mamba, a modern RNN. Both TTT-Linear and TTT-MLP match or exceed the baselines. Similar to Transformer, they can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. With preliminary systems optimization, TTT-Linear is already faster than Transformer at 8k context and matches Mamba in wall-clock time. TTT-MLP still faces challenges in memory I/O, but shows larger potential in long context, pointing to a promising direction for future research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a new RNN layer they call Test-Time Training (TTT) layer. The hidden state in this layer is a machine learning model itself, in practice a linear layer or an MLP. The update for this layer is a single step of "self-supervised" learning using SGD. The authors show how to efficiently implement their layer on the GPU, or similar hardware. Finally a number of different models using TTT layers are empirically evaluate it terms of perplexity, training FLOPs and inference time on natural language tasks.

### Strengths
The paper is well written and easy to follow.

It presents a novel idea which it thoroughly investigates at a number of different scales.

Some of the new models using TTT layers seem to present gains over Mamba and Transformers in terms of training efficiency, and inference latency thought the significance of these gains is questionable.

### Weaknesses
The motivation for the self-supervised loss used is a little weak. The self supervised loss described seem to be learning a mapping from one linear projection of a vector to another. I believe this mapping would have a closed from solution as a function of the projections, at least for the linear case. I would like to see the author expand their justification here and if relevant give examples of other inner loop losses experimented with. Self supervised losses for sequence data conventionally try to predict the next item in a sequence from the preceding elements. Did you try this? If so why not?

The efficiency gains are not really put into a human understandable form. Additionally the significance of the efficiency gains seem to be very marginal. While I appreciate multiple runs is likely out of scope for experiments of this size its hard to know if the increases are statistically significant.

It would be great to see if it was indeed the convolution through time that was responsible for the gains when using the Mamba backbone, or if it was some other aspect of the architecture.

The conclusion section is very weak.

### Questions
Did you experiment with other inner loop losses? What is the intuition behind the mutli-view loss, was the loss selected to produce Theorem 1?

It does not seem obvious why you'd want to run and iterative optimisation algorithm on the inner loop objective that has a closed form solution, can you give more intuition here? 

Did you have any strong justification to think it convolution through time was responsible for the gains when using the Mamba backbone?

I'm not super familiar with the hardware side of LLMs. In section 3.3 you mention wrote two kernels for the TTTlayers. Were these necessary to i) get a fair comparison with the other models? or ii) necessary to make the TTT layers cooperative with other baselines? Without how did the wall clock time compare without these kernels? Can you explain why were these kernel necessary?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes Test-Time Training (TTT) layers. TTT outperforms Transformers and Mamba for models up to 1.3B parameters. In terms of runtime, TTT-Linear is more efficient than Transfromers and is competitive with that of Mamba.

### Strengths
This paper explores a gradient-based alternative to modern RNNs and self-attention. The approach is uncommon and allows for a more expressive hidden state, which results in better performances. 

TTTs achieve competitive or slightly better performance compared to Mamba and Transformers. 

Connecting TTT with linear and self attention is a nice touch.

### Weaknesses
Mamba is parallelized with a parallel scan. However, TTT requires some sequential computations controlled by a hyperparameter $b$, potentially making it significantly slower during training. Unfortunately, the paper lacks memory and runtime analyses during training, making it rather hard to determine the practicality of training a TTT model compared to other modern RNNs that leverage parallel scan (e.g., Mamba, xLSTM, GLA-Transformers, etc...).

Meta-learning is closely related to TTT, but the connection is not properly discussed. The paper briefly mentions a connection on page 3, but the discussion is lacking. For example, TTT using "online gradient descent" is reminiscent of MAML, where the task is a sequence and each sample is a timestamp. 

When initially reading, the reviewer found the usage of the terms "batch GD" and "mini-batch GD" to be confusing and took some time to get used to. For example, when mentioning "mini-batch GD", the reviewer initially thought of its typical use case (per outer-loop training iteration). However, in this work, the mini-batch GD is done temporally with varying timestamps.

### Questions
In meta-learning, a common issue of the inner/outer loop formulation with gradient descent is the second-order gradients which are memory intensive. How does TTT resolve this issue?

Could the authors distinguish their usage of (mini-)batch GD from the commonly associated connotation? For example, prefacing the term with "temporal" could potentially reduce the initial confusion. 

The training efficiency of TTT appears to be highly dependent on $b$ as it determines the amount of parallelism TTT can leverage. Furthermore, $b$ also significantly affects the performance. In contrast, methods such as Mamba are not dependent on this hyperparameter. Plots comparing the training runtime, memory, and performance of TTT with varying values of $b$ and methods such as Mamba are important. Could the authors include these?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Test-Time Training (TTT) layers, which make the hidden state a model updated via self-supervised learning, enabling efficient long-context processing with linear complexity. Two implementations, TTT-Linear and TTT-MLP, outperform Transformers and Mamba on long-context tasks.

### Strengths
TTT layers offer efficient long-context processing with linear complexity, outperforming traditional Transformers and Mamba in terms of scalability and performance.

### Weaknesses
The paper shows limited improvement over Mamba, and its evaluation is restricted to language modeling tasks. Furthermore, it lacks comparative experiments with related methods like Fast Weight Layers and Hidden-State Optimization. 

Reconsidering the Past: Optimizing Hidden States in Language Models
Davis Yoshida, Kevin Gimpel, EMNLP2021 findings
Meta-Learning Fast Weight Language Models
Kevin Clark, Kelvin Guu, Ming-Wei Chang, Panupong Pasupat, Geoffrey Hinton, Mohammad Norouzi, EMNLP2022

### Questions
Q1. Could you demonstrate statistically significant performance improvement by adding error bars to the experimental results? While TTT-Linear shows similar computational efficiency to Mamba, its perplexity appears nearly identical. Without error bars in the results, it is difficult to determine how effective this method truly is.

Q2. Is it possible to include evaluations on downstream tasks as part of the experiment?
In Mamba’s original paper, experiments were conducted on downstream tasks to demonstrate performance beyond language modeling alone. At a minimum, it seems necessary to include experiments similar to those in Section 4.2.2 of the following paper:
https://arxiv.org/abs/2312.00752

Q3. Should comparative experiments with the following studies also be conducted? Also,one seems to be a missing citation.

1. Reconsidering the Past: Optimizing Hidden States in Language Models, Davis Yoshida, Kevin Gimpel, EMNLP 2021 Findings
2. Meta-Learning Fast Weight Language Models, Kevin Clark, Kelvin Guu, Ming-Wei Chang, Panupong Pasupat, Geoffrey Hinton, Mohammad Norouzi, EMNLP 2022

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces sequence modeling layers called Test-Time Training (TTT) layers. The key contributions are:
1. Introduction of a machine learning model (e.g., linear model or MLP) that is trained through self-supervised learning during inference/test time. This allows the model to learn and adapt to patterns in the input sequence on the fly.
2. Two practical instantiations: TTT-Linear and TTT-MLP, which use a linear model and a two-layer MLP respectively as their hidden states. 
3. Empirical validation showing that:
    * Both TTT variants match or exceed performance of Transformers and Mamba models across different model sizes (125M to 1.3B parameters)
    * TTT-Linear achieves faster inference than Transformers at 8k context and matches Mamba's speed
The authors also introduce practical optimizations like mini-batch TTT and a dual form implementation that significantly improve wall-clock time performance, making TTT-Linear immediately practical for real applications.

### Strengths
The paper is tackles an interesting problem but lacks novelty and experimental rigour.

### Weaknesses
The major weakness of the paper is the missing novelty, as well as comparison to appropriate baselines, in my opinion.  

* Novelty: Linear self-attention and its variants are clearly motivated already by learning at inference. This is discussed in e.g.  Schlag et al., 2021 (Linear Transformers Are Secretly Fast Weight Programmers) and citations within. Additionally, and building on Schlag, are at least 3 missing citations i.e. Yang et al. 2024 (Parallelizing Linear Transformers with the Delta Rule over Sequence Length) as well as von Oswald et al. 2022, (Transformers Learn In-Context by Gradient Descent) and von Oswald et al. 2023 (Uncovering mesa-optimization algorithms in Transformers) which also build on this framework. 
	
* Please provide a comprehensive analysis how TTT-linear differs from these layers. 

* Yang et al. here is therefore also one obvious baseline, which is missing. First, the parallelized delta net implementation has better throughput as well as perplexity scores than Mamba and is therefore a crucial and missing baseline here. Additionally, and probably  even closer baseline are results of gated linear self-attention layers e.g. Yang et al. 2023 (Gated Linear Attention Transformers with Hardware-Efficient Training), which seem to be an extension of TTT-linear, because of the gates. 
	
* These missing references also highlight that the following statements might be untrue:
	1) The “novel” class of layers is in my opinion a false statement and needs correction. 
	2) Related literature, see references, is able to parallelize across time (gated) linear self-attention and the delta rule, showing that they are able, to an extent, to leverage matmul units on GPUs (and TPUs). What is the relationship to Mini-Batch TTT, is this identical? 
 
	

* Further things: 
	
	- Figure 3: 1) Perplexity seems very low here on the Pile. 2) Perplexity should be lower for 8k than for 2k, especially for Transformers. 
	- Missing evals: Please Incorporate evals such as Table 1 and 2 in  Yang et al. 2024 (Parallelizing Linear Transformers with the Delta Rule over Sequence Length)

### Questions
Please see weakness

### Soundness
1

### Presentation
2

### Contribution
1
