# Pretraining LLM with Latent Thoughts in Continuous Space

- Decision: Reject
- Scores: 4, 6, 2, 8, 4

## Abstract
The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts. Our approach pretrains a language model (LM) to first generate an intermediate latent thought—the last hidden
state of the current position—which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, ours-1.4B (Pythia Arch), pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token—forming a chain analogous to CoT—consistently improves the model's performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a technique to interleave input tokens of an LLM and their corresponding hidden states during pre-training and inference. With such a horizontal scaling approach, the authors show that Pythia-1.4B models can be trained with a relatively smaller amount of tokens from the Pile dataset and attain similar validation loss as the Pythia-2.8B model. For the same computation budget of 300B tokens, the proposed pre-training approach and inference leads to better downstream performance across a variety of tasks. Similar experiments were conducted for GPT2 and Llama models.

### Strengths
The technique is quite simple, yet well presented with extensive experiments, and is also shown to present a new avenue for scaling with continuous/hidden layer features.

### Weaknesses
FLOP utilization during pre-training with the new approach is not discussed at all. This is important since the Jacobi iteration needs to be performed at least 2 or 3 times for hidden state convergence of every input sequence. This means 2/3 forward passes per sequence with double the sequence length (input tokens + hidden states). Additionally, pre-training resources, compute budget, and time are not discussed. Claims regarding the comparison with bigger models will have to be revisited based on the FLOP analysis. The details about the context lengths for inference are currently missing, so I am unsure about the claims on horizontal scalability/ complementing existing CoT prompting approaches.

### Questions
1. In Figure 1 (b), if the Pythia-1.4B model trained with the proposed approach for 62\% fewer tokens achieves the same loss as the vanilla pre-training, how does the downstream performance of this checkpoint compare with fully trained checkpoint?

2. In section 4.2.2, was instruction finetuning done with the proposed approach or vanilla?

3. How does the approach complement standard chain-of-thought? It seems that long chains will be a bottleneck for this approach. More importantly, can you provide details about the sequence lengths for training/inference? How long are the outputs for the results in (say) Figure 6?

4. What happens if we train with the proposed approach and a varying number of latent thoughts, but try to avoid latent thoughts during inference and leverage the standard chain of thought? This tradeoff is important to consider when reporting the results on math/reasoning tasks, as the steps can be lengthy and the proposed approach might result in fewer discrete tokens, potentially failing to provide the answer within a reasonable inference token budget.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a "continuous" pretraining technique for language models where, before predicting each token, the model generate intermediate "latent thoughts" hidden states in continuous space. It is essentially a pretraining extension of chain-of-continuous thought (coconut) paper. 

To make the sequential refinement of these hidden states efficient during training, the authors introduce a method that allows for parallel updates. 

Their results show that a 1.4B parameter model achieves performance comparable to a conventional 2.8B parameter model trained on the same data. This demonstrates that "horizontal thinking", or doubling the reasoning steps per token can help.

### Strengths
- Presents novel pretraining method, extending coconut to pretraining
- Parallel training procedure is nice
- Baseline comparisons are good (i have a few questions in later section)
- The results seem strong

### Weaknesses
- It's unclear the difference between parallel training and sequential inference
- Fig 1 shows similar results at lower #params, but double the compute. We know that compute can be more beneficial than #params
- Lacking in ablations and discussion on why this method outperforms the baselines
- Doesn't compare to training a model with double the amount of layers. That's a more proper comparison than just double param count. 
- It would be nice to see larger models or some math evals, but these are somewhat minor.

### Questions
- Training uses parallel, but inference uses sequential (feedback). Why isn't there a train/test mismatch?
- For the models that use double the amount of parameters, they don't use double the amount of layers, correct? That seems to be the proper comparison
- It's a bit surprising how much better your method is than other methods in table 3. Do you have insights as to why this is the case?
- The Pythia-1.4B in table 2 is not pretrained by you, correct? It uses exactly all of the same hyperparams though?
- All of the models in Table 3 are trained by you?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new pretraining methodology, where language models at each step are used to first generate hidden states in continuous space and then resend as input to generate the next token. Training uses Jacobi iteration to update hidden states iteratively. Results show improved performance in general LM benchmarks, as compared to baseline models of similar sizes.

### Strengths
* Authors propose a conceptually lightweight method to add computational depth per token.
* Jacobi iteration enables efficient parallel training despite the inherently sequential inference process.
* Experiments indicate better performance across benchmarks, as well as in parameter and data efficiency.

### Weaknesses
* The name "latent thought" is potentially an overclaim. The term "latent thought" suggests high-level reasoning or abstraction, but the method simply generates one additional hidden state per token without evidence of global deliberation or multi-token planning processes. It is also unclear from the experiment what role the hidden states actually play in reasoning.
* The method requires 2N KV cache positions for N tokens, effectively cutting the usable context window to half. The paper lacks discussion on how the model can handle longer context, which is what modern LLM applications increasingly rely on.
* Authors apply Jacobi iteration to approximate hidden states, but there is no analysis of convergence conditions, fixed-point existence guarantees, or motivation-wise, why this approximation should work for learned non-linear transformations.
* Ablation studies in Figure 8 use Pythia-70M which is 20× smaller than main experiments, and it is expected that additional computation helps more. It is unclear whether the scaling behavior can transfer to larger models.

### Questions
* It is unclear what the hidden states are actually learning through iterations. Can the authors provide some experiments or analysis of what "latent thoughts" compute. For instance, probing studies, attention pattern visualizations, or examples showing how thoughts evolve through iterations)
* How does performance scale with longer thought chains (K > 3)? Is it influenced by different task types?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a novel method to incorporate continuous thoughts into the model parameterization without changing the model architecture or number of parameters. The core idea is to pass the last hidden state of the model as input embedding at the next time step and repeat the process N times. This way model can reflect on its own representation multiple times before predicting the next token. 
Authors proposed an elegant way of speeding up the training process by using jacobian iterations which allowed them to keep the parallel training of transformer by only repeating the forward pass N times according to N continuous extra steps.
They conducted a set of experiments including pretraining, mid-training and evaluated resulted models across different benchmarks. They compared their models to known baselines that involved both discrete and continuous thoughts. Their experiments confirm that their approach is superior to related work as well as their model at smaller size surpass larger models trained without continuous thoughts.

### Strengths
1. The effectiveness of the method will allow community to use it as a mid training scenario in many future work.
2. Ability to keep the efficiency of training makes it especially attractive.

### Weaknesses
1. Novelty is not extreme given related work, but good execution here compensate that making it a solid contribution to the community.

### Questions
1. I am very curious to see how models with continuous thoughts show itself with different test time scaling approaches i.e. majority voting or combined with external reward models. If authors can perform a simple experiment with that, that would be awesome. For instance, how much different decoding hyper parameters affect results with these models (are these models more collapsed in their distributions or not).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new pretraining methodology where the language model performs an extra computation step to generate an intermediate "latent thought" (its last hidden state) before using that state to predict the next token. This "think-before-you-speak" mechanism is applied to every token. To make this sequential process trainable, the authors use a Jacobi iteration to parallelize the computation, approximating the "latent thought" in a fixed number of steps. The authors claim this method produces models that are far more efficient, with their 1.4B parameter model, which runs two passes per token, outperforming a standard 2.8B parameter model at a comparable inference throughput.

### Strengths
1. Well-Motivated Problem: The goal of creating smaller, more compute-efficient models that can match the performance of larger models is highly valuable and a key area of research.

2. Strong Empirical Results: Taken at face value, the results are impressive. Table 3, for instance, shows the proposed 1.4B LLaMA model outperforming not only a 2.8B LLaMA baseline but also other "increased-compute" methods like PonderLM and Looped Transformer, even when those are given a larger (4x) compute budget.

3. Broad Applicability: The method is demonstrated on three different architectures (Pythia, LLaMA, GPT-2) and is also shown to be effective as a continual pretraining technique for existing models (LLaMA-3-3B), suggesting it is a potentially general-purpose method.

### Weaknesses
1. The paper's claims of efficiency are one-sided and misleading. It completely omits an analysis of training cost, which appears to be substantially higher (e.g., 6-8x per step) due to the 2x sequence length and K=2-4 passes. The paper is therefore missing the most critical baseline: a vanilla model trained for the same total training FLOPs. Without this, it's impossible to know if the proposed method is superior, or if the authors simply trained their model with far more total compute.

2. The core training method is presented without theoretical or empirical justification. The Jacobi iteration is a pragmatic hack, but the paper provides no analysis of its approximation error or why this fixed-point iteration should converge to a meaningful linguistic representation. Furthermore, key design choices, like sampling K from {2,3,4}, are arbitrary and presented without any ablation or analysis.

3. Also the choice of thought tokens (the last hidden unit and the interleaved sequence) needs to be justified over other choices like PonderLM. The training cost seems to be significantly enhanced. It is unclear if the benefit comes from the Jacobi training, the choice of the hidden state as the thought tokens, or the specific interleaving strategy.

### Questions
1. Can you clarify the inference memory requirements? The two-pass generation process at inference suggests the KV-cache size may be doubled.

### Soundness
2

### Presentation
3

### Contribution
2
