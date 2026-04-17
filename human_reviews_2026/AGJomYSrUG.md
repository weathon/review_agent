# Parallel Token Prediction for  Language Models

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Autoregressive decoding in language models is inherently slow, generating only one token per forward pass. We propose Parallel Token Prediction (PTP), a general-purpose framework for predicting multiple tokens in a single model call. PTP moves the source of randomness from post-hoc sampling to random input variables, making future tokens deterministic functions of those inputs and thus jointly predictable in a single forward pass. We prove that a single PTP call can represent arbitrary dependencies between tokens. PTP is trained by distilling an existing model or through inverse autoregressive training without a teacher. Experimentally, PTP achieves a 2.4$\times$ speedup on a diverse-task speculative decoding benchmark. We provide code and checkpoints at https://github.com/mandt-lab/ptp.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel architecture for parallel token generation, says $K$ next token simultaneously. Conventional large language models (LLMs) generate text in an autoregressive manner—given $T$ input tokens, the model predicts the probability distribution of the next token, then repeats this process sequentially. In contrast, the authors observe that the sampling step can be reformulated: rather than drawing directly from a categorical distribution, one can apply the inverse cumulative distribution function (CDF) to a uniform random variable, enabling a more parallelizable generation procedure. So, given a $K$ uniform random variable as input to the model (hypothetically), we can fully determine the next $K$ tokens without the need to perform autoregressive sampling. This paper then presents a training paradigm to train such architecture: input $T$ tokens and $K$ uniform random variables, outputs the next $K$ tokens.

### Strengths
Up to my knowledge, the idea presented in this work is original and interesting, drawing inspiration from the probability/sampling theory. 

The paper distinguished itself clearly from other multi-token generation frameworks, including speculative decoding and discrete diffusion model. Though I'm not an expert in discrete diffusion, I think the independent generation issues of prior approach detailed in section 2.4 is an important contribution.

### Weaknesses
No wall-clock time experiment/ speed comparison reported.

The discussion on model architecture should be emphasized more.

It is a bit unclear what is the tradeoff between the number parallel token to be generated affects the performance. A small graph showing this would strengthen the paper.

The error correction part (section 2.3) seems to say to the use of this framework is to serve as a draft model in speculative decoding (correct me if I'm wrong). If this is the case, there should be an ablation on this. 

Overall, I believe this is a neat idea. There might be some challenges training such model so I suggest the experiments to be more focused instead of spreading over multiple separate issues.

### Questions
In section 3.1, why are we using the taxi dataset instead of other standard dataset for finetuning LLM such as GSM8K for ablation?

Is there any tradeoff between the number parallel token to be generated and the downstream performance?

Suppose I would like to do greedy sampling or changing the sampling temperature, is there any way to do this with your framework? Is there anyway to compute the uncertainty in this framework? 

Is it correct to say that the C-PTP framework can let us recover this information? 


Section 2.3 seems to suggest that the model can produces mismatch between the generated tokens and the actual probability. The error correction part looks very similar to the speculative decoding process, is there anyway to get rid of this?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a framework called Parallel Token Prediction (PTP) to accelerate the inference speed of autoregressive language models like transformers. The authors' key insight is to reframe the generation process: instead of the model predicting a probability distribution from which a token is then sampled, they make the sampling process itself part of the model.

By feeding a sequence of random numbers (called auxiliary variables) as input along with the text prompt, the model learns to deterministically convert these random numbers into a sequence of multiple output tokens in a single forward pass. This breaks the strict one-by-one generation bottleneck and allows for significant parallelization without making the limiting independence assumptions that plagued previous parallel decoding methods.

### Strengths
On a coding dataset, they distill a TinyLlama-1.1B model. Their resulting PTP model can generate an average of 5.1 correct tokens that are identical to the teacher's output, all within a single forward pass. This represents a 5x speedup over the standard autoregressive baseline.
They show that their method significantly outperforms naive parallel methods that assume token independence, which produce invalid token pairs over 60% of the time in their example.

### Weaknesses
Novelty: This approach has been adopted by a few other works like: 
1. BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models
2. Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential 
I believe the authors are doing the same thing here.

Typically such multi-token prediction tasks are evaluated on Spec-Bench. These results are necessary to justify real gains at inference time.

### Questions
1. How do you guarantee accuracy of the teacher model during PTP?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method to improve **multi-token prediction (MTP)** by using $k$ auxiliary variables in the context which “represent” the next $k$ tokens. These auxiliary variables (sampled uniformly in $[0,1]$) contain information about the next generated tokens and are used to sample the $k$-th token ahead from the corresponding cumulative distribution function.

To train the model, the authors propose methods to retrieve these auxiliary random variables (e.g., from a teacher autoregressive model). The student model (the multi-token predictor) learns to reverse-engineer the process by predicting the same tokens as the teacher with the additional auxiliary variables. This way, training and inference match.

Experiments show improvements over standard MTP baselines.

### Strengths
- The approach is conceptually sound and addresses a relevant challenge: reducing autoregressive inference latency by enabling multiple token predictions in one forward pass.

- The method provides the model with additional structured information about future sampling decisions, potentially allowing more informed multi-token predictions than standard MTP approaches.

- Several design dimensions are explored, including loss functions and auxiliary-variable extraction strategies.

### Weaknesses
- **(Major) Role of auxiliary variables.** A stronger baseline would include the same number of additional “special” tokens as placeholders for future tokens (similar in spirit to the Pause-token approach [https://arxiv.org/abs/2310.02226]), trained to predict future tokens directly. Such a baseline would help clarify whether the auxiliary variables provide unique modeling benefits or primarily serve as additional learned positions.

- **(Minor) Training efficiency.** The highest-performing configuration relies on generating full sequences from a teacher model in order to extract auxiliary variables, which is computationally expensive.

- While the evaluation demonstrates gains on the selected setups, the method appears broadly applicable; including results on standard reasoning and coding benchmarks (e.g., TruthfulQA, HumanEval, GSM8K) would strengthen the empirical validation and illustrate generality.

Small typo: “which jointly predicts future tokens $t_k$” → “which jointly predicts future tokens $t_i, \ldots, t_k$”.

### Questions
How does performance change as the number of auxiliary variables (i.e., prediction horizon) increases?
Is there a point at which adding more auxiliary variables yields diminishing returns?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Modern autoregressive models estimate the conditional probability of the next token given all the past tokens available in the context. One limitation of this class of models is that generating a sequence of $N$ tokens requires $N$ forward passes, hence resulting in possibly high latency. The paper proposes a solution to this problem by introducing a new approach, Parallel Token Prediction (PTP), enabling parallel generation of multiple tokens in parallel in a single forward pass. By incorporating auxiliary random variables into the model -- borrowing ideas from inverse autoregressive flows --  in PTP the generated tokens are not generated independently, in contrast to existing multi-token-prediction models. PTP models can be either trained from scratch or via distillation from standard autoregressive models. The experiments show some evidence that PTP models can effectively match the performance of their autoregressive counterparts yet significantly reducing latency.

### Strengths
- The paper presents a simple yet, to the best of the reviewer’s knowledge, novel approach in which the model learns to sample directly by conditioning on random variables, enabling it to anticipate and jointly predict future tokens.

### Weaknesses
- Paper clarity: the presentation of the paper can be improved in multiple points. See questions.
- The experimental evaluation is quite limited, focusing on a toy dataset and a small-scale distillation task. Given the potential of the idea, stronger empirical evidence would help validate its impact.
- While the provided arguments about the theoretical speedups guaranteed by PTP seem convincing, I believe the paper would greatly benefit from an empirical comparison between the latency of standard autoregressive models compared to the proposed frameworks. 
- At multiple points in the paper (lines: 054-056, 441-442, 465-466) it is mentioned that the model proposed in [1] generates multiple independent tokens. I am not sure this is the case as in section 2.2 of [1] the authors state that "Different from Gloeckle et al. (2024), which parallelly predicts $D$ additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth". I would appreciate it if the authors can clarify this point.


[1] Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv preprint arXiv:2412.19437 (2024).

### Questions
- In Eq. (6), shouldn't it be $P_{i+1} = P(t_{i+1}|t_{<i}, u_i)$?
- The l.h.s. of Eq. (7) does not seem correct. Rather, I think it should be written as $P(t_{k}|t_{<i}, u_i, ..., u_{k-1})$. Could the authors clarify?
- Could the authors provide intuition or evidence for potential performance gains relative to [1] (see weaknesses)?
- Have you explored how the performance of PTP evolves with scale, and how that compares to AR baselines?
- Can a PTP model trained to predict $N$ tokens be adapted at inference time to predict $M$ (with $M \neq N$) tokens?

[1] Liu, Aixin, et al. "Deepseek-v3 technical report." arXiv preprint arXiv:2412.19437 (2024).

### Soundness
2

### Presentation
2

### Contribution
2
