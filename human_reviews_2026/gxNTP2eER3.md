# Sparsity Forcing: Reinforcing Token Sparsity of MLLMs

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Sparse attention mechanisms aim to reduce computational overhead with minimal accuracy loss by selectively processing salient tokens. Despite their effectiveness, most methods merely exploit a model’s inherent sparsity and thus plateau at moderate budgets (about 50\% token reduction), with little headroom to push budget lower without hurting accuracy. 
Other approaches attempt to enforce sparsity through trainable sparse attention or sharpness-inducing regularizers, but these either fix rigid patterns that ignore input and layer dynamics, or optimize proxy objectives without direct control over token budgets.
In this paper, we explicitly reinforce token sparsity in well-posed multimodal large language models (MLLMs) through a simple RL-based post-training framework named  $\textit{Sparsity Forcing}$. 
Our method explores the efficiency-accuracy trade-off by running multiple rollouts with different token budgets, where both efficiency (token reduction ratio) and performance (answer correctness) are formulated as joint rewards.
By contrasting rollouts within each group, the more efficient and correct answer is rewarded while less efficient or incorrect ones are penalized, thereby turning token saving into an end-to-end, inference-consistent optimization objective.
Across thirteen image and video benchmarks, Sparsity Forcing raises token reduction ratio on Qwen2-VL/Qwen2.5-VL from 20\% to 75\% with minimal accuracy decline, 
significantly reducing long-context inference memory by up to 3$\times$  while speeding up decoding by up to 3.3$\times$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a token-sparse attention for VL with RL to enhance sparsity. It can be combined with different sparsity methods.

### Strengths
1. Sparsity is combined with RL, which is a new direction of sparsity training.
2. Evaluation of methods are solid, with different sparsity methods and models.

### Weaknesses
Personally I do not believe in token-sparsity with pruned KV cache as you never know what information is going to be used in the future. The situation is more severe when you have interaction with human, e.g. multi-round questioning/chating. Once you prune your KV cache, it's going to be lossy. As a result, it would be better either to use sparse attention kernel with full kv cache or you have certain ability to retrieve KV cache.

### Questions
1. What's the motivation behind "sharpness-inducing regularization/loss"? It makes no sense to me as you can tweek temperature (softmax scale) to adjust shapeness. The reason we use softmax scale in attention is that we want it to be "less sharp" otherwise zero attention score will lead to no grad for certain tokens and you miss the chance to correct your attention distribution when the model makes mistakes. Besides, in LLM, people use all kinds of tricks like QK norm or QK clip to reduce max-attn-logits (prevent too sharp) for training stablity. 
2. How is moba baseline being trained? I believe moba is a pre-training method. If you apply its attn prob on a dense pretrain model, it should have large accuracy degradation. 
3. Have you tried any non-pruned kv dynamic sparse method? For example. using SeerAttention like methods in post-training setting.
4. What's the cost of RL training-speed slowdown and GPU memory increase in current design? Similar to previous discussion, if you do not prune KV, your RL should also be efficient with a shared copy of KV cache. You only need to have differnt sparse index selections on different rollouts. Correct me if I am wrong.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Sparsity Forcing, a post-training, RL-based framework that explicitly trades off answer correctness and token savings for MLLMs. The method performs grouped, multi-budget rollouts using top-p sparse attention; a joint reward promotes correctness and token reduction, and GRPO (group-relative PPO) updates a sparse policy model while anchoring to a reference (full-attention) model for stability. Experiments on 13 image/video benchmarks claim to raise token reduction from ~20% to ~75% with minimal accuracy loss, yielding up to 3.0× memoryand 3.3× decoding speedups in long-context settings.

### Strengths
1. The proposed effieicency-aware RL post-training is important and novel.
2. The experimental results show the proposed method is very promising. 
3. The paper writing is very clear and easy to follow.

### Weaknesses
1. The method needs to compute a_sort, nnz, topk_index. Some of these functions have linear complexity. It is hard to compute in parallel. How did the authors compute this function efficiently on GPU in Algorithm 1&2 ?
2. MOBA keeps 25% of full attention. Sparsity Forcing keeps 26.4%. It seems dynamic sparsity cannot achieve lower compression rate. Can the proposed method be applied to block-based patterns to achieve the best of both worlds?
3. How did the authors deal with the irregularity introduced by the dynamic sparsity? It is not very friendly to system deployment, especially considering the skewness of KVCache, tensor core optimization and continuous batching.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Sparsity Forcing, an RL-based post-training framework that promotes token sparsity in multimodal LLMs using GRPO. The method treats token reduction and correctness as joint rewards. Applied on QwenVL and LLaVA-Video, it improves sparsity with minor accuracy loss.

### Strengths
Demonstrates sparsity gains across many MLLM tasks.

Clear implementation using grouped rollouts and ZipVL backbone.

Experiments across benchmarks covering both image and video. There are some interesting discussions in ablation study.

### Weaknesses
However, while the results are promising, I find the methodological novelty to be somewhat limited. The work mainly applies the existing GRPO framework to a known sparse attention mechanism (ZipVL/MOBA style) and does not clearly articulate what new algorithmic insights it introduces beyond this combination. 

The paper also does not sufficiently discuss how sparsity behaviors differ between text-only LLMs and multimodal models, even though modality-specific sparsity patterns and cross-frame visual dependencies are central challenges in MLLMs.

The comparison to baselines leaves unanswered questions. Although the related work section lists many sparse attention approaches, only a small subset is used in experiments. It is not clear whether the baselines are retrained or used as training-free methods, nor whether the training and rollout costs are comparable. Since the method uses RL and multiple rollouts with a reference model, training cost is a key factor, and it would be helpful to discuss fairness and overhead. 

In addition, the improvements over ZipVL are sometimes modest (Fig 4 cd), particularly in latency, which suggests that the marginal efficiency gain relative to the additional training cost may be less significant in some settings.

There are also concerns regarding generality and deployment. It appears that the method largely supports single-turn inference and requires recomputing context when multiple iterations occur, since the KV cache pruning changes across runs. This limits applicability to interactive or long-horizon reasoning settings. 

The method also seems to require training for each sparsity level, which affects flexibility in real-world deployment scenarios where dynamic sparsity adjustment may be desired. 

For video tasks, the approach applies the same sparsity policy as images and does not attempt to leverage temporal redundancy or cross-frame consistency, which misses an opportunity.

Finally, although the paper reports latency and memory results, the reward signal itself uses token ratio rather than actual measured runtime efficiency. This gap raises questions regarding hardware alignment, especially given that sparse attention can have non-linear speed behavior depending on the implementation.

### Questions
Do baseline methods also involve training? If so, what is the training and rollout cost for each? If not, how do you ensure comparison fairness?

Does this approach support multi-turn conversations with incremental KV cache usage, or does each turn require full recomputation?

Is the model retrained for each desired sparsity level, or can sparsity be adjusted at inference time without retraining?

How does the method handle video temporal redundancy, and could temporal attention provide additional efficiency gains?

Why not include real hardware latency as part of the efficiency reward instead of only token reduction?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper identifies a real and important limitation of current sparse-token methods for MLLMs and they rely on emergent sparsity and plateau around ~50% token reduction.

### Strengths
1. Transforms token sparsity into an explicit joint reward (accuracy + sparsity) and performs GRPO over multiple budget rollouts, effectively avoiding the token-sparsity mismatch seen in SFT-based methods.

2. General and practical framework that requires no architectural modifications and remains fully compatible with ZipVL and other sparse-attention approaches.

### Weaknesses
1. Is the reward shaping universally effective, or does it depend on dataset and model scale? Additional experiments or analysis on scalability and robustness would strengthen the claims.

2. Although the method’s goal is to improve inference efficiency, the paper does not disclose critical training cost details. Specifically:

    2.1 Total training time?

    2.2 GPU resources (e.g., A100 hours)?

    2.3 Wall-clock comparison vs. ZipVL, LoRA, or R1-style RL fine-tuning?

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
2
