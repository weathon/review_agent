# TARE: Lightweight Token-Aware Representation Editing for Fine-tuning Transformer

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
Parameter-efficient fine-tuning (PEFT) of large Transformers often struggles to balance effectiveness with efficiency. Methods based on low-rank adaptation can be resource-intensive, while representation-editing techniques that apply a single, global transformation tend to underfit fine-grained, token-level contexts. The core challenge is achieving token-aware, fine-grained edits while keeping inference overhead and the hyperparameter tuning burden negligible. Our work introduce Token-Aware Representation Editing (TARE), a novel PEFT method. After each feed-forward network (FFN) block, TARE employs a lightweight selector that scores a small pool of "editors" for each token's hidden representation. It sparsely activates only the top-scoring editors and mixes their element-wise edits to update the representation. Because the edits are computationally minimal diagonal operations and are sparsely activated, TARE adds near-zero inference overhead and introduces no rank or scaling hyperparameters. Our work conduct extensive experiments on LLaMA-3-8B across eight knowledge reasoning and seven mathematical reasoning tasks, and on RoBERTa-base/large for the GLUE benchmark. Compared to strong baselines like LoRA, DoRA, MiLoRA, LoReFT, and RED, TARE achieves state-of-the-art results. It attains an 86.7% average on knowledge reasoning tasks, 76.7% on mathematical reasoning tasks, and 88.3% on the GLUE benchmark. These results are achieved while tuning only 0.0392% of the model's parameters and using approximately 20 GiB of memory, surpassing prior methods by several percentage points and demonstrating exceptional resource efficiency. An anonymized implementation is available at: https://anonymous.4open.science/r/tare-BCF5/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper proposes a new method called TARE which enables token-aware representation editing using a codebook of editing parameters.
- The editing parameters are formulated as element-wise scaling and shifting of the token representations.
- If $n$ is the number of codebook entries (editing parameters), TARE first obtains the edited representations using all the $n$ editing parameters, and then selects the top $k$ edited representations.
- The final edited representation is obtained using a weighted sum of the top $k$ edited representations with the weights determined using a softmax over the top k values.

### Strengths
- TARE performs better than prior representation editing methods (LoReFT and RED), and proposes a novel token-aware parameter selection method.
- The method is easy to implement and lightweight, making it feasible for low-resource settings.
- The method is evaluated on a variety of tasks (commonsense reasoning, arithmetic reasoning, code generation, NLU, conditional text generation).

### Weaknesses
1. The comparisons with LoRA, DoRA and MiLoRA are not completely fair.
    - LoRA, DoRA and MiLoRA are applied to all the linear layers in the model, while TARE is applied to only the hidden representations of the MLP blocks.
    - A more fair comparison would be to apply LoRA, DoRA and MiLoRA to the MLP down-projection layer.
    - This is necessary because the lower performance of baseline methods could be due to overfitting or optimization difficulties, and reducing the parameter count and fine-tuning only the down-projection layer could increase the performance.
2. The hyperparameter tuning strategy and details are not provided.
    - The performance gap between TARE and baseline methods is suspiciously large. It is possible that the hyperparameters of the baseline methods are not tuned well. For e.g., LoRA gets 68.7 average accuracy vs. 76.7 average accuracy for TARE on arithmetic reasoning. While it could be possible to achieve slightly better performance with reduced parameter count, this gap is too large especially given that the parameter count is reduced by ~17 times.
    - Similar is the case with conditional text generation and code synthesis tasks.
3. The paper claims that the hyperparameter tuning burden is negligible and the method does not introduce rank/scaling hyperparameters. However, the method introduces a new hyperparameters, the codebook size $n$ and the top $k$ selection parameter. From Section 4.3, these hyperparameters also need to be tuned, similar to rank and scaling hyperparameters in LoRA. While this is not a disadvantage of the method (all methods need some sort of hyperparameter tuning), it is not an advantage, and the paper should not claim it as such.
4. Comparisons with recent PEFT techniques like LoRA-GA and LoRA-One is missing.
5. Finally, LoRA is an established method with a large practical impact. The contribution of a new fine-tuning method does not have much impact without a significant advantage other than performance alone. In that sense, the contribution of the paper is limited.

### Questions
- In Table 2, is the VRAM usage reported for training or inference?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents TARE (Token-Aware Representation Editing), a PEFT approach that adapts pretrained Transformers by directly editing their token representations instead of weight matrices as done by LORA and variants. Each Transformer layer is equipped with a pool of lightweight “editors”, each defined by simple per-feature scaling $\gamma$ and bias $\beta$ parameters. A token-wise selector computes soft attention over these editors, activates the top-k editors per token, and mixes their outputs to form a small, token-dependent edit to the hidden representation. The method thus performs fine-grained, adaptive modulation of intermediate activations via an extremely small parameter budget (~0.04 % of model parameters on LLaMA-3-8B). The experimental results show promising comparisons.

### Strengths
+ It worths exploring PEFT beyond weight-space adaptation (LoRA, DoRA,etc.) and beyond layer-wise scaling (IA$^3$, RED). By making token-specific and dimension-specific adjustments, it aligns better with the local gradient structure of each token representation.

+ The reported results seem promising across different tasks.

### Weaknesses
- It's a bit surprising that the paper misunderstood LORA in terms of inference overhead statement in the introduction and motivates the proposed TARE based on that misunderstanding. There is no overhead at all for methods such as LORA since weights have been merged. 

- The claimed near-zero overhead of the proposed TARE seems to be problematic. What TARE does per layer includes: (1) Compute a token-dependent routing score over n “editors”, which requires a linear projection from the hidden state of every token (dimension d) to an n-dimensional logits vector, resulting in the complexity: $O(B \times T \times d \times n)$ — nontrivial for large $B,T,d$. (2) Top-k and softmax are per-token and per-layer operations, which are memory-bound and cause branching inefficiency (divergent threads), thus adding synchronization and extra kernel calls. (3) For each selected editor, apply per-feature scale and bias (diagonal affine transform), for which even if each edit is diagonal (element-wise), doing k of these per token requires multiple reads/writes over the hidden representation — heavy on memory bandwidth, not FLOPs. In experiments, evaluations use short sequences (e.g., 256–512 tokens).
But, for real LLM workloads (2k–8k tokens), the selector’s per-token cost scales linearly with context length. Token-wise selectors also prevent efficient tensor-core matmuls. In sum, TARE’s design inherently couples every token with a dynamic routing decision and multiple per-token transforms. Even if the FLOPs are modest, the memory traffic and kernel fragmentation will dominate runtime cost. The paper’s measurements appear to ignore wall-time and sequence length scaling, so in practice, the “lightweight” claim doesn’t hold.

- It's also unclear how and why TARE will work for autoregressive models in the sense that  it perturbs the hidden representation dynamically and token-wise in a way that changes across time steps and cannot be merged into weights, which is potentially destructive for an autoregressive generator, which depends on stable internal feature statistics. For downstream tasks, especially when the data is not very large as in pretraining, it is not easy to train stable and meaningful selector at the token level, unlike MoE in pretraining. 

- Despite of the above unclear aspects, for the method itself, its mechanics largely combine diagonal adaptation and MoE routing concepts already explored in RED, IA$^3$, and sparse activation work. The step from global to token-level editors is incremental.

- Some LORA variants are missing in comparisons:  LORA-One (ICML'25, https://github.com/YuanheZ/LoRA-One) and WeGeFT (ICML'25, https://openreview.net/pdf?id=K0sv5T2usb).

### Questions
Please consider to address the questions in the weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes TARE, a new parameter-efficient fine-tuning method that edits each token’s hidden representation using a lightweight selector that activates only a few tiny “editors” (per-dimension scale and bias) after each Transformer feed-forward block, giving token-aware, context-specific adaptation with almost no inference overhead and no rank hyperparameters. TARE matches or outperforms strong PEFT baselines like LoRA, DoRA, MiLoRA, LoReFT, and RED on knowledge reasoning, math reasoning, GLUE, code generation, and more, while updating only 0.0392% of model parameters and keeping memory usage around 20 GiB.

### Strengths
1. The paper is well written.
2. Across LLaMA-3-8B and RoBERTa , TARE matches or beats strong PEFT baselines under good efficiency constraints.
3.  I thin TARE  is  smart and lightweight. It doesn’t slap a big adapter on top of the model. Instead, for each token it picks just a few tiny “editors” that do simple per-dimension scale + bias tweaks, mixes them, and moves on.

### Weaknesses
**If the authors address the issues in the Weaknesses and Questions sections, I will  increase the score.**

1. Training stability depends on extra mechanisms (like load-balancing and top-k gating), which hints at brittleness. the paper notes this top-k routing can collapse if a few editors get picked too often, so they introduce a load-balancing regularizer to force different editors to be used.

2. I find that the paper is missing comparisons against some recent PEFT methods, including weight adapters and memory-efficient sparse fine-tuning methods such as [1][2][3]. It would be better if the authors could include comparisons to these approaches.  

3. For large language models, the experiments were only conducted on the LLaMA family, so it's unclear how well the method would perform on other LLM model families.


[1] Meng, Fanxu, Zhaohui Wang, and Muhan Zhang. "Pissa: Principal singular values and singular vectors adaptation of large language models." Advances in Neural Information Processing Systems 37 (2024): 121038-121072.

[2] Zhang, Fangzhao, and Mert Pilanci. "Spectral adapter: Fine-tuning in spectral space." arXiv preprint arXiv:2405.13952 (2024).


[3] Liu, Zihang, et al. "LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning." arXiv preprint arXiv:2506.00772 (2025).

### Questions
Please refer to Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TARE is a new fine-tuning method that makes large language models adapt better by editing hidden representations at the token/channel level. Instead of using the same transformation for all tokens, it adds small “editors” that adjust each token’s features with lightweight scaling and bias operations. Only a few editors are activated per token, keeping computation very low. This design gives the model fine-grained control without slowing inference or adding many trainable parameters. Experiments on a diverse set of benchmarks show that it outperforms previous PeFT approaches.

### Strengths
1. Diverse Benchmarks: The paper provides extensive evaluations across a wide range of models and benchmarks, including both encoder (RoBERTa) and decoder (LLaMA-3) architectures. It covers multiple task families -- knowledge reasoning, mathematical reasoning, GLUE, text generation, code synthesis, and symbolic reasoning -- demonstrating that TARE generalizes effectively across domains and scales well with different model types.

2. Strong Empirical Results: TARE consistently achieves superior or state-of-the-art performance compared to leading PEFT baselines. The method delivers notable gains in accuracy across diverse tasks while tuning a fraction of the parameters, highlighting its efficiency and the strength of its token-aware editing mechanism.

3. Minimal Inference Overhead: Despite introducing token-level adaptivity, TARE’s operations are strictly diagonal and sparsely activated, adding virtually no latency during inference. Its lightweight design requires no additional matrix multiplications or rank-based hyperparameters, allowing it to maintain near-identical runtime performance to the frozen base model.

### Weaknesses
1. The tested benchmarks seem to focus on short output sequences; it is important to include some more diverse sequence lengths, given that editing is token/channel dependent (see Q1).

2. The impact of editor placement or layer-wise variation is not deeply analyzed, limiting insight into optimal integration. There should be some further analysis into the best location to place the editor -- with conventional PEFT strategies, we similarly evaluate different subsets of trainable parameters such a QKVO/GUD.

### Questions
1. Most of the benchmarks used in this work, such as GLUE and commonsense reasoning datasets, involve short outputs. While the inclusion of specialized domains like code generation (e.g., HumanEval) adds diversity, evaluating TARE on long-context, multi-turn dialogue tasks such as MT-Bench would provide stronger evidence of its practicality and robustness in extended conversational or reasoning settings.

2. Ablation on cross-layer dependencies (see W2).

3. Although the paper frequently refers to “near-zero inference overhead,” it does not provide explicit latency or throughput measurements to substantiate this claim. Including numerical comparisons and where they are derived from would strengthen the argument for efficiency and make the claim more transparent and verifiable.

### Soundness
3

### Presentation
3

### Contribution
3
