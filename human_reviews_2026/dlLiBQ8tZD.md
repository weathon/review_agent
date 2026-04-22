# LoLA: Low-Rank Linear Attention With Sparse Caching

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
The inference cost of transformer-based large language models is proportional to the context length. This prevents their application to lifelong in-context learning. Linear attention is an efficient alternative that maintains a constant memory footprint, even on infinite context lengths. While this is a potential candidate for lifelong learning, it falls short in memory capacity. In this paper, we propose LoLA, a novel inference augmentation to linear attention that boosts associative recall. LoLA distributes past key-value pairs from context into three memory systems: (i) recent pairs in a local sliding window cache; (ii) difficult-to-memorize pairs in a sparse, global cache; and (iii) generic pairs in the recurrent hidden state of linear attention. We empirically show through ablations that our self-recall error metric is crucial to efficiently manage long-term associative memories. LoLA improves the long-context performance of the base model on the RULER benchmark, with improvements from 0.6% to 97.4%  on pass-key retrieval. This is achieved with a 4.6x smaller cache than Llama-3.1 8B on 4K context length. LoLA also outperforms other 1B and 8B parameter subquadratic models on zero-shot commonsense reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposes LoLA, a training-free enhancement to linear attention that improves long-term associative memory. It introduces a three-part memory system: (i) a local sliding window for recent key-value pairs, (ii) a sparse global cache for hard-to-recall items, and (iii) a low-rank recurrent hidden state for generic context, which is guided by a self-recall error metric to detect and mitigate memory collisions.

### Strengths
The proposed method is effective at NIAH recall.

### Weaknesses
1. Limited evaluation: The current evaluation focuses on synthetic task and short-context NLP benchmarks. Could authors also evaluate their method on at least one long-context NLP benchmark, e.g. LongBench [1]?
2. Work related to this paper to be discussed:
   - The authors claim that current hybrid approaches only focus on local information with sliding window attention, which is not true. Could authors discuss the recent hybrid architectures with sparse attention, e.g. B'MOJO [2], Se-Attn [3], HAX [4]?
   - For the minimization of Self-Recall Error (Eq. 10) in the paper, could authors discuss the correlation of their method with recent test-time training method, e.g. GatedDeltaNet [5], TTT [6], Titans [7], Lattice [8]?

[1]. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. ACL2024.

[2]. B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory. NeurIPS 2024.

[3]. Expansion Span: Combining Fading Memory and Retrieval in Hybrid State Space Models. NeuS 2025.

[4]. Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention. NeurIPS 2025.

[5]. Gated Delta Networks: Improving Mamba2 with Delta Rule. NeurIPS 2024.

[6]. Learning to (Learn at Test Time): RNNs with Expressive Hidden States. ICML 2025.

[7]. Titans: Learning to Memorize at Test Time. https://arxiv.org/abs/2501.00663

[8]. Lattice: Learning to Efficiently Compress the Memory. CoLM 2025.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes **LoLA**, a *training-free* inference-time augmentation for hybrid sliding-window + linear attention models. LoLA keeps three memory systems: (i) a local sliding-window cache, (ii) a sparse global cache, and (iii) a recurrent linear-attention hidden state. It scores KV pairs using a **self-recall error (SRE)** how poorly linear attention alone can reconstruct a value from its key and promotes “hard” pairs into the sparse cache while easier ones flow into the recurrent state. On synthetic long-context recall (RULER/S-NIAH) and some zero-shot commonsense tasks, LoLA improves over the authors’ distilled LoLCATs baselines under fixed cache budgets.

### Strengths
The SRE criterion is intuitive and easy to compute given $\Phi(k)$, H, s; the paper supplies pseudo-code and a useful efficiency study (TTFT and VRAM) versus sliding-window size η and sparse-cache size $\lambda$, which practitioners can adopt to tune deployments.

### Weaknesses
1. **Positioning / novelty is narrow and tied to a special base model.** Although billed as “training-free,” LoLA **assumes** a specific *subquadratic* base (sliding-window + linear attention) obtained via distillation/LoRA (40M tokens) before LoLA can be used. It is therefore not a drop-in for standard Transformers and the headline “training-free” risks misinterpretation. Please clarify scope and re-title accordingly; also separate the cost/benefit of distillation from LoLA’s cache policy.  

2. **Claims vs. baselines and calculation clarity.** The abstract’s “**4.6$\times$ smaller cache than Llama-3.1-8B on 4K**” is not explained precisely; Table 1 lists the Transformer as full-attention (effectively a 4K-token KV cache at 4K), while LoLA uses η+λ (e.g., 256+256). Please specify exactly which Llama cache size is used for that factor and how it’s computed; as written, it’s hard to reconcile. More broadly, comparisons mix distilled subquadratic models and different training budgets; ensure apples-to-apples controls (same teacher, tokens, and evaluation protocol).   

3. **Evidence focuses on synthetic recall; limited breadth on real tasks and ablations.** The large wins are concentrated on RULER/S-NIAH; while there are commonsense results, the absolute gains vs. LoLCATs are modest in places and still trail Llama on some tasks. Scoring ablations for SRE are shown but only on S-NIAH @ 0.5K; please broaden ablations (e.g., alternative importance metrics, cache-update cadence, chunk size sensitivity) on harder and real-world long-context tasks (multi-needle, multi-hop QA with citations).

### Questions
* **Grammar in Fig. caption.** “**LoLA stores past KV pairs in three forms memory**” → *three forms of memory*. 
* **Informal wording.** “**anyways**” → *anyway*. (Figure text) 
* **Spelling.** “**pruple**” → *purple*. (Figure text) 
* **Apostrophe.** “**Its more efficient**” → *It’s more efficient*. (Figure text) 
* **Spelling.** “**occurence**” → *occurrence*. (Appendix F text) 
* **Consistency.** Use either *commonsense* or *common sense* uniformly; currently the abstract uses “commonsense” while Table 3 caption uses “common sense.”  
* **Section label consistency.** TOC line says “**Algorithm Psuedo-code**” (again typo); ensure capitalization and spelling consistent with the section header. 
* Reframe “training-free” to “inference-only augmentation **for subquadratic models**” and clearly delineate the *pre-training/distillation* of the base from the *LoLA* mechanism. Provide a small *plug-compatibility* study across multiple linear-attention bases to evidence generality. 
* Make the cache-size factor claims auditable: define the exact KV token counts behind “× smaller” in each table/setting, and add a per-token throughput plot (not only TTFT) with and without sparse rescoring, end-to-end (I/O included).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces LoLA, a training-free inference strategy designed to solve the poor memory capacity of linear attention models. Standard linear attention suffers from "memory collisions," where new information overwrites past associations in its fixed-size hidden state. LoLA addresses this by augmenting hybrid linear attention models with a novel three-part memory system: (i) a local sliding window cache for recent key-value (KV) pairs, (ii) a sparse, global cache for difficult-to-memorize pairs, and (iii) the standard recurrent hidden state for all other generic pairs. The core of LoLA is a new importance metric called Self-Recall Error (SRE), which is used to manage the sparse cache. SRE measures how well the current linear hidden state can already retrieve a KV pair's correct value. Pairs that are difficult to recall (high SRE) are kept in the full-rank sparse cache to prevent them from being corrupted, while easy-to-recall pairs (low SRE) are compressed into the hidden state. This method is highly effective, improving a base model's accuracy on pass-key retrieval tasks from 0.6% to 97.4% and achieving strong performance on general commonsense reasoning benchmarks.

### Strengths
1. The paper's core idea is highly original. Instead of using query similarity or softmax scores for sparse attention, it introduces the Self Recall Error (SRE) . This query agnostic metric provides a principled, data driven way to determine which tokens are "difficult to memorize" for the linear state and should be cached in full rank. This is a clever and novel approach to mitigating memory collisions.

2. The claims are supported by exceptionally strong and well targeted experiments. The method provides a near perfect fix for associative recall on Needle in a Haystack (NIAH) tasks, boosting accuracy from a failing 0.6% to 97.4%. Furthermore, the ablation study in Table 4 is critical, proving that LoLA's SRE metric (99.0% accuracy) is vastly superior to intuitive alternatives like extending the sliding window (29.4%) or using other sparse attention metrics (e.g., 11.4%, 20.0%).

3. The work is highly significant because it is a training free inference strategy. This allows it to be applied directly to existing, pretrained hybrid models like LoLCATs to immediately unlock their long context recall capabilities.

### Weaknesses
1. The paper states the scoring introduces a "small overhead compute cost". However, the proposed algorithm re-scores all $\lambda$ elements in the sparse cache plus the new candidate token(s) at every generation step. This overhead is non trivial, especially when $\lambda$ is large. The Time to First Token (TTFT) in Figure 4 confirms this. For a 64 token window, TTFT increases from 0.99s ($\lambda=0$) to 1.46s ($\lambda=512$). This is a roughly 47% slowdown. This trade off is not sufficiently addressed in the main paper.

2. The paper positions LoLA as a solution to memory collisions in linear attention but fails to compare its approach to other architectures designed to solve the exact same problem. Specifically, Gated DeltaNet, a highly relevant model, combines gating and a delta rule to precisely manage its recurrent state, mitigate collisions, and improve associative recall. While Gated DeltaNet is a different architecture (requiring training) and LoLA is an inference strategy, they are both direct solutions to memory collisions. The paper should have discussed, if not benchmarked against, such advanced recurrent models.

3. The method is described as "training free". While true that LoLA itself is an inference strategy, its SRE metric (Equation 10) is entirely dependent on the feature map $\phi$, which was learned during the distillation of the LoLCATs base model. It is unclear if LoLA would be as effective if applied to a linear attention model trained from scratch, or one using a different $\phi$.

### Questions
1. The paper's goal is to mitigate memory collisions, which is the same goal as advanced recurrent models like Gated DeltaNet. How do the authors view LoLA's "sparse caching" approach in comparison to Gated DeltaNet's "gating + delta rule" approach? 

2. The TTFT analysis in Figure 4 shows a clear slowdown as $\lambda$ increases. Could the authors provide a more direct comparison of this overhead? 

3. The SRE metric relies on the $\phi$ map learned by the LoLCATs base model. Have the authors tested whether LoLA is effective when applied to other linear attention models that were not trained with LoLCATs' specific distillation process, such as a model pretrained from scratch?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LoLA, a training-free inference strategy that enhances linear attention models by mitigating their poor memory capacity. It distributes key-value pairs into three systems: a local sliding window, the low-rank recurrent state, and a new sparse global cache that stores "difficult-to-memorize" pairs identified by a self-recall error metric, drastically improving associative recall.

### Strengths
1. The paper empirically demonstrates an improvement on tasks where the base model completely fails
2. The propose of a new sparse global cache is novel

### Weaknesses
1. The method adds a new computational cost not present in standard linear attention, specifically for scoring and managing the sparse cache. The paper says its overhead is O(λd) which itself needs to be larger for more complex, long-context tasks.
2. LoLA moves away from the simplicity of linear attention by requiring three distinct memory systems that must be managed. 
3.  This caching strategy cannot fully compensate for the knowledge lost during the base model's efficient distillation from attention

### Questions
Can author provides the comparison on the decoding latency their LoLA and original LA?

### Soundness
3

### Presentation
3

### Contribution
3
