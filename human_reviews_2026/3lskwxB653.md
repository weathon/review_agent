# Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 2, 4, 4

## Abstract
Sparse Mixture-of-Experts (MoE) have been widely adopted in recent large language models since it can efficiently scale up the model capability without increasing the inference cost. However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.g., 10-20% in accuracy) to the optimal routing. In this paper, we show that aligning the manifold of routing weights with that of task embedding via post-training can effectively reduce the gap and improve MoE LLMs’ generalization performance. Our method, “Routing Manifold Alignment (RoMA)”, introduces an additional manifold regularization term in the post-training objective and only requires lightweight finetuning of routers (with other parameters frozen). Specifically, the regularization encourages the routing weights of each sample to be close to those of its successful neighbors (whose routing weights lead to correct answers) in a task embedding space. Consequently, samples targeting similar tasks will share similar expert choices across layers. Building such bindings between tasks and experts over different samples is essential to achieve better generalization. Moreover, RoMA demonstrates the advantage of unifying the task understanding (by embedding models) with solution generation (by MoE LLMs). In experiments, we finetune routers in two recent MoE LLMs using RoMA. Evaluations on diverse benchmarks and extensive comparisons with baselines show the substantial improvement brought by RoMA.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel post-training method, termed Routing Manifold Alignment (RoMA), that fine-tunes routing parameters to address the misalignment between routing weights and task semantics in MoE models. The authors first analyze the mismatch between routing selection and tasks in existing MoE models. Based on this, RoMA aligns the manifold of task and routing. Sufficient experiments have demonstrated the effectiveness of the proposed method.

### Strengths
1. The proposed post-routing training idea of ​​explicitly aligning the "task embedding manifold" with the "routing weight manifold" is very novel.

2. Experiments have fully demonstrated the effectiveness of the proposed method and achieved significant improvements on different datasets.

### Weaknesses
1. The proposed method requires an additional embedding model to provide annotations, and its performance may be limited by the additional model.

2. Further quantitative analysis is needed to illustrate the impact of the proposed method. (detailed in questions)

### Questions
1. Although it achieves good performance on a range of downstream tasks, does this adjustment to the expert distribution compromise the model's expressive power?

2. The authors need to provide some theoretical analysis of the ablation study. The current analysis is limited to experimental phenomena, and the authors should provide some theoretical analysis of the causes.

### Soundness
3

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
4

### Summary
The paper proposes RoMA (Routing Manifold Alignment), a post-training method for Mixture-of-Experts (MoE) LLMs that updates only the routers. The idea is to build a neighbor graph in a task-embedding space over a subset of “successful” training examples (those the current model gets right), then add a *manifold regularizer* that encourages a sample’s routing pattern ($r_i$) to resemble those of its successful neighbors ($r_j$), with similarity-normalized weights ($W_{i,j}$). The method adds no inference-time cost and is reported to improve accuracy across benchmarks on MoE backbones. The paper’s motivation leans on a claimed gap to an “oracle routing” upper bound and on UMAP visualizations of routing vs task-embedding manifolds.

### Strengths
- Clear, practical objective: The manifold regularizer is standard, well-posed, and easy to implement for router-only fine-tuning.
- **Empirical breadth:** The paper reports consistent gains over a wide suite (MMLU, ARC-C/E, HellaSwag, etc.) and shows competitive performance relative to a strong test-time routing method (C3PO).
- **Focused ablations:** The paper varies the number of routed layers, token choice, and neighborhood size, which are some of the design levers that plausibly matter for this approach (though see “Weaknesses” on explanations/variance).
- Writing is generally clear at the high level; figures are visually appealing. The method section is readable.

### Weaknesses
**Related works section is really sparse.** 
A huge body of related works are present in the literature that are missing and many works such as Moefication etc. are relevant. 

**Oracle” routing is defined but not computed, yet used as an empirical anchor.**
The paper defines a per-example oracle  $r_i^*=\arg\min_r \mathcal L_{\mathrm{CE}}(f(x_i,r),y_i)$ to claim a 10–20% “oracle gap,” and it reports Oracle rows in Table 1. However, no procedure is provided for obtaining these oracle numbers (search/relaxation/heuristic/optimization), nor is the compute or fairness (e.g., use of labels at evaluation) specified. As written, the “oracle” is kind of like a tautological bound, not really a reproducible baseline. Using it in tables materially affects the paper’s narrative but is methodologically unclear.

**Heavy reliance on UMAP visuals without quantitative alignment metrics**
Figure 3 is central to the paper’s motivation (misalignment before RoMA, alignment after, similarity to “oracle” etc.), but UMAP is a non-linear, hyperparameter-sensitive projection, and the paper provides *no quantitative* alignment measures nor any robustness checks across UMAP settings/datasets. As such, the figure is suggestive but insufficient as primary evidence.

**Incomplete reproducibility for key components**
Important knobs/levers are underspecified in the main text:
- **Embedding model ($E(\cdot$)):** name/version, what text is embedded (task descriptions vs inputs), metric/σ and sensitivity.
- **Neighbor graph schedule:** fixed vs recomputed during training; frequency/criteria.
- **Router differentiability:** how gradients traverse the top-(k) gating (STE, Gumbel-Softmax, soft relaxation), and whether training uses hard or soft routing during FT.
- **Hyper-parameters & tuning protocol:** $\lambda$ for the regularizer, $\sigma$, ($k/\epsilon$ seems to be provided in the appendix)..
Without these, independent reproduction will be difficult and also makes the method incredibly opaque and the results hard to interpret. 

*Successful neighbors* may introduce *confirmation bias*
The method imitates routing only from samples the current model already solves correctly ((\mathcal S)). This is a plausible heuristic, but it risks **locking in** existing modes and under-serving hard/rare cases; the paper does not probe this trade-off. Can the authors comment on this?

**Statistical rigor and significance**
Main tables lack error bars and multiple seeds. Router-only fine-tuning and MoE routing can be high-variance and single-run improvements, especially single-digit deltas, need **CIs**/**p-values** to rule out noise.

**In general** the main proposal of this paper is a manifold regularizer for routing. My main issue with this paper is that, although the paper proposes this regularizer, it does not study it well. It adds it and gets "good results". But whether it is because of that is not well studied nor are the experimental details clearly expressed.

### Questions
1. **Oracle:** How were the Oracle numbers in Table 1 produced? Please detail the optimization/relaxation, steps, etc.
2. **Router gradients:** What is the exact differentiability strategy for top-(k) routing during FT (STE vs Gumbel vs soft routing)?
3. **Embedding ($E(\cdot)$):** Which model, inputs (instruction vs raw text), metric/σ, and sensitivity? Is the neighbor graph static or periodically recomputed?
4. **Stats:** Please report means error bars over $\geq3$ seeds and the tuning protocol for $\lambda$, $k/\epsilon$, $\sigma$, LR.
5. Providing the following would really benefit the paper's claims: quantitative evidence that routing space aligns with task-embedding space e.g. through subspace similarity (CCA/CKA) and neighbor consistency (k-NN overlap, trustworthiness) reported pre/post RoMA on at least two datasets to substantiate Figure 3.
6. **Successful-neighbor filter:** Have you tried variants that include a small fraction of incorrect but semantically close neighbors or a curriculum that relaxes the filter over time?

### Soundness
1

### Presentation
2

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
This paper proposes Routing Manifold Alignment (RoMA), a lightweight post-training method for Mixture-of-Experts (MoE) large language models. RoMA addresses the suboptimal routing problem in MoE LLMs by aligning the routing weight manifold with the task embedding manifold via a novel manifold regularization term. The method fine-tunes only the router parameters (a tiny fraction of the total model) while keeping experts frozen, leading to significant accuracy improvements (7–15%) across diverse benchmarks without increasing inference cost.

### Strengths
1. RoMA introduces a novel manifold alignment perspective to MoE routing, unifying task understanding (via embeddings) with expert selection.
2. The paper extensively evaluates RoMA on two recent MoE architectures (OLMoE and DeepSeekMoE) across eight diverse benchmarks. It outperforms strong baselines, including C3PO, Dense BP, and tuning methods, and shows competitive or superior performance to dense models with up to 34B parameters.
3. The authors systematically analyze key design choices—layer selection, token positions, neighborhood strategies, training set size, and regularization methods.

### Weaknesses
1. While inference cost is unchanged, the training cost of RoMA—especially the nearest-neighbor search over the training set—is not discussed. 
2. The paper uses a pre-trained embedding model 
E(⋅) to compute task embeddings but does not justify the choice or explore its sensitivity. The impact of different embedding models on RoMA’s performance remains unclear.

### Questions
1. How did you compute the Inference cost
2. It is good to show the routing weights in Table 1. 
3. For Figure 6-10, did you use the average accuracy for the 8 tasks in Table 1? If the answer is yes, how did you select these hyperparameters (k, layers, token position) in zero-shot tasks? 
4. For Figure 7, why can the last token obtain good performance?

### Soundness
3

### Presentation
3

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
The authors propose a lightweight finetuning method for MoE models built upon the insight that there is typically misalignment between the clustering structure of the task embedding space and the routing weights, which undermines expert specialization as the semantic structure of the inputs is not preserved across the experts. The authors thereby propose a regularizer termed 'Routing Manifold Alignment', which trains the router to align the task embedding manifold, obtained by embedding task samples with a pretrained embedding model, with the routing weights manifold.

### Strengths
1. The high level idea of manifold alignment is intuitive and original. 

2. The result that a substantive performance gain is attainable purely through better routing is compelling.

3. The authors present their work in two frontier backbones, OLMoE and DeepSeekMoE, and extensively validate their work across a host of model sizes. The authors also do a good job of validating their method against a range of downstream tasks.

### Weaknesses
**Missing baselines**. Perhaps the largest issue is that the authors essentially present a parameter-efficient finetuning method for MoE but do no compare with existing MoE-PEFT / PEFT works. The authors do a good job of comparing with a wide range of lightweight baselines including prompt and prefix tuning up to Dense BP and C3PO, but, in my view, comparison with LoRA [1] or one of its new sota variants (perhaps DoRA [2]), and a PEFT-MoE method, such as MoLE [3] is vital to assessing the validity of the performance gains and efficiency. 

**Unclear methodology**. The authors refer extensively to 'routing weights' as $\\{r_i\\}_{i=1}^n$ where $r_i$ is the concatenated routing weights across multiple layers. But this sounds more like the authors are describing the (concatenated) *projections* of the $n$ samples through the router, not the weights themselves. Indeed, there are not $n$ different routing weights for each sample, but shared routing weights that project the $n$ samples in each layer. Furthermore, without any preliminaries on what kind of router architecture the authors are referencing, it becomes harder to work this out. To me, the method still makes sense if we are referring to routed projections, i.e hidden representations of the tokens after applying routing weights $h_i = W_r x_i$, where $W_r \in \mathbb R^{K \times d}, x_i \in \mathbb R^d$ for $K$ experts, but it seems this is a misreference to refer to $n$ routing weights. Alternatively, it could be the authors mean the hidden representation after the entire MoE layer (i.e after both routing projection and expert feed forward), but, again, without preliminaries this is unknown. If the authors could clarify this that would be much appreciated.

[1] LoRA: Low-Rank Adaptation of Large Language Models (Hu et al, 2021)

[2] DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al, 2024)

[3] Mixture of LoRA Experts (Wu, 2024)

### Questions
1. See the note about unclear methodology. Clarification on preliminaries and notation would be helpful. 

2. The authors' key insight is that samples with similar task embeddings should share similar routing patterns. There's a conceptual distinction here though between a sample-level task embedding and the token-level routing patterns. Could the authors clarify or add some intuition here why these conceptually distinct objects can be unified?

### Soundness
2

### Presentation
2

### Contribution
3
