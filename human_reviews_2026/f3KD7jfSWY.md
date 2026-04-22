# CERSA: Cumulative Energy-Retaining Subspace Adaptation for Memory-Efficient Fine-Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
To mitigate the memory constraints associated with fine-tuning large pre-trained models, existing parameter-efficient fine-tuning (PEFT) methods, such as LoRA, rely on low-rank updates. However, such updates fail to fully capture the rank characteristics of the weight modifications observed in full-parameter fine-tuning, resulting in a performance gap. Furthermore, LoRA and other existing PEFT methods still require substantial memory to store the full set of frozen weights, limiting their efficiency in resource-constrained settings. To addres these limitations, we introduce Cumulative Energy-Retaining Subspace Adaptation (CERSA), a novel fine-tuning paradigm that leverages singular value decomposition (SVD) to retain only the principal components responsible for 90% to 95% of the spectral energy. By fine-tuning low-rank representations derived from this principal subspace, CERSA significantly reduces memory consumption. We conduct extensive evaluations of CERSA across models of varying scales and domains, including image recognition, text-to-image generation, and natural language understanding. Empirical results demonstrate that CERSA consistently outperforms state-of-the-art PEFT methods while achieving substantially lower memory requirements. The code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes CERSA, a parameter/memory‑efficient fine‑tuning (PEFT) framework. The core idea is to compute an SVD of each pretrained weight matrix, truncate it to retain a target fraction of cumulative spectral energy (typically 90–95%), freeze the truncated left/right singular vectors and train a dense “core” matrix inside the retained principal subspace (with optional thresholds alpha for retained energy and beta for the trainable subset). The claim is that this achieves a better accuracy–memory trade‑off than LoRA and recent SVD‑based methods (SVFit, SVFT), because it compresses away low‑energy components and avoids storing the full frozen weights. Empirically, the paper reports results on ViT‑Base/Large for eight image classification datasets and on DeBERTaV3‑Base for GLUE, showing consistent or improved accuracy at lower total memory than baselines. Theory is provided in the form of a simple basis‑change existence result and an assumption that principal subspaces are preserved across fine‑tuning (supported by appendix measurements of Grassmann similarity).

### Strengths
Clear, pragmatic goal (memory)
The paper focuses on reducing total fine‑tuning memory and presents an explicit accounting model O(mr + nr + 4r^2) contrasting it with FT/LoRA/SVFit/SVFT in Table 1. This moves beyond only counting trainable parameters and is valuable to practitioners dealing with GPU limits. 

Principled subspace selection via cumulative energy
Using a layer‑wise cumulative‑energy threshold leverages heavy‑tailed singular spectra of pretrained weights, with Figure 3 illustrating that many layers can be aggressively truncated while retaining 90–95% energy. The ablation (Table 5) shows layer‑wise rank selection outperforms a uniform rank at equal memory. 

Simple mechanism with broader expressiveness than SVFit
Training a full core matrix (not just singular values) plausibly increases expressive power while staying inside the principal subspace; the loss curve (Figure 6) suggests faster optimization than several baselines on one dataset. 

Empirical breadth (vision + NLU) at reasonable scale
With ViT‑Large on eight image benchmarks (Table 7) and DeBERTaV3‑Base on GLUE (Table 8), CERSA is competitive and often the best among PEFT methods under the paper’s specified memory budgets. The best vision average (90.6%) slightly surpasses FT (90.2%) at ~3.8× lower reported memory; on GLUE, the reported average (89.5%) matches or exceeds others at the smallest stated memory footprint among baselines. 

Transparent acknowledgement of scope
The conclusion explicitly notes a limitation: performance may degrade when downstream tasks diverge from pretraining and proposes future work on dynamically adjusting the learned subspace. This is honest and useful to readers.

### Weaknesses
Novelty over closely related SVD‑subspace methods is under‑argued
The method is very close in spirit to prior SVD‑based PEFT (SVFit, SVFT) and to weight‑driven approaches using principal components (e.g., PiSSA). The difference—training a dense matrix in the retained subspace after truncation—feels incremental on top of “internal factors,” and the paper lacks a crisp, formal distinction and necessity argument beyond empirical gains. The conceptual piece (Theorem 3.1) is essentially a basis‑change statement that any matrix can be written once spans are fixed; it does not substantively justify why subspace preservation should hold across tasks. 

Key assumption left untested in main text
The central claim is that principal subspaces largely coincide (99–99.99% similarity), which motivates freezing and training only matrices​. But evidence is relegated to the appendix; the main paper neither analyzes sensitivity when this assumption fails nor quantifies task regimes where subspace drift is large (e.g., domain shift, compositional generalization). Without these stress tests, the assumption risks being tautological on the chosen benchmarks. 

Compute–memory trade‑off and wall‑clock are not characterized
While memory accounting is detailed, compute overhead for forward/backward with factorized weights is unreported. In particular, the term for training can be non‑trivial when alpha is high (Table 4 chooses α=0.95\alpha=0.95α=0.95 as “best”), potentially negating real‑world speedups versus LoRA with small ranks. No throughput or training‑time measurements are provided, nor any activation‑memory analysis under checkpointing—both are crucial to on‑device fine‑tuning claims. 

Fairness of baselines and budgets is unclear
The paper compares total memory numbers but does not fully specify optimizer choices (e.g., Adam states vs. 8‑bit optimizers), precision (bf16/fp16), or whether baselines are permitted standard optimizations (e.g., LoRA rank scaling per layer, dropout, bias tuning, or combining with quantization). For example, LoRA is presented largely at rank=32 on specific matrices, and Figure 7 uses a dashed line for LoRA (Q,V, r=32) as a fixed reference; it’s not obvious that comparisons are under matched accuracy or matched memory constraints with tuned hyperparameters for each method. Conclusions about a “clearly superior accuracy–memory trade‑off” (Fig. 2) therefore feel premature. 

Memory accounting contains assumptions that need justification
Table 1 assumes particular gradient/optimizer state sizes for each method and claims that SVFit/SVFT effectively double memory because of storing U,VU,VU,V, whereas CERSA stores truncated U,VU,VU,V and a trainable SSS. But critical details are missing:

Are U,VU,VU,V stored in full precision or quantized?
Are optimizer states fp32 or 8‑bit?
Figure 1 shows absolute MB for ViT‑Large but omits per‑layer ranks.

Limited scale and task diversity relative to stated ambition
The abstract claims evaluation across “models of varying scales and domains, including image recognition, text‑to‑image generation, and natural language understanding,” yet the main paper contains no T2I results and only one NLU backbone (DeBERTaV3‑Base). Given the method targets large models, absence of results on standard LLMs (e.g., 7B–70B) or modern vision backbones (e.g., ViT‑Huge in full experiments, not just in compression curves) weakens the external validity of the claims. The text says T2I and OOD are in the appendix; key findings should be surfaced in the main paper. 

Statistical rigor is insufficient
The results tables lack variance/confidence intervals, numbers of runs, and seed sensitivity. Several reported gains over strong baselines are on the order of 0.2–0.5%, which can be within run‑to‑run noise on these datasets; without error bars, it is hard to judge whether improvements are robust. 

Ablations do not probe critical design choices enough
The ablations touch on matrix choices but following are missing:
per‑layer auto‑tuning of alpha under a global memory budget;
the effect of freezing part of the principal subspace on catastrophic forgetting across tasks.
the impact of quantizing U,VU,VU,V or regularizing (e.g., orthogonality/low‑rank priors) to mitigate over‑parameterization inside the subspace. 

Theory is lightweight and does not predict when CERSA helps/hurts
Theorem 3.1 formalizes a basis‑change factorization, but it does not connect spectral energy retention to task loss or generalization. The crucial empirical premise (subspace preservation) is asserted, not derived; there is no analysis of how large alpha must be to bound approximation error in terms relevant to downstream accuracy. This limits the theory’s explanatory power.

### Questions
Compute/time vs. memory.
Please report end‑to‑end wall‑clock training time and throughput (images/s or tokens/s) under the same hardware/precision for FT, LoRA, SVFit/SVFT, and CERSA. How does the extra cost for training affect speed at α=0.95\alpha=0.95α=0.95 on ViT‑Large and DeBERTaV3‑Base? Include activation memory under checkpointing if used. 

Baseline fairness and hyperparameters.
What optimizer, precision (bf16/fp16), gradient scaling, and 8‑bit optimizer usage were allowed for each method when computing Table 1 and Figures 1–2? Were LoRA ranks and target matrices tuned per dataset under matched memory budgets (not fixed r=32)? Provide a table mapping memory budgets ↔ ranks for each baseline.

Memory accounting details.
How are U,VU,VU,V stored (precision/quantization)? Are optimizer states 32‑bit? In Table 1, why is e>me>me>m assumed for SVFT, and how sensitive are the totals? Please include per‑layer values and the precision used to reproduce Figure 1. 

When does subspace preservation fail?
Can you present main‑paper results (not only appendix) quantifying Grassmann similarity across diverse domain shifts and show CERSA’s performance as similarity decreases? For example, train on ImageNet‑21K‑pretrained ViT but fine‑tune on a strongly out‑of‑distribution dataset and compare to LoRA / FT. 

Large‑model applicability.
Do results hold for LLMs (7B–13B) and larger ViTs beyond illustrative compression curves (e.g., full experiments on ViT‑Huge)? Please include at least one autoregressive language task to validate claims in the generative regime, or move such claims from the abstract if not supported. 

Regularization and stability inside the subspace.
Did you try constraining (e.g., spectral norm, low‑rank penalty, orthogonality) to reduce over‑parameterization? Does such regularization close the gap to FT on tasks where CERSA underperforms or improve OOD robustness? 

Automatic rank allocation.
Instead of fixed alpha, can you optimize per‑layer under a global memory budget via a knapsack‑style procedure or gradient‑based sensitivity, and does this improve the accuracy–memory Pareto frontier ? 

Inference‑time cost and deployment.
At inference, do you keep the factorized form (U, S, V) or recompose? Please report latency and peak memory for both options, and comment on compatibility with fused kernels. 

Variance and statistical significance.
Provide mean±std over ≥3 seeds for Tables 7–8 and mark statistically significant wins. Several deltas are small and could be noise without this. 

Ablations on β\betaβ and forgetting.
You note that β<α\beta<\alphaβ<α can freeze more principal components to preserve pretrained knowledge. Please include a forgetting measure (e.g., zero‑shot or linear‑probe evaluations on pretraining‑like tasks) versus β\betaβ, and compare to LoRA.

### Soundness
2

### Presentation
2

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
This paper proposes a PEFT method that leverages SVD to compress pre-trained weights by retaining only principal components that preserve 90%-95% of cumulative spectral energy. CERSA introduces a trainable matrix initialized with the top-k singular values, enabling better expressiveness while maintaining memory efficiency. Extensive experiments on vision and NLU tasks demonstrate that CERSA outperforms other PEFT methods while achieving comparable or lower memory consumption than these baselines.

### Strengths
1. The paper proposes a straightforward solution via layer-wise cumulative energy retention. The three-way decomposition (discarded, frozen, trainable components) is intuitive and well-visualized in Figure 4.

2. Theorem 3.1 provides reasonable theoretical grounding by arguing that the principal subspaces remain largely invariant during fine-tuning, which justifies freezing $U$ and $V$ while training an intermediate matrix $S$.

3. The paper includes extensive ablations across multiple dimensions and evaluates on diverse benchmarks.

### Weaknesses
1. The paper does not compare against other strong LoRA variants like DoRA, AdaLoRA, HiRA, etc., or discuss why training a full $k \times k$ matrix $S$ is always preferable to other parameterizations. Additionally, there is no analysis of computational overhead during SVD preprocessing or fine-tuning speed beyond the single loss curve in Figure 6.

2. The core innovation—using cumulative energy retention for layer-wise rank selection instead of uniform rank—is relatively incremental. While the trainable matrix $S$ is an improvement, it is a straightforward extension.

3. No error bars, confidence intervals, or multiple runs are reported for any results. Given the relatively small performance differences between CERSA and strong baselines (e.g., PiSSA: 89.5% vs. CERSA: 89.5% on GLUE), it is unclear whether differences are statistically significant or due to random variation.

4. Beyond memory footprint, the paper lacks wall-clock training time comparison with LoRA and other baselines. Since CERSA requires SVD preprocessing and changes the forward pass from $W+BA$ to $U·S·V^T$, it's unclear whether the method is practically faster or slower than LoRA in terms of training throughput (samples/second) and total training time. This is crucial for practitioners deciding whether to adopt CERSA.

### Questions
How does CERSA perform on tasks with significant distribution shift from pre-training? Can the authors characterize the boundary conditions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a parameter-efficient finetuning (peft) method based on singular value decomposition (svd). The pretrained weights are decomposed into $U, V, \Sigma$ matrices. Then, the singular values and corresponding rows/columns in $U, V$ that do not contribute to the top $p%$ of cumulative energy are entirely discarded. A ratio $\beta$ determines how many of the retained subspaces are trained. The final trainable parameters is a $r\times r$ matrix that maps between the top $r$ subspaces in $U, V$ while keeping the remaining subspaces and corresponding singular values frozen.

Experiments on image classification, text classification, and subject-driven image generation evaluate the advantages and disadvantages of the proposed method, finding improved performance at lower memory consumption.

### Strengths
**(S1)** The evaluation is broad, covering different models, modalities, and tasks.

**(S2)** Memory savings are demonstrated convincingly, and actual memory requirements are shown throughout the paper.

**(S3)** The method, its motivation, and properties are discussed in detail, and the motivation is convincing.

**(S4)**  The paper includes an ablation of catastrophic forgetting in Appendix E, where CERSA is preserves performance on unrelated tasks better than other fine-tuning methods. This is briefly mentioned in the main paper, but this is actually a very interesting and important analysis.

**(S5)** The visual in Fig. 4 is very helpful in clarifying the method

### Weaknesses
**(W1)** One main concern is how statistically significant the performance improvement is: Both the GLUE benchmark and the FGVC benchmark (Tab. 7 and Tab. 8) are close to saturation, so we cannot expect to observe large differences between methods. In fact, this is also not the case. Advantages of CERSA are generally < 1% average accuracy. A more challenging benchmark could help highlight the strengths of CERSA better.

**(W2)** The evaluation is broad, but one important domain that is missing is language generation. Given its importance, experiments on LLM fine-tuning, such as instruction tuning, would make the paper significantly stronger.

**(W3)** One aspect I am concerned about is performance degradation through discarding low-energy spaces in SVD. I think a separate analysis without any training, where the effects of SVD truncation on pre-trained weights are compared to base model performance, would be helpful in understanding this better. This analysis should also encompass different domains in classification and generation.

**(W4)** Minor: F.2 does not state the theorem, only a proof. This is confusing to readers.

**(W5)** Minor: In Appendix F.1, it would be interesting to see a comparison of subspace similarity for other methods as well, using the same tools. Currently, we do not have any point of reference, so it is unclear what the measured similarities mean concretely and how to put them in relation.

### Questions
The paper is already comprehensive, and the supplementary material contains many interesting analyses. However, I have concerns regarding the performance advantage of the proposed method that I hope will be addressed in the rebuttal:
  * Can the significant advantages of CERSA over baselines be demonstrated more clearly, e.g., by using less saturated benchmarks?
  * Does CERSA also improve language generation fine-tuning?
  * How much does SVD truncation affect pre-trained weights without any training, i.e,. how catastrophic is this intervention?

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
5

### Summary
This paper proposes CERSA, a PEFT method within a principal energy subspace.

This energy subspace is obtained by he truncated SVD of pre-trained weights, keeping only the components that retain 90\%–95% of spectral energy, yielding large memory savings over existing full FT and LoRA-style PEFT methods.

Besides, the proposed method shows a very competitive performance against the state-of-the-art methods on several benchmarks.

### Strengths
+ This paper clearly presents a layer-wise truncation mechanism based on cumulative energy.

+ The experiments and visualization in this paper are extensive and solid.

+ This paper reports competitive results on a diverse set of tasks, with efficiency.

+ This paper is overall easy-to-follow.

### Weaknesses
- More theory justification on the subspace invariance should be made. This is becauuse, the update geometry and useful directions are often not fixed low-rank, or align neatly with pre-trained singular directions.

- Following up this issue, CERSA is evaluated where pre-trained features plausibly align with downstream needs (e.g., standard benchmark GLUE). But the core risk of freezing subspaces is distribution shift. Therefore, the proposed method should do some validation on this aspect to justify its feasliblity and robustness, for example on fine-grained or domain-shifted vision, and/ or retrieval-augmented settings.

- he compared state-of-the-art PEFT methods are significantly missing. Some more recent and much stronger PEFT methods are mssing for comparison, for example:

[1] VeRA: Vector-based Random Matrix Adaptation. ICLR 2024.

[2] Foura: Fourier low-rank adaptation. NeurIPS 2024.

[3] SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation. NAACL 2024.

- A sensivity analysis on $r$ value, in conjunction with different methods, should be necessary to show its robustness. 

- More experimental justification is needed on whether the proposed method is dependent on specific ViT shapes (or not).

- More implementation details, for example, on how $U$, $V$ are materialized and shared, should be made.

### Questions
Please refer to the weakness section, and address the concerns point-by-point.

### Soundness
3

### Presentation
3

### Contribution
3
