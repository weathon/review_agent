# Matrix-Driven Detection and Reconstruction of LLM Weight Homology

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Recently, concerns about intellectual property in large language models (LLMs) have grown significantly,
particularly around the unattributed reuse or replication of model weights.
However, existing methods for detecting LLM weight homology fall short in key areas, including recovering the correspondence between weights and computing significance measures such as $p$-values.
We propose Matrix-Driven Instant Review (MDIR), leveraging matrix analysis and Large Deviation Theory.
MDIR achieves accurate reconstruction of weight relationships, provides rigorous $p$-value estimation, and focuses exclusively on homologous weights without requiring full model inference.
We demonstrate that MDIR reliably detects homology even after extensive mutations, such as random permutations and continual pretraining with trillions of tokens.
Moreover, all detections can be performed on a single consumer PC, making MDIR efficient and accessible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MDIR, a method for detecting weight homology between LLMs using matrix analysis and Large Deviation Theory. The approach analyzes weight matrices directly to identify if models share common ancestry through transformations like pruning, fine-tuning, or permutations, providing rigorous p-value estimates without requiring model inference.

### Strengths
- The mathematical framework connecting invariant transformation groups, polar decomposition, and Large Deviation Theory to LLM weight detection appears novel, with detailed theoretical derivations provided in the appendices.

- The method demonstrates interpretability by reconstructing specific transformations (e.g., Figure 8a shows layer correspondence in Llama-3.2-1B pruning), which could provide insights beyond binary homology detection.

### Weaknesses
- The theoretical foundation relies on assumptions that may not hold in practice. The method assumes training dynamics preserve G-coordinates under "idealized conditions (infinite numerical precision and G-invariant optimizers)" (line 131), but modern training uses Adam/AdamW (not G-invariant) and low-precision formats (fp16/bf16). While the authors note "up to 1% discrepancy" between fp32 and fp64 (line 480), there is no systematic analysis of how Adam optimization or quantization affects the invariance assumptions. The claim that "α remains constant at its initialization value" (line 134) would benefit from empirical validation across different optimizers, learning rates, and training durations.

- The evaluation of false positive and false negative rates appears limited. The study tests only 25 models primarily from 4 families (Llama, Qwen, DeepSeek, RWKV) without systematic analysis of negative pairs from diverse architectures. The threshold p < 2×10^{-23} is stated but not justified through ROC analysis or comparison with alternative thresholds. The paper would benefit from: (1) testing on models deliberately trained to be similar (e.g., same architecture and data, different seeds) to establish false positive rates, (2) varying the similarity threshold to construct precision-recall curves, and (3) explaining whether the extreme p-values (10^{-104}) reported are necessary or if detection could work with less extreme thresholds.

- The robustness to adversarial evasion is not empirically tested. Section 5 acknowledges "potential ways of evading detection" as future work, but the paper only evaluates models not designed to avoid detection. Critical untested scenarios include: (1) deliberately retraining embedding layers with higher learning rates to break the orthogonal relationship, (2) applying non-orthogonal transformations before fine-tuning, (3) adding strategic noise to weights. The claim that the method is "exceedingly difficult for adversaries to bypass" (line 71) would be strengthened by adversarial experiments. Additionally, the changed tokenizer case (Section 3.2) assumes shared tokens retain aligned embeddings, which may not hold if embeddings are retrained.

- The experimental comparison with existing methods appears incomplete. Section E.1 compares with REEF on only 2 model pairs through visual inspection, without quantitative metrics (accuracy, F1, precision-recall). The ablation study (Section 4.3) tests only same-seed initialization on two small datasets, but does not systematically vary: (1) amount of training data (would 100B tokens break detection?), (2) different learning rates, (3) continued pretraining duration, (4) mixing weights from multiple models. The computational cost is stated as "single consumer PC" but no wall-clock time or memory comparisons with baselines are provided. For the GQA transformation group construction (Section 3.1), the paper states this is a "sufficient subset" but does not prove completeness—unexplored transformations could enable evasion.

I will reconsider my score in the rebuttal.

### Questions
see weaknesses

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
2

### Summary
The paper studies model weight 'homology'. In particular, they propose a method to infer whether a model parameterized by theta_a is derived from another model theta_b. Authors consider a broad range of what constitutes 'derivation', including continual pretraining, finetuning, pruning etc (5 in total, and a combination of all). 

They propose a mathematically grounded method called MDIR to compare the similarity between the matrices of parameters, leveraging SVD and polar decomposition. Instead of setting a threshold for this similarity metric, or learning it using some ground truth, they further use Large Deviation Theory (LDT) and random matrix theory to estimate p-values (where the null hypothesis is that both models are not homologous). 

Authors then consider 25 open-source models and use MDIR to estimate whether each combination of these models can be considered homologous with respect to a certain p-value. They find their method to give strong signal for models with a known homology relationship, while giving no signal for independently developed models. 

They further add an ablation experiment studying whether their method picks up on weight initialisation versus training data similarity, confirming that it is the former. 

As a **disclaimer,** I want to clarify that, as a reviewer, my mathematical background likely does not suffice to thoroughly critique or fully appreciate the proposed method in Section 3. I therefore focus most of my review on the motivation, experimental setup, results and conclusions.

### Strengths
- The problem of identifying model homology is interesting, and the paper proposes a novel way, grounded in mathematics, to address this. 
- The fact that the method allows for the computation of a p-value a priori (no ground truth is needed) is valuable, and remains overlooked by other methods. 
- The ablation experiment in section 4.3 is very nice, and cleanly distinguishes between initialization and training data similarity.

### Weaknesses
- Authors do not consider any baselines, including some representation-based methods from prior work. I understand from the introduction that these methods "generally lack the ability to reconstruct the weight correspondence mapping", but when it comes to vizualizing the similarity metric as in Figure 3a, these methods could be presented as a valuable baseline. Moreover, to justify the mathematical complexity of the method, some more naive baselines (e.g. norm of the difference between the weights) would further ground the significance of the results. 
- While I find Section 2.2 easy to follow, Section 3 quickly becomes hard to understand. The paper could be appreciated by a wider audience if the most relevant pieces of the method remain in the main body of the paper, while the other pieces could be put in the appendix. On that note, it might also be useful to summarize the core of the method either at the end or at the start of section 3. 
- Sections 4.1 and 4.2 would benefit from more elaborate description of the results and what they mean.

### Questions
- What do authors mean by "Independently developed models" on line 366? 
- In figure 3B, could you explain why there seems to be homologous relationship between Qwen-2.5-14B and Pang-Pro-MOE? 
- An interesting set of additional experiments would be to use the pretrained models from section 4.3 and apply multiple transformations to their weights (e.g. continued pretraining, SFT+DPO, pruning) and evaluate how the similarity metric and p-value compare across transformations, and for e.g. dataset size or hyperparameters in the transformations. Such results would further illustrate the robustness of the method.

### Soundness
2

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
The paper introduces Kernelized Dynamics Pruning (KDP), a new method for pruning layers in large language models by viewing their forward computation as a discrete-time dynamical system. It observes that consecutive layers often produce highly similar representations, indicating redundancy. To exploit this, KDP projects representations into a kernel space where nonlinear transformations become approximately linear, allowing simpler modeling. A linear operator and an inverse mapping network then replace entire Transformer blocks. The authors present a theoretical error bound showing that multi-layer dynamics can be linearly approximated in this kernel space and prove that it provides superior fitting capacity compared to the original representation space. Extensive experiments on fifteen benchmarks demonstrate that KDP effectively prunes models while maintaining performance, without requiring fine-tuning. Overall, KDP provides a geometric and theoretically principled framework for simplifying internal model dynamics and reducing redundancy in large language models.

### Strengths
- Originality: Previous methods rely either on vendors implanting specific proprietary key-value pairs, or similarity measures of representations/weights. This work operates along the line of the second method, but also identifies the transformation involved (based on known symmetries of the weight matrices), which allow for a more fundamental detection of similarity. Moreover, they improve the statistical soundness of evaluating detection through a better evaluation of p-value based on large deviation theory arguments.
- Quality: The theoretical part seems solid. The experiments are extensive, showing the method for a large number of examples. The algorithm is computationally efficient and runs on a laptop, allowing broad use. The appendix also includes comparison with previous methods (REES).
- Clarity: The problem to solve is well posed and the writing is understandable and linear.
- Significance: The method seems to improve significantly the state-of-the-art of identifying unattributed reuse of LLMs.

### Weaknesses
As a general comment, it is not completely clear to me the extent to which weight relationships are included in the algorithm. The authors list a few of them, saying the list is not exhaustive. This is fine, as it is probably hard to write down an exhaustive list, but then I would expect that at least mathematically we have a clear statement of which groups of transformations satisfy the hypotheses of the method. The authors say that “we don’t need to characterize all totally invariant transformations; We need a subset with sufficiently large dimension,  which is enough for subsequent analysis with high confidence”. This is in principle explained in section 3.1, but I don’t see this specific point clearly adressed. Since this is a key point of the papers, the authors might consider refining this part to improve the manuscript.

### Questions
- Along the line of the comments above, why are transformations from theta_A to theta_B being considered linear (cfr. line 176) ?
- Just below, the authors show that Uq, Uk and Uv must be orthogonal matrices, but they don’t say anything about Uo. Should we consider this to be a generic matrix?
- The subset of W transformations is declared to be sufficient on lines 196-200. Is this proven?

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
To detect LLM weight homology, this paper proposes Matrix-Driven Instant Review (MDIR), leveraging matrix analysis and Large Deviation Theory to identify weight reuse, reconstruct transformation relationships between weights, and provide rigorous p-value estimates.

### Strengths
- The proposed method is novel and contributes to the protection of open-source model IP.
- Experiments demonstrate the robustness and practicality of the proposed method.

### Weaknesses
- The paper's core motivation relies on an "idealized" assumption (Lines 106-107). Is there related experimental evidence or research to support this?
- MDIR depends on first reliably computing $U$ from the embedding layers $E$ and $E'$. If an adversary intentionally injects large-scale, non-orthogonal noise into or retrains only the embedding layers, it could lead to an incorrect $U$ calculation, causing all subsequent layer detections to fail. There is a lack of discussion on the lower bounds of MDIR's robustness.
- There are potential overclaims, such as an undefined "consumer PC" and no discussion on computational resource consumption.
- The method is not applicable to closed-source models. In real-world scenarios, malicious users often provide API services, which limits its application.
- There are clarity issues with the paper's presentation, such as excessively small text in figures (e.g., Figures 4, 5, 6, and more) and an illogical presentation of text content in Section 4.2.

### Questions
- If multiple transformations (e.g., permutation and quantization) exist in the model simultaneously, can MDIR reconstruct them?
- Why is there no specific pattern in Figure 5(c)? The pattern in Figure 4(c) is also unclear. Is this related to the properties of MLPs?
- Is this method applicable to model merging scenarios?

### Soundness
3

### Presentation
2

### Contribution
3
