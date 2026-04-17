# It's all in the heads: An investigation of domain knowledge infusion into LLMs

- Decision: Reject
- Scores: 4, 4, 8

## Abstract
While large language models (LLMs) are widely studied, the mechanisms by which they internalize knowledge from specialized domains remain poorly understood. To investigate this, we analyze the continual pre-training (CPT) paradigm, where a base model is further pre-trained on a curated, domain-specific corpus. Through a study on diverse domains, including mathematics, instruction, code, and text data, we uncover novel properties of this process. By analyzing SVD decompositions of model weights we determine that the difference before and after CPT can be attributed predominantly to changes in singular vectors. We identify **head heterogeneity** in the behavior of attention weight matrices. We investigate the effect of rewinding attention heads on model quality by ordering them according to various scalar criteria. Based on our analysis we propose a novel head importance criterion which allows to either truncate up to **60**% heads in the model increment or to achieve up to **4**% quality increase upon partial head rewinding to the pre-train state. Further, we discover **domain connectivity** — *i.e.*, the ability to linearly interpolate between CPT checkpoints on different domains without significant quality loss, and discuss key quality drivers of this phenomenon. To foster further research, we provide an open-source scalable toolkit for performing spectral analysis on models with billions of parameters - NetInspect. The code is
available at https://anonymous.4open.science/r/netinspect-EF67

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates how continual pre-training (CPT) infuses domain knowledge into large language models through spectral analysis of attention heads. Using OLMo-2-1B as a testbed, the authors find that domain adaptation primarily rotates singular vectors rather than changing singular values, and that many heads or singular components can be pruned with minimal loss. The analysis is supported by the open-source toolkit NetInspect.

### Strengths
- **Relevant topic:** Understanding how domain-specific knowledge is absorbed in LLMs is timely and valuable.
- **Systematic analysis:** Provides multiple spectral diagnostics (norms, ranks, SVD fits, vector agreement).
- **Open-source toolkit:** The release of *NetInspect* adds reproducibility and utility for future research.
- **Empirical findings:** Observations on head-level redundancy and domain connectivity are potentially insightful for model merging and interpretability.

### Weaknesses
1. **Statistical robustness & reproducibility.** Report mean ± std (or bootstrap CIs) across multiple random seeds for all key curves/thresholds (e.g., head removal, SVD truncation, interpolation results). Add exact seeds and number of repeats in the appendix.
2. **Dataset / evaluation choices.** If feasible, add other benchmarks (additional math datasets, code or reasoning tasks) to test generality.
3. **Architecture / scale generality.** Demonstrate (or empirically argue) that the reported spectral phenomena and domain-connectivity scaling hold beyond OLMo-2 1B — e.g., a different backbone or model scale, or provide a clear justification why OLMo-2 is representative.
4. **Mechanistic claims need stronger evidence.**  Provide a clearer mechanistic hypothesis or conduct targeted control experiments (shuffled/noise data, optimizer ablations, perturbation tests) before asserting causal explanations. 
5. **Practical guidance & safety of sparsification.** Expand discussion on how to choose heads/singular values safely (per-layer heterogeneity, CPT token budget dependence), and evaluate risks for downstream / rare tasks to avoid misapplication of pruning advice.

### Questions
See above.

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
4

### Summary
This paper investigates the key phenomena occurring during Continual Pre-training (CPT), an important stage in modern LLM development. The authors observe that the pre-training stage alters the internal structure of the model's weight matrices, forming a complex singular value spectrum in the attention heads that simple random matrix theories cannot adequately describe. Furthermore, the paper posits that CPT injects new knowledge primarily by rotating the singular vectors of the weight matrices, rather than altering their singular values. Key findings include the identification of domain connectivity, which is the ability to interpolate between checkpoints after CPT on different domains, and the discovery of head-wise sparsity in the model increment that encodes new domain knowledge. The paper also provides an open source implementation to ensure reproducibility.

### Strengths
1. Significance: The CPT stage has received far less attention than supervised fine-tuning (SFT). This study offers valuable new insights into the mechanisms of domain adaptation that occur during CPT.  
2. Quality and Clarity: The experimental design is rigorous and thorough. The figures are clear and informative.  
3. Reproducibility: The appendix provides extensive details supporting reproducibility, and the authors offer an open-source implementation to ensure that the results can be reliably reproduced.

### Weaknesses
1. Generalisability: The study's generalisability is a concern, as the analysis is limited to a single model architecture (OLMo 2, 1B) and primarily the target domain of Mathematics.  
2. Lack of Theoretical Depth: Whilst the experiments are comprehensive, the theoretical explanations for the observed phenomena, such as the complex spectral structures and the improved interpolation quality, remain preliminary. The work would benefit from more rigorous mathematical modelling and analysis.  
3. Practical Implications of Sparsity: While the paper's main contribution is clearly its scientific insight into how continual pre-training works, it would be even stronger if it also discussed how these findings could be applied in practice, especially the "high rank yet compressible" characteristic of the CPT delta. Such a discussion would make the work more useful to a wider audience.

### Questions
1. Could the authors provide supplementary results from more/larger models and more target domains to strengthen the support for the generalisability of the observed phenomena?   
2. Regarding the geometric interpretation of the "domain connectivity" interpolation path: The transition of the interpolation path from concave to convex in Figure 6, as pre-training duration increases, is a particularly interesting phenomenon. The authors briefly suggest this may be related to the decreasing Frobenius norm of the difference between the weights of the maths and text checkpoints. Could the authors elaborate on this hypothesis in more detail?  
3. On the redundancy and compressibility of the CPT delta: The paper finds the CPT delta to be high rank, yet simultaneously compressible. This appears to contrast with the fundamental assumptions of low rank adaptation methods like LoRA, which are mainly used in supervised fine tuning (SFT). Could the authors discuss the similarities and differences between this "high rank yet compressible" characteristic of the CPT delta and the common observation that SFT delta are typically low rank?  Furthermore, could insights from this comparison be leveraged to develop more efficient CPT strategies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper analyzes how LLMs learn domain knowledge during Continual Pre-Training (CPT) using spectral analysis (SVD) on the weight deltas of 1B models. The authors identify two key findings: (1) "domain connectivity," showing that models CPT-trained on different domains (e.g., math and text) can be linearly interpolated, and this interpolation quality improves significantly with longer base model pre-training. (2) Head-wise sparsity, demonstrating that a significant fraction (up to 35%) of attention heads in the CPT delta are redundant and can be rewound to their pre-trained values without significant performance loss. The paper also provides the important insight that domain adaptation occurs primarily via singular vector rotation, while the singular value spectrum remains stable.

### Strengths
1. Novelty and Significance of "Domain Connectivity": The paper's contribution is identifying "domain connectivity" in CPT models and showing that the quality of this linear interpolation becomes more convex (i.e., better) with longer pre-training. This is a key insight for the entire lifecycle of LLM development.

2. The paper provides concrete, quantitative results on the redundancy of CPT deltas. Finding that 35% of heads can be rewound and 20% of the delta's singular values can be truncated is impactful for future work on efficient adaptation and merging.

3. The paper provides code for both the experiments and the NetInspect analysis library.

### Weaknesses
1. The study is conducted on a single model size (1B) and a single architecture (OLMo 2). It is unclear whether the findings generalize to other model architecture
2. The most effective head-pruning strategy presented is a "greedy" rewind based on single-head impact (Fig 7a). This method appears to be computationally prohibitive, as it seems to require a separate evaluation for every head.

### Questions
1. Does the increasing convexity has anything to do with learning rate? Did you run different token budget in totally separate experiments, or using early checkpoints as model for smaller token budget?

### Soundness
3

### Presentation
4

### Contribution
4
