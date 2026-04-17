# From Compression to Expression: A Layerwise Analysis of In-Context Learning

- Decision: Reject
- Scores: 2, 8, 4, 6

## Abstract
In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks without weight updates by learning from demonstration sequences. While ICL shows strong empirical performance, its internal representational mechanisms are not yet well understood. In this work, we conduct a statistical geometric analysis of ICL representations to investigate how task-specific information is captured across layers. Our analysis reveals an intriguing phenomenon, which we term *Layerwise Compression-Expression*: early layers progressively produce compact and discriminative representations that encode task information from the input demonstrations, while later layers express these representations to incorporate the query and generate the prediction. This phenomenon is observed consistently across diverse tasks and a range of contemporary LLM architectures. We demonstrate that it has important implications for ICL performance---improving with model size and the number of demonstrations---and for robustness in the presence of noisy examples. To further understand the effect of the compact task representation, we propose a bias-variance decomposition and provide a theoretical analysis showing how attention mechanisms contribute to reducing both variance and bias, thereby enhancing performance as the number of demonstrations increases. Our findings reveal an intriguing layerwise dynamic in ICL, highlight how structured representations emerge within LLMs, and showcase that analyzing internal representations can facilitate a deeper understanding of model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the "layerwise" spread of information compression in ICL, where early layers compress task information and later layers express it. The authors' proposed approach TDNV (Task-Distance Normalized Variance) quantifies compression and observe U-shaped patterns across models and tasks. While the empirical scope is extensive, concerns about novelty relative to recent work on layerwise information dynamics, generalizability across modern architectures, and lack of statistical rigor in the experiments limit the contribution.

### Strengths
- Comprehensive empirical evaluation across architectures and modalities
- Clear presentation with intuitive visualizations
- Practical applications such as TDNV-based layer selection

### Weaknesses
### **Major**

**Generalizability and credibility of the findings:** The motivation and method appear very similar to Skean et al.'s work [1], which studies information compression across layers with variance-based technical analysis. The authors essentially replicate this using the same models. Studying ICL in this context offers limited novelty, as ICL is merely one instance of task/information encoded in text.  

Furthermore, recent work [2] demonstrates that information compression doesn't necessarily follow a U-shaped pattern, despite common claims [1, 3, 4]. Saglam et al. [2] show that while the U-shaped pattern exists in models tested here (Llama, GPT), it doesn't hold in others like Mistral and Gemma. This indicates information compression patterns heavily depend on model architecture, rendering the paper's findings questionable.   

The authors observe U-shaped patterns across all tested models. Without counter-evidence, they should expand their model selection to ensure academic rigor. So I don't think this work is sufficiently concrete and novel for publication.

**Causal vs. geometric interpretation:** The paper claims compression "extracts task information from demonstrations" and expression "applies to the query," but TDNV measures geometric properties (clustering, separation), not causal information flow. Saliency maps (Appendix D.1) provide suggestive evidence, but more rigorous connections are needed:
- information-theoretic analysis (e.g., mutual information between demonstrations and task vectors),
- causal interventions (e.g., ablating early vs. late layers and measuring downstream effects),
- or explicit framing of geometric patterns as correlates rather than causes of information compression.

**Statistical rigor in experiments:** Results lack error bars, confidence intervals, or statistical significance tests.

---

### **Minor**
**Context length terminology:** The variable $K$ is defined as the number of ICL demonstrations (lines 353-354) but later termed "context length" (lines 1199-1200). These are fundamentally different. If $K$ represents context length, the experiments become questionable as $K \leq 20$ is too low for meaningful ICL.

---

### **Suggestions for Improvement**
- **More models:** I strongly recommend including at least one of Gemma or Mistral, as they reportedly lack the U-shaped pattern [2].

- **"Theorem" framing:** Presenting a simple bias-variance decomposition as a "theorem" is an overclaim. I suggest framing it as a "proposition" or "analytical result," especially given the linear attention and single-layer assumptions, though I acknowledge this is common practice. Theorem 1's assumptions (linear attention, single layer, i.i.d. demonstrations) diverge significantly from practice. Empirical validation of $\mathcal{O}(\frac{1}{K})$ rates in actual transformers remains incomplete.

- **Other geometric metrics:** Comparing TDNV to alternative metrics (e.g., nearest-class-center variance, silhouette score) would strengthen the methodological justification. Is TDNV uniquely informative or do other metrics show similar patterns?

---
**References**

[1] Skean et al. _Layer by Layer: Uncovering Hidden Representations in Language Models._ ICML 2025.    

[2] Saglam et al. _Large Language Models Encode Semantics in Low-Dimensional Linear Subspaces._ arXiv preprint (2025).    

[3] Valeriani et al. _The geometry of hidden representations of large transformer models._ NeurIPS 2023.

[4] Razzhigaev et al. _The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models._ Findings of EACL 2024.

### Questions
None

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper analyzes how Large Language Models (LLMs) perform in-context learning (ICL) by examining internal representations across layers. The authors discover a "Layerwise Compression-Expression" phenomenon: early layers progressively compress demonstration information into compact, discriminative task representations, while later layers express these representations using the query to generate predictions. Using Task-Distance Normalized Variance (TDNV) metrics, they show this pattern occurs consistently across diverse architectures, tasks, and modalities. The compression improves with model size and more demonstrations, explaining ICL performance gains. They provide theoretical analysis showing attention mechanisms reduce both bias and variance. The findings reveal how LLMs extract task information from demonstrations, offering insights for improving interpretability and robustness.

### Strengths
1. The "Layerwise Compression-Expression" phenomenon is a genuinely interesting discovery that provides new understanding of how ICL works internally. The observation that models split into distinct compression and expression phases is intuitive yet non-obvious. 
2. Extensive experiments across multiple architectures (Transformers, Mamba), model scales, and task domains (symbolic, language understanding, multimodality). Shows the phenomenon emerges during training and isn't present in random models. 
3. Provides mathematical analysis through bias-variance decomposition. Theorem showing how attention reduces both bias and variance with more demonstrations. 
4. Clear figures that effectively illustrate the phenomenon

### Weaknesses
1. TDNV metric assumes tasks are well-separated and distinct, which may not hold for subtle task variations. The metric may not capture all aspects of task representation quality. Reliance on the last separator token as the task vector is somewhat arbitrary (though they do ablate this). 
2. Experiments primarily focus on relatively simple, well-defined tasks.  Limited exploration of more complex, real-world ICL scenarios where task boundaries are unclear. Most experiments use relatively small models (except for a few with larger models). 
3. While the paper describes WHAT happens, it provides limited insight into WHY this specific pattern emerges. Doesn't fully explain why compression must precede expression or why the transition occurs at specific layers. 
4. Unclear how findings apply to tasks requiring complex reasoning or multi-step inference. Limited investigation of how the phenomenon varies with different prompt formats or instruction styles.

### Questions
1. How does the compression-expression pattern change with task complexity? For instance, in multi-hop reasoning tasks, do you observe multiple compression-expression cycles?
2.  How sensitive is TDNV to the choice of distance metric and number of samples? Have you validated that low TDNV actually correlates with better downstream performance across all your tasks?
3.  Could the U-shaped TDNV curve be explained by simpler factors like attention patterns becoming more focused toward the end, rather than a fundamental compression-expression mechanism?
4.  How do your findings relate to the "induction heads"? Do these heads primarily appear in compression or expression layers?

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
3

### Summary
The paper provides a layerwise analysis of in-context learning (ICL), reporting Layerwise Compression-Expression (LCE). Early layers of transformer progressively compress the demonstration examples into a compact representation, and later layers of transformer express this information and incorporate this with the query to make the prediction. Using LCE, the authors analyze various ICL phenomenon, such as (1) larger model and more demo improves ICL (2) noisy demonstration hurts ICL. Finally authors propose task-vector contrastive fine-tuning method to improve ICL.

### Strengths
- The paper is overall well-written and structurally sound.
- The paper provides carefully designed metrics and experiments to analyze ICL.
- The fact that decoder-only LLM also acts as encoder-decoder model is interesting.

### Weaknesses
- The ICL tasks are synthetic and simple. Also models are small even compared to sLMs. 
- Since this is tested on models solely trained for specific ICL tasks, LCE phenomenon might not be applicable to actual general-purpose LLM, which is of real interest.
- In the same manner "explains why larger models and longer contexts yield better performance" could be an over-statement.

### Questions
- Prior works have shown that there exists a positional bias for general-purpose LLM on ICL tasks, which can vary depending on the tasks and models. I wonder "perturbing demonstrations
that appear later in the sequence causes a larger performance drop" would be true for those models as well.
- Would it be possible to observe the encoder-decoder structure in pretrained LLM as well?
- The proposed task-vector contrastive fine-tuning is solely proposed for ICL fine-tuning. What would be its practical application?

### Soundness
3

### Presentation
4

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
This paper investigates the layerwise dynamics of in-context learning (ICL) in large language models. It introduces a metric called Task-Distance Normalized Variance (TDNV) to characterize how task information is compressed in early layers and expressed in later layers. The study includes experiments across multiple model architectures and task types, and analyzes how factors such as model size and demonstration count affect this pattern. A bias–variance decomposition of task vectors is also presented, along with applications for identifying optimal layers and improving task representations.

### Strengths
1. The paper proposes TDNV as a new metric for analyzing in-context learning (ICL), and presents a clear two-phase compression–expression pattern in layerwise representations. The experimental setup is thorough and supports the empirical observations across different model sizes and settings.

2. The authors provide a bias–variance decomposition of task vectors, offering a structured interpretation of how representation quality evolves with the number of demonstrations. This analysis contributes to a more systematic understanding of ICL dynamics.

3. The work applies its findings to practical objectives by identifying the optimal layer for extracting task vectors using TDNV and introducing a contrastive fine-tuning strategy that improves task-vector performance. These applications demonstrate the potential utility of the proposed analyses.

### Weaknesses
1. The paper adopts TDNV as a central metric but provides limited theoretical support for why this specific ratio captures task compression or predicts ICL performance. The analysis is primarily empirical, and while correlations such as the alignment of minimum TDNV with peak task-vector accuracy are demonstrated, the underlying mechanisms remain unexplained.

2. The experimental evaluation primarily focuses on symbolic tasks such as letter transformations and simple list operations. While the paper includes additional experiments on natural language and multimodal settings, these are conducted on relatively small synthetic datasets.

### Questions
1. The paper exclusively uses TDNV to demonstrate the compression-expression phenomenon, without examining whether the pattern depends on this specific metric. Have the authors investigated whether the observed trend holds under alternative metrics?

2. The paper's evaluation focuses on symbolic tasks and small synthetic datasets, leaving the generalizability to complex reasoning tasks unclear. Have the authors evaluated whether the compression-expression phenomenon 
holds on standard ICL benchmarks such as GSM8K, MMLU, or BBH?

### Soundness
3

### Presentation
3

### Contribution
3
