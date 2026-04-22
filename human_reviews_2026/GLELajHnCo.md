# GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
Domain-specific embedding models have shown promise for applications that require specialized semantic understanding, such as coding agents and financial retrieval systems, often achieving higher performance gains than general models. However, state-of-the-art embedding models are typically based on LLMs, which contain billions of parameters, making deployment challenging in resource-constrained environments. Model compression through pruning offers a promising solution, but existing pruning methods treat all parameters uniformly, failing to distinguish between general semantic representations and domain-specific patterns, leading to suboptimal pruning decisions. Thus, we propose GAPrune, a pruning framework that addresses this challenge by considering both domain importance and preserving general linguistic foundation. Our method uses Fisher Information to measure importance and general-domain gradient alignment to assess parameter behavior, then combines these signals using our Domain Alignment Importance (DAI) scoring. Lower DAI scores indicate that the parameter is either less important for the domain task or creates conflicts between domain and general objectives. Experiments on two domain benchmarks, FinMTEB and ChemTEB, show that GAPrune maintains performance within 2.5\% of dense models in one-shot pruning at 50\% sparsity, while outperforming all baselines. With retraining in 100 steps, GAPrune achieves +4.51\% improvement on FinMTEB and +1.73\% on ChemTEB, demonstrating that our pruning strategy not only preserves but enhances domain-specific capabilities. Our findings demonstrate that principled pruning strategies can achieve model compression and enhanced domain specialization, providing the research community with a new approach for development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method for conducting domain-aware pruning for embedding models.The method comes in three different stages: (1) gathering representative data points from general domain dataset and domain-specific dataset using k-means sampling ($k = 5000$); (2) calculating parameter importance as measured by gradient over InfoNCE loss on general domain samples and domain-specific samples, as well as gradient alignment between the general and specific domain; (3) computing a domain-aware importance (DAI) score using the importance and alignment measures above, then prune parameters based on the DAI score. Experimental results show better performance compared to baselines such as vanilla dense embedding and magnitude pruning, and better compatibility with domain-specific re-training after pruning.

### Strengths
1. The paper is very well-written. The approach has many components, but all the steps are very well explained and mostly well-motivated.
2. The method achieved significantly better performance over strong baselines under the re-training setup.

### Weaknesses
1. Under the setup without re-training, the performance improvement over more simplistic methods like magnitude pruning is minimal, so I'm not entirely convinced if the proposal is really worth it when no re-training is performed.
2. The necessity of each steps would be better justified if more rigorous ablation is performed. For example, while I understand the motivation of the gradient alignment term, I'm not sure if pruning "domain specific" parameters serves the stated goal of domain-aware embedding pruning. In defense of this proposal, this pruning choice might carry some benefit when re-training is performed (because re-training can recover domain-specific knowledge), but this needs to be validated by empirical experiments.

### Questions
Can the authors further comment on the motivation behind the $(1 + \alpha s^j_g)$ term? I might have missed or misunderstood some details.

Minor nit comment: L629 in Algorithm 1 seems like a self-assignment? I think you can either get rid of this line, or assign the magnitude to a variable.

### Soundness
3

### Presentation
4

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
The submission proposes a pruning method for domain-specific embedding models that scores parameters with a Domain-Alignment Importance (DAI) metric combining (i) domain Fisher importance and (ii) gradient alignment between general and domain objectives. On FinMTEB and ChemTEB, GAPrune outperforms magnitude/Fisher baselines at 30–50% sparsity and, after 100 retraining steps, exceeds dense performance in several settings.

### Strengths
This submission Clear formulation of DAI with intuitive justification.

Strong results showing their work outperforms some existing methods.

Helpful auxiliary analyses (layer-wise correlations/performance; embedding geometry) in the Appendix.

### Weaknesses
Assumption support is limited. The central claim that parameters exhibit domain-dependent behavior is plausible but not directly validated; Either citing this conclusion from existing literature or providing theoretical or empirical analyses to support this assumption would be more convincing. A targeted test (e.g., per-parameter behavior shifts across domains) would help.

Lack of interpretability. While the multi-component DAI framework appears well-motivated, it lacks formal empirical or theoretical analysis on each component. Consequently, the work appears to be reliant on intuition rather than on rigorous validation.

Hyperparameter sensitivity concerns. The designed approach introduces three new hyperparameters: $\alpha$, $\beta$, and $\gamma$. The submission does not have a discussion about the sensitivity to these hyperparameters. An ablation study or sensitivity analysis is crucial to understand this designed approach.

Reproducibility. Datasets/models/sparsity and “100 steps” are stated, but key training details (optimizers, LR, batch sizes, seeds) and code availability are missing.

Missing KD comparison. Knowledge Distillation is a prevalent deployment strategy for embedders and isn’t discussed or compared.

Related work concerns. Motivation cites several non-peer-reviewed sources to show “The demand for domain-specific embedding models has grown significantly ”, which is not convincing. Adding stronger peer-reviewed evidence would strengthen the case for domain-specific embedders’ importance.

### Questions
Empirically validate domain-dependent parameters (e.g., measure per-parameter gradient/activation shifts across domains and tie to pruning outcomes).

Add sensitivity/ablation for $\alpha$, $\beta$, and $\gamma$ and sampling choices (k-means subset size).

Discuss/benchmark against KD baselines for efficiency.

Tighten related-work positioning with more peer-reviewed citations backing domain-specific embedders.

Provide fuller training details and release code.

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
3

### Summary
The author proposes GAPrune, a pruning method for embedding models. GAPrune has two main components: they use Fisher information to measure the importance of parameters. In addition, they use gradient alignment to identify general and domain-specific parameters.

### Strengths
1. The authors show consistent improvements over random pruning.
2. The authors study pruning for embedding models, which can be very useful.

### Weaknesses
1. This paper has no ablation study. The authors combines Fisher information with their DAI scores. However, it's unclear how much improvement each contribute individually.

2. The authors unnecessarily restrict their scope to embedding models, but their approach is not specific to embedding models to my understanding. It would be ideal if the authors can test their approach in general LLMs, and compare their method with SOTA pruning methods.

3. The results are generally close to magnitude-based pruning. Looking at the Table 1, the performance is very close to a simple magnitude-based pruning.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
