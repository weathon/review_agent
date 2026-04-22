# The Achilles’ Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6

## Abstract
Large Language Models (LLMs) have become foundational tools in natural language processing, powering a wide range of applications and research. Many studies have shown that LLMs share significant similarities with the human brain. Neuroscience research has found that a small subset of biological neurons in the human brain are crucial for core cognitive functions, which raises a fundamental question: do LLMs also contain a small subset of critical neurons? In this paper, we investigate this question by proposing a Perturbation-based Causal Identification of Critical Neurons method to systematically locate such critical neurons in LLMs. Our findings reveal three key insights:
(1) LLMs contain ultra-sparse critical neuron sets. Disrupting these critical neurons can cause a 72B-parameter model with over 1.1 billion neurons to completely collapse, with perplexity increasing by up to 20 orders of magnitude;
(2) These critical neurons are not uniformly distributed, but tend to concentrate in the outer layers, particularly within the MLP down\_proj components;
(3) Performance degradation exhibits sharp phase transitions, rather than a gradual decline, when these critical neurons are disrupted.
Through comprehensive experiments across diverse model architectures and scales, we provide deeper analysis of these phenomena and their implications for LLM robustness and interpretability. These findings can offer guidance for developing more robust model architectures and improving deployment security in safety-critical applications. Our code is available at https://github.com/qqqqqqqzx/The-Achilles-Heel-of-LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper finds that LLMs contain a sparse set of critical neurons whose disruption can cause catastrophic performance failure. The authors propose a perturbation-based method to identify these neurons and evaluate it across multiple models. They report three main findings: (1) disrupting even three neurons can collapse models with billions of parameters, (2) critical neurons concentrate in outer layers and MLP down_proj components, and (3) performance degradation exhibits sharp phase transitions rather than gradual decline.

### Strengths
1. Comprehensive experimental scope: The evaluation spans 21 models across multiple architectures (Llama, Gemma, DeepSeek, Phi, Qwen) and diverse benchmarks, providing strong empirical evidence for the phenomena.

2. Clear methodology: The two-stage approach (sensitivity analysis + causal verification) is clearly presented with algorithmic details.

3. Robust validation: The paper includes thorough ablation studies examining parameter sensitivity (α, K), input length requirements (T>10), cross-lingual consistency, and fine-tuning robustness.

### Weaknesses
1. Limited Novelty of Core Findings
The paper acknowledges that concentration in MLP down_proj layers "is consistent with the observations reported in Yu et al. (2025)". The existence of "super weights" in these components is already documented. What fundamentally new insight does this work provide beyond confirming known vulnerabilities at the neuron level rather than weight level?

**Recommendation:** I would encourage to distinguish more clearly how this work advances beyond Yu et al. (2025) on "The Super Weight in Large Language Models"

2. Insufficient Theoretical Depth
While the empirical observations are extensively documented across many models, the paper lacks sufficient explanation of the underlying mechanisms. The phenomenon is thoroughly characterized in terms of what happens (phase transitions, concentration patterns, catastrophic failure), but the *why* and *how* remain underexplored.

**Recommendation:** To strengthen the contribution, I recommend providing deeper theoretical analysis grounded in optimization dynamics, information theory, or circuit analysis to explain why these vulnerabilities emerge and what they fundamentally reveal about transformer computation.

3. Unclear Practical Implications
The paper frames its findings as identifying vulnerabilities with implications for "deployment security in safety-critical applications" (line 22), yet the practical actionability of these findings remains unclear. Two critical questions are unaddressed: (1) How can practitioners suppress or mitigate the emergence of these critical neurons during training or deployment? (2) What is the realistic threat model—who would have the capability to selectively mask individual neurons in production LLMs, and under what circumstances? While Section A.9 discusses potential defenses, it reads as speculative rather than validated. The gap between identifying the vulnerability and demonstrating exploitability weakens the security framing. 

**Recommendation:** To strengthen practical impact, clarify the threat model with concrete attack scenarios, and either demonstrate mitigation strategies empirically or reframe the contribution as primarily advancing scientific understanding rather than addressing immediate security concerns.

4. Overextended Neuroscience Analogy
The neuroscience motivation (Section 1, lines 6-30) establishes useful context by drawing parallels to sparse criticality in biological neural networks. However, the analogy is invoked repeatedly throughout the paper without deepening the connection beyond surface-level metaphor. This creates an expectation for substantive interdisciplinary dialogue—for example, demonstrating whether LLM critical neurons exhibit functional properties analogous to biological critical neurons, or whether insights from neuroscience can inform mitigation strategies. As currently presented, the neuroscience framing risks appearing cosmetic rather than integral to the contribution. 

**Recommendation:** Either (1) reduce the emphasis on neuroscience to a brief motivational context in the introduction, or (2) strengthen the interdisciplinary connection by establishing concrete links—such as comparing your findings to specific neuroscience phenomena (e.g., do critical neurons in LLMs exhibit analogous plasticity, recovery patterns, or functional specialization documented in biological systems?).

### Questions
Connection to Related Phenomena: Attention Sinks
Your findings bear notable similarity to prior work on attention sinks, which identified highly important neurons critical for model fluency. Specifically, Xiao et al. (2023) documented the attention sink phenomenon (https://arxiv.org/abs/2309.17453), which was subsequently traced to highly activated neurons in specific architectural locations (https://arxiv.org/pdf/2503.08908). This mechanistic understanding has enabled cleaner interpretability analyses (https://arxiv.org/abs/2506.08010) and deeper theoretical investigation of the underlying dynamics (https://arxiv.org/abs/2504.02732).
Questions for the authors:

Is there a connection between the critical neurons you identify and the attention sink mechanism? Given that both phenomena involve sparse, highly influential neurons whose disruption causes catastrophic failure, investigating potential overlap could be illuminating.
If a relationship exists, the extensive theoretical framework developed for attention sinks (particularly https://arxiv.org/abs/2504.02732) might provide a foundation for explaining why your critical neurons emerge and how they function. This could address the theoretical depth concern raised above.

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper claims that ultra-sparse sets of “critical neurons” exist in LLMs such that masking only 3–9 neurons can catastrophically cripple models up to 72B parameters, spiking perplexity by many orders of magnitude and driving zero scores on diverse downstream tasks. These neurons are found via a two-stage procedure: (i) rank neurons by Monte-Carlo sensitivity to Gaussian input perturbations, then (ii) greedily mask neurons in rank order until a log-perplexity ratio threshold is crossed. The paper further reports that critical neurons cluster in outer layers, especially MLP down-proj, and that degradation follows a sharp phase transition.

### Strengths
1. A simple, reproducible perturbation to ranking to masking pipeline with explicit formulas and hyperparameters.

2. Broad model sweep (0.5B–72B) and a wide set of benchmarks; clear figures/tables illustrating phase transitions and architectural concentration.

3. Comparisons to random/activation/gradient ranking baselines show the proposed method finds far more damaging neuron sets.

### Weaknesses
1. The reported magnitudes are extraordinary and risk reflecting measurement artifacts rather than semantic criticality. My intuition is that if the model truly loses ability (functional), the outputs degrade smoothly and consistently across precisions. If it’s mostly numerical, the collapse often looks phase-like and precision-dependent. Particularly, given mixed-precision (fp16) inference, zeroing channels could induce norm/scale pathologies (e.g., RMSNorm statistics, residual balancing) that blow up loss numerically. Given this, pleasure (1) replicate in fp32, (b) re-compute perplexity with clamp/NaN filtering diagnostics.

2. Regarding task evaluation, I have the same concern. Repetitive punctuation is a classic sign of softmax/scale pathologies (one token gets an extreme logit and repeats, for example "\\\\\\\\\\\\\\"), which is exactly what you’d expect if masking perturbs RMSNorm/residual calibration in mixed precision—--not necessarily evidence that a few semantically critical neurons were removed.

3. The architectural concentration in outer-layer MLP down-proj echoes prior reports on outliers/super-weights/massive activations. The paper cites this, but does not tease apart whether your “critical neurons” are simply those scale outliers.

### Questions
1. Could you do a numeric robustness check by rerunning the main part of your experiments with fp32 and implement NaN/Inf filtering for perplexity calculation?

2. Can you explain how your critical neurons story relates and differs with the super-weights story essentially?

3. As I discussed above, LayerNorm can be a confounder. You may consider add controls to do a per-layer normalization vs. no-norm ablation test to isolate normalization as the failure source.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Inspired by findings from neuroscience, the paper investigates whether Large Language Models (LLMs) contain ultra-sparse critical neurons whose disruption can cause catastrophic performance collapse. The authors propose a Perturbation-based Causal Identification method with two stages: noise-based sensitivity analysis to rank neuron importance, and greedy masking to identify minimal critical sets. The authors conduct comprehensive experiments across 21 models and find three main phenomenons: masking as few as 3 neurons can increase perplexity by 20 orders of magnitude in 72B models; critical neurons concentrate in outer layers, particularly MLP down_proj components; performance degradation exhibits sharp phase transitions rather than gradual decline.

### Strengths
1. The two-stage methodology is well-motivated by neuroscience lesion studies.
2. Extensive experimental validation across 21 diverse models demonstrates robustness. Meanwhile thorough ablation studies and detailed hyperparameter analysis ensure reproducibility.
3. The greedy search algorithm reduces the computational complexity from $\mathcal{O}(2^{|N|})$ to $\mathcal{O}(|N|)$, making the search process significantly more efficient.

### Weaknesses
1. The greedy search assumes critical neurons can be identified independently, but the phase transition behavior suggests strong interdependence. The paper doesn't guarantee whether the greedy solution approximates the global optimum.

### Questions
1. How close is your greedy solution to the true minimal critical set? Have you validated this on smaller models where exhaustive search is feasible?
2. Do different input types activate different critical neurons? Your robustness analysis shows consistent identification, but does this mean the same neurons are always critical, or just that 3 neurons are always sufficient?
3. In appendix A.5, Tables 4, 5 and 6 exhibits failure modes of different models which raises a question that do critical neurons implement the same computational mechanisms across architectures, or are the mechanisms model-specific?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a small subset of neurons in language models that, when disrupted, can significantly alter the performance and comprehension of the model. These findings are tested for several model sizes.

### Strengths
1) The paper overall is written very well, the findings are clearly presented, and the ideas are well grounded in the literature. The paper flows well without any verbose/terse sections. 
2) The overall idea of the work, identifying critical neurons (model components), although not novel, is certainly an important direction of research. The methodology presented seems to hold well against the past baselines. 
3) The evaluation and ablation study presented is sound and consistent with the claims presented. 
4) The work is generalized well across diverse model families, sizes, and architectural choices.

### Weaknesses
1) The findings presented are consistent with the prior work highlighted in the paper, and hence make them unsurprising, calling into question the novelty of the work. While I do think this work has merit, primarily due to the generalization and breadth of experiments vis-à-vis prior literature, the presence of similar literature that draws similar conclusions undermines the contributions presented. 

2) The analogy to the legion experiments in neuroscience is unnecessary, as prior literature in language model research [1,2] uses similar methods as reference. Identifying critical regions in MLPs has been a research direction studied for a while, so I would highly recommend amending the analogy. 

3) The issue with the 2nd finding, the location of the clusters of critical neurons being primarily in MLP_down_proj, is that it asks more questions than are answered in the paper. Firstly, would a slightly larger set of neurons in the early layers of MLPs cause system-wide failures? As the late layer neuron compression high-dimensional attributes. Would this hold true for the middle layers? The importance of the late-layer neurons, although central to the claims made in the paper, needs a deeper analysis of the early/middle layers. If the number of neurons needed for failure in the early/middle layers is significantly higher than in the later layers, it could provide some more depth and robustness to the claims made in the paper. 

4) I would also like to see an experiment of randomly perturbing a similar cluster size of random neurons as a baseline to understand the expected degradation in model capabilities. 

5) Although I do think the work shows promise and is in a publishable state, addressing these weaknesses would add to the claims presented.  



[1] Zhang, Fred, and Neel Nanda. "Towards best practices of activation patching in language models: Metrics and methods." arXiv preprint arXiv:2309.16042 (2023).
[2 Hanna, Michael, Ollie Liu, and Alexandre Variengien. "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model." Advances in Neural Information Processing Systems 36 (2023): 76033-76060.]

### Questions
Have you performed any experiments on the attention weights of the models? 
In general, do we find these neurons in the early/middle/later layers of the models?

### Soundness
3

### Presentation
3

### Contribution
3
