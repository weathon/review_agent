# Scaling Multimodal Temporal Graphs with Event-Adaptive Compression and Sparse Connectivity

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multimodal temporal data analysis presents a challenge: it needs to strike a balance between high resolution for capturing sudden events and a wide temporal range for scalability. This often results in vast graph models that can be computationally intractable. Current approaches tend to either break the sequences into fixed-length segments or trim edges to stay within budget constraints, often at the cost of fidelity.  

We introduce EAMC–C2SG, a novel framework that dynamically compresses temporal streams into segments tailored to events and creates a sparse graph model that respects temporal ordering. By curbing the proliferation of nodes and edges, our design achieves strict budget control while reducing complexity from a quadratic to a near-linear scale with respect to sequence length.  

Our framework preserves valuable information in multimodal temporal data and, when tested on extensive clinical datasets (MIMIC-IV + CXR) and diverse cross-domain benchmarks (TimeMMD), achieves state-of-the-art predictive accuracy with markedly lower latency and memory usage. Beyond raw performance, EAMC–C2SG also offers interpretable segmentations and insightful graph diagnostics, making it a scalable and transparent solution for multimodal temporal learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces EAMC-C2SG, a framework for scalable multimodal temporal graph learning that addresses the "node explosion" problem by dynamically compressing time-series data into event-adaptive segments and enforcing strict node and edge budgets. The method combines event-salient segmentation, a node budget controller, and a causally constrained sparse graph to reduce computational complexity from quadratic to near-linear while preserving key information. Extensive experiments on clinical (MIMIC-IV + CXR) and cross-domain (TimeMMD) benchmarks demonstrate that EAMC-C2SG achieves superior performance with significantly lower latency and memory usage compared to state-of-the-art baselines.

### Strengths
1. Clear Problem Formulation: It precisely identifies and targets the critical "node explosion" problem in multimodal temporal graphs.

2. Integrated and Novel Framework: It introduces a cohesive solution that uniquely combines event-adaptive compression, strict budget control, and causally-constrained sparse graphs.

3. Compelling Efficiency Gains: It demonstrates significantly lower latency and memory usage while maintaining superior accuracy on complex benchmarks, effectively shifting the Pareto frontier.

### Weaknesses
1. Limited Analysis of Multimodal Fusion and Cross-Modal Leakage: While the paper strongly emphasizes and validates temporal causality with its C2SG module, the handling of cross-modal causality is less rigorous. The method allows edges between temporal nodes and auxiliary image/text nodes based on cosine similarity, without any explicit causal constraints (e.g., ensuring an image is only connected to past temporal events). In a clinical setting, a chest X-ray taken at time t should not inform the model's understanding of a physiological state at time t-1. The paper does not discuss this potential for cross-modal temporal leakage or provide an ablation studying the effect of applying causal masks to cross-modal edges. This is a significant oversight for a method claiming "leakage-free message passing."

2. Sensitivity and Justification of Key Heuristics: The event-adaptive segmentation relies on a predefined threshold (θ) for boundary detection and a fixed lag window (ε) for the causal graph. The paper does not present a sensitivity analysis for these critical hyperparameters. How does performance degrade if θ is set too high or low? Is the optimal ε consistent across the clinical (MIMIC) and diverse (TimeMMD) domains, or is it highly dataset-dependent? The choice of these parameters feels somewhat arbitrary, and their impact on the final model's fidelity and efficiency should be quantified. A robustness analysis or a discussion on how to set these in a domain-agnostic way would significantly strengthen the method.

### Questions
See weaknesses

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
This paper introduces a framework that dynamically compresses temporal streams into segments tailored to events and creates a sparse graph model that respects temporal ordering. Evaluation on two datasets demonstrate the superior performance on the evaluated tasks with markedly lower latency and memory usage.

### Strengths
1. The paper proposes a multimodal temporal graph framework with an event-adaptive compression module. The overall architecture is well-motivated and technically sound.


2. It introduces a node-budget controller and a causally sparse connectivity scheme, both of which are novel and likely to be of interest to the community.


3. Comprehensive experiments and ablation studies are presented, providing strong evidence for the effectiveness of the approach.

### Weaknesses
1. The paper lists broad limitations of prior methods—e.g., “over-tokenize stable regions,” “boundary-cutting artifacts,” and “excessive edges with risk of temporal leakage”—without concrete examples, measurements, or citations. These statements should be grounded in referenced studies or small, controlled comparisons.

2. The presentation of method is confused. Key terms and losses are undefined (e.g., $L_{CL}$, $L_{BCE}$ in Line 343). Section 3 would benefit from a brief end-to-end overview, clear definitions at first use, and pointers to the appendix for implementation details.

3. Most compared methods were published before 2022; CSFformer is the only recent baseline. Including more up-to-date approaches would make the evaluation fairer and more convincing.

### Questions
1. The approach relies on several hand-tuned settings whose values and effects are not reported: the threshold (Line 225; is it shared across modalities?),  B (Line 278), and the lag window (Line 315). Please report the chosen values and include simple sensitivity studies to show robustness.

2. Equation numbers are missing; the abstract spans multiple paragraphs rather than a concise single paragraph; and Section 3 lacks a roadmap. Notation is also inconsistent—for example, if $Z_s$ denotes a segment slice (Line 210), define the corresponding set and keep it consistent with items (i)–(iii).

3. Figure 2 appears to depict an undirected graph, while the text describes a directed one. The term “candidate temporal edge” is not defined in the paper. Important elements—such as modality fusion and loss terms—are absent from the diagram, and the pie chart seems unnecessary.

4. Will the code be open-sourced? 

4. It will be more clear to explicitly show which approaches are TS-specific and which approaches are multimodal in Tables 1 and 3.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes EAMC-C2SG, a framework to scale multimodal temporal graphs. It dynamically compresses event streams, enforces a node budget, and builds a sparse causal graph. This achieves state-of-the-art accuracy with significantly lower latency and memory on MIMIC and TimeMMD benchmarks.

### Strengths
Novel Framework: Jointly optimizes event-adaptive segmentation, budget control, and sparse graph construction to solve node explosion.

Strong Empirical Results: Achieves state-of-the-art accuracy with significantly lower latency on MIMIC-IV and TimeMMD benchmarks.

Thorough Ablations: Ablation studies clearly validate the essential contribution of each component (EAMC, NBC, C2SG).

Principled Causal Design: Causal masking with a lag window prevents temporal leakage, a common flaw in sequence models.

Budget-Aware Fidelity: Maintains high fidelity and information retention even under strict node budgets.

### Weaknesses
Hyperparameter Sensitivity: EAMC boundary detection relies on multiple hyperparameters (e.g., $\theta$, top-k) that may require careful tuning

Boundary Detection Heuristic: The proxy for boundary detection (change in segmentation probability) is not strongly justified against alternatives

Simplistic Node Merging: The NBC merges unselected nodes to the *nearest* neighbor, which may be a suboptimal strategy

Fixed Causal Lag Window: The causal lag window $\epsilon$ is fixed and set manually, not learned or adapted

Narrow Event Definition: The focus on "abrupt, high-entropy" events may not generalize to tasks needing low-entropy pattern detection

Interpretablity: The link between and analysis like "Segment importance distribution," and a true interpretable method is unclear.

Error bars: Most results seem to miss reporting error ranges (like STD or CI).

### Questions
What shows that the claim scalability of the method is significant?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles what the authors call the node explosion problem, which occurs when graph-based models become too large to handle in long, multimodal time-series data. The main ideas are: (1) compressing long sequences into a smaller set of “super-nodes” based on detected events, and (2) constructing a sparse, causally ordered graph on top of those to avoid temporal leakage and improve efficiency.

### Strengths
The idea of event-adaptive compression is interesting and makes sense — it’s a more data-driven alternative to fixed window segmentation. Also, I like that the causal constraint explicitly prevents information from future timesteps from leaking backward. That design choice adds credibility to the claims about interpretability and temporal correctness. The empirical results on clinical datasets seem strong, especially given that the model reportedly runs faster and uses less memory than the baselines.

### Weaknesses
The paper has several issues that make it hard to follow and evaluate properly.

W1. Key symbols are never defined. For example, in the EAMC module, the merging weights γis\gamma_{is}γis​ depend on csc_scs​ and cic_ici​, but these aren’t defined anywhere (not in the text or Table 4). This makes it unnecessarily difficult to follow the math.

W2. The paper refers to several loss terms (task loss, contrastive loss, missingness loss) but doesn’t provide their explicit forms. It also introduces two different “total loss” equations — one in Section 3.4 and another in the appendix — which don’t match. The second version adds extra penalty terms that aren’t explained in the main text. It’s unclear which version was actually used during training.

W3. Some hyperparameters and design choices are missing. For instance, the “short moving average” used for smoothing isn’t specified (no window size), and key parameters like the causal lag window or the per-node edge budget are never mentioned. Table 5 lists only the standard ones. A short discussion or sensitivity analysis would help a lot.

W4. The section titled “Mathematical Proofs and Derivations” is misleading—it doesn’t contain any proofs, just restates previous equations.

W5. The experiments are limited to two datasets (MIMIC-IV + CXR and TimeMMD) and a small set of baselines. Given the lack of theoretical justification, a broader empirical validation would strengthen the paper. Also, the claim that the code will be released is fine, but at review time there’s no code or sufficient hyperparameter info to reproduce the results.

Overall, while the idea is solid and timely, the paper feels incomplete. The contributions are interesting, but the missing definitions, unclear objectives, and limited experiments make it difficult to assess the method’s real impact.

### Questions
Q1. Since the method has numerous hyperparameters, how would the authors justify the practicality of their approach?
Q2. Appendix C is for data processing and SOTA settings. But what is the SOTA setting in this section? I don't find the detailed experimental setup for the baselines. What does SOTA refer to?

### Soundness
2

### Presentation
3

### Contribution
3
