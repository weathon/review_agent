# SHADE: Spectral Hallucination Detection via Dual Spectral Decompositions

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Large language models often produce confident but unsupported statements.
Detecting such hallucinations from internal signals is essential for reliable
systems. Existing attention-based methods either summarize weights with local
statistics or adopt Laplacians (e.g., \(D_{\text{out}}-A\)) whose guarantees and
applicability break outside strictly causal, square attention. We seek a single,
rigorous framework that scales across architectures and attention types while
yielding interpretable indicators of grounding. We introduce \textsc{SHADE}, a
unified spectral approach that models attention with two standard operators: (i)
a random-walk operator \(L_{\mathrm{rw}}=I-A\) that quantifies
diffusion/leakiness, and (ii) a degree-normalized cross-operator
\(M=D_Q^{-1/2}AD_K^{-1/2}\) that quantifies query--key coupling. The resulting
features are mathematically rigorous and physically interpretable, with clear
operator semantics that map to failure modes associated with hallucination:
\emph{diffusion} (from \(L_{\mathrm{rw}}\)) quantifies probability leakiness;
\emph{conductance} (via a symmetric/PSD Laplacian, e.g., Chung's) captures
expansion/connectedness; and \emph{energy/alignment strength} (from the SVD of
\(M\)) quantify total coupling and the dominance or fragmentation of coupling
modes. The formulation applies unchanged to encoder, decoder, and
encoder--decoder settings, including rectangular cross-attention and masked
sub-blocks. Evaluated on GPT-2, FLAN-T5, and Phi-2 across HaluEval and
TruthfulQA, \textsc{SHADE} consistently surpasses token-probability and
LapEigvals-style baselines, delivering strong discrimination and calibration
alongside interpretable spectra. By grounding hallucination detection in
standard spectral operators with physically meaningful interpretations,
\textsc{SHADE} offers an explanatory basis for \emph{where} and \emph{how}
hallucinations originate. This suggests two practical avenues: (i) training-time
regularization to suppress emergent hallucinatory patterns, and (ii) a
deployable \emph{hallucination risk score} with mode-level rationales (by
layer/head and prompt span) that end users and developers can act on.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SHADE, a novel framework for detecting hallucinations in Large Language Models by analyzing the internal structure of their attention mechanisms. The method uniquely employs a dual-view approach to diagnose distinct failure modes.

The first view analyzes the information flow within the attention network, identifying when the model detaches from the source prompt and begins to recycle its own generated content. The second view assesses the query-key coupling, measuring the alignment strength between what the model is looking for and the information it finds. This detects brittle connections, such as when attention collapses onto a few non-informative tokens.

The authors demonstrate that these two perspectives are complementary and capture different types of structural failures. Through extensive experiments on various models and benchmarks, SHADE is shown to significantly outperform previous detection methods. The framework is notable for being theoretically grounded, applicable across diverse model architectures, and providing interpretable signals for why a model is hallucinating.

### Strengths
1. Addresses a Timely and Critical Problem: The authors tackle the problem of hallucination in Large Language Models, which remains a significant and pressing challenge for the field. The work's focus on developing principled, internal methods for detecting such failures is highly relevant and contributes to a crucial line of research aimed at improving model safety and reliability.

2. Conceptually Interesting Premise: The core intuition to diagnose attention failures by modeling the mechanism from two distinct structural viewpoints—information flow and query-key coupling—is conceptually novel and interesting. This dual-perspective approach offers a potentially more holistic way to interpret the internal state of the model compared to methods that rely on a single analytical lens.

### Weaknesses
1. The core weakness is the inherent ambiguity of its spectral features, which are incapable of distinguishing between attention patterns essential for correct reasoning and pathological ones that lead to hallucinations. For instance, a strong, necessary focus on a key entity to answer a factual question produces the same "over-concentration" signal that the method flags as high-risk. Similarly, a simple and correct copying mechanism, often used in summarization, results in a "rank collapse", a pattern the method incorrectly associates with pathological over-fitting. Because these features are fundamentally context-blind—describing only the geometric shape of the attention distribution rather than its semantic appropriateness for the task—they cannot serve as reliable, standalone indicators of hallucination.

2. A significant number of hallucinations emerge dynamically over the course of generation. Therefore, for complex outputs, a methodology that relies exclusively on a static analysis of the final attention distribution is theoretically insufficient for accurate detection. Consider a "snowball effect," where an initial, single-step hallucination leads to a cascade of subsequent errors. Given the final length of the text, the original trigger for this failure mode would likely be obscured and thus undetectable by a post-hoc static analysis. This scenario is particularly critical as it more closely mirrors real-world failure cases.

### Questions
1. I am skeptical about the practical viability of using attention features as a primary mechanism for hallucination detection. My main concern stems from the significant computational overhead required to access and process attention distributions during the inference phase. For each token generated, this method necessitates retrieving the attention scores from multiple layers and heads, which is a computationally expensive operation. This introduces non-trivial latency and reduces throughput, especially when compared to standard, optimized inference processes that do not require inspecting these intermediate states. Does the potential gain in detection accuracy justify the substantial performance cost? In many application scenarios, such as real-time chatbots or large-scale content generation, inference speed is a paramount constraint. Therefore, a method that significantly slows down generation may not be a feasible solution, regardless of its theoretical effectiveness.

2. A reliance on the final static attention distribution for detection, without the capacity to integrate the temporal dynamics of attention features in long sequences, raises a critical challenge: how can such an approach detect the "snowball effect" originating from a single-step hallucination within a long text? This class of cascading hallucination represents a more prevalent and practical scenario, yet the proposed methodology appears to completely circumvent the handling of this realistic failure mode.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
If I'm understanding correctly- and this paper has a lot of (what I think is) unnecessary jargon, so this has been difficult- 

The core idea this paper proposes for detecting hallucinations is essentially to use the transformer's attention matrices to compute spectral statistics (eigenvalues, singular values, entropies, etc.) and then feed these handcrafted feature numbers into logistic regression to classify whether the output is a hallucination or not. 

They test a series of models like GPT2/Phi2 on HaluEval and TruthfulQA (which are quite different datasets, given that HaluEval is looking at in-context grounded hallucination while TruthfulQA is moreso factuality without checking against a grounding source), and compare their handcrafted feature logistic regression model against token probability detectors and LapEigvals (a similar spectral baseline that uses a triangular Laplacian). TLDR, the authors method outperforms all them both, though the improvement is something like 1% absolute improvement against LapEigvals on most comparisons.

The core claim seems to be that simple logistic regression on spectral features from attention matrices detects hallucinations better than token-probability baselines.

### Strengths
The empirical finding seems relatively solid- that SVD-based coupling features consistently outperform token-probability and older spectral baselines. This seems like a relatively lightweight pipeline.

### Weaknesses
The main issue with this paper is that it tries to overcomplicate what is essentially a very simple approach. It buries this under heavy and unnecessary math, but it's really just simple linear algebra. 

The baselines comparisons aren't exactly convincing- many, many hallucination detection methods have been proposed, like semantic entropy, self-consistency, embedding-based factuality, surface based classification, etc. Comparing against only 2 baselines with no justification as to why they were selected seems like cherry picking.

Furthermore, there is little analysis on cross-modal or cross-dataset generalization. the logistic regression classifiers are trained per model and per dataset; how extensive this extends to other models and domains remains a question- it can be very convincingly argued that token-probability (uncertainty-based) metrics are much more generalizable across domains and models, and more lightweight to compute, than needing to train the authors' approach across every single model and domain.

### Questions
NA

### Soundness
2

### Presentation
1

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
This paper proposes SHADE (Spectral Hallucination Detection), a framework for detecting LLM hallucinations by analyzing the internal attention matrices. The authors critique existing methods like LapEigvals for having limited applicability (e.g., only causal, square attention) and lacking rigorous spectral guarantees. Paper's claim is a "unified spectral approach" built on a "dual" decomposition:
1.  **Flow Analysis:** Using a random-walk operator ($L_{rw} = I - A$) to quantify "diffusion" or "leakiness" (i.e., information flow anomalies).
2.  **Coupling Analysis:** Using a degree-normalized cross-operator ($M = D_Q^{-1/2} A D_K^{-1/2}$) and its SVD to quantify query-key "coupling" (i.e., alignment anomalies).

### Strengths
1.  The paper successfully identifies and validates a powerful, architecture-agnostic hallucination detector based on the SVD of the normalized cross-operator $M$. This method consistently achieves near-ceiling performance on HaluEval and strong performance on TruthfulQA.
2. The paper provides a valuable service by thoroughly dismantling the LapEigvals baseline, pointing out its non-PSD nature, lack of spectral guarantees, and information collapse on the diagonal.
3.  The SVD Coupling features show excellent discrimination (AUROC) and calibration across all models and datasets.

### Weaknesses
1.  The paper's primary weakness is the bait-and-switch between the theoretically-motivated **random-walk Laplacian ($L_{rw} = I - A$)** in Section 3.2.1 and the experimentally-implemented **symmetric normalized Laplacian ($L_{sym}$)** in Section 4.1. The "flow" operator that was motivated at length is never tested.
2.  Because of the weakness above, the central conceptual claim of a "dual" framework with "complementary" operators is unproven. The paper's own complementarity analysis (Proposition 3.1) is for $L_{rw}$ and $M$, which are not the operators compared in Table 1 or 5.
3.  Even ignoring the operator-swap, the paper's *empirical* data contradicts its conclusion. The conclusion claims "low RV coefficients". However, Table 5 reports an RV coefficient of **0.944** for "EIGEN-FEATURES" vs. "SVD" on GPT-2/TruthfulQA. This indicates *near-perfect linear association* (i.e., redundancy), the exact opposite of complementarity.
4.  The $L_{sym}$ operator that *was* tested ("Eigen-Features") performed very poorly, and in one case (GPT-2/HaluEval), was no better than chance (AUROC 0.492). This undermines the "dual" claim from an empirical standpoint as well.

### Questions
1.  Why does Section 3.2.1 motivate the $L_{rw} = I - A$ operator, citing its specific properties, if the experiment in Section 4.1 uses the $L_{sym} = I - D^{-1/2}\overline{A}D^{-1/2}$ operator for its "Eigen-Features"?
2.  Given this disconnect, how can the paper validate its central "dual framework" thesis, since the "flow" operator from the theory was never empirically tested against the "coupling" operator?
3.  How does the conclusion claim "low RV coefficients" when Table 5 clearly shows an RV coefficient of **0.944** for GPT-2 on TruthfulQA? This high value suggests the features are redundant, not complementary.
4.  Given that the SVD-coupling path performs exceptionally well and the "Eigen-Features" path performs poorly (e.g., 0.492 AUROC) and is theoretically disconnected, wouldn't this paper be stronger if it were simply presented as a single, novel SVD-based method?

### Soundness
1

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
2

### Summary
This paper proposes SHADE, a universal illusion detection framework based on dual spectral decomposition: the attention "diffusion/leakage" is measured by the random walk operator, and the query-key coupling strength and pattern are measured by the SVD of the degree normalized crossover operator. The two operators respectively reveal information flow imbalance and evidence alignment anomaly. The features possess mathematical rigor and physical interpretability, and can be applied to self-attention, cross-attention, rectangular blocks, and various types of Transformers without modification. Experiments on HaluEval and TruthfulQA of GPT-2, Phi-2, and FLAN-T5 show that the SVD coupling characteristics of SHADE are consistently superior to the baselines such as token probability and LapEigvals, demonstrating both high discriminative power and calibration accuracy. It can provide interpretable risk scores at the layer/head/prompt fragment level, support regular expression suppression hallucinations during training or real-time alerts during deployment.

### Strengths
- This method provides a cross-architecture, interpretable, plug-and-play unified spectral operator. With just two lines of code, it can be integrated into any Transformer architecture and immediately output a clear physical layer-head-token-level illusion risk score without the need for additional training or external knowledge. Achieve detection performance close to the upper limit directly on the three major models and two benchmarks.

### Weaknesses
- All experiments were only for correlation classification and did not prove through intervention (such as suppressing high $\sigma_1$ or low leakage patterns) that "once these spectral anomalies are corrected, the hallucination rate will decrease". Therefore, it is impossible to establish that spectral characteristics are the cause of hallucinations

- Not sufficient ablations. PCA retains the key hyper-parameters such as 85% variance, temperature scaling, and the selection of $\tau$ with Youden J, and only provides the final values, without showing the sensitivity of the ROC curve to these hyperparameters. Nor has it completely dissolved the respective contributions to performance of "using only $L_rw$", "using only $M$", and "combining the two".

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
