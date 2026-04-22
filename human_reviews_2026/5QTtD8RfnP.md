# GNPA-DIL: Unveiling the Vulnerability Genome Through Semantic Graph Distillation and Invariant Neural Reasoning

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Software vulnerabilities constitute an escalating security crisis with over 25,000 new CVEs documented annually, demanding detection models capable of identifying complex vulnerability patterns across evolving codebases. Contemporary vulnerability detection models exhibit catastrophic brittleness when deployed beyond controlled benchmarks, failing to maintain accuracy on rigorously-validated samples and collapsing entirely when confronted with routine syntactic variations or cross-function vulnerability patterns. The GNPA-DIL model overcomes these limitations through a neural architecture trained on vulnerability-centric program slices extracted via Code Property Graphs, learning domain-invariant representations that capture fundamental vulnerability semantics rather than superficial code patterns. By learning to process dramatically compressed program representations, the GNPA-DIL model transcends the context limitations plaguing existing architectures while preserving the critical information flows that characterize actual vulnerabilities. This fundamental advance in vulnerability representation learning enables the model to generalize beyond its training distribution, detecting previously unseen vulnerability types with 63.48\% accuracy on Emerging-Post-Vulnerability CVEs. On the SVEN benchmark, GNPA-DIL achieves 73.58\% F1-score compared to the best baseline's 54\%, representing a 36\% relative improvement, while maintaining 67.63\% accuracy on cross-function vulnerabilities despite being trained only on function-level data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a graph-guided neural program analysis framework (GNPA-DIL) that effectively bridges symbolic static analysis and neural learning. By combining Code Property Graph (CPG) analysis with domain-invariant learning, the model captures the semantics of vulnerabilities rather than superficial syntax patterns.

### Strengths
The motivation is well sounded and has significant practical value

The manuscript makes an attempt to introduce a rigorous mathematical, formal definitions on the vulnerability detection task, a rare and commendable feature in vulnerability detection research.

### Weaknesses
The manuscript forms a structured and sounded motivations, but the paper’s formality may obscure accessibility. Theoretical sections (e.g., Theorems 1–6, Definitions 1–4) dominate the presentation but lack intuitive explanation or ablation to confirm practical contribution of each formal component.

The architectural novelty, while well-motivated, could be viewed as an incremental synthesis of prior CPG + invariant-learning approaches rather than a fundamentally new paradigm.

Minor issue: Duplicate references (line 538, 541)

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The proposed GNPA-DIL model is trained on vulnerability-centric program slices extracted using Code Property Graphs, enabling it to learn domain-invariant representations that capture fundamental vulnerability semantics rather than superficial code patterns. The robustness analysis is conducted thoroughly.

### Strengths
The method is rigorously evaluated across multiple benchmarks, demonstrating consistent performance gains.

### Weaknesses
1. This paper is hard to follow as the mathematical formalisms in Section 3 are highly dense and presented without sufficient intuitive explanation. 

2. Minor inconsistencies in citation formatting are present.

### Questions
How is the Wasserstein constraint (Theorem 5) actually enforced during training?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a neural network approach for vulnerability detection. The GNPA-DIL model presents a neural architecture trained on vulnerability-centric program slices extracted via Code Property Graphs, learning domain-invariant representations. The model achieves accuracy of 73.58% on SVEN.

### Strengths
The use of neural networks for vulnerability detection is an active area of research. The paper contributes in that space.

### Weaknesses
The paper os hard to read. It consists of introduction, related work and a set of theorems with limited explanations. The authors appear to have made heavy use of LLMs when writing the paper (which they acknowledged0.

Experiments are not convincing as accuracy seems low.

### Questions
I could not understand the high level approach-- please descrine.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes GNPA-DIL, a vulnerability detection model that combines CPG with domain-invariant neural learning. The approach extracts vulnerability-centric program slices from CPGs and trains neural networks with domain-invariant constraints to achieve robustness against semantic-preserving transformations. The authors claim significant improvements over baselines on multiple benchmarks including SVEN  and demonstrate cross-function generalization capabilities.

### Strengths
**Relevant Problem**: Addressing robustness of vulnerability detectors to semantic-preserving transformations is important and timely

**Cross-Function Generalization**: The ability to detect project-level vulnerabilities despite function-level training (67.63% on ReposVul) is potentially valuable if validated properly

**Multi-Benchmark Evaluation**: Testing across diverse datasets with different characteristics (synthetic vs. real-world, function-level vs. project-level) is valuable

### Weaknesses
Soundness:

1. Mathematical Rigor vs. Practical Implementation Gap: The paper presents extensive mathematical formalism (Sections 3.1-3.7) involving Riemannian manifolds, variational methods, Wasserstein distances, and wavelet decompositions. However, there is a complete disconnect between this theoretical framework and the actual implementation. The paper did not explain:

- How the "vulnerability manifold" (Eq. 6) is constructed in practice
- How the Bellman operator (Eq. 15) is computed
- Whether the wavelet decomposition (Eq. 25) is actually used in the model

This suggests the mathematical framework may be decorative rather than functional.

Missing Architecture Details: Despite the heavy mathematical notation, basic implementation details are absent:

- What neural architecture is used? (GNN? Transformer? RNN?)
- How are CPG slices encoded as neural network inputs?
- What is the model size and computational complexity?
- How is domain-invariant learning actually implemented in the training procedure?

Dataset Quality Concerns: The paper reduces FormAI from 331,000 to 8,259 samples (97.5% reduction) and PrimeVul from 235,768 to 2,096 samples (99.1% reduction) through a "three-phase refinement". This extreme filtering raises concerns:

- Is the model learning from a representative sample or cherry-picked easy cases?
- How do baselines perform when trained on the same filtered dataset?
- The paper doesn't provide fair comparisons with baselines on identical training data

Incomplete Experimental Validation:

- No ablation study showing the contribution of CPG slicing vs. domain-invariant learning vs. other components
- No comparison on identical training data with baselines
- No analysis of failure cases or error types

Contribution:

The claimed contribution “unveiling the vulnerability genome” is ambitious, but the delivered contribution is a flawed experiment, an unverified slicing method, and a confused methodology.

- **“Vulnerability genome” :**

  This metaphor is exaggerated and unscientific. On the expert-validated PrimeVul benchmark, the recall is only about 40%, directly disproving the claim that the model has “unveiled the genome.” At best, it captures some invariant features in some cases. The overstated framing hurts the paper’s credibility.

- **Actual potential contribution:**

  The true potential lies in cross-granularity generalization. However, this value is undermined by other flaws, most notably the following contradiction:

  The paper reports opposite behaviors on PrimeVul (high-precision / low-recall) and SVEN (high-recall / medium-precision) without any explanation, revealing a serious inconsistency that undermines the validity of its experimental results.

Presentation:

1. **Excessive Mathematics Without Justification:**

   Theoretical complexity (Riemannian geometry, measure theory, functional analysis) is introduced without showing why it is necessary. For example:

   - Why view vulnerabilities as a *“Riemannian substructure”* (Def. 2)?
   - How is the Hausdorff distance to the *“vulnerability manifold”* (Eq. 14) computed?

2. **Unclear Architecture Figure:**

   Figure 1 shows pipeline stages but lacks detail on what each component actually performs.

3. **Weak Related Work:**

   The related-work section is thin and does not adequately situate this work within prior literature.

4. **Missing or Redundant Figures/Tables:**

   Figures 3–5 are redundant, and several “Tables” are referenced but not actually present in the paper.

### Questions
1. Are the mathematical definitions and theorems actually implemented, or are they just conceptual analogies? Please clarify their connection to the implementation.
2. Could you include ablations isolating the impact of:

- CPG slicing (vs. full code input),
- domain invariance (with/without Wasserstein regularizer),
- and the graph neural architecture choice?

3. Can you explain the opposite behaviors on PrimeVul (high-precision/low-recall) and SVEN (high-recall/medium-precision)?

### Soundness
2

### Presentation
2

### Contribution
2
