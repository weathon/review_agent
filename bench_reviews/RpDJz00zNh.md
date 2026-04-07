## Summary
This paper introduces ConciseHint, a framework to improve the efficiency of large reasoning models (LRMs) by performing *in-reasoning intervention*. The core idea is to continuously inject learnable hints (either manual text or embeddings trained on concise data) during the token-by-token generation of the reasoning chain. A key innovation is an adaptive mechanism that adjusts the hint intensity based on the current reasoning length (as a proxy for query complexity) and dynamically positions the hints to balance accuracy and computational overhead.

## Strengths
*   **Novel and Well-Motivated Paradigm:** The paper clearly identifies and targets the under-explored direction of intervening *during* the reasoning generation, contrasting it with established "before-reasoning" methods (prompting, SFT, RL). This framing establishes a distinct and compelling research niche.
*   **Strong and Extensive Empirical Results:** The method demonstrates substantial token reduction (often 40-65%) while maintaining or slightly improving accuracy across multiple state-of-the-art open-source LRMs (Qwen-3 series, DeepSeek-R1) and diverse benchmarks (GSM8K, AIME24, GPQA-Diamond). Critically, it also shows strong composability, further boosting efficiency when combined with existing methods like Deer and NoWait.
*   **Rigorous and Transparent Evaluation:** The evaluation includes end-to-end latency measurements demonstrating practical utility, thorough ablation studies justifying the adaptive components, and analyses on transition word reduction, hyperparameter sensitivity, and generalization to code/commonsense tasks. The training-free variant (ConciseHint) and the trainable variant (ConciseHint-T) are both explored.

## Weaknesses
*   **Heuristic Design with Limited Justification:** The adaptive rules for hint interval (\(\tau_k = \alpha + \beta \cdot l_k\)) and injection position (Eq. 3) are empirically motivated but heavily heuristic. While ablation studies show they work, the paper provides no exploration of the design space (e.g., non-linear functions) or theoretical grounding for why these specific forms are effective, which limits a deeper understanding of the method.
*   **Insufficient Statistical Reporting:** The paper reports average accuracy and token usage over multiple runs but does not provide measures of variance (e.g., standard deviations) or statistical significance tests. This makes it difficult to assess the reliability of small accuracy changes, which is important for a conference with high standards like ICLR.
*   **Superficial Analysis of Mechanism:** The paper convincingly shows *that* hint injection reduces tokens but provides limited analysis into *how* it changes the reasoning process. A more fine-grained analysis (e.g., categorizing pruned tokens as redundant coherence phrases, elaborated calculations, or self-checks) would strengthen the claim of improving *reasoning efficiency* rather than just textual brevity.

## Nice-to-Haves
*   A brief exploration of alternative adaptive rules (e.g., a step function) or a sensitivity analysis on the manual hint's wording beyond the provided ablation would help better characterize the method's design space.
*   Expanding the evaluation to a wider range of reasoning types (e.g., symbolic reasoning, multi-hop QA) would more robustly support the claim of a general "in-reasoning" paradigm beyond STEM tasks.
*   A more detailed discussion of potential failure modes or boundary conditions (e.g., for extremely complex queries) would provide a clearer picture of the method's limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
*   **"Lack of comparison to strongest recent baselines"**: The paper includes several strong and relevant baselines (BeConcise, Prompt, Deer, NoWait). Demanding comparison against every recent method is scope creep.
*   **"Need for evaluation on closed-source models"**: The paper's focus on open-source models is valid and sufficient; applicability to closed-source APIs is not a core requirement for establishing the method's contribution.
*   **"Missing training details for ConciseHint-T"**: The paper describes the training procedure sufficiently in Section 3 (SFT on concise data, embedding initialization, next-token prediction). Further hyperparameter details are appropriate for code release, not the paper.
*   **"Requires theoretical justification for interpolation-based controllability"**: The empirical demonstration of smooth control via embedding interpolation (Figure 3) is sufficient for an empirical paper; a theoretical guarantee is not standard or expected.
*   **"The complexity proxy is circular"**: The paper acknowledges this prior (length correlates with complexity) and the adaptive mechanism is designed to handle this dynamic; it is a reasoned design choice, not a flaw.
*   **"Potential overhead not explored in batched inference"**: The paper includes analysis of relative latency overhead (<0.3%) and end-to-end latency reduction. A full systems analysis for batched serving is outside the paper's core scope.

## Novel Insights
The paper's primary novel insight is the viability and effectiveness of the *in-reasoning intervention* paradigm for improving efficiency. It demonstrates that continuously influencing an LRM during its generation process—via simple, adaptive hint injection—can yield substantial compression of the reasoning chain without compromising accuracy. A secondary insight is that a trained hint embedding (ConciseHint-T) can capture generalized "conciseness" patterns that transfer effectively to out-of-domain tasks, and that controllability can be achieved smoothly through embedding space interpolation.

## Suggestions
*   Add standard deviations or confidence intervals for accuracy and token usage metrics across the multiple runs already performed to substantiate the reported improvements.
*   Include a qualitative analysis comparing original and ConciseHint-modified reasoning chains for a few examples, annotating what types of content (e.g., transitional phrases, repeated calculations, verbose explanations) were reduced to provide concrete evidence of how conciseness is achieved.