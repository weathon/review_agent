# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
While large language models (LLMs) demonstrate impressive performance across various tasks, their deployment in real-world scenarios is still constrained by high computational demands. Layer-wise pruning, a commonly employed strategy to mitigate inference costs, can partially address this challenge. However, existing approaches generally depend on static heuristic rules and fail to account for the interdependencies among layers, thereby limiting the effectiveness of the pruning process. To this end, this paper proposes a game-theoretic framework that formulates layer pruning as a cooperative game in which each layer acts as a player and model performance serves as the utility. As computing exact Shapley values is computationally infeasible for large language models (LLMs), we propose using a lightweight surrogate network to estimate layer-wise marginal contributions. This network can predict LLM performance for arbitrary layer combinations at a low computational cost. Additionally, we employ stratified Monte Carlo mask sampling to further reduce the cost of Sharpley value estimation. This approach captures inter-layer dependencies and dynamically identifies critical layers for pruning. Extensive experiments demonstrate the consistent superiority of our method in terms of perplexity and zero-shot accuracy, achieving more efficient and effective layer-wise pruning for large language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper points out that existing layer pruning methods mostly rely on static heuristic rules, overlooking the dynamic interdependencies between layers, which leads to suboptimal results. To address this issue, the authors propose a game-theoretic approach that formulates the layer pruning problem as a cooperative game. In this game, each Transformer layer is treated as a “player,” and the overall model performance (measured by perplexity, PPL) represents the collective “utility” produced through cooperation among all players. Since computing the exact contribution of each player (i.e., the Shapley value) is computationally intractable, the authors further design an efficient two-stage approximation framework to estimate these contributions and prune layers with lower importance. Experimental results show that this method consistently and significantly outperforms existing depth-wise and width-wise pruning baselines on both language modeling (PPL) and zero-shot reasoning tasks. Moreover, it generalizes well to non-Transformer architectures and demonstrates strong compatibility with quantization techniques such as GPTQ.

### Strengths
1. The motivation is clear, and the results in Figure 1 clearly demonstrate the interdependence among layers.
2. The paper proposes an efficient two-stage approximation framework that significantly reduces the computational complexity of solving the cooperative game.
3. The experimental evaluation is comprehensive, covering both Transformer and non-Transformer model architectures.

### Weaknesses
1. The paper’s core innovation is insufficient. The idea of viewing pruning from a cooperative game theory perspective has already been explored in prior work, such as “Using Cooperative Game Theory to Prune Neural Networks.” In addition, “Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding” also formulates the layer pruning problem as an optimization task through binary pruning masks.
2. The experimental validation is not sufficiently comprehensive. The benchmarks used are limited to perplexity (PPL) and multiple-choice tasks, without evaluation on generation benchmarks such as GSM8K. In the generalization experiments on non-Transformer architectures, only PPL was reported, lacking broader task evaluation.

### Questions
1. Why did the authors choose to evaluate on the ANLI benchmark?
2. Why was MMLU not included in the evaluation? According to the results reported in the ShortGPT paper, its pruning method performs well on MMLU. Given that MMLU is a standard benchmark for evaluating reasoning and knowledge retention in large language models, the authors should at least include results on this task to make the evaluation more complete.
3. Regarding the implementation of iterative pruning, the paper mentions “iteratively removing the least contributive layers.” Could the authors clarify how this iterative process is carried out?
(A) Are all layers’ Shapley values computed once, and then layers are removed in batches (e.g., first pruning the three least contributive layers, then the next three)?
(B) Or after each batch of layers is removed (e.g., three layers), are both Stage 1 and Stage 2 rerun to recompute the Shapley values for the remaining layers?
If the process follows (A), it seems inconsistent with the paper’s main motivation that pruning changes the relative importance of other layers. If it follows (B), the overall computational cost would increase significantly.

### Soundness
3

### Presentation
3

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
This paper introduces a game-theoretic framework for pruning LLMs, aiming to reduce computational cost while preserving model performance. Instead of treating layers independently, the authors model pruning as a cooperative game, where each transformer layer is a “player” and the model’s performance serves as the utility function.

### Strengths
1. This paper reformulates LLM pruning as a cooperative game, capturing inter-layer dependencies ignored by static heuristics.
2. This paper proposes a scalable Shapley-based pruning framework using stratified sampling and a surrogate model for efficient layer contribution estimation.
3. This paper demonstrates consistent improvements over depth- and width-wise pruning baselines on multiple benchmarks, including WikiText2, PTB, C4, and zero-shot reasoning tasks.

### Weaknesses
1. While the paper proposes a surrogate-assisted approach to estimate Shapley values efficiently, it lacks a clear theoretical analysis quantifying how well the surrogate approximates true layer contributions.
2. The method relies on a small calibration set, which may not adequately represent diverse data distributions or downstream task requirements. The resulting Shapley estimates might therefore be dataset-dependent and unstable across domains.
3. The study focuses solely on one-shot pruning without retraining, which may restrict achievable performance. Many recent works (e.g., ShortGPT) benefit from lightweight fine-tuning. It will be helpful to investigate how minor retraining after pruning interacts with the cooperative-game framework.

### Questions
1. The paper uses 10 BookCorpus samples for calibration. How was this number chosen, and how does increasing or diversifying the calibration set affect Shapley estimation quality?
2. Have you explored whether minimal fine-tuning after pruning further improves performance?
3. Have you profiled the wall-clock speedups and memory reductions on real hardware (e.g., A100, H100, or consumer GPUs)? How does the pruning affect model latency when combined with quantization in real-time inference settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a game-theoretic approach to layer-wise pruning in large language models, casting the problem as a cooperative game where each layer is a player and model performance serves as utility. Since computing exact Shapley values to measure each layer’s marginal contribution is infeasible at scale, the authors design a two-stage approximation using stratified Monte Carlo mask sampling and a lightweight surrogate network to efficiently estimate layer importance. The framework reportedly captures inter-layer dependencies better than prior methods and consistently surpasses strong depth-wise and width-wise pruning baselines across multiple downstream and generative benchmarks for Transformer and non-Transformer models.

### Strengths
- **Principled Formulation & Theoretical Motivation**: The paper introduces a compelling game-theoretic framing for layer pruning, challenging the prevalent assumption of independent layer importance and instead recognizing context-dependent inter-layer dynamics. This principled approach addresses a core limitation of widely-used heuristics.
- **Efficient Approximation of Shapley Values**: By incorporating a lightweight surrogate network trained on stratified Monte Carlo mask samples, the method approximates Shapley values efficiently, enabling practical application to large-scale LLMs—this is articulated with clear algorithmic details and demonstrated scalability (see Algorithm 1 & Figure 2).
- **Conceptual Generality**: The approach is validated across Transformer and non-Transformer architectures, and shown to integrate compatibly with quantization and LoRA fine-tuning (Figure 7), suggesting broad applicability for LLM deployment.

### Weaknesses
- **Surrogate Network Limitations and Validation**: While Figure 6 and Table 7 specify the surrogate’s structure, the paper lacks a rigorous quantitative evaluation of its prediction fidelity, especially for masks far from the training distribution. There is little discussion of failure modes, e.g., overfitting to calibration samples, brittleness under extreme masking, or calibration data misspecification (see Appendix F.1), limiting confidence for highly compressed regimes or out-of-domain settings. For instance, the impact of surrogate error on Shapley ranking stability is not systematically assessed.
- **Potential Optimization and Mask Generation Shortcomings**: The stratified Monte Carlo mask sampling strategy (Section 3.3, Table 21) is justified primarily by ablation, with theoretical explanations on sampling sufficiency or representativeness lacking. While empirical results (Table 21) suggest an advantage over random sampling, the method’s robustness to choice of Hamming weights, number of samples, and potential for bias due to nonuniform coverage of important layer subsets remains underexplored. The mask set’s coverage and the potential for missed critical coalitions, particularly as $L$ increases, are not discussed in depth.
- **Missing Related Work and Baselines**: The paper mentions SparseGPT but the citation to SparseGPT is pointing to the wrong paper (from the same research group).  Adding SparseGPT to the results would also make the experiments more complete. Furthermore, there has been research using Shapely values [1,2] and Influence Functions [3] for LLM pruning and LLM layer importance estimation that the paper fails to acknowledge.
- **Ambiguity in Theoretical Guarantees**: While the game-theoretic formulation is elegant, there is no quantification of the approximation gap between true Shapley values and those estimated by the surrogate. No guarantees or bounds are provided regarding the surrogate’s reliability or the stability of resulting pruning strategies as masking regimes change. This undermines full confidence in the method’s reliability for critical compression tasks.
- **Additional Minor Concerns**: Certain tables (such as Table 2) require careful reading to parse due to excessive fragmentation of sub-columns; legend clarity could be improved (see also Figure 3, Figure 4); some experimental settings (e.g., LoRA fine-tuning, ablation tasks) are relegated to the appendix without high-level results in the main paper.

### Questions
- How robust is the surrogate model to mismatches between the calibration data distribution and the actual deployment or test distribution? Have the authors quantified the surrogate’s error rate, particularly for masks not seen during training?

- Can the authors provide theoretical or empirical insights into the approximation gap between surrogate-predicted and true Shapley values, especially regarding stability of the ranking under different sampling or masking regimes?

- Can authors provide results for SparseGPT as well and acknowledge or compare against the newer pruning methods that were left out.

- Could the authors provide more interpretability on which types of layers (e.g., attention, feed-forward, early vs. late) are most frequently pruned, and relate this to observed task degradations

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This is a well-executed paper with a creative idea (cooperative game formulation + surrogate Shapley approximation) and extensive empirical validation. While more theoretical analysis of the surrogate approximation would strengthen the paper, the methodological novelty and practical effectiveness make it strong.

### Strengths
1. Viewing layer pruning as a cooperative game is original and well-motivated. It captures inter-layer dependencies often ignored in prior pruning methods based on static heuristics. The surrogate-assisted estimation is elegant and computationally practical, bridging theory and application.
2. Experiments are thorough, covering multiple models (transformer and non-transformer), datasets, and both generative and reasoning tasks, and generalization to quantization. 
3. Consistently outperforms strong baselines (SliceGPT, SLEB, ShortGPT, Shortened-LLaMA) across tasks and pruning ratios. The improvements are meaningful, especially at high pruning levels.

### Weaknesses
1. While inspired by cooperative game theory, the connection remains mostly heuristic. The surrogate model approximates marginal contributions but lacks analysis of approximation error or variance bounds.
2. There is limited discussion of the surrogate’s accuracy or potential biases (e.g., overfitting to sampled masks). Reporting R² or correlation between predicted and true perplexities would strengthen the claim.
3. Although the method reduces evaluation costs compared to naive Shapley computation, 8k–80k mask evaluations and 200-epoch surrogate training are still substantial for large models. Quantitative runtime comparisons to baselines would help.

### Questions
1. What is the computational overhead (GPU hours) compared to simpler pruning baselines like ShortGPT or SliceGPT?
2. How sensitive is the method to the number of sampled masks or surrogate capacity?

### Soundness
3

### Presentation
3

### Contribution
3
