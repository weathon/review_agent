=== CALIBRATION EXAMPLE 6 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
*   **Title:** The title clearly captures the paper's core conceptual reframing—"curriculum as selective data acquisition." The typo ("GOAL- CONDITIONED") is a parser artifact. The subtitle "Toward Reliable Generalization" is appropriate.
*   **Abstract:** The abstract accurately summarizes the study's approach, environment, and primary findings. However, the claim that curricula "reduce approximation error" is not substantiated by any reported metric in the results. The paper only shows success rates (policy performance), not value function error. This is a significant overstatement that needs correction.

### Introduction & Motivation
*   **Quality:** The introduction is well-written and effectively sets up the problem. It clearly identifies a gap: prior work treats curriculum as an exploration heuristic, whereas this work reframes it as a mechanism for shaping the *training data distribution*. This perspective is well-motivated by the challenge of open-ended learning. The link to Hughes et al. (2024) provides a contemporary anchor.
*   **Contributions:** The contributions are implied but not explicitly listed in a bullet-point format. This is acceptable, but the paper's claims would be sharper with a clear "Our contributions are:" statement.

### Methods
*   **Reproducibility Gaps:** Several key details are missing, making exact replication difficult.
    1.  **Curriculum Specification:** The exact sampling distributions for "baseline curriculum" and "weighted curriculum" are not defined. What is the probability of sampling an edge vs. interior goal? The phrase "increased edge sampling to match their empirical difficulty under NoCurr" is too vague.
    2.  **UVFA Training:** The "pseudo-reward targets" are not defined. While PBRS is described, it's unclear how the target for the value function is computed from the shaped reward. Is it the discounted sum of shaped rewards? The terminal bonus of +1 is mentioned but not integrated into the PBRS equation.
    3.  **Data Collection:** Using "greedy action selection" during data collection for a value function being learned is atypical and raises concerns. Greedy with respect to what policy? If the value function is untrained initially, this is essentially random action selection. If it's an iterative process, this needs to be described (e.g., is data collected from a continually updated policy?). The protocol describes a single batch of data collection, but this is ambiguous.
*   **Logical Gaps:** The choice of Manhattan distance for the potential function in PBRS is reasonable for GridWorld but is a strong shaping prior. The success of the curriculum might be contingent on this dense, informative shaping. The paper does not discuss how the curriculum's effect might change with sparser reward signals.

### Experiments & Results
*   **Adequacy of Experiments:** The experiments are minimal and serve as a proof-of-concept. The GridWorld setting is extremely simple, which is fine for an initial study but severely limits the strength of the claims about "generalization" and "open-ended learning."
*   **Evaluation of Claims:**
    *   **Claim: Curriculum reshapes the state-goal distribution.** Figure 2 is cited for this, but the figure caption in the text is generic. The visual evidence for a distributional shift is assumed but not explicitly presented in the provided text. The authors must ensure the figure clearly shows the difference in goal visitation frequencies.
    *   **Claim: Curriculum improves performance on hard goals.** The data in Table 1 and the textual results support this, but with major caveats. The absolute improvements are very small (e.g., edge-goal success increases from 0.060 to 0.143). More critically, the standard deviations are *extremely* high relative to the mean differences. For edge-goal success, the standard deviation for the curriculum condition (0.107) is larger than the mean improvement over uniform (0.083). This suggests the result is not statistically reliable. **No statistical significance testing is reported**, which is a critical omission for ICLR. The claim of "consistent improvements" is not backed by the presented data.
    *   **Claim: Curriculum reduces approximation error.** This claim, made in the abstract and Section 3.1, is **not tested at all**. The paper reports only policy success rates, not any measure of value function error (e.g., MSE against Monte Carlo returns) across the state-goal space. This is the most serious flaw in the experimental validation.
*   **Baselines and Ablations:** The comparison between uniform and edge-biased sampling is appropriate. However, a more informative baseline would be a simple "oracle" that samples goals uniformly from the set of currently unattainable goals, to distinguish the effect of targeting "hard" goals from the specific heuristic of targeting "edge" goals.
*   **Results Presentation:** References to figures are inconsistent (Figure 1 and Figure 2 are described identically). Table 1 appears truncated in the text ("Pc"). This must be fixed for clarity.

### Discussion & Limitations
*   **Limitations:** The discussion correctly identifies key limitations: the simplicity of the environment, the manual nature of the curriculum, and the modest/inconsistent gains. This is honest and appropriate.
*   **Broader Impact:** Not discussed, which is acceptable for this type of methodological paper.

### Writing & Clarity
*   **Overall Clarity:** The writing is generally clear and concise. The central idea is communicated effectively.
*   **Major Confusion:** The description of the training protocol is the most confusing part, as noted in the Methods section. The relationship between data collection, policy used, and UVFA training needs a step-by-step explanation.

### Overall Assessment
The paper proposes a thoughtful conceptual reframing of curriculum learning in GCRL as "selective data acquisition," linking it to distributional shifts and function approximation bias. This perspective is valid and could be valuable for the field. However, the empirical validation is critically insufficient for an ICLR publication. The experiments are conducted in an overly simplistic environment, the results show small effects with high variance and no statistical testing, and the core claim about reducing approximation error is entirely unsupported by evidence. The paper reads as a promising but very preliminary proof-of-concept. In its current form, the contribution is primarily conceptual, lacking the rigorous experimental evidence required to substantiate the claims and demonstrate a meaningful advance. Major revisions—including more robust experiments, statistical analysis, and direct measurement of approximation error—are necessary before it could meet ICLR's bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper reframes curriculum learning in goal-conditioned reinforcement learning (GCRL) as a form of selective data acquisition, arguing that biasing goal sampling toward underachieved regions (e.g., edge cells in GridWorld) structurally reshapes the state–goal distribution and improves function approximation. Using Universal Value Function Approximators (UVFAs) with potential-based reward shaping, the authors demonstrate that a simple edge-biased curriculum yields modest but consistent improvements on harder goals while maintaining overall performance. The work positions curriculum learning as a principled mechanism for guiding data distribution, with implications for open-ended and lifelong learning.

### Strengths
1. **Clear conceptual framing**: The paper articulates a coherent perspective—curriculum as selective data acquisition—that connects curriculum design to distributional shifts in training data and their effect on function approximation. This reframing is well-motivated and clearly presented in the introduction and discussion.
2. **Controlled experimental design**: The use of a deterministic GridWorld with UVFAs and potential-based reward shaping allows for a focused investigation of distributional biases. The comparison between uniform and edge-weighted sampling cleanly isolates the effect of curriculum on data composition and downstream performance.
3. **Transparent reporting**: Results are reported with means and standard deviations across multiple seeds, and the evaluation separates overall success from edge-goal success. The training protocol and hyperparameters are described in sufficient detail for reproducibility.

### Weaknesses
1. **Limited novelty and incremental contribution**: The core idea that curricula bias data distributions is well-established in prior curriculum learning literature (e.g., Florensa et al., 2017; Portelas et al., 2020). The paper’s reframing as “selective data acquisition” offers a useful perspective but does not introduce a fundamentally new technique or theoretical insight.
2. **Toy domain and modest results**: Experiments are confined to a simple GridWorld, and performance gains are small (e.g., edge-goal success improves by +0.083 in Table 1). The improvements are inconsistent across seeds (large standard deviations), and the study lacks validation in more complex, high-dimensional environments, limiting the generality of the claims.
3. **Absence of comparisons to advanced baselines**: The paper only compares uniform sampling against a hand-designed curriculum. It does not benchmark against state-of-the-art automated curriculum methods (e.g., reverse curriculum generation, teacher-student frameworks) or alternative exploration strategies, making it difficult to assess the practical advantage of the proposed viewpoint.

### Novelty & Significance
**Novelty**: The paper’s main novelty lies in explicitly linking curriculum design in GCRL to the concept of selective data acquisition and demonstrating this link empirically with UVFAs. However, this connection is conceptually straightforward and builds directly on prior work.

**Significance**: The work offers a modest step toward understanding how curricula affect function approximation in GCRL. While the perspective could inform future research on data-driven curriculum design, the empirical findings are preliminary and limited to a toy domain, reducing immediate impact.

**Clarity**: The writing is clear and well-structured. The methodology, results, and limitations are presented in an accessible manner.

**Reproducibility**: The experimental setup is described in enough detail to permit replication, though the exact grid size and random seeds are not specified. Code is not provided, but the simplicity of the environment and models likely allows reproduction.

### Suggestions for Improvement
1. **Scale experiments to more challenging domains**: To strengthen the claim, evaluate the curriculum-as-data-acquisition perspective in environments with continuous state/action spaces (e.g., MuJoCo) or pixel-based observations, and with deeper function approximators.
2. **Compare against automated curriculum methods**: Include baselines such as reverse curriculum generation (Florensa et al., 2017) or goal-generating adversarial networks to contextualize the hand-designed curriculum and demonstrate the utility of the data-selection viewpoint relative to existing approaches.
3. **Perform deeper analysis of approximation error**: Quantify and visualize how the curriculum affects UVFA generalization error across the entire state-goal space (e.g., via error heatmaps) to better support the claim that curricula improve approximation in underachieved regions.
4. **Investigate adaptive curricula**: Extend the manual curriculum to an adaptive strategy that dynamically adjusts the goal distribution based on agent progress (e.g., tracking learning progress or value error), which would better align with the open-ended learning motivation.
5. **Clarify experimental details and release code**: Specify the grid dimensions, architecture details (number of layers, hidden units), and random seeds. Releasing code would enhance reproducibility and facilitate future work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against automated curriculum baselines (e.g., ALP-GMM, GoalGAN).** Without showing the handcrafted edge curriculum is competitive with established adaptive methods, the claim that curriculum acts as a "principled mechanism" is unsupported.
2. **Ablation on curriculum weighting schemes.** The paper uses a fixed edge bias but does not test whether weighting by empirical success rate or distance is critical. This is necessary to validate that the effect is due to targeting underachieved goals, not arbitrary bias.
3. **Scale to a more complex environment (e.g., MiniGrid).** The claims about open-ended learning and generalization are not credible when only tested on a trivial deterministic GridWorld. A single more complex domain is needed to suggest broader applicability.
4. **Vary dataset size to measure data efficiency.** If curriculum is selective data acquisition, it should achieve similar performance with less data or better performance with the same data. Without this, the claimed benefit is merely a redistribution effect.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify UVFA approximation error across state-goal space.** The paper claims curricula reduce approximation error but only reports success rates. Show MSE on a held-out set split by goal difficulty to directly link distribution shift to improved function approximation.
2. **Analyze training dynamics (e.g., success rate over time).** It is unclear whether curriculum helps learning converge faster or just shifts final performance. Plot learning curves for edge vs. interior goals to show if curriculum accelerates learning on hard goals.
3. **Perform a causal analysis: retrain on uniform data but oversample edge goals.** Does performance improve solely because edge goals appear more often? This would isolate the effect of data frequency from other factors like exploration.

### Visualizations & Case Studies
1. **Heatmaps of goal sampling frequency and success rates per cell.** A simple bar chart of edge vs. interior is insufficient. Show the spatial distribution of where the curriculum allocates data and where success improves/degrades.
2. **Visualize learned value functions for representative goals.** Plot the predicted value landscape across the grid for a few goals (edge and interior) under both training regimes. This would reveal if curriculum leads to smoother or more accurate value estimates.
3. **Show trajectory rollouts for failing cases.** Illustrate specific goals where curriculum underperforms uniform sampling to understand the trade-offs and failure modes of the biased distribution.

### Obvious Next Steps
1. **Implement a simple adaptive curriculum (e.g., based on recent success rates).** The paper advocates for curricula as selective data acquisition but uses a static handcrafted one. A dynamic rule would strengthen the connection to open-ended learning.
2. **Test generalization to unseen goal distributions.** Evaluate the UVFA on goals sampled from a different distribution (e.g., only corners) to see if curriculum improves out-of-distribution generalization, which is key for persistence.
3. **Analyze the effect of curriculum on exploration.** Since the paper argues curriculum is not just an exploration heuristic, measure exploration metrics (e.g., state coverage) to disentangle data distribution effects from exploration.

# Final Consolidated Review
## Summary
This paper reframes curriculum learning in goal-conditioned reinforcement learning as a mechanism for selective data acquisition, arguing that biasing goal sampling shapes the training distribution and improves function approximation. Using UVFAs in a GridWorld, the authors show that an edge-biased curriculum yields modest improvements on harder goals. The contribution is primarily a conceptual perspective linking curriculum design to data distribution management.

## Strengths
- **Clear conceptual framing:** The paper coherently articulates curriculum learning as selective data acquisition, connecting it to distributional shifts in training data and their effect on function approximation. This reframing is well-motivated and presented.
- **Controlled experimental design:** The use of a simple deterministic GridWorld with UVFAs and potential-based reward shaping allows for a focused, isolated investigation of how a curriculum bias affects data composition and downstream performance.

## Weaknesses
- **Unsubstantiated core claim:** The abstract and text claim curricula "reduce approximation error," but the paper provides no measurement or analysis of value function error (e.g., MSE). Only policy success rates are reported, leaving a key theoretical claim experimentally unsupported.
- **High variance undermines claimed consistency:** The reported improvements, particularly on edge goals, are small (e.g., +0.083 in Table 1), and the standard deviations are large relative to the mean differences (e.g., ±0.107 for curriculum edge success). Without statistical significance testing, the claim of "consistent improvements" is not convincingly demonstrated by the presented data.
- **Overly simplistic experimental setting:** The proof-of-concept uses a trivial, deterministic GridWorld and a static, hand-designed curriculum. This severely limits the strength and generality of the conclusions regarding "reliable generalization" and the pathway to "open-ended learning."

## Nice-to-Haves
- A deeper analysis quantifying UVFA approximation error across the state-goal space to directly link the distribution shift to function approximation quality.
- Experimentation in a slightly more complex environment (e.g., MiniGrid) to better suggest the perspective's broader applicability.
- Comparison against a simple adaptive curriculum baseline to strengthen the connection to open-ended learning.

## Novel Insights
The paper's novel insight is the explicit articulation of curriculum learning in GCRL as a structural mechanism for biasing the training data distribution—"selective data acquisition"—rather than merely an exploration heuristic. This perspective cleanly connects curriculum design to the inductive biases of the function approximator (the UVFA), offering a distinct conceptual lens through which to analyze and design curricula, particularly for challenges in persistent, open-ended learning.

## Suggestions
- Revise the text to remove the unsupported claim about reducing approximation error or add experiments that directly measure and report value function error.
- Report statistical significance tests (e.g., p-values or confidence intervals) for the performance differences to substantiate claims of consistency.
- Clarify the training protocol in the Methods section, explicitly stating if data collection is a single batch from an initially random policy or an iterative process.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
