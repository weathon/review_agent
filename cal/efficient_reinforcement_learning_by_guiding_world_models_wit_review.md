=== CALIBRATION EXAMPLE 87 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title is appropriate and clearly reflects the core contribution: improving RL efficiency by using non-curated data to guide world models. The abstract succinctly states the problem, proposed solution, and key results. The claim of achieving "nearly twice the aggregate score" under a limited budget (150k samples) is a strong, quantifiable claim that the results section must substantiate.

**Introduction & Motivation:** The introduction effectively motivates the problem by contrasting the expense of curated, reward-labeled data with the abundance of non-curated data (reward-free, mixed-quality, multi-embodiment). The research question is clearly stated. The identified gap—prior world model methods pay limited attention to fine-tuning and show marginal gains—is well-supported by citations. The four contributions (C1-C4) are specific and accurately summarize the paper's narrative.

**Method / Approach:** The two-stage method (pre-training + fine-tuning) is clearly described. The choice of RSSM is justified as a standard, reproducible baseline.
*   **Pre-training:** The modifications (removing task losses, zero-padding actions) are sensible for handling multi-embodiment data. The scale (280M parameters) is noted.
*   **Fine-tuning & Proposed Techniques:** The diagnosis of distributional shift (Fig. 2) is a strong motivator. However, the description of the retrieval mechanism (Eq. 5) is sparse. It states the encoder `e_θ` is used, but it's unclear if this is the full world model encoder `q_θ` or a separate network. The claim that retrieval happens "in seconds" using Faiss is plausible but lacks detail on indexing time/complexity given the ~10M sample dataset.
*   **Execution Guidance:** The process of alternating between `π_bc` and `π_ϕ` with random start timestep and duration is clear and differentiates it from JSRL. The theoretical analysis in Sec. B.2 (Proposition 2) is problematic. It aims to show performance improvement but relies on a bound from Kakade & Langford (2002) that assumes access to the advantage function `A^{π^e}` of the online policy, which is precisely what is hard to estimate. The proposition does not convincingly demonstrate why guidance with a BC policy from *non-curated* data should help, as it assumes `E[Σ π^g(·|s) A^{π^e}(s,a)] ≥ 0`, which is not guaranteed with suboptimal data. This analysis feels like an afterthought and does not strengthen the methodological contribution.
*   **Reproducibility:** Algorithm 1 provides a good overview. Appendix G offers implementation details for BC and baselines. Crucial details for the main method (e.g., the `λ` schedule for mixing online and retrieved data in the loss, the exact `α` probability schedule for execution guidance) are in hyperparameter tables (Appendix J), which is appropriate.

**Experiments & Results:** The experimental scope is a major strength, evaluating 72 pixel-based tasks across two benchmarks.
*   **Baselines:** The choice of baselines is comprehensive, covering training-from-scratch (DrQ-v2, DreamerV3), representation learning (R3M), and offline-data reuse methods (UDS, ExPLORe, JSRL-BC). The note that baselines are given task-relevant data (a favorable condition for them) while NCRL handles the full non-curated set is a fair point in NCRL's favor.
*   **Results:** The aggregate results (Fig. 3 left, Tables 3-5) show NCRL at 150k steps outperforming or matching baselines at 150k and often rivaling them at 500k/1M steps. This supports the sample efficiency claim. The comparison with iVideoGPT (Fig. 4) is compelling, showing gains without reward shaping or demonstration pre-filling. The task adaptation experiment (Fig. 5) is a nice additional demonstration.
*   **Concerns:** 1) **Statistical Significance:** While confidence intervals are shown in learning curves, the aggregate score comparisons in tables lack any measure of variance (standard deviation, confidence intervals) across seeds. For ICLR, reporting mean ± std or similar is essential to assess significance, especially when comparing means like 0.748 (NCRL) vs. 0.656 (DreamerV3 @1M). 2) **Ablations:** The ablation (Fig. 6) clearly shows the contribution of each component (P, ER, G). However, an ablation on the *retrieval mechanism itself* is missing. How does performance degrade with random retrieval or a simpler (e.g., random) sampling from `D_off`? Fig. 14 tests robustness to *irrelevant* data but not the necessity of intelligent retrieval. 3) **Baseline Calibration:** The DreamerV3 baseline uses a "commonly used but relatively small model-size configuration." The authors show in Fig. 15 that a larger DreamerV3 model does not consistently beat NCRL, which is good, but this should be more prominently discussed in the main text to preempt concerns about an unfair comparison.

**Writing & Clarity:** The paper is generally well-written and logically structured. Some minor points: The caption for Fig. 2 references a Wasserstein distance plot that is not clearly visible in the provided text (likely a formatting artifact). The phrase "non-curated" is sometimes hyphenated, sometimes not. The theoretical analysis section (B) is the least clear and, as noted, arguably the weakest part.

**Limitations & Broader Impact:** The limitations section (Appendix D) is good, acknowledging architectural constraints (RNN), generalization challenges, the in-domain nature of the data, and the simulation-only experiments. The broader impact statement is appropriately cautious. A more concrete limitation is the computational cost of pre-training a 280M parameter world model, which is noted in Appendix F (~48 GPU hours) but could be emphasized as a barrier to entry.

### Overall Assessment
This paper presents a strong, well-motivated contribution. The core idea—actively reusing non-curated data during fine-tuning via retrieval (experience rehearsal) and guided exploration (execution guidance)—is simple, intuitive, and demonstrated to be highly effective across an extensive suite of tasks. The empirical results are the paper's greatest strength, providing compelling evidence for the method's sample efficiency and robustness. The main weaknesses are the unconvincing theoretical analysis and some missing analysis on statistical significance and retrieval ablations. For ICLR, which values empirical rigor and novelty, the paper is likely above the acceptance bar provided the authors can address the concerns about statistical reporting and provide a more thorough ablation for the retrieval component. The theoretical section should either be substantially improved or its claims toned down.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces NCRL, a method that leverages non-curated (reward-free, mixed-quality, multi-embodiment) offline data to improve the sample efficiency of online reinforcement learning. The approach pre-trains a task-agnostic world model on such data and, during fine-tuning, employs two novel techniques—experience rehearsal (retrieving task-relevant trajectories) and execution guidance (using a behavior-cloned policy from retrieved data)—to mitigate distributional shift and guide exploration. Extensive evaluation on 72 visuomotor tasks demonstrates that NCRL nearly doubles the aggregate performance of training-from-scratch baselines under a limited sample budget and outperforms prior offline-data methods.

### Strengths
1. **Extensive and Rigorous Evaluation**: The paper evaluates NCRL across 72 diverse tasks (22 DMControl locomotion tasks and 50 Meta-World manipulation tasks) with pixel-based observations, using multiple seeds and consistent metrics. This large-scale benchmark convincingly demonstrates the method’s broad applicability and robustness, going beyond the limited task sets common in prior work.
2. **Well-Motivated and Novel Techniques**: The authors identify a key failure mode of naive world-model fine-tuning—distributional shift between offline and online data—and propose two novel, intuitive solutions. Experience rehearsal mitigates shift by retrieving relevant offline trajectories, while execution guidance leverages a behavior-cloned policy to steer exploration. Both are supported by empirical analysis (e.g., t-SNE visualizations) and theoretical justifications in the appendix.
3. **Strong Empirical Comparisons**: NCRL is compared against a comprehensive set of baselines, including training-from-scratch methods (DrQ-v2, DreamerV3), prior offline-to-online RL approaches (R3M, UDS-RLPD, ExPLORe, JSRL-BC), and recent world-model pre-training methods (iVideoGPT). NCRL consistently outperforms all, often by significant margins, and thorough ablations validate each component’s contribution.

### Weaknesses
1. **Limited Generalization Assessment**: While the method uses multi-embodiment data, evaluation is limited to embodiments seen during pre-training. There is no systematic study of generalization to entirely new embodiments or out-of-distribution tasks, which is critical for real-world applicability.
2. **Incomplete Comparison with Contemporary Methods**: The paper could better situate itself by comparing with recent large-scale world models or foundation models for robotics (e.g., Cosmos, GAIA-1, RT-2, Octo). Although iVideoGPT is included, the field is rapidly evolving, and such comparisons would strengthen the paper’s relevance.
3. **Overhead of Retrieval Mechanism**: Experience rehearsal relies on nearest-neighbor search over the offline dataset using Faiss. While retrieval is reported to take seconds, the computational and memory costs of indexing and storing embeddings for very large (e.g., internet-scale) datasets are not analyzed. This could become a bottleneck in scaling.

### Novelty & Significance
The paper makes a clear and novel contribution by addressing the under-explored problem of leveraging non-curated offline data—reward-free, mixed-quality, and multi-embodiment—for RL. The identification of distributional shift during fine-tuning and the proposed techniques to counteract it are novel and well-motivated. The extensive evaluation across 72 tasks significantly advances beyond prior work that often uses smaller, curated datasets. Given the growing interest in leveraging diverse offline data for RL, this work is likely to influence future research in offline-to-online RL and world model pre-training.

### Suggestions for Improvement
1. **Evaluate Generalization to Unseen Embodiments**: Conduct experiments where the pre-trained world model is fine-tuned on tasks involving embodiments not present in the offline data. This would better demonstrate the method’s ability to handle true multi-embodiment generalization.
2. **Expand Comparisons to Recent Foundation Models**: Include comparisons with state-of-the-art large-scale world models or robotic foundation models (e.g., Cosmos, RT-2) if feasible, or at least discuss how NCRL relates to these approaches in the related work.
3. **Analyze Scalability of Retrieval**: Provide a more detailed analysis of the computational overhead of the retrieval step, including indexing time, memory footprint, and retrieval latency as the offline dataset size grows. Discuss potential optimizations (e.g., hierarchical indexing) for scaling to very large datasets.
4. **Elaborate on Limitations and Societal Impact**: The limitations section is brief. Expand on assumptions (in-domain data, RSSM architecture), computational costs of pre-training, and potential negative societal impacts (e.g., misuse in autonomous systems) with more concrete examples.
5. **Improve Figure Clarity and Accessibility**: Some figures (e.g., Figure 1) are blurry and contain small text. Ensure all figures are legible and consider adding a summary table of key results for quick reference. Also, clarify the axes and legends in learning curves where necessary.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with state-of-the-art world model pre-training methods that use non-curated data.** The paper compares with iVideoGPT but only in a limited setting. A direct, head-to-head comparison on the same tasks and datasets with modern, scalable world models (e.g., transformer or diffusion-based) is missing. This is critical to show the contribution is not just about better fine-tuning, but a meaningful advance over the strongest existing architectures.

2. **Ablation on the importance of multi-embodiment pre-training.** The paper trains a single world model across 6 embodiments, but there's no control experiment comparing against separate, embodiment-specific pre-trained models. This gap undermines the claim that a unified model effectively leverages multi-embodiment data and does not suffer from interference.

3. **Experiments scaling the amount and diversity of offline data.** The paper uses a fixed dataset of 60k trajectories. To validate the core premise—that leveraging abundant non-curated data improves efficiency—the paper needs experiments varying dataset size (e.g., 10k vs 100k vs 1M trajectories) and quality mix. Without this, it's unclear if the gains are due to the method or simply having more data.

4. **Evaluation on truly out-of-domain offline data.** The offline data is "in-domain" (from the same benchmarks). To substantiate the claim of using "non-curated" data, an experiment using offline data from a different source (e.g., random robot videos or a different simulator) is necessary. Its absence leaves the method's robustness and generalizability in doubt.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of distribution shift and its mitigation.** The paper uses t-SNE and Wasserstein distance qualitatively. A rigorous analysis is missing: how does the world model's prediction error (e.g., MSE in latent or pixel space) correlate with online performance, and how does experience rehearsal reduce this error? This is needed to trust the claimed root cause of failure.

2. **Analysis of what makes retrieval successful or fail.** The paper reports high precision but doesn't analyze the content of retrieved trajectories. A breakdown of retrieved data by task success, embodiment, or behavior policy is needed to understand whether performance gains come from near-expert data, diverse coverage, or both. This directly affects the claim about leveraging "mixed-quality" data.

3. **Understanding the contribution of execution guidance vs. simply better exploration.** The comparison with OTS is insufficient. A deeper analysis should measure state-space coverage (e.g., coverage metrics) with and without guidance, and disentangle whether gains are from steering to high-confidence regions or simply injecting more diverse actions from the prior.

### Visualizations & Case Studies
1. **Visualization of world model rollouts before and after fine-tuning.** Showing side-by-side video predictions of the pre-trained model, the fine-tuned model (with/without rehearsal) on key tasks would reveal whether the model actually learns task-relevant dynamics and if rehearsal prevents degradation.

2. **Case studies of failure modes on the hardest tasks.** The paper notes some Meta-World tasks have low success rates. Detailed qualitative analysis (e.g., video of agent behavior) on why the method fails on specific tasks (e.g., Disassemble, Pick Place) is essential to identify limitations and whether they are due to exploration, dynamics modeling, or retrieval.

### Obvious Next Steps
1. **Apply the method to a real-world robotic setup with existing "non-curated" logged data.** The most compelling next step for impact is demonstrating the method on a physical robot using offline data from previous experiments or human teleoperation. The paper's claims about practicality remain speculative without this.

2. **Integrate with a more scalable world model architecture (e.g., transformer).** The paper uses an RSSM. Given the trend toward scalable architectures for world models, the obvious technical next step is to combine the proposed fine-tuning techniques with a more modern backbone to handle even more diverse data.

3. **Extend the retrieval mechanism to be adaptive during training.** The current retrieval is static (based on initial observation). An obvious improvement is to continuously update the retrieved set based on the agent's current state or learned task representation, which would better handle non-stationary online data distributions.

# Final Consolidated Review
## Summary
This paper introduces NCRL, a method for sample-efficient reinforcement learning that leverages non-curated (reward-free, mixed-quality, multi-embodiment) offline data. It pre-trains a task-agnostic world model and, during online fine-tuning, employs two novel techniques—experience rehearsal (retrieving relevant trajectories) and execution guidance (using a behavior-cloned prior)—to mitigate distributional shift and guide exploration. Evaluated across 72 visuomotor tasks, NCRL significantly outperforms training-from-scratch and prior offline-data methods under limited sample budgets.

## Strengths
- **Extensive and convincing empirical evaluation.** The paper demonstrates effectiveness across 72 diverse pixel-based tasks (22 DMControl, 50 Meta-World) spanning 6 embodiments, using multiple seeds. This large-scale benchmark robustly shows NCRL nearly doubles aggregate performance of strong baselines (DrQ-v2, DreamerV3) at 150k steps and matches their performance with 3-6x more samples.
- **Well-motivated and simple technical contributions.** The paper identifies distributional shift during world model fine-tuning as a key failure mode (supported by analysis in Fig. 2) and proposes two intuitive, novel solutions—experience rehearsal and execution guidance—to reuse offline data effectively during fine-tuning. Ablations confirm each component's contribution.

## Weaknesses
- **Insufficient statistical reporting for key aggregate claims.** The paper reports mean success rates/episodic returns in tables but does not include measures of variance (e.g., standard deviation, confidence intervals) across seeds. For ICLR, this is essential to assess the significance of claims, such as NCRL's mean score of 0.748 vs. DreamerV3's 0.656 at 1M steps on Meta-World.
- **Theoretical analysis is unconvincing and adds little.** Proposition 2 (Appendix B.2) aims to justify execution guidance but relies on the assumption that the behavior-cloned policy from potentially suboptimal data has non-negative advantage relative to the online policy, which is not guaranteed and not demonstrated. This analysis does not strengthen the methodological contribution.
- **Retrieval mechanism description is sparse.** While experience rehearsal is central, the description of the retrieval process is minimal. Equation 5 states the encoder `e_θ` is used but does not specify if this is the world model encoder `q_θ` or a separate network. Details on indexing and search efficiency given a ~10M sample dataset are lacking.

## Nice-to-Haves
- **Ablation on the necessity of intelligent retrieval.** The paper shows robustness to injecting irrelevant data (Fig. 14) but does not ablate against a baseline that randomly samples from the offline dataset instead of performing nearest-neighbor retrieval. This would help isolate the contribution of the retrieval mechanism itself.
- **Analysis of what makes retrieved trajectories helpful.** A deeper breakdown of retrieved data (e.g., by success rate, embodiment, or behavior policy) could clarify whether gains stem from near-expert snippets, diverse state coverage, or both, strengthening the claim about leveraging "mixed-quality" data.
- **Discussion of computational cost trade-offs.** Pre-training a 280M parameter world model (~48 GPU hours) and maintaining an index for retrieval incur non-trivial costs. A brief discussion of these trade-offs relative to the sample efficiency gains would provide practical context.

## Novel Insights
The paper's core insight is that naive fine-tuning of world models pre-trained on non-curated data often fails due to distributional shift between offline and online data, and that this can be effectively addressed by continuing to leverage the offline dataset during fine-tuning. Specifically, it demonstrates that retrieving task-relevant trajectories (experience rehearsal) acts as a regularizer to prevent catastrophic forgetting and augment the state distribution for model rollouts, while a simple behavior-cloned prior policy (execution guidance) can robustly steer exploration without complex reward shaping.

## Suggestions
- Report mean ± standard deviation or confidence intervals for all aggregate metrics in tables (e.g., Tables 3-5) to allow proper assessment of statistical significance.
- Either substantially improve the theoretical analysis in Section B.2 by grounding it in the practical setting (e.g., analyzing when the BC policy might provide a good prior) or remove/clearly frame it as informal motivation rather than a formal guarantee.
- Clarify the retrieval mechanism in Section 3.3: specify the encoder architecture, discuss indexing costs, and mention the retrieval frequency (e.g., once at the start of fine-tuning or periodically).

# Actual Human Scores
Individual reviewer scores: [6.0, 10.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
