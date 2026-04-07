=== CALIBRATION EXAMPLE 88 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: using non-curated data to guide world models for sample-efficient RL. The abstract clearly states the problem setting (reward-free, mixed-quality, multi-embodiment data), identifies a key failure mode (distributional shift during fine-tuning), and proposes two techniques (experience rehearsal and execution guidance) to address it. The claims of improved sample efficiency (nearly double aggregate score across 72 tasks) are specific and appear supported by the experiments detailed later. No major concerns.

### Introduction & Motivation
The introduction effectively motivates the problem by highlighting the limitations of prior work that relies on curated, reward-labeled data. The research question (“How can we effectively leverage non-curated offline data for efficient RL?”) is well-defined. The contributions (C1–C4) are clearly stated and substantial. The introduction could be strengthened by a more explicit discussion of why prior world-model pre-training methods (e.g., iVideoGPT) show marginal gains despite massive datasets, but it adequately sets up the distribution-shift issue.

### Related Work (Section 2)
The related work covers relevant areas: task-specific offline RL, multi-task offline RL, representation learning, and world-model pre-training. It succinctly positions NCRL relative to these lines of work. The extended discussion in Appendix C is appropriate. A minor point: the table reference (Table 1) appears in the text but the table itself is missing from the extracted content; however, this is likely a parsing artifact and not a substantive issue.

### Methods (Section 3)
The method is described in two stages: world-model pre-training and RL fine-tuning with experience rehearsal and execution guidance.

**Pre-training**: The use of a modified RSSM (removing task losses, zero-padding actions, scaling to 280M parameters) is straightforward. The loss function is standard. The authors correctly note that architectural improvements are orthogonal.

**Fine-tuning**: The integration of DreamerV3-style updates with the proposed techniques is clear. Key innovations:
- **Experience rehearsal**: Retrieves task-relevant trajectories via neural feature distance (Eq. 5). This addresses distributional shift by continuing to train the world model on relevant offline data, augmenting initial states for rollouts, and providing data for behavior cloning. The theoretical analysis in Appendix B.1 supports the claim that retrieval reduces distribution shift.
- **Execution guidance**: Uses a BC policy (trained on retrieved data) to guide exploration, with a schedule for switching between the prior and RL policy. Proposition 2 (Appendix B.2) provides a theoretical justification for performance improvement.

**Potential concerns**:
- The retrieval mechanism relies on the pre-trained encoder’s features. While Table 2 shows high precision, robustness across all tasks (e.g., Door Open with 68% precision@500) could be discussed further. The authors note that RL training remains effective despite some irrelevant retrievals, but a sensitivity analysis (Fig. 14) shows performance degrades with increasing task-irrelevant data. This is acknowledged but could be elaborated.
- Equation (3) defines the λ-return using a distributional critic, but the notation is slightly confusing (e.g., \(v_t^\lambda\) is the expectation of the value distribution, yet \(V_t^\lambda\) is used recursively). This could be clarified for better readability.
- The weights \(\beta_1, \beta_2, \beta_3\) in Eq. (1) are not specified; presumably they follow DreamerV3 defaults, but this should be noted.

Overall, the method is novel, well-motivated, and technically sound.

### Experiments & Results (Section 4)
The experimental evaluation is extensive, covering 72 tasks from DMControl and Meta-World. Comparisons include:
1. Offline-data methods (R3M, UDS-RLPD, ExPLORe, JSRL-BC).
2. Training-from-scratch baselines (DrQ-v2, DreamerV3).
3. Model-based methods (iVideoGPT, pre-trained DreamerV3).
4. Continual adaptation (PackNet).

Results show NCRL significantly outperforms all baselines in sample efficiency, achieving at 150k steps what baselines require 3.3–6.7× more steps to match. Ablation studies (Fig. 6) confirm the contribution of each component. Additional analyses (hyperparameter sensitivity, retrieval robustness, model size) are thorough.

**Potential concerns**:
- Some DMControl tasks (e.g., Acrobot Swingup Hard, Cheetah Run) show lower performance for NCRL compared to DreamerV3 at 150k. The authors should briefly discuss possible reasons (e.g., inadequate coverage by offline data) to provide a balanced view.
- The comparison with iVideoGPT in Fig. 4 uses the original iVideoGPT results, which include reward shaping and demonstration pre-filling. The authors address this in Appendix A.2 with an aligned version (iVideoGPT-align) and show NCRL still outperforms, which strengthens the claim.
- The number of seeds differs (5 for NCRL, 3 for some baselines). While 3 seeds are acceptable, consistency would be ideal. Confidence intervals are reported, which is good.

**Overall**, the experiments are comprehensive, support the claims, and adhere to ICLR standards.

### Writing & Clarity
The paper is generally well-written and logically structured. Figures are informative. Minor issues likely due to PDF extraction exist (e.g., “mix-quality” should be “mixed-quality”, formatting glitches in Fig. 1 caption). The notation in Eq. (3) could be clarified for easier parsing. These do not impede understanding.

### Limitations & Broader Impact
Limitations are discussed in Appendix D: reliance on RNN-based world models (scaling challenges), limited generalization to new embodiments, use of in-domain (not in-the-wild) data, and simulation-only experiments. These are honest and reasonable. The ethics statement is appropriate. A minor point: computational cost of pre-training a 280M-parameter model is non-trivial but acknowledged.

### Overall Assessment
This paper makes a significant contribution by effectively leveraging non-curated offline data for RL via a world-model approach. The identification of distributional shift as a key failure mode and the proposed techniques (experience rehearsal and execution guidance) are novel and well-supported by extensive experiments across 72 tasks. The paper is technically sound, clearly written, and meets ICLR’s standards for novelty, empirical rigor, and impact. Minor weaknesses (clarity of notation, discussion of a few underperforming tasks) do not detract from the overall contribution. **This paper is a strong candidate for acceptance.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes NCRL, a method to leverage non-curated offline data (reward-free, mixed-quality, multi-embodiment) for sample-efficient online reinforcement learning. It identifies that naive fine-tuning of a world model pre-trained on such data often fails due to distributional shift and proposes two key techniques: experience rehearsal (retrieving task-relevant trajectories) and execution guidance (using a behavior-cloned prior policy) to mitigate this issue during RL fine-tuning.

### Strengths
1.  **Realistic and Valuable Problem Setting:** The paper focuses on a highly practical yet under-explored setting: leveraging abundant, non-curated offline data that lacks rewards and contains mixed-quality trajectories from multiple embodiments. This significantly expands the pool of usable data compared to prior work that assumes curated, reward-labeled datasets.
2.  **Rigorous and Extensive Empirical Evaluation:** The evaluation is comprehensive, spanning 72 visuomotor tasks across 6 embodiments from DMControl and Meta-World. The method is compared against a wide array of strong baselines (e.g., DrQ-v2, DreamerV3, ExPLORe, JSRL-BC, iVideoGPT). The results are compelling, showing that NCRL nearly doubles the aggregate score of from-scratch baselines under a limited sample budget (150k) and outperforms prior offline-data methods by a decent margin.
3.  **Clear Diagnosis and Well-Motivated Solutions:** The paper provides a clear analysis (visualized via t-SNE) identifying distributional shift between offline pre-training and online fine-tuning data as a key failure mode. The proposed techniques—experience rehearsal and execution guidance—are directly motivated by this analysis to regularize the world model and guide exploration.
4.  **Strong Ablation Studies and Analysis:** Ablation studies (Fig. 6, 13) convincingly demonstrate the contribution of each component (pre-training, rehearsal, guidance). Additional analyses on hyperparameter sensitivity (Fig. 12), robustness to retrieval quality (Fig. 14), and adaptation for continual learning (Fig. 5) add depth to the understanding of the method.

### Weaknesses
1.  **Limited World Model Architecture:** The core world model is a scaled-up RSSM (RNN-based). While sufficient for the presented results, the paper acknowledges that this architecture may not be optimal for scaling to even larger and more diverse datasets. More modern architectures (e.g., transformers, diffusion models) are mentioned as future work but not integrated.
2.  **In-Domain (Not In-the-Wild) Data:** The "non-curated" offline data, while mixed-quality and reward-free, is still collected from the same benchmark domains (DMControl, Meta-World) as the downstream tasks. The paper does not demonstrate the use of truly in-the-wild, internet-scale video data, which is the ultimate promise of this research direction. The limitation is acknowledged but not addressed.
3.  **High Computational Cost and Lack of Real-World Validation:** Pre-training requires significant resources (~48 GPU hours), and full reproduction of the 72-task results would take ~1700 GPU hours. Furthermore, all experiments are conducted in simulation. While this is common, validation on a real robotic system would significantly strengthen the practical impact claim.
4.  **Incomplete Success on All Tasks:** While aggregate performance is strong, some challenging Meta-World tasks (e.g., Disassemble, Pick Place) still show very low success rates even with NCRL (Table 3, 4). This indicates remaining frontiers for exploration and manipulation tasks with long horizons or precise requirements.

### Novelty & Significance
**Novelty:** The paper's primary novelty lies in the specific problem formulation—leveraging non-curated, multi-embodiment data without rewards—and the subsequent identification and mitigation of the distributional shift problem during fine-tuning. While individual components like trajectory retrieval and guided rollout have precedents, their integration into a coherent framework to solve this specific problem is novel.
**Significance:** The work is highly significant for the field of sample-efficient RL. It provides a clear pathway to leverage vast, readily available datasets that lack rewards, moving beyond the bottleneck of curated data. The strong empirical results across a very broad task suite convincingly demonstrate the effectiveness of the proposed approach, setting a new benchmark for methods in this setting. It meets ICLR's expectation for solid empirical validation of a well-motivated idea.

### Suggestions for Improvement
1.  **Integrate a More Scalable World Model:** To better prepare for future scaling, the next version could include a comparative experiment using a transformer- or diffusion-based world model backbone alongside the RSSM, discussing trade-offs in performance, training stability, and scaling potential.
2.  **Explore In-the-Wild Pre-training:** A compelling extension would be to first pre-train on a large, diverse, in-the-wild video dataset (e.g., something akin to Ego4D or internet videos), and then further pre-train or adapt using the in-domain, embodiment-specific data used in this paper. This would test the method's ability to handle a truly large distribution shift.
3.  **Provide a Deeper Theoretical Analysis:** The theoretical analysis in Appendix B, while a good start, is relatively brief. Expanding this section to provide formal guarantees or more detailed insights (e.g., on the convergence properties of the mixed policy or the conditions under which retrieval is most beneficial) would strengthen the methodological foundation.
4.  **Address Remaining Challenging Tasks:** The paper could include a brief discussion or small experiment analyzing why certain Meta-World tasks remain difficult (e.g., is it a model prediction error, an exploration issue, or a credit assignment problem?) and suggest potential avenues to address them, perhaps by modifying the retrieval or guidance strategy for such tasks.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation without retrieval:** Compare experience rehearsal against simply mixing the entire offline dataset (or random subsets) into the replay buffer. This is necessary to isolate whether retrieval is crucial or if any offline data suffices.
2. **Comparison with recent offline-to-online RL methods:** Omitted baselines like Cal-QL, AWAC, or MOReL—which can be adapted to reward-free settings—weaken the claim that NCRL outperforms prior work. Including them would strengthen the comparison.
3. **Generalization to unseen embodiments:** Pre-training on a subset of embodiments and testing on a held-out one is missing. This is critical to validate the claim that multi-embodiment pre-training generalizes.
4. **Guidance vs. intrinsic exploration:** Ablate execution guidance against standard exploration techniques (e.g., RND, curiosity) to show that guidance is uniquely beneficial, not just any exploration bonus.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative distribution shift analysis:** Measure divergence (e.g., Wasserstein distance) between offline and online data across tasks and correlate it with performance drop. This would substantiate the core motivation.
2. **World model accuracy during fine-tuning:** Track prediction error on online data with/without experience rehearsal to demonstrate that rehearsal prevents degradation, directly supporting the stabilization claim.
3. **Quality of the behavior-cloned policy:** Report the success rate or return of the guidance policy and correlate it with the benefit of execution guidance. This would validate when and why guidance helps.

### Visualizations & Case Studies
1. **Visualize retrieved trajectories:** Show example trajectories (e.g., keyframes) retrieved for a few tasks to confirm they are task-relevant and not trivial.
2. **State visitation during early training:** Plot state visits with/without execution guidance to illustrate how guidance expands exploration, especially for hard tasks.

### Obvious Next Steps
1. **Scale to larger, in-the-wild datasets:** The offline data is still domain-specific. Using truly diverse internet videos would better test the "non-curated" claim and show broader applicability.
2. **Real-world robotic validation:** All experiments are in simulation. Demonstrating sample-efficient learning on a real robot would significantly increase impact.

# Final Consolidated Review
## Summary
This paper introduces NCRL, a method for sample-efficient reinforcement learning that leverages non-curated offline data—reward-free, mixed-quality, and multi-embodiment—by pre-training a world model and fine-tuning it with two novel techniques: experience rehearsal (to mitigate distributional shift) and execution guidance (to steer exploration). The method is extensively validated across 72 visuomotor tasks, showing significant improvements in sample efficiency over strong training-from-scratch and offline-data baselines.

## Strengths
- **Addresses a Realistic and Valuable Setting:** The paper tackles the practical problem of utilizing abundant, non-curated data that lacks reward labels and contains trajectories of varying quality from multiple embodiments, significantly expanding the usable data pool for RL compared to prior work that relies on curated, reward-labeled datasets.
- **Rigorous and Comprehensive Empirical Validation:** The evaluation is exceptionally thorough, spanning 72 diverse tasks from DMControl and Meta-World. NCRL is compared against a wide array of strong baselines, including training-from-scratch methods (DrQ-v2, DreamerV3), prior offline-data methods (R3M, ExPLORe, JSRL-BC, UDS-RLPD), and model-based approaches (iVideoGPT). The results convincingly demonstrate that NCRL nearly doubles the aggregate score of from-scratch baselines under a tight sample budget (150k) and outperforms prior methods by a decent margin.
- **Clear Mechanistic Diagnosis and Well-Motivated Solutions:** The paper provides a clear, visualized analysis identifying distributional shift between offline pre-training and online fine-tuning data as a key failure mode of naive fine-tuning. The proposed techniques—experience rehearsal (retrieving task-relevant trajectories) and execution guidance (using a behavior-cloned prior)—are directly motivated by this diagnosis to regularize the world model and guide exploration, supported by theoretical analysis in the appendix.

## Weaknesses
- **Reliance on a Potentially Less Scalable World Model Architecture:** The core world model is a scaled-up Recurrent State Space Model (RSSM). While effective for the presented scale, the paper acknowledges this RNN-based architecture may face challenges when scaling to even larger and more diverse datasets compared to more modern architectures like Transformers or diffusion models. This is a limitation for future scaling ambitions.
- **Offline Data is In-Domain, Not Truly "In-the-Wild":** The "non-curated" offline data, while reward-free and mixed-quality, is still collected from the same benchmark domains (DMControl, Meta-World) as the downstream tasks. The method is not demonstrated on truly in-the-wild, internet-scale video data, which is the ultimate promise of leveraging non-curated data. This limits the immediate claim about handling vast, unstructured data sources.
- **High Computational Cost and Simulation-Only Validation:** Pre-training a 280M-parameter model requires significant resources (~48 GPU hours), and full reproduction across 72 tasks would be expensive (~1700 GPU hours). Furthermore, all experiments are conducted in simulation. While common, validation on a real robotic platform would significantly strengthen the claim of practical impact for robotic applications.
- **Incomplete Success on Some Challenging Tasks:** While aggregate performance is strong, some challenging Meta-World tasks (e.g., Disassemble, Pick Place, Soccer) still show very low success rates with NCRL (see Tables 3 & 4). This indicates remaining frontiers for tasks with long horizons, small objects, or precise requirements, suggesting the method's limitations in the hardest zero-shot exploration settings.

## Nice-to-Haves
- Including a brief qualitative analysis or discussion of the specific failure modes on the challenging Meta-World tasks (e.g., is it a model prediction error, an exploration issue, or a credit assignment problem?) could provide clearer directions for future work.
- A small experiment comparing the RSSM backbone to a more modern, scalable architecture (e.g., a Transformer) could better illustrate the trade-offs and prepare the method for future scaling.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness about notation clarity in Equation (3):** This is a minor presentation issue that does not impede understanding of the method or its results.
- **Weakness about missing comparison with specific offline-to-online RL methods (Cal-QL, AWAC):** The paper explicitly focuses on the reward-free, non-curated setting, which these methods are not designed for. Including them would be scope creep. The provided comparison against adapted baselines (UDS-RLPD, ExPLORe, JSRL-BC) is appropriate.
- **Weakness about inconsistent number of seeds (5 vs. 3):** While ideal, using 3 seeds for baselines is standard and sufficient, and confidence intervals are reported. This does not undermine the validity of the results.
- **Weakness about the need for "deeper theoretical analysis":** The paper includes theoretical propositions in the appendix supporting the core claims about retrieval and guidance. Demanding extensive formal guarantees is beyond the standard for a primarily empirical systems paper in RL.
- **Strength about "the paper is well-written":** This is a generic strength that applies to many papers and is not specific to this work's contributions.

## Novel Insights
The paper's core novel insight is the identification and mechanistic analysis of distributional shift as the primary failure mode when fine-tuning world models pre-trained on non-curated offline data for downstream RL tasks. This insight directly motivates the integrated two-stage solution: using the same offline data not just for pre-training but also during fine-tuning via task-relevant retrieval (experience rehearsal) to stabilize the model and a learned policy prior (execution guidance) to guide exploration. This framework demonstrates that effectively leveraging such unstructured data requires mitigating the shift between pre-training and online fine-tuning distributions, a point under-explored in prior world-model pre-training work.

## Suggestions
- To address the scalability limitation, future work could integrate a more scalable world model architecture (e.g., Transformer-based) within the NCRL framework and briefly discuss the comparative trade-offs in performance and training stability.
- Conduct a focused analysis on the tasks where NCRL still struggles (e.g., Disassemble), perhaps visualizing state visitation or model prediction error, to better understand the remaining challenges and inform improvements to the retrieval or guidance mechanisms.

# Actual Human Scores
Individual reviewer scores: [6.0, 10.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
