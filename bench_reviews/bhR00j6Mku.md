## Summary
This paper systematically investigates benchmark contamination in Large Reasoning Models (LRMs). It identifies two critical vulnerabilities: in Stage I, contamination evidence introduced via Supervised Fine-Tuning (SFT) can be concealed by subsequent Reinforcement Learning (RL) training (specifically methods with PPO-style objectives); in Stage II, extensive contamination with Chain-of-Thought (CoT) data on advanced LRMs leaves almost no detectable evidence for existing memorization-based detection methods. The findings reveal that current detection paradigms are fragile, posing a significant threat to the integrity of LRM leaderboards.

## Strengths
- **Identifies a novel and significant vulnerability:** The paper provides the first systematic evidence that standard RL training (GRPO) can actively conceal prior SFT contamination, a finding with serious implications for evaluation fairness. The theoretical analysis linking this to PPO-style importance sampling and clipping offers a principled explanation.
- **Comprehensive empirical evaluation across realistic scenarios:** The study rigorously tests two high-stakes contamination scenarios ("pre-LRM" and "post-LRM") across six reasoning benchmarks, four base/advanced models, and ten representative detection methods, providing strong, reproducible evidence for its core claims.
- **Strong technical depth with theory and ablation:** The work goes beyond pure empiricism by providing a theoretical analysis of why PPO-style objectives contract the log-probability gap between members and non-members. The ablation studies (e.g., RAFT vs. RAFT++) convincingly isolate clipping as a key driver of the concealment effect.

## Weaknesses

### Major:
- **Unverified claim of broad RL class vulnerability:** The paper concludes that "a broad class of RL methods may inherently exhibit similar concealment capability," but this is only supported by analysis of PPO-style objectives (GRPO, RAFT++). Crucially, the paper does not test RL methods from fundamentally different paradigms (e.g., REINFORCE, Q-learning variants) that lack importance sampling and clipping. Without this, the claim of broad-class vulnerability is overstated and not empirically substantiated.
- **Incomplete exploration of the threat model for Stage I:** The experiments focus on SFT contamination concealed by subsequent *clean* RL. A more potent and realistic evasion strategy for a malicious actor would be to include benchmark data *during* the RL phase itself. The paper does not test whether RL training can conceal contamination it simultaneously introduces, which is a critical gap in assessing the full scope of the vulnerability.
- **Under-analysis of why CoT contamination evades detection (Stage II):** The finding that CoT contamination leaves minimal evidence is critical but poorly explained. The suggestion that models "internalize knowledge" and generalize is plausible but not rigorously tested. A deeper analysis is needed—for instance, comparing the verbatim overlap or semantic similarity of generated reasoning steps for members vs. non-members—to distinguish between sophisticated memorization and true generalization. This limits understanding of the failure mode for detectors.

### Minor
- **Limited model scale and architecture exploration:** Experiments are conducted on 7B-8B and 14B parameter models. While sufficient to demonstrate the effect, the community's push toward larger LRMs raises the question of how these concealment and detection failures scale. A discussion or preliminary experiments on scaling trends would strengthen the long-term relevance of the findings.
- **Lack of adaptive detection baselines:** The evaluation uses static, off-the-shelf detection methods. A more rigorous stress test of the fragility claim would involve testing against adaptive detection strategies (e.g., LiRA modified to use intermediate checkpoints, or detectors fine-tuned on LRM outputs) that an evaluation platform might deploy in response to such threats. Their failure would further underscore the paper's message.

### Trivial
- **Repetition in textual descriptions of figures/tables:** Some findings are stated in the text and then immediately restated in the caption of the referenced figure or table (e.g., descriptions of Table 2 and Figure 2). This is a minor clarity issue that could be streamlined.

## Nice-to-Haves
- A simple, proof-of-concept experiment testing whether the release of intermediate checkpoints (as suggested in the conclusion) could aid detection by comparing metrics from pre-RL and post-RL stages.
- A more detailed visualization of how token-level loss distributions evolve over the course of RL training, rather than just the before/after snapshots shown, to better illustrate the dynamics of concealment.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

**Strengths Removed:**
- "The paper is well-written and easy to follow." (Removed: Generic strength applicable to any competent paper.)
- "The topic of benchmark contamination is important for the community." (Removed: Generic strength, does not identify what *this* paper does specifically.)

**Weaknesses Removed:**
- **"The paper does not test contamination introduced for the first time during RL."** (Removed: While a valid experimental direction, this is a scope expansion. The paper explicitly defines its two scenarios (Fig 1, Sec 3 & 4). Criticizing it for not studying a third, different scenario is scope creep. The paper's claims are valid within its stated scope.)
- **"The evaluation suffers from temporal bias in data splits."** (Removed: The paper does not use time-based splits. It states, "For each dataset, we randomly sample half of the questions as the member set... and leave the remaining half as the non-member set" (Sec 3). The criticism misreads the methodology.)
- **"Insufficient controls for CoT-related factors like prompt length."** (Removed: This is a highly speculative criticism. The paper's finding is that detection fails *despite* the models' behavior; demanding an analysis of every potential confounding variable in the CoT data is not a standard requirement for the core claim being made.)
- **"The discussion on practical implications for leaderboards is inadequate."** (Removed: Weakened to "Nice-to-Have". Section 5 provides concrete, actionable directions (releasing checkpoints, moving beyond memorization-based detection). Demanding a full protocol specification is beyond the paper's contribution of identifying the vulnerability.)

## Suggestions
- To support the "broad class of RL methods" claim, add an experiment with a non-PPO-style RL algorithm (e.g., REINFORCE or a simple policy gradient without clipping). If concealment does not occur, the claim can be refined to be specific to PPO-style objectives, which is still a highly significant and widely used class.
- Conduct a focused analysis for Stage II: For a subset of questions, compute metrics like ROUGE-L or BLEU between the generated CoT for a member question and its original training CoT, and compare this to the same metric for a non-member's CoT against *any* training CoT. This would provide concrete evidence on whether the undetectability stems from lack of verbatim memorization.
- In the conclusion or discussion, briefly hypothesize how the findings might change with larger models (e.g., 70B+ parameters) to frame the work within the ongoing scaling trends.

## Evaluation
- **Novelty:** High. This is the first work to systematically study contamination and concealment specifically in the LRM paradigm, uncovering two previously unreported and serious vulnerabilities.
- **Technical Soundness:** Strong. The methodology is robust, experiments are well-controlled (e.g., ruling out simple forgetting), and the theoretical analysis provides meaningful insight into the Stage I results.
- **Empirical Support:** Very Strong. The evaluation is extensive across scenarios, models, benchmarks, and detection methods. Results are clear, statistically supported (AUROC), and visualized effectively.
- **Significance:** Very High. The paper identifies a concrete and easily exploitable threat to the integrity of LRM leaderboards, a central fixture in contemporary AI research. It convincingly argues that current detection safeguards are inadequate, necessitating a community response.
- **Clarity:** Good. The paper is logically structured and clearly written, though some passages are slightly repetitive. Figures and tables effectively support the narrative.

**Overall, this is a strong paper that makes a timely, important, and well-supported contribution. It reveals critical flaws in the current ecosystem for evaluating reasoning models and provides a solid foundation for future work on robust evaluation.** The major weaknesses pertain primarily to the breadth of one claim and the depth of analysis for one finding, but they do not undermine the paper's core, valuable contributions.