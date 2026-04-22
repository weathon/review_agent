# Repurposing Synthetic Data for Fine-grained Search Agent Supervision

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
LLM-based search agents are increasingly trained on entity-centric synthetic data to solve complex, knowledge-intensive tasks. However, prevailing training methods like Group Relative Policy Optimization (GRPO) discard this rich entity information, relying instead on sparse, outcome-based rewards. This critical limitation renders them unable to distinguish informative "near-miss" samples—those with substantially correct reasoning but a flawed final answer—from complete failures, thus discarding valuable learning signals. We address this by leveraging the very entities discarded during training. Our empirical analysis reveals a strong positive correlation between the number of ground-truth entities identified during an agent's reasoning process and final answer accuracy. Building on this insight, we introduce Entity-aware Group Relative Policy Optimization (E-GRPO), a novel framework that formulates a dense entity-aware reward function. E-GRPO assigns partial rewards to incorrect samples proportional to their entity match rate, enabling the model to effectively learn from these ''near-misses''. Experiments on diverse question-answering (QA) and deep research benchmarks show that E-GRPO consistently and significantly outperforms the GRPO baseline. Furthermore, our analysis reveals that E-GRPO not only achieves superior accuracy but also induces more efficient reasoning policies that require fewer tool calls, demonstrating a more effective and sample-efficient approach to aligning search agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a reinforcement learning framework Entity-aware Group Relative Policy Optimization (E-GRPO), to enhance the training of LLM-based search agents. Existing GRPO methods usually rely on sparse, outcome-based rewards, treating all incorrect outputs equally and overlooking partially correct cases. In contrast, E-GRPO uses entity-centric information from synthetic training data by assigning partial rewards based on the entity match rate, i.e., the proportion of ground-truth entities correctly identified during reasoning. Experiments also show that entity-aware rewards provide a computationally efficient and semantically rich supervision signal for training search agents.

### Strengths
1.	Comprehensive experimental results are reported, including diverse LLM-based search agents and datasets.

2.	The experimental settings and evaluation details are clearly provided to facilitate reproducing.

3.	The paper is generally clearly-written and easy to follow.

### Weaknesses
1.	The empirical finding of “strong positive correlation between the number of ground-truth entities identified during the agent’s reasoning process and final answer accuracy” is relatively trivial and expectable. This simply follows the fact that correct reasoning paths naturally tend to include more correct entities.

2.	While the idea of entity-aware rewards is useful, the contribution seems an incremental improvement on GRPO rather than a fundamental paradigm shift. The core update is scaling reward by entity match rate, which is conceptually straightforward.

3.	In the introduction, it argues that PRM methods cannot handle open-ended, dynamic nature due to the need for step-wise annotation. However, the E-GRPO also needs the ground-truth entities for the complete reasoning path to compute the partial reward. This seems to suggest that E-GPRO also cannot handle the tasks that PRM cannot. The utility needs justification.

4.	The entity annotations in synthetic datasets are assumed correct and reliable. If entity extraction or labeling is noisy, the reward signal may mislead training. The paper could better address this potential sensitivity to data quality and generalization to non-synthetic data.

5.	The paper does not sufficiently address how well E-GRPO generalizes to real-world search questions that differ substantially from the synthetic data distribution.


Minor:

1.	In the abstract, “We address this by leveraging the very entities discarded during training.”, is the word “very” a typo?

2.	The related work section should be a part of the main content to make it self-contained, instead of appearing only in the appendix.

### Questions
1.	The notion of “ground-truth entities” in reasoning requires clarification. Are there cases where multiple reasoning paths are valid, and if so, how does the method define or handle ground-truth entities in those scenarios?

2.	How well does E-GRPO agent generalize to other **question** domains or verticals that differ from the synthetic training data?

### Soundness
2

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
4

### Summary
This paper introduces E-GRPO that improves the performance of LLM-based search agents. E-GRPO extends the vanilla GRPO with a denser, entity-aware reward function. It reuses the ground-truth entities from the synthetic training data to assign partial credit to incorrect answers based on how many key entities they successfully identified. Experiments show this approach allows the model to learn more effectively, consistently outperforming the GRPO baseline in accuracy and learning more efficient policies that use fewer tool calls.

### Strengths
- The reward formulation originated from a good observation in the data and its design is clear and reasonable.
- The empirical performance appears to be strong and better than vanilla SFT & GRPO.

### Weaknesses
- The research contribution appears to be incremental. While E-GRPO seems to outperform GRPO, I am not fully convinced E-GRPO is a fundamentally “novel framework” compared to the original GRPO given the fact that it only customizes the reward function. Partial entity matching in RL is also an established method, especially in NL2SQL field (e.g. Reasoning-SQL).
- It is nuanced whether the performance comparison with other baseline models in the main results is a fair comparison using same training data.

### Questions
- It appears that the training datasets for E-GRPO and other baseline models are different because the performance scores are directly cited from those external publications. Can authors clarify how much performance improvement comes from discrepancy in training data?
- While uniform credit assignment is a common practice in many GRPO-like RL training algorithms, is it technically sound and mathematically stable in E-GRPO when the partial entity matching reward is also uniformly assigned to the tokens, regardless of whether that specific token's "turn" was the one that found an entity or the one that made a mistake?
- What is the accuracy trend of GRPO and E-GRPO after 80 steps in fig. 3? Is it possible that the training happened to stop at the point where the difference is large, given the large fluctuation in both training curves?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the reward sparsity problem in training LLM-based search agents with reinforcement learning. Current methods totally assign a zero reward even if the trajectory contains partially correct information. The authors use an entity-centric method to synthesize a dataset and save the entities as ground-truth information. Then they add an extra entity-matching reward to improve performance.

Contributions: 
* Identifying the "near-miss" problem: The paper shows that standard GRPO fails to distinguish informative partially correct samples from complete failures in search-agent training.
* Entity-aware reward function: A novel reward formulation that assigns partial credit based on entity match rate

### Strengths
* **Well-motivated problem**. Although partial correctness has been studied in RL, introducing it in training search agents is a good problem to facilitate this community.
* **Technical soundness**. The solution using the entity-matching score is reasonable, and the analysis of the relationship between accuracy and the matching score supports the insights of the proposed method.

### Weaknesses
1. **Handling of incorrect reasoning with correct entities**. The entity-matching reward may inadvertently credit erroneous reasoning paths that happen to mention correct entities without proper understanding. For instance, a model might generate factually incorrect statements while coincidentally including the right entity names. The paper does not address how to distinguish between genuine entity identification and spurious mentions.
2. **Overclaim**. While the entity-aware reward is effective, the core contribution is to augment GRPO with an additional reward term. The branding as "E-GRPO" may overstate the methodological advance, as the fundamental GRPO framework remains unchanged. A more modest framing as a reward technique might be more appropriate.
3. **Insufficient baseline comparisons in controlled settings**. The Local environment experiments (Table 1) would benefit from comparisons with more approaches that are feasible in controlled settings.
4. **Unexplained performance degradation**. The results show E-GRPO slightly underperforms on PopQA. The paper does not analyze or explain this anomaly.
5. **Absence of systematic error taxonomy and failure analysis**. Given the lengthy reasoning traces, the paper would benefit from categorizing failure modes (e.g., entity identification errors, reasoning chain breaks, etc.). Besides, the paper does not provide any concrete failure cases of the E-GRPO. While the case study in Appendix E shows a successful E-GRPO trajectory compared to a failed GRPO one, there is no analysis of cases in which E-GRPO fails despite its entity-aware reward.
6. **Entity matching robustness concerns**. While the authors justify exact string matching, they do not address practical challenges such as spelling variations, abbreviations, etc. These issues could significantly degrade the quality of the reward signal in real-world applications.

### Questions
1. **Figure 1 (upper-right) requires clarification**. I cannot fully understand the upper-right panel of Figure 1, especially the meaning of the x-axis labels. I would like to request more details and straightforward explanations of it.
2. **Figure 1 (lower-right) distribution analysis**. In the lower-right panel, the distribution shows cases where the entity match rates are not 1 but still lead to correct answers. Are these instances of hallucination where the model just happened to get it right?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an improvement to search agent RL algorithms such as Group Relative Policy Optimization (GRPO), which treat so-called near-miss rollouts as bad as complete miss rollouts. The authors propose entity-awareness for the reward function, which assigns partial credit to rollouts, leading to a denser reward function.

The approach is evaluated for QA datasets such as TriviaQA or PopQA, and Deep Research datasets such as GAIA or BrowseComp. A local (Wikipedia) and web setting (Google Search, Jina) are evaluated. Baselines are GRPO as well as ReAct-based agent variants.

The results show that the approach improves the performance of the search agents compared to GRPO, and is also on par or superior when compared to other agents.

### Strengths
- Well-motivated improvement: Model intermediate rewards for partially correct rollouts
- Improvement over GRPO visible in empirical evaluation
- Stabilization of training process increases efficiency

### Weaknesses
- Empirical gain rather low compared to GRPO

### Questions
- Did you experiment with decreasing alpha during training?
- Can this approach be transfered to other domains with less entities?

### Soundness
4

### Presentation
3

### Contribution
2
