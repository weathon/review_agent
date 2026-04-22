# Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
A prevailing view in Reinforcement Learning with Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named ER Velocity and ER Acceleration, to capture exploitation dynamics. Our analysis reveals that in the semantic space, exploration and exploitation could be decoupled (Sec.~4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4\% absolute accuracy improvement on the challenging Gaokao 2024 dataset. The code is available at https://anonymous.4open.science/r/coding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper mentions that the trade-off of exploration and exploitation is not an intrinsic issue, it may exist in the context of entropy/confidence situation, but may not exist in other spaces. In particular, the authors argue that in the hidden-state space, this trade off does not exist. The authors use effective rank (ER) to measure exploration and effective rank velocity (ERV) to meaure exploitation, and provide empirical evidence that these two are not negatively correlated. This motivates the authors to shape the advantage function to encourage both high ER and high ERV, i.e., encourage both exploration and exploitation. In essence, the core claim in this paper is that the exploration and exploitation are not a trade off but can be encouraged simultaneously.

### Strengths
The idea of studying ER and ERV in the hidden state space is interesting, and encouraging both ER and ERV leads to a better performance. The paper provides theoretical justification of the proposed approach.

### Weaknesses
The major issue of this paper is the misuse of "exploitation". In standard RL [1], exploitation means greedy action, i.e., always choose the action that has the highest (current) Q-value where the Q-value is approximated either by Monte-carlo sampling or a critic model.  Exploration means that you choose some other actions that may not have the highest Q-value. In RL, the trade-off between exploration and exploitation always exists because you either choose the greedy action or non-greedy action.

In this paper, the authors find that there is no trade-off between ER and ERV. However, this does not mean that there is no exploration and exploitation trade off. Specifically, ERV measures the change in ER between successive hidden states. I do not see any relation between ERV and greedy action. That says, I highly doubt that ERV can represent exploitation.

In other words, I think this paper is not studying the exploration-exploitation trade off, but studying something else, i.e., ER-ERV relation. I do not see the reason of mentioning exploration-exploitation as a foundation in this paper as you are studying something else.

Another weakness is that the code is not provided. In the anonymous link, there is only a readme saying "hahahah". I do not see the intention of the authors to include such a "code".



[1] Reinforcement learning: An introduction

### Questions
Can the authors elaborate the relation between ERV and greedy action, or explain why ERV can be "exploitation"?

### Soundness
1

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
This paper introduces VERL, a novel method for RLVR in LLM reasoning tasks. Unlike traditional RL, which assumes an inherent token-level exploration–exploitation trade-off, VERL shows that this trade-off may be an artifact of token-level metrics. The method quantifies exploration via ER of hidden states and exploitation via its first- and second-order temporal changes, i.e., ERV and ERA, which are nearly uncorrelated, allowing exploration and exploitation to be decoupled. VERL incorporates these metrics into a dynamic, dual-channel advantage function, with ERA modulating the balance to prevent overconfidence and improve reasoning efficiency. Experiments on diverse benchmarks, i.e., ASDiv and OlympiadBench, demonstrate consistent improvements, including up to 21.4% absolute accuracy gain on Gaokao 2024, and show strong generalization across RL algorithms and models. Both exploration and exploitation benefits are confirmed via Pass@k and Pass@1 metrics, and ablation studies indicate robustness to key hyperparameters.

### Strengths
1. Novelty: this is the first work to analyze exploration–exploitation dynamics in the hidden-state space. It introduces the ER, ERV, and ERA metrics, overcoming the limitations of token-level trade-offs.


2. Strong theoretical and empirical support: the paper provides theoretical analysis of ER, ERV, and ERA scalability, and demonstrates consistent performance improvements across multiple models, tasks, and RL algorithms.


3. General and robust: VERL is a plug-and-play method compatible with various RL frameworks, maintaining stable performance across key hyperparameters.


4. Extensive evaluation: the method has been thoroughly evaluated on diverse reasoning benchmarks and multi-step reasoning tasks, including detailed ablation studies and case analyses.

### Weaknesses
1. Concern about computational cost: ER, ERV, and ERA rely on SVD of hidden states, with a computational complexity of $O(TD^2)$. Since these metrics must be computed frequently during training, VERL could become computationally expensive for long sequences or large models. Although the appendix discusses efficiency considerations, no practical runtime results are reported, suggesting that computational cost may still pose a bottleneck for real-world, efficient implementation.

2. Theoretical generalization: the exploration–exploitation decoupling is primarily observed empirically and lacks a theoretical explanation. As a result, its applicability to other models, tasks, or datasets remains unverified?

### Questions
Q1. While VERL introduces novel hidden-state metrics to enhance exploration, the paper does not compare its performance against existing exploration strategies, such as entropy regularization methods. As a result, the relative advantage of VERL remains unclear.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the exploration–exploitation tradeoff in RLVR, focusing on representations (i.e., the *hidden state space*) of LLMs. The authors propose using Effective Rank (ER)—the entropy of the singular value spectrum of hidden representations—as an exploration metric, and the velocity of effective rank (ERV), its first-order difference, as an exploitation metric. They empirically observe that, surprisingly, both ER and ERV can improve simultaneously for certain base models under RLVR fine-tuning. Based on these findings, they introduce VERL, a new RLVR approach that augments the advantage function with ER- and ERV-based auxiliary terms. Experiments on math reasoning tasks show consistent gains for Llama-3.2-3B and Qwen2.5-7B models when VERL is combined with GRPO and PPO.

### Strengths
- The examination of exploration in the hidden state space rather than in the action space is a fresh perspective and adds conceptual depth to RLVR research.  
- The use of effective rank, borrowed from signal processing, offers an interpretable way to quantify diversity in model representations.
- The empirical results on math domains demonstrate that VERL can yield nontrivial performance gains, particularly on challenging out-of-domain tasks.

### Weaknesses
1. **Questionable definition of the exploitation metric (ERV):** (top concern)

The interpretation of ERV as “exploitation” lacks grounding in classic RL. Conventionally, exploitation involves selecting actions that maximize estimated value (i.e., $\arg\max_a \hat{Q}(s,a)$), while exploration refers to deviating from that policy. Without an explicit value function, it is unclear how ERV captures exploitation behavior. Moreover, defining ERV as deviation from a historical average (rather than consecutive-step difference) causes it to converge over time, which weakens its interpretation as a “velocity” metric. 

If ERV is not conceptually tied to exploitation, the finding that exploration and exploitation can co-improve becomes unsurprising. The paper’s framing around “beyond the exploration–exploitation tradeoff” could be overstated given an incorrect measurement of exploitation. Clarifying the relationship between ER/ERV and traditional exploration–exploitation trade-off would strengthen the contribution and avoid conceptual misuse in the field.

2. **Lack of justification for focusing on the final hidden layer:**  
   As acknowledged by (Skean et al., 2025) and cited in this work, mid-layer representations often encode richer features. The decision to use only the final layer should be justified or complemented by additional analysis on intermediate layers.

3. **Insufficient clarity in VERL formulation:**  
   The proposed VERL objective is spread across multiple subsections. Presenting the full training objective in consolidated equations would improve clarity and allow readers to better assess how ER, ERV, ERA interact with the base RL loss.

4. **Mixed empirical outcomes not sufficiently discussed:**  
   While VERL shows large improvements on out-of-domain sets (e.g., +17.9 on CMATH and +21.4 on Gaokao2024_I), it also produces performance drops (−7.1, −1.0) for a different base model. These drop should be acknowledged and analyzed in the main text to present a balanced view.

### Questions
Consider citing *“Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning”* (https://arxiv.org/abs/2010.14498), which proposes a different "effective rank" and connects it to representation diversity. How do you view the relationship between exploration and representational diversity in this context?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes reinterpreting the exploration–exploitation relationship in the reasoning of large language models (LLMs) within reinforcement learning by looking at the hidden state space. The authors use the hidden-layer Effective Rank (ER) and its first- and second-order derivatives (ERV, ERA) to measure the diversity of the semantic space and the speed of information gain. They find that exploration and exploitation can be decoupled at the hidden-state level, and thus propose a new method called Velocity-Exploiting Rank-Learning (VERL). VERL uses ERA to dynamically shape the advantage function, improving reasoning performance across different RL algorithms (GRPO/PPO). Experiments show that on several mathematical reasoning benchmarks (such as Gaokao 2024, AIME), VERL achieves improvements relative to GRPO and PPO.

### Strengths
The paper has a complete structure and is generally well-written.

### Weaknesses
1.	The main innovation of the paper is described as recharacterizing the exploration-exploitation trade-off through information sources in the hidden state space and proposing a new method, VERL, to improve the performance of different RL algorithms and large models. However, the paper claims to surpass traditional balance, but in fact, it only provides a more fine-grained way to adjust the ratio of exploration to exploitation, not a real reconstruction of the trade-off. A rename is suggested.
2.	The paper uses Effective Rank (ER) to quantify the semantic diversity of hidden states and introduces ER-derived quantities ERV and ERA to capture information-gain speed and acceleration, extending the notion of exploration metrics in RL. However, the core assumption that the singular value decomposition (SVD) of hidden states reflects exploration behavior lacks rigorous mathematical support; SVD reflects changes in the representation space’s dimensions, not necessarily exploration.
3.	VERL employs several hand-designed indicators, dynamic weights, and sigmoid-regulated adaptive parameters. While these work in experiments, their causal relationships are not clear, and it is not obvious which parts contribute what. Could the method be simplified? This coupling leads to weaker interpretability, so adding theoretical analysis is recommended.
4.	For long sequences or large models, performing frequent SVDs and multi-order statistics can be costly. The paper only mentions time cost in general and lacks detailed cost/efficiency comparisons, which affects the method’s usability.
5.	The main tables and curves report only single values or average curves, without multiple independent runs, means, standard deviations, or significance tests. This makes it hard to assess robustness and reliability. Significance testing data should be added.
6.	In ablation analysis, the separate contribution of internal VERL indicators such as ERV and ERA is insufficient, lacking experiments where removing one indicator (e.g., ERA) changes performance for comparison.
7.	In Fig. 5’s three groups of experiments (a), (b), (c), the pink curves (GRPO + ours (Full Version)) nearly coincide across all three subfigures; the data points, slopes, and inflection points align. Statistically, this is unlikely because different hyperparameters should cause slight fluctuations. The main text only states that performance is not sensitive to hyperparameters and does not explain why the three curves are almost identical. Please clarify: a)	Do the three curves come from the same experimental data reuse?  b)	If they are independent experiments, were seeds fixed, or were results averaged over multiple runs? c)	If they are example curves, should this be noted in the caption? If not clarified, this may affect reproducibility and credibility.

### Questions
Please refer to the “Weakness” section for related questions.

### Soundness
2

### Presentation
2

### Contribution
2
