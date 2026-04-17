# PAWS: Preference Learning with Advantage-Weighted Segments

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Training agents that align with human intentions is a central challenge in machine learning. Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm by leveraging human feedback in the form of trajectory-level comparisons, thereby avoiding the need for explicit reward design or expert demonstrations. However, existing PbRL methods typically rely on per-step reward assignments inferred from trajectory preferences, which introduces inconsistencies and exacerbates the temporal credit assignment problem. In this work, we analyze this issue and demonstrate its adverse impact on policy learning. To address this problem, we propose Preference Learning with Advantage-weighted Segments (PAWS), a novel segment-based preference learning method that updates policies directly with segment-level advantage functions. By preserving segment-level preference information, PAWS ensures stable policy updates while avoiding misleading per-step reward signals. Empirical evaluations on a diverse set of simulated robot manipulation tasks, as well as locomotion tasks, show that PAWS achieves higher task-specific performance over existing PbRL approaches, highlighting the effectiveness of our method in aligning policies with human preferences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to use learning an advantage function from preference data to address the credit assignment problem faced when learning a reward model from preference data. The proposed solution is a method named PAWS that aims to preserve segment-level preference information instead of inferring values for single state-action pairs when provided segment level preference information.  The method is applied to the offline data case. The advantage function is learned using the Bradley-Terry objective, and the policy using a combination of standard approaches to learning from offline data. Results are presented for ten tasks from MetaWorld for two different amounts of preference data (50 vs. 500 samples) and for two different advantage function architectures (transformer vs. MLP). Across tasks there is no clear winner between the transformer and MLP architecture. When using 50 preference samples, PAWS performs best on 6/10 tasks and on 8/10 tasks when using 500 samples.

### Strengths
- The paper is well written and easy to follow.
- The paper aims to address an important problem in PbRL, which is accounting for the credit assignment problem when learning from preference feedback.
- Multiple baselines are compared against, all of which operate in the same offline setting as PAWS. 
- The results demonstrate that PAWS improves performance relative to baselines on 6/10 MetaWorld tasks give 50 preference samples and 8/10 tasks given 500 samples. When PAWS performs best, the gains are not small suggesting the approach improves performance more often than not.

### Weaknesses
- No mention of any prior work in PbRL specifically aimed at addressing the credit assignment problem. Please update the related work section to mention prior work has attempted to tackle this problem. Some papers focusing on the credit assignment problem in reward learning include:
     - Zou, Haosheng, et al. "Reward shaping via meta-learning." arXiv preprint arXiv:1901.09330 (2019).
     - Novoseller, Ellen, et al. "Dueling posterior sampling for preference-based reinforcement learning." Conference on Uncertainty in Artificial Intelligence. PMLR, 2020.
     - Gangwani, Tanmay, Yuan Zhou, and Jian Peng. "Learning guidance rewards with trajectory-space smoothing." Advances in neural information processing systems 33 (2020): 822-832.
     - Verma, Mudit, and Katherine Metcalf. "Hindsight PRIORs for Reward Learning from Human Preferences." The Twelfth International Conference on Learning Representations (2024).
- The exact connection between the advantage function and the credit assignment problem needs to be better formulated and motivated in the paper. For example, why/how does advantage address the credit assignment problem?
- The approach to learning the advantage function in Section 3.1 is exactly the approach and method used to learn reward functions. It is not clearly spelled out how the resulting function is an advantage function, and not a reward function used like an advantage function. This calls into question the validity of applying the preference learned model as an advantage. Therefore, how exactly the model learned from the BT objective is an advantage function instead of a reward function needs to be addressed.
- The paper uses many existing optimizations and approaches, but applied to the problem of PbRL. Without a clear motivation for how advantage specifically addresses the credit assignment problem, the contributions of the paper are minimal.
- Results are only presented on MetaWorld. It is common for PbRL papers to present results on both DMC tasks and MetaWorld to demonstrate the algorithm works on goal and non-goal-based tasks. Therefore, to be inline with prior work, results should be presented on non-goal-based tasks.
- The paper does not present results suggesting that the credit assignment problem specifically is addressed. Please include experiments highlighting 
- The use of BC and P-IQL as baselines for PAWS is not clear as neither method relies on learning from preference data. Please motivate why these are appropriate baselines.
- Small issues:
     - In Table 2 both PAWS(MLP) and PAWS(Transformer) should have their 500 results bolded. For 50 preferences, is the differences of 51 versus 52 meaningful?

### Questions
- For Figure 3 (b) and (c), why are different tasks shown for the 500 versus 50 preference samples cases? Why aggregate over 3 tasks for 500 samples and only present results for one task for 50 samples?

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
This paper analyzes how per-step reward inference in preference-based reinforcement learning (PbRL) leads to inconsistencies and instability because of the temporal credit assignment problem. To address this, it proposes PAWS (Preference Learning with Advantage-weighted Segments), an offline preference learning framework that learns a segment-level advantage function and performs segment-wise trust-region constrained policy optimization with adaptive scaling for stable updates within the data distribution. Experiments on robot manipulation tasks demonstrate that PAWS effectively mitigates credit assignment errors and outperforms existing PbRL methods.

### Strengths
1. The paper provides excellent visualization (Figures 1 and 2) of the temporal credit assignment problem in preference learning. The empirical demonstration that trajectory-level advantages are well-predicted while per-step advantages are inconsistent effectively motivates the segment-based approach. 
2. The paper derives the optimal policy through Lagrangian optimization (Proposition 1) and shows that the gradient does not depend on environment dynamics (Proposition 2), providing solid theoretical foundation.

### Weaknesses
1. The temporal credit assignment problem in PbRL has been extensively explored in recent works that explicitly model temporal dependencies through transformer-based reward estimator or world models [1,2,3]. While [2] was proposed for online PbRL, the reward learning component could be adapted to offline settings for comparison. The authors should more clearly articulate the fundamental difference between their segment-based policy update approach and these existing methods that address TCA.
2. Several recent and relevant methods are missing, such as IPL [4], DPPO [5], LiRE [6], and APPO [7]. At minimum, comparison with IPL and DPPO (both avoiding reward modeling) would strengthen the empirical contribution.
3. The paper generates preference labels by comparing the log probabilities under the best policy, claiming it “models the expert nature of the labeler”. This design choice is neither deterministic (as in reward-based oracle preferences for fair comparison) nor genuinely human (as in real-world situation), and no prior work is cited that justifies this design choice. The log probability approach is unconventional and its validity is not established.
4. All experiments use purely synthetic proxy labels. As highlighted by the Preference Transformer [1], synthetic preferences differ substantially from human feedback, which tends to be subjective, noisy, and context-dependent.

References

[1] Kim, C., Park, J., Shin, J., Lee, H., Abbeel, P., & Lee, K. Preference Transformer: Modeling Human Preferences using Transformers for RL. In The Eleventh International Conference on Learning Representations.

[2] Verma, M., & Metcalf, K. Hindsight PRIORs for Reward Learning from Human Preferences. In The Twelfth International Conference on Learning Representations.

[3] Gao, C. X., Fang, S., Xiao, C., Yu, Y., & Zhang, Z. (2024). Hindsight preference learning for offline preference-based reinforcement learning. arXiv preprint arXiv:2407.04451.

[4] Hejna, J., & Sadigh, D. (2023). Inverse preference learning: Preference-based rl without a reward function. Advances in Neural Information Processing Systems, 36, 18806-18827.

[5] An, G., Lee, J., Zuo, X., Kosaka, N., Kim, K. M., & Song, H. O. (2023). Direct preference-based policy optimization without reward modeling. Advances in Neural Information Processing Systems, 36, 70247-70266.

[6] Choi, H., Jung, S., Ahn, H., & Moon, T. (2024, July). Listwise Reward Estimation for Offline Preference-based Reinforcement Learning. In International Conference on Machine Learning (pp. 8651-8671). PMLR

[7] Kang, H., & Oh, M. H. Adversarial Policy Optimization for Offline Preference-based Reinforcement Learning. In The Thirteenth International Conference on Learning Representations.

[8] Lee, K., Smith, L., Dragan, A., & Abbeel, P. B-Pref: Benchmarking Preference-Based Reinforcement Learning. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

### Questions
1. The proposed objective appears quite similar to CPL, which also builds on the regret-based preference model, and the authors use it as a comparison method. Could you explicitly clarify the fundamental algorithmic difference and corresponding advantage?
2. Could you provide theoretical or empirical justification for using log probabilities of the best policy as preference labels? How does it compare to stochastic teacher [8] or using the best policy’s Q-values?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PAWS, a novel approach to preference-based reinforcement learning (PbRL) designed to address the temporal credit assignment problem. PAWS first learns a segment-level advantage function and then updates the policy through advantage-weighted segment optimization. Extensive experiments on diverse simulated tasks demonstrate the effectiveness of the proposed method compared to existing baselines.

### Strengths
1. The paper is well-structured, with a clear and meaningful motivation that highlights the importance of addressing the temporal credit assignment problem.

2. The ablation studies are insightful, providing a deeper understanding of the proposed method.

3. The code is provided, which is very appreciated.

### Weaknesses
1. The innovation is relatively incremental, as the two main components, i.e., the advantage learning formulation and the Lagrangian approach for constrained policy optimization, have already been proposed in prior work.

2. It remains unclear whether the proposed method can truly address the temporal credit assignment problem as claimed by authors. First, the paper provides no theoretical analysis to support this claim. The method appears to learn a state-wise advantage function, aggregate it into a trajectory-level advantage, and then optimize the policy through a constrained procedure. However, it is not evident why this formulation effectively resolves the temporal credit assignment issue. Second, the experimental section lacks dedicated analyses or ablation studies specifically designed to verify this aspect.

3. The paper lacks a formal definition of the advantage function, which makes this part difficult to follow for readers unfamiliar with the concept. It is recommended to explicitly define or describe the advantage function around Lines 162-163.

4. The set of baseline methods in the experiments is relatively limited. It is recommended to include additional representative baselines such as P-CQL [1] and Preference Transformer [2] to enable a more comprehensive and fair comparison.

[1] Kumar A, Zhou A, Tucker G, et al. Conservative q-learning for offline reinforcement learning[J]. Advances in neural information processing systems, 2020, 33: 1179-1191.

[2] Kim C, Park J, Shin J, et al. Preference transformer: Modeling human preferences using transformers for rl[J]. arXiv preprint arXiv:2303.00957, 2023.

### Questions
1. In Eq. (4), the definition of $\pi(\tau)$ is unclear. Why is the policy defined over entire trajectories rather than individual state-action pairs?

2. In Table 1, it is not clear why the Transformer-based version of PAWS outperforms the MLP-based version when the number of preferences is small, yet the trend reverses when more preferences are available. Please provide an analysis or discussion to explain this behavior.

### Soundness
2

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
This paper presents PAWS: an offline preference learning algorithm that avoids the credit assignment problem by training an advantage network explicitly on trajectory, and subsequently deriving a policy through a combination of Lagrangian optimisation, monte-carlo estimation and stochastic gradient descent.

PAWS is evaluated against 10 metaworld tasks where it obtains better success rates (and cumulative returns) than recent offline preference learning baselines (P-IQL, CPL, and CPL+KL).

### Strengths
* PAWS proposes an original solution to an important problem in the preference learning domain: the credit assignment problem.
* The paper features a robust mathematical derivation of the advantage function and the corresponding policy (though see questions below).
* The evaluation is carried out against many recent baselines and presents strong results (+7.6% success rate across the 10 metaworld tasks with respect to the next best baseline: P-IQL).
* Many of the key algorithmic decisions are ablated (including neural network architecture, number of preferences, segment length, number of effective samples, etc).

### Weaknesses
* **W1**: The manuscript can be at times hard to read, particularly section 3, and especially if –like me– you don't have a background in optimisation. Questions **Q1**–**Q8** below are intended to remedy this weakness (which is the main issue bringing the rating down).
* **W2**: Similarly some aspects of the evaluation are confusing, making a fair assessment of the manuscript hard. See questions **Q9**–**Q16** for  more details.
* **W3**: Though very thorough, the current experiments are limited to MetaWorld. It would be interesting to see how PAWS behaves in other tasks (for instance from D4RL [1])

_Overall_, this is a strong submission solving an interesting problem in a mathematically robust way, but that is currently held back by missing details in both the mathematical derivation, and the training and evaluation regimes.

--------

[1] Fu et al. (2020) "D4RL: Datasets for Deep Data-Driven Reinforcement Learning" ArXiV preprint.

### Questions
* **Q1**: In sec 3.1 I would point out the differences between advantage learning and classical PbRL under Bradley-Terry. As far as I can tell, the main difference is that the advantage network is accumulated over trajectories rather than consuming individual state-action pairs. However the maximum likelihood estimation of the loss seems to be equivalent for BT and for advantage learning.
* **Q2**: The order of section 3 does not follow the order in which PAWS is applied (first train A, then iterate fitting $\lambda$, and finally train $\pi_\theta$). The order makes sense to introduce the different elements of PAWS, but I would still explicitly state how PAWS operates, perhaps adding a pseudo-code listing.
* **Q3**: In sec 3.2 $\pi$ is defined as a mapping: $\mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^+$, but in equation 4 is is shown operating on trajectories $\tau$. It would be good to clarify earlier that this is a simplification of the notation referring to: $\\pi_\\theta(\\tau) = p(s_0)\\prod_{t=0}^T p(s_{t+1}|s_t, a_t) \\pi_\\theta(a_t|s_t)$.
* **Q4**: Could you expand on why "it is not straightforward to generate samples for new states for inference" from the optimal policy $\\pi^\\star$? (Line 275) This needs to be better explained as the use of a parametric policy is a key component of PAWS.
* **Q5**: I would better explain how we arrive to eq (6). As I understand it, eq (6) is obtained by plugging eq (14) into $\\mathcal{L}(\\theta) = \\mathbb{E} \\left[ \\log \\pi_\\theta (\\tau) \\right]$, and then ignoring the constant term $\\exp(\\frac{-\\beta - \\lambda}{\\lambda})$ which we can do since $\\mathcal{L}$ is optimised via SGD, and gradients are not affected by constants.
* **Q6**: How many iterations of setting $\\epsilon$, minimising eq. (8) and computing  $n_{eff}$ were necessary for your experiments?
* **Q7**: Why is $n_{eff} = \\frac{(\sum_i \omega_i)^2}{\sum_i \omega_i^2}$ a measure of the support between $\\pi^\\star$ and  $p_D$?
* **Q8**: What value of the discount factor, $\\gamma$, did you use for the experiments? Did you ablate it? Does it have any significant effect on performance?
* **Q9**: Why was it necessary to remove the proprioceptive history from the preference datasets?
* **Q10**: How are the samples from the four policies mixed in the preference dataset? Uniformly? Are the preferred and non-preferred trajectory pairs always from different policies?
* **Q11**: Why do you need an evaluation dataset? Is it not enough to rollout the learnt policy $\\pi_\\theta$? How big is the evaluation dataset?
* **Q12**: Why use log-probs of the best policies to create the preferences? Why not use the underlying reward function that you used to train SAC instead?
* **Q13**: I would suggest adding SAC as an upper-bound to Table 1.
* **Q14**: For the first ablation in sec 4.1 ($n_{eff}$): did you use the transformer or MLP backbone?
* **Q15**: For the second ablation in sec 4.1 (segment vs states): what was the training setup? I understood it as the same advantage function (trained on segments), but with a policy trained only on state-actions? If so, how did you train the policy?
* **Q16**: How many parameters do the transformer and ML advantage networks have? What is the architecture of the actor network? MLPs?
* **Q17**: How was Fig.1 computed? I could not find any details about it in the manuscript.

-------

### Nitpicks (do not affect rating, no need to follow up during rebuttal).

* **N1**: In sec 2 (related work), line 154: please provide examples of other feedback types.
* **N2**: In sec 3 (notation): use $\\mathbb{R}$ instead of $\\mathcal{R}$ to refer to the set of real numbers.
* **N3**: In sec 3.2, line 255 (and again in line 258): I suggest using "benefits" instead of "advantages". "Advantages" may be confused with "advantage networks".
* **N4**: In line 336: refer to eq (8) instead of eq (33). They are the same equation but eq (8) is closer.
* **N5**: In fig. 3 (b)–(c): can you add a horizontal line with the performance of BC? 
* **N6**: In the appendix, line 735 the text should be "inserting the optimal solution in eq (14) instead of eq (13).

### Soundness
3

### Presentation
1

### Contribution
2
