# Repairing Reward Functions with Human Feedback to Mitigate Reward Hacking

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
Human-designed reward functions for reinforcement learning (RL) agents are frequently misaligned with the humans' true, unobservable objectives, and thus act only as proxies. Optimizing for a misspecified proxy reward function often induces reward hacking, resulting in a policy misaligned with the human's true objectives. An alternative is to perform RL from human feedback, which involves learning a reward function from scratch by collecting human preferences over pairs of trajectories. However, building such datasets is costly. To address the limitations of both approaches,  we propose Preference-Based Reward Repair (PBRR): an automated iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from preferences. A manually specified reward function can yield policies that are highly suboptimal under the ground-truth objective, yet corrections on only a few transitions may suffice to recover optimal performance. To identify and correct for those transitions, PBRR uses a targeted exploration strategy and a new preference-learning objective. We prove in tabular domains PBRR has a cumulative regret that matches, up to constants, that of prior preference-based RL methods. In addition, on a suite of reward-hacking benchmarks, PBRR consistently outperforms baselines that learn a reward function from scratch from preferences or modify the proxy reward function using other approaches, requiring substantially fewer preferences to learn high performing policies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method, PBRR, that adjusts a human-provided proxy reward function by learning an additive correction term. This additive term is learned from human preferences using a novel preference learning objective (not the typical cross entropy one). The authors demonstrate in reward hacking benchmarks that PBRR can perform similarly and might outperform strict preference learning methods or other methods that modify a proxy reward.

### Strengths
The paper tackles a relevant and important problem: improving human-provided proxy reward functions to make them better aligned with true human objectives. Reward design is very difficult; therefore, developing methods to improve human designed reward functions is useful. 

The proposed loss function is conceptually simple (that's a good thing!) and the decomposition into three loss terms provides interpretability about how the correction is applied.

The authors provide both theoretical and empirical support to their approach.

### Weaknesses
**Assumptions about human-provided rewards**:
The paper assumes that humans can readily specify proxy reward functions that reflect the ground-truth objective but lack robustness. However, existing empirical evidence (e.g., Booth et al., 2023; Muslimani et al., 2025) suggests that humans often provide misaligned or poorly specified reward functions. The assumption that a proxy reward is “close to optimal” appears unjustified.

**Strong modeling assumptions**:
The method assumes that the ground-truth reward can be expressed as the sum of the proxy reward and a correction term. I'm not sure how realistic this assumption is, especially in high-dimensional or partially observable tasks where misspecification can be complex and non-additive.

**Loss function**:
The paper introduces three loss terms, but the behavior of the second and third terms is unclear.

The second loss term, $L_{+}$, is intended to regularize the correction term $g$ towards zero on trajectory pairs where the proxy reward function agrees with the human preference. However, from the definition provided, this loss would also be minimized when
$g(\tau_1) = -g(\tau_2)$,
which could alter the overall reward function and potentially invert or misclassify previously correct preferences.

The third loss term, $L_{-}$, explicitly prioritizes decreasing the reward for undesirable behaviors rather than increasing the reward for desirable behaviors. The motivation for this asymmetric treatment is unclear. The authors should discuss what effect this design choice has on the learned correction $g$ and the resulting policy behavior.

**Assumptions about the reference policy**:
The method assumes access to a “good enough” reference policy that can distinguish failures of the proxy reward-induced policy. This also seems like a strong assumption. This means that the reference policy can’t make the same mistakes that the proxy induced policy makes. Moreover, in the regret analysis, the authors further assume that the reference policy lies in the set of possibly optimal policies induced by the ground truth reward function. Lastly, the appendix mentions that some reference policies were obtained from “only a handful” of demonstrations. It is unclear how many demonstrations this entails.

**Experimental design and baselines:**
- The choice of reward learning baselines is limited. The paper omits more recent or preference-efficient algorithms such as PEBBLE (Lee et al., 2021) or recent RLHF approaches, which would provide stronger comparisons.The RLHF baseline provided in the text (Christiano et al. (2017)) is notoriously preference inefficient. 

- The comparison between PBRR and Online-RLHF seems unfair: PBRR benefits from a hand-designed proxy reward (which itself requires human effort), while Online-RLHF learns from scratch, so it makes sense that it would require more human preferences. 

- The normalization strategy for the figures seems unconventional and lacks justification.
- Statistical significance is unclear: results are based on only three seeds, and shaded regions represent standard error rather than 95% confidence intervals. Therefore, claims that PBRR “matches or exceeds” all baselines are not statistically supported.

**References**:

Booth et al (2023): https://scholar.google.com/citations?view_op=view_citation&hl=en&user=sf3ROEUAAAAJ&citation_for_view=sf3ROEUAAAAJ:KlAtU1dfN6UC

Muslimani et al (2025): https://arxiv.org/abs/2503.05996

Lee et al (2021): https://arxiv.org/pdf/2106.05091

Christiano et al. (2017): https://arxiv.org/abs/1706.03741

### Questions
- What empirical evidence supports the assumption that humans can easily specify a proxy reward that reflects the true objective (albeit imperfectly)?
- Can the authors clarify how the $L_{+}$ loss avoids pathological cases where $g(\tau_1) = -g(\tau_2)$?
- Why did the authors decide to focus on correcting undesirable transitions  (that have a high reward) rather than correcting desirable ones (that may incorrectly have a low reward)?
- In the online RLHF baseline, the authors note that they use the reference policy for initial exploration. What does that mean? Do the authors initialize the starting policy with the reference policy?
- Could the authors justify the normalization strategy used in their plots?
- Are the reported improvements statistically significant?
- Why were stronger baselines such as PEBBLE not included?

### Soundness
2

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
This paper proposes Preference-Based Reward Repair (PBRR), a method to mitigate reward hacking in reinforcement learning by iteratively correcting a human-specified proxy reward function instead of learning a reward model from scratch. PBRR learns an additive correction term over transitions based on human trajectory-pair preferences and uses a targeted exploration strategy comparing the learned policy and a reference policy. The method introduces a tailored objective that focuses corrections on transitions where the proxy reward conflicts with preferences. The authors provide theoretical cumulative regret bounds in tabular MDPs comparable to uncertainty-based RLHF methods and empirically evaluate PBRR on reward-hacking benchmarks. Results show improved preference efficiency and policy performance over standard RLHF and proxy-modification baselines.

### Strengths
Strengths:
- Reward hacking and costly preference collection are established issues. The framing of “repairing” instead of replacing reward functions is interesting
- The proposed loss explicitly distinguishes between aligned and misaligned transitions, encouraging targeted corrections rather than global reward learning.
- Empirical results demonstrate that PBRR requires fewer human preferences to achieve high-performing policies than RLHF baselines.
- Regret bounds: Theoretical analysis establishes sublinear cumulative regret comparable to prior strategic preference-based RL methods.
- Method leverages existing proxy reward functions and can use imperfect reference policies which is more realistic.

### Weaknesses
Weakness:
- Though the formulation of "repairing" is interesting, the paper can benefit from a more thorough discussion between this paradigm and algorithms that learn a human reward in addition to the original (sparse) environment reward. Some references that can help illustrate this:
	- Zhang et al., GUIDE: Real-Time Human-Shaped Agents (continuous feedback shaping and simulator-based human reward modeling combined with existing environment reward)
	- Peng et al., Learning from Active Human Involvement through Proxy Value Propagation (active intervention and human-guided policy shaping, alternative to preference-based repair)
- The method presumes the proxy reward is “aligned or overly optimistic”, which may not hold universally. The effects of pessimistic or adversarial proxies are underexplored or need a discussion.
- Experiments mainly use reward-hacking benchmarks; lack of evaluation on tasks with continuous control or high-dimensional observations. Could the authors discuss more on this and how the proposed method can be generalized to these domains? Many references mentioned in the related works have already conducted experiments in these domains.
- The algorithm relies on a reference policy for exploration guidance, but quality requirements and failure cases are not evaluated.
- Preferences are assumed to follow a Bradley–Terry model; discussions on the robustness to noisy or inconsistent human feedback should be better addressed.

### Questions
Addressing some of the weakness would be very helpful.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the problem of reward hacking in RL. The paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that aims to "repair" an initial, imperfect proxy reward function. PBRR works by learning an additive, transition-dependent correction term.

The method has two core component: (1) a targeted exploration strategy that elicits human preferences on trajectory pairs generated by the current (repaired) policy; (2) a new preference-learning objective that consists of three terms: $\mathcal{L}_{pref}$ (the standard cross-entropy preference loss), $\mathcal{L}^+$ (a regularization term that penalizes corrections for trajectories where the proxy reward already agrees with the human preference, and $\mathcal{L}^-$ (a regularization term that, when the proxy disagrees with the preference, penalizes corrections on the preferred trajectory)

The authors provide a theoretical regret analysis in tabular domains, showing a variant of PBRR matches sublinear regret bounds of prior work. Empirically, PBRR is shown to be significantly more data-efficient and stable, outperforming baselines that learn from scratch (RLHF) or use alternative repair methods (like the concurrent Residual Reward Modeling, RRM). The authors also thoroughly ablate their results

### Strengths
- The two key components of PBRR are novel and well-justified. The optimism-based loss function (Eq. 3) is a clever piece of mechanism design, directly targeting the assumed cause of reward hacking (over-optimism) by prioritizing negative corrections
- The paper presents good empirical results and thorough ablations
- The paper is well-written and presents good additional information in the Appendix, eg in App. G, explaining why strong baselines like RRM and Online-RLHF fail

### Weaknesses
- The empirical evaluation is somewhat limited in complexity. The environments are useful but relatively simple. The paper would be significantly strengthened by demonstrating PBRR's effectiveness on more complex, high-dimensional continuous control domains
- The main regret analysis in Section 5 is for a complex algorithm variant (with C_1 > 0) that is not used in the experiments (which set C_1 = 0)
- The method relies on $\pi_{ref}$. The paper uses reasonable BC-derived policies. However, the sensitivity to the quality of $\pi_{ref}$ is not explored. What if $\pi_{ref}$ is a random policy? An ablation on the quality of $\pi_{ref}$ would strengthen the paper's claims

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The work presents a method for correcting a misspecified proxy reward function in the broad framework of PBRL and RLHF automatically by designing a new loss function. The work’s proposed PBRR (preference based reward repair) is claimed to be better than RLHF as it requires less data than full RLHF, and easily incorporate a human specified, potentially misaligned reward. Empirically, the work demonstrates their method can learn better than learning a reward function from scratch using preference data using fewer data samples, and archives high performing policies.

### Strengths
The work is well motivated. I also liked the idea of humans specifying an initial, potentially misaligned rewards. Then using targeted exploration, the methods can automatically correct the reward function to get true, hidden human prerefences.

### Weaknesses
There are some concerns in the problem formulation, solution design and empirical setup that are noted below.

One main issue in section 4 is that the authors make an assumption in lines 182-190 that humans provide an overly optimistic reward function. This insight is certainly not true. While in some cases humans can be overly optimistic which may cover some past cases of reward hacking in Krakovna et al, understanding how humans misspecify the reward is itself a monumental challenge. Some humans can be more risk sensitive, some more risk seeker. Thus, framing the entire learning objective, which is the main contribution of the work, in Eq 3 on the overly optimistic scenario is not general, and can be misleading in a range of real world domains. Thus, this is one major issue with the  optimistic reward hypothesis with the propose formulation.

Unfortunately, the para “Constructing a preference dataset” is described only in passing, and several important details missed. This step, how to construct a pref dataset to repair the proxy reward is highly important and what is the novel contribution of the work over prior cited work needs to be specified clearly, and proper ablations must be done over different methods to acquire data to correct proxy reward function.

Experimentally, the work is tested only on 4 domains, which is quite limited evaluation. Furthermore, despite the main claim of the paper as humans specifying the proxy reward function \hat{r}, in the tested domains the proxy reward seems to be hardcoded using the logic in table 1. This is not inspiring confidence that the approach can really learn from real world human feedback as humans are often not good at providing numerical score to their preferences. Thus, empirical evaluation currently is not strong.

Furthermore, experiments on all the real human labeled data in the context on RLHF for LLMs is required. It also solves the problem of getting real human rewards, as human specified preferences are already collected in various publicly available dataset. This dataset can be used to get a proxy reward.

Better justification for when having access to a base policy \pi_ref that specifies safe behavior is possible. In the context of LLMs, it is not clear if such a based policy will exist.

Overall, I find the work addresses an important problem. I find value in the approach. Main outstanding issues remain using strong assumptions on how humans specify rewards to design a solution method, and experiments that are limited and do not use real human feedback, which seems inconsistent with the main premise of the paper.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2
