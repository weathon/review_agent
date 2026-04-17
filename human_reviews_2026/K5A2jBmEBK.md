# DeepCompress: A Dual Reward Strategy for Dynamically Exploring and Compressing Reasoning Chains

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large Reasoning Models (LRMs) have demonstrated impressive capabilities but suffer from cognitive inefficiencies like ''overthinking'' simple problems and ''underthinking'' complex ones. While existing methods that use supervised fine-tuning (SFT) or reinforcement learning (RL) with token-length rewards can improve efficiency, they often do so at the cost of accuracy. This paper introduces DeepCompress, a novel framework that simultaneously enhances both the accuracy and efficiency of LRMs. We challenge the prevailing approach of consistently favoring shorter reasoning paths, showing that longer responses can contain a broader range of correct solutions for difficult problems. DeepCompress employs an adaptive length reward mechanism that dynamically classifies problems as "Simple" or "Hard" in real-time based on the model's evolving capability. It encourages shorter, more efficient reasoning for "Simple" problems while promoting longer, more exploratory thought chains for "Hard" problems. This dual-reward strategy enables the model to autonomously adjust its Chain-of-Thought (CoT) length, compressing reasoning for well-mastered problems and extending it for those it finds challenging. Experimental results on challenging mathematical benchmarks show that DeepCompress consistently outperforms baseline methods, achieving superior accuracy while significantly improving token efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DeepCompress, a novel framework that enhances Large Reasoning Models (LRMs) by dynamically classifying problems as "Simple" or "Hard" based on the model's real-time performance. By applying a dual-reward strategy that encourages shorter reasoning for "Simple" problems and longer exploration for "Hard" ones, DeepCompress achieves improved accuracy while simultaneously improving token efficiency on multiple mathematical benchmarks.

### Strengths
1. The problem studied in this paper is critical.
2. The paper is easy to follow and understand.
3. Experimental results show that the performance of the proposed method improves consistently over base models.

### Weaknesses
1. The paper lacks important baselines to compare with other RL-based reasoning chain compression works, for example [1, 2, 3], and does not position the proposed method among them.
2. The difficulty measure completely relies on batches, which may lead to instability. In a batch containing only simple problems, a problem that is simply "not that simple" may also be incorrectly marked as "difficult" because $P_g $is lower than the extremely high $P_b $. Vice versa. This mechanism cannot enable the model to learn a stable and absolute internal representation of the difficulty of the problem, but instead causes the reward signal to fluctuate dramatically with the random fluctuations of batch data.
3. It is unclear how the hyperparameters are determined, e.g., $\beta$ is set to 1 and -1 without explanation or investigation.
4. How will the performance change with a different group size $G$?

References:

[1] Tu S, Lin J, Zhang Q, et al. Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL[J]. arXiv preprint arXiv:2505.10832, 2025.

[2] Shrivastava V, Awadallah A, Balachandran V, et al. Sample more to think less: Group filtered policy optimization for concise reasoning[J]. arXiv preprint arXiv:2508.09726, 2025.

[3] Yuan D, Xie T, Huang S, et al. Efficient RL Training for Reasoning Models via Length-Aware Optimization[J]. arXiv preprint arXiv:2505.12284, 2025.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes DeepCompress, a reinforcement learning (RL) framework for large reasoning models (LRMs) that introduces a dual reward strategy—encouraging shorter reasoning for “simple” problems and longer reasoning for “hard” ones. The method dynamically classifies questions based on the group pass ratio within RL batches, and adjusts a sigmoid-shaped length reward accordingly. The authors claim improved accuracy and token efficiency across mathematical reasoning benchmarks such as MATH-500, AIME, and OlympiadBench.

### Strengths
- The problem setup—balancing reasoning efficiency and accuracy—is interesting and timely.

- The adaptive length reward (shorter for easy, longer for hard) is conceptually reasonable and aligns with observed overthinking/underthinking phenomena in LLMs.

- The paper provides extensive experimental evaluation on multiple math reasoning benchmarks.

- Figures and tables (e.g., Table 1, Fig. 4) show measurable improvements over baseline RL models like DeepMath-Zero.

### Weaknesses
1. Missing ablation and causal analysis: Although the authors attribute improvements to the dual reward, there are no clear ablations isolating which component (dual reward vs. model-aware difficulty) contributes most to the gain. The provided variants (length bonus / penalty) are simplistic and insufficient to explain why the accuracy improves. There’s also no sensitivity study on α, β, or λ—key hyperparameters controlling reward magnitude.

2. It seems the results contradict the motivation on length vs. accuracy: Figure 1 and Table 4(c) still show that pass@1 accuracy decreases with increasing reasoning length. This raises doubts about the claim that longer reasoning improves correctness for hard problems. The authors interpret longer reasoning as beneficial in the RL phase, but the empirical results suggest diminishing returns for long outputs.

3. Ambiguity in difficulty classification: The definition of “simple” vs. “hard” via group pass ratio is circular—model competence determines the classification, which in turn affects training rewards. This may cause instability or reinforce existing biases rather than lead to genuine adaptivity.

4. Unclear explanation of policy gradients and learning signal: It is not clear how the “dual reward” integrates into the policy gradient objective beyond heuristic addition. The paper repeatedly mentions DAPO and GRPO but doesn’t formally show how the modified reward affects gradient updates or stability. It reads more as an empirical tweak than a well-motivated RL formulation.

### Questions
1. How does the model perform if we train only with the model-aware difficulty component but no explicit length reward?
2. Does the adaptive β term (Eq. 8) introduce additional variance into the policy gradient estimator? If so, how do you ensure stable convergence?
3. How much of the gain comes from model-aware difficulty versus dual reward?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an an adaptive length reward mechanism that trade-off the efficiency and performance for the large reasoning models (LRM). Different from the previous length-penality method, this method introcue the length reward according to the “difficulty” of problem at hand. However, while the core idea has merit, there are several concerns regarding experimental rigor, theoretical justification, and presentation clarity that need to be addressed.

### Strengths
+ Observation: The observation that Pass@1 decreases while Pass@32 increases with length (Figure 1) is genuinely insightful.

+ The experiments is fully evaluated acorss different becnhmark (7 benchmark) and is hoslitc enough for the math problem.

+ The paper presents a coherent and compelling research: clear motivation, principled method design, and  in-depth analysis. The writing clearly connects empirical findings to design choices to performance outcomes, making the contribution well-motivated.

### Weaknesses
1. The authors observe that Pass@1 diminishes with length increase, while Pass@32 generally increases. The paper attributes this to "longer responses contain a wider coverage of potentially correct solutions." However, this interpretation requires deeper scrutiny:
- **Is this truly about difficulty?** Or is it simply a statistical artifact? When you have 32 samples, longer responses might just have more opportunities to "stumble upon" correct solutions through increased exploration, not necessarily because the problem is inherently harder.
- What is the correlation between problem difficulty (ground truth labels from MATH dataset) and average response length? This would validate whether length truly reflects difficulty or model capability.

2. The classification depends on batch composition. A problem could be "Simple" in one batch and "Hard" in another depending on what other problems are sampled. This inconsistency could lead to unstable training signals. And moreover, there is a lot of recent study on dynamic batching and rollout filtering, the usage of batch and group-level metric for defining “difficulty” may be somewhat unrealiable.

3. Experimetns, since the author demonstrate the adaptive length rewards, I think at least should compare with some popular length-penality such as those in Kimi-1.5 or over-long filtering method. But in current experiments, the author just experimenet with several baseline models.

4. Moreover, I would recommand using Qwen-3 series for further validation since Qwen-3 is more prone to think longer comapred to Qwen-2.5.

5. I will be glad to see the ablation experiment accordong to different hyper-parameters such as $\beta$ in Eq.4, the reward weight $\alpha$.

6. In Figure 4, Length Bonus baseline shows higher entropy and longer responses but lower final performance than DeepCompress. This contradicts the paper's claim that longer responses help with hard problems. Why doesn't pure length bonus work better?

7. Typo

    + Lien 58: infeasbile -> infeasible
    + Line 59: effciency -> efficiency

### Questions
+ How does DeepCompress perform on problems of known, fixed difficulty (e.g., using MATH's difficulty labels)? This would separate the "model-aware" aspect from actual difficulty.

    + Or use similar difficulty assessment similar to [1]

        [1] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. Zihe Liu, et al.

+ Have you tested on non-mathematical reasoning tasks? (e.g., LiveCodeBench)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DeepCompress, a new framework designed to enhance both the accuracy and efficiency of Large Reasoning Models. The authors identify that LRMs often "overthink" simple problems and "underthink" complex ones, and that current methods to shorten reasoning paths often sacrifice accuracy.

DeepCompress addresses this with a dual reward strategy in an RL setup. It dynamically classifies problems as "Simple" or "Hard" based on the model's real-time performance. For "Simple" problems, it rewards shorter, more efficient reasoning chains. For "Hard" problems, it encourages longer, more exploratory thought processes.

Experimental results on challenging mathematical benchmarks show that models trained with DeepCompress consistently outperform baselines, achieving higher accuracy while simultaneously generating more concise responses, thus improving token efficiency.

### Strengths
- This paper introduces a method that rewards shorter reasoning for simple problems and encourages longer, exploratory reasoning for difficult ones. This challenges the standard approach of always favoring conciseness.

- This paper demonstrates that DeepCompress models achieve state-of-the-art results, outperforming strong baseline models across multiple challenging mathematical reasoning benchmarks, especially on difficult problems like AIME.

- The research provides a thorough analysis showing that DeepCompress encourages a more effective learning process by balancing exploration (via high policy entropy) and efficiency, leading to more frequent and effective "reflection" behaviors.

### Weaknesses
- The core mechanism hinges on classifying a problem as "Simple" or "Hard" based on whether its group pass ratio (Pg) is above or below the batch pass ratio (Pb). This is a noisy and relative metric. A problem isn't inherently "Hard"; it's just harder than the batch average. This could lead to suboptimal rewards, especially in batches with skewed difficulty distributions. 

- The paper proposes conditioning the length reward on the correctness of a solution to prevent reward hacking. However, this creates a paradox for the most challenging problems. If a problem is so hard that the model fails to generate any correct solutions in a group (Pg = 0), the length reward mechanism (which is designed to encourage longer, exploratory paths for hard problems) will never be triggered. The model receives no signal to try harder, precisely when it needs it most.

- The authors themselves acknowledge that the method's effectiveness relies on sufficient length variation among the sampled responses. If the model's sampling process (governed by temperature and top-p) produces responses of very similar lengths for a given problem, the standard deviation (σi) will be close to zero, making the standardized length (zi) unstable and the resulting length reward signal meaningless. This makes the framework's success sensitive to the initial state of the model and the choice of decoding hyperparameters.

- The preliminary analysis (Figure 1) shows that pass@32 (a proxy for the training objective) generally improves with longer responses, while pass@1 (the primary evaluation metric) decreases. The framework is trained to find at least one correct answer within a group, which favors exploration. However, it is evaluated on its ability to produce a correct answer on the first attempt (pass@1), which favors exploitation and conciseness. While the paper shows success on both, this fundamental tension between the training signal and the desired inference behavior is a subtle but significant weakness.

- The main comparison is against DeepMath-Zero, a vanilla reinforcement learning approach. The ablation studies use fixed "Length Penalty" and "Length Bonus" strategies, which are arguably strawman arguments. The paper would be stronger if it compared DeepCompress against other adaptive length-reward strategies mentioned in its own related work section, which would provide a more rigorous test of its claimed superiority.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
