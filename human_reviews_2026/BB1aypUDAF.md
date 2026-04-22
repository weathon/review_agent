# Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human values. 
However, noisy preferences in human feedback can lead to reward misgeneralization – a phenomenon where reward models learn spurious correlations or overfit to noisy preferences, which poses important challenges to the generalization of RMs.
This paper systematically analyzes the characteristics of preference pairs and aims to identify how noisy preferences differ from human-aligned preferences in reward modeling. 
Our analysis reveals that noisy preferences are difficult for RMs to fit, as they cause sharp training fluctuations and irregular gradient updates.
These distinctive dynamics suggest the feasibility of identifying and excluding such noisy preferences.
Empirical studies clarify that policy LLM optimized with a reward model trained on the full preference dataset, which includes substantial noise, performs worse than the one trained on a subset of exclusively high-quality data.
To address this challenge, we propose an online Collaborative Reward Modeling (CRM) framework to achieve robust preference learning through peer review and curriculum learning. 
In particular, CRM maintains two RMs that collaboratively filter potential noisy preferences by peer-reviewing each other’s data selections.
Curriculum learning establishes a well-defined learning trajectory to synchronize the capabilities of two RMs, further promoting the utility of peer review.
Extensive experiments demonstrate that CRM significantly enhances RM generalization, with up to $9.94$-points improvement on RewardBench under an extreme 40\% noise. 
Moreover, CRM can seamlessly extend to implicit-reward alignment methods, offering a robust and versatile alignment strategy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Collaborative Reward Modeling (CRM), a framework for improving the robustness and generalization of reward models (RMs) in reinforcement learning from human feedback (RLHF). The authors observe that noisy or inconsistent human preferences lead to reward instability in downstream alignment. To address this, CRM trains two reward models jointly through two complementary mechanisms:

1.Peer Review: At the batch level, each RM evaluates the other’s data and selects high quality preference pairs for training, filtering out potential noise and avoiding self-confirmation bias.

2.Curriculum Learning: At the epoch level, samples are ordered from easy to hard based on reward margins, synchronizing the learning trajectories of the two models and stabilizing collaboration.

The framework can be applied to both explicit reward modeling and implicit reward methods. Experiments using Llama3-based RMs on HH-RLHF, Ultrafeedback, Skywork-Reward, and RewardBench datasets show that CRM achieves consistent improvements in preference accuracy and downstream win-rates, especially under synthetic noise levels of 20–40%. 
The main contributions are: (1) a data-centric analysis of noisy preferences, (2) the design of the CRM framework combining peer review and curriculum learning, (3) experiments validating improved generalization under noise, and (4) preliminary extension to implicit-reward alignment (CRM-DPO).

### Strengths
The paper presents an original and creative reformulation of reward model training by introducing collaborative supervision into RLHF. Rather than improving the objective function or adding regularization, as prior work does, the proposed Collaborative Reward Modeling (CRM) adopts a data-centric approach where two reward models peer-review each other’s data and learn through curriculum scheduling. This perspective is novel and conceptually appealing.

In terms of technical quality, the paper provides a solid empirical foundation. The authors conduct extensive experiments across several benchmarks, covering both explicit and implicit reward settings. 

The clarity of presentation is generally good. The motivation is well established, the figures are informative, and the algorithmic framework is easy to follow. Some implementation details, particularly for the DPO extension, could be clearer, but the core method is explained with sufficient precision to reproduce the explicit-RM results.

Regarding significance, the proposed method contributes valuable insight into how data quality and model interaction affect reward generalization, but its practical impact on large-scale RLHF pipelines is likely limited. Training two RMs simultaneously substantially increases computational cost, making the method less feasible for large models or industrial-scale alignment settings. Nonetheless, as a research contribution, CRM highlights a new collaborative direction for improving robustness in preference learning and reward modeling.

### Weaknesses
1. The paper’s most critical weakness is the lack of a fair experimental baseline, which undermines the central claim that collaboration itself drives robustness. CRM uses two reward models, but all baselines employ only one. Without comparing to a two-independent-RM setup (two RMs trained separately and averaged or ensembled in policy training), the results cannot disentangle whether performance gains arise from the collaborative mechanism or simply from increased capacity and variance smoothing. Adding this control is essential to substantiate the core contribution.

2. Another weakness is the unclear handling of model imbalance: if one RM is substantially stronger, its peer feedback could dominate training, negating the intended collaboration. The authors should clarify how they ensure balanced influence—e.g., alternating updates, confidence weighting, or divergence regularization.

3. Finally, the implicit-reward (DPO) extension lacks methodological clarity. The paper presents results but omits key details about loss formulation, data exchange. Providing pseudocode or an explicit algorithm would make the extension reproducible.

4. Addressing these issues, especially the missing baseline, would make the contribution more rigorous and credible.

### Questions
1. Two-Model Baseline (Critical): The central claim of CRM is that collaboration between two reward models improves robustness. However, all baselines use a single RM. Could the authors include a two-independent-RM baseline, training two RMs separately on the same data and either averaging or ensembling their scores? This would isolate the effect of collaboration from the benefit of having two models. If results for this baseline are similar, the claimed advantage of CRM may be overstated.

2. Balancing Between the Two RMs: How is the contribution of each RM controlled during training? If one model becomes consistently stronger, does it dominate the peer-review process? Please clarify whether CRM uses alternating updates, normalization, or confidence-based filtering to maintain symmetry.

3. Implicit-Reward (CRM-DPO) Implementation: The paper states that CRM extends to implicit reward methods (DPO), but the algorithmic details are unclear. How is peer review defined when there is no explicit reward function? Providing explicit pseudocode or a step-by-step description would help clarify this.

4. Typographical Error (Minor issue): in Table 5 (ROPO objective), a ‘+’ sign is missing.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of reward misgeneralization in LLM alignment caused by noisy human feedback and proposes Collaborative Reward Modeling (CRM), an online dual-model framework. CRM introduces a Peer Review mechanism where two reward models mutually select low-loss samples to reduce confirmation bias, combined with Curriculum Learning to synchronize training and filter noisy preferences. Overall, CRM effectively enhances the robustness and generalization of reward models against noisy feedback.

### Strengths
1. The paper systematically analyzes the intrinsic characteristics of preference pairs and proposes an online noise filtering method rather than merely improving training objectives based on this analysis.
2. The method demonstrates strong generalizability, being applicable to both explicit and implicit reward alignment approaches, and the paper is overall clear, coherent, and well-written.

### Weaknesses
1. In the Peer Review stage, two reward models are trained collaboratively; however, this dual-model setup approximately doubles the training cost. The paper does not analyze training efficiency, convergence speed, or scalability with respect to data size, and such analysis is recommended.
2. In Section 3.2, the paper states that the two reward models determine their sample selection ratio based on the noise rate to mutually update and improve performance, but it does not explain how the noise rate is estimated.
3. Although the method is intuitive, the paper lacks theoretical analysis on convergence or the effect of reducing confirmation bias.
4. The experiments are mainly conducted on LLaMA3-3B (partly 8B), and validating the generalization performance on larger-scale models would make the conclusions more convincing.

### Questions
1. It remains unclear how sensitive CRM’s performance is to inaccurate estimates of the noise rate, and whether an adaptive sample selection ratio or uncertainty-based adjustment could improve robustness.
2. It is not specified whether CRM has been tested on larger reward models, and whether the benefits of collaboration diminish as model capacity increases.
3. The paper does not analyze potential failure cases where both RMs select noisy samples early on, and it is unclear if curriculum learning alone can prevent such collapse.
4. Since CRM resembles the Co-teaching method[1], the authors should clarify their algorithmic differences and the specific adaptations made for LLM reward modeling.

**Reference**

[1] Han B, Yao Q, Yu X, et al. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NeurIPS 2018.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a new collaborative method of reward modelling.

### Strengths
The paper is well written and is an important problem. The methodology described is novel as per to my knowledge. Several detailed experiments are provided

### Weaknesses
I have one primary issue with the experiment results described.

Technically, in CRM, one is training twice the number of reward parameters. Thus, it seems that comparing it with other methodologies that just use a single reward model is not fair. I think for an effective demonstration, the authors should compare the results with a 6B reward model. Otherwise, I am not convinced that the performance is solely due to the increased number of parameters being trained.

If the authors can elaborate on this point I shall consider raising my score

### Questions
Can you describe the sorting process (Curriculum Sort)

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
4

### Summary
This paper studies how noisy preference data in RLHF harms reward model generalization and proposes Collaborative Reward Modeling (CRM), where two reward models perform peer review with curriculum learning to filter noisy pairs. Experiments report higher preference accuracy and improved win rates over standard training and robust preference learning baselines, under intentionally added label-flip noise.

### Strengths
1. The selected problem setting, i.e., noisy preference data in RLHF is interesting, and is known to impact downstream alignment performance. This paper gives a demonstration through the definition of robust preference pairs, incorrect preference pairs and ambiguous preference pairs, though the the provided results in Figure 2 and 3 may be inaccurate because of "self-loss".
2. The proposed CRM framework is very simple and easy to implement: two reward models select low-loss pairs for each other and follow an easy-to-hard curriculum learning method, which though inevitably brings additional training complexity and costs into the pipeline.

### Weaknesses
1. The proposed CRM is limited in theory, and its designs including using self-loss, data filtering strategy and etc. are mostly heuristic, which indicates it may not be effective or extendable in other settings. For example, why choose two models evolve instead of 3 or more?
2. The empirical settings may have fetal flaw that the noise in the data is manually added by symmetric label flipping, which cannot reflect the complex situation in alignment of real world. The used "self-loss" for distinguishing the noised preference pairs may not effective when applied for realistic settings. The effectiveness and performance gains may come from the chosen label flipping. In another way, the self-loss for identifying noisy data is not fair or practical. I would like to know the ratio of noise data in the original preference dataset/benchmark often labeled by human or superior LLMs, and whether how many noisy pairs the self-loss could identity in this situation. In addition, the authors mention that real datasets already include noise, the evaluation does not convincingly separate gains from intentional corruption versus truly realistic noise.
3. The proposed method depends on a prior noise-rate estimate \(\eta\) to set the selection ratio, but this paper gives limited practical guidance on how to estimate this quantity in real deployment.
4. The proposed method treats ambiguous and non-robust pairs as noise to be suppressed, which may introduce additional bias, since ambiguous pairs may not be ambiguous if handled by more powerful LLM or human. Also, the provided cases are very few and limited. Again, it would be better to see the statistics in original dataset.
5. The evaluation is constrained to Llama-3-3B backbones and a single alignment pipeline per setting, so it is unclear whether CRM still brings clear gains at larger model sizes. To my understanding, the reward model should not become the computation burden especially in offline RL settings (e.g., DPO). Also, it will be interesting to see the two models are from different sizes and different model families. I also want to know whether it can be applied to LLM-as-a-Judge setting. It is also suggested to use reasoning models for alignment evaluation instead of relatively outdated GPT-4.
6. To understand the method in depth, It would be useful to see diagnostics or qualitative examples of pairs that are frequently discarded, to understand any systematic bias introduced by the filtering.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
1
