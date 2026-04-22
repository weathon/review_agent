# XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Sample efficiency is a central property of effective deep reinforcement learning algorithms. Recent work has improved this through added complexity, such as larger models, exotic network architectures, and more complex algorithms, which are typically motivated purely by empirical performance. We take a more principled approach by focusing on the optimization landscape of the critic network. Using the eigenspectrum and condition number of the critic’s Hessian, we systematically investigate the impact of common architectural design decisions on training dynamics. Our analysis reveals that a novel combination of batch normalization (BN), weight normalization (WN), and a distributional cross-entropy (CE) loss produces condition numbers orders of magnitude smaller than baselines. This combination also naturally bounds gradient norms, a property critical for maintaining a stable effective learning rate under non-stationary targets and bootstrapping. Based on these insights, we introduce XQC: a well-motivated, sample-efficient deep actor-critic algorithm built upon soft actor-critic that embodies these optimization-aware principles. We achieve state-of-the-art sample efficiency across 55 proprioception and 15 vision-based continuous control tasks, all while using significantly fewer parameters than competing methods. Our code is available at danielpalenicek.github.io/projects/xqc.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper starts with a systematic evaluation of three design choices in RL (normalization layer, MSE-CE loss, and weight normalization), based on the smoothness of the critic loss curvature. Their results suggest that the combination of batch normalization, CE loss, and weight normalization yields the smoothest loss curvature and thus stable optimization process. Theoretical insights are additionally provided on why CE loss generates much more stable gradients than standard MSE loss.

Based on the results, authors propose a simple architecture design that encorporates the three components. With significantly smaller number of parameters compared to prior works, their architecture shows competitive or even superior performance in diverse benchmarks comprising up to 55 tasks. Their results also extend to vision-based RL, where the architecture leads to significant improvement of DrQ-v2, especially in challenging humanoid environments.

Finally, their ablation results reveal that switching on of the components lead to degradation of performance, validating their design choices.

### Strengths
- **Well-motivated design choices**. Authors have verified their design choices based on thorought and systematic evaluation of the smoothness of the critic loss curvature.
- **Strong performance across diverse benchmarks**. The proposed architecture shows strong results compared to prior SOTAs in total 55 tasks.

### Weaknesses
- I only see minor questions and weaknesses; please see the questions.

### Questions
- I believe comparing the actual train/inference speed (FPS) with prior SOTAs would also be meaningful, as it directly reflect the speed expected in practice, and may differ from FLOPs depending on aspects such as number of layers.
- To my understanding, IQM was used to remove outlier tasks in Atari benchmark [1], and it feels unnatural to use IQM for learning curves of individual tasks. I don't necessarily go fully against using IQM, but I suggest the authors provide average results for both individual learning curves and aggregate results along with IQM.
- In Figure 8, XQC LN seems to have twice as much ELR as XQC. Would it be possible that this difference caused the performance drop shown in Figure 9 (right)? Have you tried decreasing the base learning rate to $1.5e-4$, half of default value of XQC?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces XQC, a simple actor-critic architecture that improves sample efficiency in deep reinforcement learning through well-conditioned optimization. Rather than relying on model scaling or complex architectures, XQC achieves stability and efficiency by combining batch normalization, weight normalization, and a categorical cross-entropy loss, which together yield lower Hessian condition numbers and bounded gradient norms. 

Theoretical analysis and large-scale empirical results - spanning 70 continuous control tasks from proprioceptive and vision-based benchmarks - demonstrate that this synergy leads to superior training stability, plasticity, and state-of-the-art performance while using significantly fewer parameters and compute than prior methods such as BRO or SIMBA-V2.

### Strengths
- Studying architectures that allow for scaling actor-critic is timely and important
- The proposed method is simple and seems to offer a more computationally efficient alternative to scaling via BRO or SimBa
- I appreciate the analysis comparing different normalization strategies

### Weaknesses
- Limited novelty. The work combines puzzle pieces from previous works (batch norm + weight norm was used in [1], weight norm + categorical loss was used in [2], layer norm + categorical loss was used in [3]). I still appreciate showcasing that this particular combination seems to be most effective for scaling actor-critic.

### Questions
1. Why do the authors use an action repeat wrapper in the humanoid bench? The humanoid bench uses joint position control, which does not make much sense with action repeat. Because of how joint position commands work, repeating that same position command multiple times has no new effect (ie. the robot just keeps tracking the same target position, no?)
2.  [4] found that BRO performance significantly changes depending on the action repeat setting. Did the authors test their method without action repeat on benchmarks other than Mujoco?
3.  Is it expected that BRC underperforms BRO? To the best of my understanding, they differ only by using categorical loss instead of MSE. Considering the presented results on the importance of the categorical loss, shouldn't BRC perform similarly to BRO?
4. Do authors have a hypothesis as to why the performance of their approach decreases when using more layers?
5. BRO and SimBa use residual architectures. Could the authors comment on why they opted for standard MLP here? Do authors think that it might matter for more complex environments?
6. It would be interesting to see how DrQ+BRO or DrQ+SimBa performs as compared to DrQ+XQC, considering that BRO and SimBa papers did not include vision-based DMC experiments. Do authors think that these methods would also perform worse than XQC in vision experiments?
7. Could the authors summarize the shortcomings of their theoretical analysis? What are the differences between the assumptions and real-world problems?
8. It would be interesting to see how scaling improves conditioning on its own as compared to using bn + wn + categorical loss.
9. Batch normalization is interesting because the batch statistics depend on the sampled batch (and the variance of these statistics depends on the batch size). To this end, it would be interesting to see how robust XQC is to different batch size settings and whether the increased variance improves the optimization landscape.

Nitpicks:
- line 48 update-to-date -> update-to-data
- line 099 bellmann -> bellman
- line 331 propioceptive -> proprioceptive


[1] Palenicek et al. Scaling CrossQ with Weight Normalization
[2] Lee et al. Hyperspherical Normalization for Scalable Deep Reinforcement Learning
[3] Nauman et al. Bigger, Regularized, Categorical
[4] Voelcker et al. MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL

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
This paper proproes XQC, studying how to make critic optimization in deep RL better conditioned by systematically revisiting three ingredients—normalization layers, weight normalization/projection, and the critic loss—and ties them together through a Hessian-based analysis. It shows that, contrary to common practice favoring LN, BN yields local loss landscapes with dramatically smaller condition numbers during learning; WN leverages scale invariance to stabilize the effective learning rate (ELR); and a categorical cross-entropy (CE) distributional loss induces a better-conditioned objective than MSE, with a bounded Hessian condition number under mild assumptions. Building on these findings, the authors propose XQC, a simple SAC extension that combines BN, WN, and a CE Bellman error, arguing the trio acts synergistically to improve conditioning and stabilize ELR, which in turn boosts sample efficiency. Extensive experiments on 55 proprioceptive and 15 vision control tasks report state-of-the-art results against larger, more complex baselines, supporting the claim that improving optimization fundamentals can outperform mere scaling.

### Strengths
**Principled theory that links design choices.** The paper gives a clear Hessian-based account of why BN (vs. LN), CE (vs. MSE), and WN jointly improve critic optimization—showing BN yields much smaller condition numbers, CE produces a better-conditioned objective, and WN stabilizes the effective learning rate; it even derives bounded eigenvalue ranges/condition numbers under mild assumptions. 

**Thorough and careful empirics at scale.** Results span 70 tasks (55 proprioceptive + 15 vision), use solid aggregate metrics (IQM AUC with 90% SBCIs, 10 seeds), and consistently show SOTA sample efficiency—including fair pixel-based comparisons by re-using DRQ-V2 encoders/hyperparameters—and diagnostic analyses that tie gains to stabilized ELR and gradients.

### Weaknesses
**Limited originality**. Similar ideas have been proposed in previous works, like BatchNorm in CrossQ (ICLR 2024); Replacing MSE with CE loss in EfficientZero (Neurips 2021), Stop Regression (ICML2024), BRC (Neurips 2025). It seems that this paper just analyzed the same problem from another perspective but reached a similar conclusion.

**Limited scope**. As mentioned before, similar conclusions have been proposed in previous work, and this paper seems to be an ensemble of them. However, it would be interesting if the authors could expand the scope of this paper to include more settings, such as model-based RL, to investigate the general efficacy of the proposed architecture rather than constraining the scope to only the actor-critic and model-free settings.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes XQC, a simple modification of SAC that targets the optimization landscape of the critic. The key design is the synergistic use of batch normalization at every MLP layer, weight normalization via post-update unit-sphere projection, and a categorical distributional critic trained with cross-entropy. The authors argue this trio yields (i) bounded gradients, (ii) a nearly constant effective learning rate, and (iii) substantially better Hessian conditioning than typical alternatives. Empirically, XQC reports state-of-the-art sample efficiency across 55 proprioceptive and 15 vision DMC tasks, using fewer parameters and lower stated FLOP/s than strong baselines.

### Strengths
- Originality: Reframes sample efficiency through loss-landscape conditioning with BN, WN, CE and a principled rationale.
- Quality: Extensive experiments (10 seeds in most main results), scaling studies (UTD/width/depth), and insightful plasticity metrics.
- Clarity: The paper is easy to follow, the role of each component is well articulated, figures are generally informative.
- Significance: If the compute/latency picture is corroborated, XQC can influence default choices for actor–critic architectures.

### Weaknesses
- Reporting parameters and FLOP/s is helpful, yet the current figures seem potentially misleading. For a fair comparison, training/inference speed should be addressed to complete the real-world efficiency story.
- For pixels, the authors state that XQC keeps the vision encoder unchanged for fairness. That isolates MLP effects, but it also under-tests the hypothesis: BN/WN/CE improvements could plausibly extend into the conv encoder. This limits the reported vision gains.
- Figure 7 highlights sample efficiency but 10M buffer size is quite unrealistic in various setups. A 1M buffer setup would make the point under practical budgets.

### Questions
- XQC’s MLPs do not include residual/skip connections, while some baselines do. Given residuals’ impact on conditioning, it would be useful to justify their omission and/or report a small ablation.
- The bounded condition number result for CE relies on weight decay to enforce positive definiteness, yet the best empirical runs often avoid weight decay. The paper acknowledges this; still, a short empirical probe on “with vs without WD” vs conditioning would tighten the loop.

### Soundness
3

### Presentation
4

### Contribution
4
