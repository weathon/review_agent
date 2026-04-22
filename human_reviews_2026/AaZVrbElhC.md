# CaRe-BN: Precise Moving Statistics for Stabilizing Spiking Neural Networks in Reinforcement Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Spiking Neural Networks (SNNs) offer low-latency and energy-efficient decision-making on neuromorphic hardware by mimicking the event-driven dynamics of biological neurons. However, the discrete and non-differentiable nature of spikes leads to unstable gradient propagation in directly trained SNNs, making Batch Normalization (BN) an important component for stabilizing training. In online Reinforcement Learning (RL), imprecise BN statistics hinder exploitation, resulting in slower convergence and suboptimal policies. While Artificial Neural Networks (ANNs) can often omit BN, SNNs critically depend on it, limiting the adoption of SNNs for energy-efficient control on resource-constrained devices. To overcome this, we propose Confidence-adaptive and Re-calibration Batch Normalization (CaRe-BN), which introduces (i) a confidence-guided adaptive update strategy for BN statistics and (ii) a re-calibration mechanism to align distributions. By providing more accurate normalization, CaRe-BN stabilizes SNN optimization without disrupting the RL training process. Importantly, CaRe-BN does not alter inference, thus preserving the energy efficiency of SNNs in deployment. Extensive experiments on both discrete and continuous control benchmarks demonstrate that CaRe-BN improves SNN performance by up to $22.6$% across different spiking neuron models and RL algorithms. Remarkably, SNNs equipped with CaRe-BN even surpass their ANN counterparts by $5.9$%. These results highlight a new direction for BN techniques tailored to RL, paving the way for neuromorphic agents that are both efficient and high-performing. Code is available at https://github.com/xuzijie32/CaRe-BN.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work presents Confidence-adaptive and Re-calibration Batch Normalization (CaRe-BN) to stabilize Spiking Neural Networks (SNNs) in Reinforcement Learning (RL). Addressing imprecise BN statistics in online RL that hinder SNN performance, CaRe-BN integrates a confidence-guided adaptive update for BN statistics and a periodic recalibration mechanism, ensuring accurate normalization without disrupting RL training or compromising SNN energy efficiency at inference. Evaluated on MuJoCo continuous control tasks, it boosts SNN performance by up to 22.6% across neuron models/RL algorithms, even outperforming ANNs by 5.9%, advancing BN techniques for SNN-RL.

### Strengths
First, it addresses SNN-RL's BN statistic imprecision via dual mechanisms, stabilizing training without disrupting RL processes or losing SNN energy efficiency .

Second, it shows strong adaptability, boosting SNN performance by up to 22.6% across diverse neuron models and RL algorithms .

Third, it outperforms ANNs by 5.9%, proving SNNs’ potential for efficient, high-performance control .

### Weaknesses
First, it only evaluates on MuJoCo continuous control tasks, lacking tests on more complex or real-world RL scenarios .

Second, it fails to report comparisons of training time and memory with existing BN methods for SNN-RL .

Third, it doesn’t explore CaRe-BN’s performance with more advanced spiking neuron models.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper highlights the misalignment between the true and computed activation distributions when batch normalization is applied in spiking neural networks (SNNs). In particular, the work discusses issues with existing batch normalization methods in SNN models that can negatively impact performance in reinforcement learning (RL) tasks.

Inspired by the Kalman filter, the authors propose a confidence-guided approach that more accurately estimates batch statistics. They also discuss a recalibration method to align the computed batch statistics with those of the training data.

The proposed method is evaluated on several RL tasks and compared against multiple baselines from both ANN- and SNN-based RL frameworks. Furthermore, an ablation study is conducted to demonstrate the performance improvements contributed by each individual component of the method.

### Strengths
Most of the paper is easy to follow. 

The problem highlighted is an interesting issue in BN for reinforcement learning. 

I believe the discussion of related work is adequate and also provides the necessary background to understand the problem.

### Weaknesses
I believe the novel part of the method is the confidence adaptive updation of batch statistics; however, the second part, recalibration, is a modification of the existing method. 

A few things are not clear, which are mostly related to the proposed method, which can be found in the questions section.

### Questions
Line 204: The authors mention that “conventional ANN-based RL algorithms do not employ BN.” This raises an important question — is batch normalization (BN) truly necessary in SNN-based RL frameworks? How does the proposed SNN model perform without BN? A comparison or ablation along these lines would strengthen the justification for including BN in SNNs.


Line 233: The authors provide a proof for computing the optimal K value. It appears that \mathbb{D} is computed recursively. Could the authors please elaborate on this derivation and clarify the recursive relationship involved?


The definition and interpretation of the measure \mathbb{D} are unclear. Could the authors provide more explanation regarding what \mathbb{D}  quantifies and how it relates to the overall estimation or confidence process?


The rationale behind approximating \mathbb{D} as shown in Equation (9) is not clearly justified. Please clarify why this specific approximation is valid and under what assumptions it holds.


The terminology “confidence-based adaptive scheme” is somewhat confusing. Why is the deviation from the actual value interpreted as confidence? A clearer explanation of how “confidence” is defined and why it is used in this context would be helpful.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a method to improve the estimation of Batch Normalization (BN) statistics. The authors apply this improved BN to spiking neural networks (SNNs) and report that it outperforms other BN variants on six reinforcement learning (RL) tasks.

### Strengths
1. The paper is relatively easy to follow.

2. Multiple tasks are evaluated.

### Weaknesses
1. The claim that “due to the discrete and non-differentiable nature of spikes, directly trained SNNs rely heavily on BN to stabilize gradient updates” is misleading. BN is not designed to solve non-differentiability in SNNs; please revise this.

2. It is unclear whether the contribution is primarily for ANNs or SNNs. Figure 1 suggests general applicability to ANNs, yet experiments are focusing on SNNs—please clarify the intended scope and update the paper.

3. Section 5.2 / Figure 3: The evidence in Figure 3 appears insufficient to conclude that the method yields more precise BN statistics, and contributes to better reward. We can only observe that CaRe-BN gets obviously higher expected rewards on three tasks out of five. It is unknown what is the root cause of such observation. 

4. How many time steps are used in the SNNs? How sensitive is performance to this hyperparameter, and what justifies the chosen value to be used in the baseline setting? Also, ablation study on the time step should be provided.

### Questions
1. Figure 1: What does the “update step” represent—training parameter updates or an SNN inference time step?

2. I could not find details of the network architectures used. Please describe the models (layers, widths, neuron types, readouts, etc.).

3. How are SNN time steps integrated with the RL policy? Do you aggregate/compress time steps to produce the actor/critic outputs? Which components are SNNs and which components are not?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets a practical instability in online RL with SNNs: BatchNorm (BN) statistics are hard to estimate under non-stationary data and replay, leading to train–inference mismatch and noisy updates. It proposes CaRe-BN, a two-part BN strategy: (i) Confidence-adaptive BN (Ca-BN), which linearly fuses the previous moving estimate and the current mini-batch estimate using variance-based gains (a Kalman-style, minimum-MSE update under stated assumptions); and (ii) Re-calibration (Re-BN), which periodically re-estimates inference statistics from larger replay samples to correct accumulated drift. CaRe-BN is drop-in and leaves the inference path unchanged. Integrated with TD3/DDPG and LIF/CLIF neurons, experiments on MuJoCo report consistent gains and, on average, SNN+CaRe-BN outperforming a vanilla ANN-TD3 baseline. The paper also argues that “more precise BN statistics” improve exploration quality.

### Strengths
1. Clear problem framing with a simple, principled fix. The paper isolates BN statistic mismatch as a key bottleneck in SNN-RL and proposes a plug-and-play estimator with an intuitive confidence-weighted fusion + periodic re-calibration backed by a tractable derivation.

2. Consistent empirical gains with informative ablations. Across several MuJoCo tasks and spiking neuron types, CaRe-BN improves stability/returns, and ablations suggest Ca-BN and Re-BN are complementary, supporting the design rationale.

3. Practical applicability and minimal engineering overhead. The method does not alter the inference graph, is easy to integrate into standard actor–critic pipelines, and is plausibly extensible beyond SNNs to other online/temporal settings where BN statistics drift.

### Weaknesses
1. Theory–data distribution mismatch in Theorem 1 (target inconsistency between moving BN and batch BN)
The theorem assumes that the “moving/prior” BN estimator and the “mini-batch/observation” BN estimator are unbiased estimates of the same target distribution with correctly specified variances. In online RL with replay, this assumption is typically violated: the moving BN statistics track the current inference distribution, while the batch BN statistics are computed from a replay mixture of (often older) policies, with temporal correlation and potentially heavy-tailed activations. Consequently, the two estimators do not share the same target, unbiasedness/i.i.d. fail, and the linear fusion with gain 𝐾 is not guaranteed to be valid or optimal (and may introduce bias). 

2. Limited experimental scope beyond MuJoCo 
Although the paper compares against common SNN normalization baselines (e.g., BNTT), the evaluation remains restricted to MuJoCo continuous control. To substantiate generality and the claimed benefits under stronger non-stationarity, the method should also be tested on exploration-hard and vision-based benchmarks (e.g., Atari).

3. Exploration claim lacks causal evidence and proper metrics
Section 5.2 infers “more precise BN statistics → better exploration” primarily from an “exploration return” curve without defining the metric, controlling confounders (action-noise schedule, value-estimation bias, policy smoothness), or reporting standard exploration measures (state-coverage/visitation, occupancy entropy, trajectory diversity, feature-space coverage). No experiments are provided on exploration-hard tasks or against classic intrinsic-motivation baselines (ICM, RND, NGU/Plan2Explore, count-based variants). The authors need to further analyze the reason why the proposed method can improve the efficiency of exploration. 

4. Unestablished advantage over strong ANN-RL and unmeasured energy claims
The only ANN comparison uses a vanilla ANN-TD3 (in the appendix). Modern, stronger ANN-RL baselines (e.g., SAC; TD3/SAC with LN/GN/RMSNorm or Batch Renorm/AdaBN; distributional critics; entropy/parameter-noise regularization) are not evaluated, so superiority over ANN remains unproven. Moreover, the touted energy advantage of SNNs is theoretical and specific to neuromorphic hardware; no deployment-phase energy/latency measurements (or credible proxies) are reported, nor is training vs. inference cost disentangled. Open discussion: beyond theoretical energy efficiency on neuromorphic hardware, what concrete advantages does SNN + CaRe-BN have over strong ANN-RL methods (e.g., sample efficiency under non-stationarity, stability under distribution shifts, low-latency control, small-batch robustness), and can the authors provide controlled evidence for them?

### Questions
1. Theory & Theorem 1 assumptions (target inconsistency and variance modeling).
Could you formally define the target distribution for BN statistics in online RL and quantify the mismatch between moving BN (inference) and batch BN (replay) (e.g., TV/Wasserstein distance over activations)? Please provide bias/variance bounds for your fused estimator under distribution shift, report effective sample size N_"eff" with autocorrelation estimates for replayed batches, and clarify how you separate process noise vs. observation noise when computing the confidence gains (consider an adaptive Kalman/EM treatment instead of a single residual EMA).
2.“Better exploration” claim: metrics, confounders, and causal evidence.
What is the exact definition of exploration return? To establish causality, could you (i) align random seeds and action-noise schedules, (ii) report standard exploration metrics—state coverage/visitation counts (or pseudo-counts), occupancy entropy, trajectory diversity, feature-space coverage, and (iii) add Actor-only vs. Critic-only CaRe-BN ablations and noise-aligned runs? Please also include tests on exploration-hard benchmarks (e.g., Atari) and comparisons to intrinsic-motivation baselines (ICM, RND).
3.External validity beyond MuJoCo and advantage over strong ANN-RL.
Beyond MuJoCo, can you evaluate on vision-based and non-stationary settings (context/reward switches, parameter drift) and compare against stronger ANN baselines (e.g., SAC; TD3/SAC with LN/GN/RMSNorm or Batch Renorm/AdaBN; distributional critics; entropy/parameter-noise regularization)? If the paper claims energy advantages, please report deployment-phase energy/latency (or credible proxies/ simulation power consumption estimation) and disentangle training vs. inference costs; otherwise, consider narrowing the energy-efficiency claims.

### Soundness
2

### Presentation
3

### Contribution
2
