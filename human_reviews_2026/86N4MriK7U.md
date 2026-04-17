# SAMG: Offline-to-Online Reinforcement Learning via State-Action-Conditional Offline Model Guidance

- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
Offline-to-online (O2O) reinforcement learning (RL) pre-trains models on offline data and refines policies through online fine-tuning. However, existing O2O RL algorithms typically require maintaining the tedious offline datasets to mitigate the effects of out-of-distribution (OOD) data, which significantly limits their efficiency in exploiting online samples. To address this deficiency, we introduce a new paradigm for O2O RL called State-Action-Conditional Offline Model Guidance (SAMG). It freezes the pre-trained offline critic to provide compact offline understanding for each state-action sample, thus eliminating the need for retraining on offline data. The frozen offline critic is incorporated with the online target critic weighted by a state-action-conditional coefficient. This coefficient aims to capture the offline degree of samples at the state-action level, and is updated adaptively during training. In practice, SAMG could be easily integrated with Q-function-based algorithms. Theoretical analysis shows good optimality and lower estimation error. Empirically, SAMG outperforms state-of-the-art O2O RL algorithms on the D4RL benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel offline-to-online (O2O) reinforcement learning algorithm called SAMG that eliminates the need for offline data to be consumed again for online fine-tuning via state-action-conditional model guidance. SAMG first trains the critic and generative conditional model (C-VAE) in an offline manner, followed by online fine-tuning where the offline critic is frozen and used for guiding the online critic weighted by the state-action-conditional coefficient. The authors provide a theoretical analysis to prove the convergence property of SAMG with the modified Bellman operator, supported by empirical experiments with SOTA baselines on the D4RL benchmark. Comparisons with vanilla methods without SAMG highlight the clear improvements of the suggested online fine-tuning approach without the offline data.

### Strengths
- The idea of weighting the offline and online target critics with a confidence coefficient is simple but effective, which is reusable across most value-based offline RL methods.
- The contribution is strengthened by highlighting the difference with WSRL, which emphasizes the importance of eliminating the consumed offline data.
- Extensive empirical margins of adopting SAMG to popular offline RL methods demonstrate strong plug-and-play capability.
- Theoretical analysis helps in understanding the modified Bellman operator with the state-action-conditional coefficient, which serves the core function in the main methodology.

### Weaknesses
- The novelty of SAMG is marginal. Confidence coefficient estimation relies on the offline generative model for judging whether a given state-action pair is in-distribution or out-of-distributional data. However, there is no guarantee that the generative model produces reliable predictions when out-of-distributional data whether the output is erroneous or not. Furthermore, the fact that distinguishing whether the given pair falls under the offline distribution depends on the task-specific, hand-engineered threshold value $p_m^\text{off}$ exacerbates the reliability problem, since it is hard to decouple between erroneous output from the C-VAE or an inappropriate threshold value when the coefficient predicts the given pair is out-of-distributional data. I believe that providing further justifications on how well SAMG can overcome when C-VAE produces unreliable predictions or authors' tips on controlling the threshold value would strengthen the novelty.
- Experimental rigor could be improved significantly.
- - In Section 5.2, lacks of the experimental setup weaken the contribution of this paper. What does "1/2 for offline and 1/2 for online in SAMG (even)" exactly stand for? How do you generate the random probability for SAMG-random? What is the dataset used for this section (I infer it is *medium-replay* by comparing the results in Table 1)
- - In L205, the authors denote that samples with $p^\text{int} < p_m^\text{off}$ are regarded as OOD. However, the authors remark that if $p_m^\text{off}$ is too small, all samples will be regarded as OOD samples, thereby the online target critic is solely updated with the offline target critic (L419). This contradicts Equation (5) where $p^\text{off}(s,a)=0$ (OOD situation) when $p^\text{int}(s,a)<p_m^\text{off}$, since extremely low $p_m^\text{off}$ value would naviely pass most $p^\text{int}(s,a)$ into $p^\text{off}(s,a)$ directly. If this extreme case does not fall into an exemplary case, I would appreciate further justifications.
- Figure 1 can be improved to provide a comprehensive view of the overall architecture. I find it hard to establish an intuition about static or adaptive coefficient updates in Section 3.3.

### Questions
- What does $\alpha$ stand for in L257? The notation of the state-action-conditional coefficient is $p(s,a)$ or $p^\text{off}$, which is different.
- It is hard to find the best algorithms across different datasets and tasks in Table 1. Could you highlight the best scores per case in the table? (e.g., bold face)
- Although the authors note that the learning curves for $p_m^\text{off}=0.2$ and $p_m^\text{off}=0.5$ cases are overlapped in L421, readers may become confused when first seeing the plot alone. Could you provide more details in the caption in Figure 3?
- Extra experiments could deliver a more pronounced contribution of SAMG. I suggest a few recommendations below:
- - Curves of $p^\text{int}(s,a)$ (or $p^\text{off}(s,a)$ during online fine-tuning across different dataset optimalities. Ideally, the offline agent should rely more on the online target critic progressively as the fine-tuning progresses and the agent acquires better knowledge of the underexplored state-action space. I wonder how the coefficient degrades as the fine-tuning process continues.
- - If it is possible to attach a PDF file during the rebuttal, t-SNE visualization of latent vectors from C-VAE under two scenarios-in-distribution and out-of-distribution data is given-would justify the above weakness 1 by showing the similar clusters regardless of the distribution.

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
4

### Summary
The paper proposes SAMG, an offline-to-online RL fine-tuning framework that fuses a frozen offline critic with the online Bellman target using a state–action probability gate estimated by a conditional VAE. The gate emphasizes the offline critic in in-distribution regions and suppresses it for out-of-distribution samples, is updated during training, and comes with convergence-style arguments.

### Strengths
No longer relies on offline data replay; online sample efficiency is high. The formulation is simple and easy to plug into various Q-learning fine-tuning methods.

With C-VAE plus adaptive updates, p can gradually expand the “mastered OOD regions” as training proceeds, and the paper provides clear implementation details.

### Weaknesses
1. The design makes several simplifications that may harm performance by introducing imprecision. First, it models the encoder outputs as independent Gaussians and linearly weights them, ignoring correlations and richer distributional structure. Forcing (p) below a threshold directly to zero (Eq. (5)) is too hard, causing non-differentiable/discontinuous use of information and sensitivity to training noise. The CQL extra penalty is altered by adding an “offline version” weighted by (p); the authors admit this “slightly deviates” from the original setup, implying less-fair comparisons and potential divergence risk.
2. The theoretical assumptions are strong. The accelerated-convergence bound depends on the ratio of offline/online sub-optimality bounds, which is unobservable in practice, and the assumption that the offline bound is significantly tighter may not always hold.
3. The O2O fine-tuning based on CQL/AWAC/IQL is not specifically optimized for the online phase. As noted in related work, “a series of Q-ensemble-based algorithms are proposed, combined with balanced experience replay.” The authors should discuss more thoroughly to demonstrate broader effectiveness and performance advantages of the approach.
4. Notational issues: please carefully check the entire paper. In Equation (2), the KL divergence is written as Enc (|) Dec, which seems unreasonable—the prior should be (p(z)), not the decoder. Also, the state–action-conditional coefficient (\alpha) should be consistent with (p(s,a)); if I’m not mistaken, this should be clarified.

### Questions
see weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SAMG, a simple way to do offline-to-online RL without keeping the offline dataset during fine-tuning. It keeps the pre-trained offline critic and blends its guidance into the online value updates using a state–action–aware weight that estimates how in-distribution each sample is (learned with a conditional VAE). As training progresses, this weight (and, if useful, the offline critic) are updated so reliance on offline guidance fades where the agent has learned. Plugged into CQL, AWAC, and IQL, SAMG improves returns and especially early online sample efficiency on standard benchmarks. The theory supports stability for policy evaluation.

### Strengths
- Removes the need to keep the offline dataset during fine-tuning by guiding updates with a frozen offline critic $Q_{\text{off}}$, so it's practical and easy to plug into standard Q-learning methods.
- Uses a state–action–conditional weight $p(s,a)$ so guidance is per-sample (high when in-distribution, low when OOD)
- The weighting model is adaptive during fine-tuning: as the policy improves, reliance on $Q_{\text{off}}$ fades.
- Works as a drop-in wrapper for multiple baselines (e.g., CQL, IQL, AWAC) and consistently improves early online sample efficiency and final returns on standard O2O benchmarks.
- Clear intuition via the intrinsic-reward view

### Weaknesses
- Theoretical guarantees cover policy evaluation; there is no convergence guarantee for control with function approximation.
- The $p(s,a)$ pipeline is complex/heuristic which may affect robustness across domains.
- The adaptive update relies on low online Bellman error to tag “mastered” OOD samples and can periodically replace the “offline” critic with the current $Q$; this feedback loop may drift or mis-label samples.
- Compute reporting is light: claims of minimal overhead aren’t backed by detailed wall-clock or memory breakdowns during online fine-tuning.
- Performance is uneven in some tasks and depends on the base algorithm and the quality of the offline critic
- Slight ambiguity around “offline-data-free” when re-creating certain offline regularisers

### Questions
1. Why the specific construction of $p(s,a)$ from the C-VAE encoder instead of simpler OOD scores (e.g., reconstruction error) or density-ratio/score-based estimates?
2. What happens if you remove the hard cutoff and use a soft schedule for $p(s,a)$ below $p_m^{\text{off}}$, does this help on OOD-heavy tasks?
3. How sensitive is performance to the offline pretraining budget and to the adaptive update cadence/thresholds?
4. Please provide wall-clock and peak memory comparisons during online fine-tuning, with an explicit breakdown of the C-VAE update cost and frequency.
5. Clarify how the “offline” regularisation terms that reference $\mathcal D$ are implemented without access to $\mathcal D$ during fine-tuning. If a learned behaviour model substitutes for $\mu(a\mid s)$, how is its error controlled?
6. Can you report a failure analysis in settings where SAMG underperforms and Bellman error over time

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
4

### Summary
The paper proposes State-Action-Conditional Offline Model Guidance (SMAG) for offline-to-online reinforcement learning. The approach eliminates the need to retain offline datasets during fine-tuning by leveraging a frozen offline critic to guide online learning. A state-action-conditional coefficient, instantiated using a CVAE, adaptively determines how much to rely on offline or online critics. Theoretical analyses demonstrate convergence speed, and experiments on the D4RL benchmark show superior empirical performance.

### Strengths
- The idea of freezing the offline critic and using an adaptive coefficient to guide online updates is well-motivated.
- SAMG outperforms base algorithms (CQL, IQL, AWAC) and advanced baselines such as Cal-QL and EDIS on diverse D4RL benchmark tasks (Mujuco locomotion, AntMaze, Adroit).
- The method is compatible with various Q-function-based algorithms, allowing easy adaptation to existing RL frameworks.

### Weaknesses
I am willing to increase the score if my concerns are addressed.

- The presentation could be improved through clearer structure, tighter writing, and better figures and tables. For example,
    - The term $p_{int}$ should be clearly defined before its first use at Line 203, i.e., $p_{int}$ = Eq. (4).
    - The name "state-action-conditional coefficient" could benefit from a simpler, more intuitive description. Similarly, the method name State-Action-Conditional Offline Model Guidance may benefit from a more precise description, as I find the current name somewhat ambiguous.
    - In Figure 1, the upper part should explicitly indicate that the frozen and inherited components correspond to the Q network.
    - Captions for Figures 2-4 are somewhat difficult to read. It might be beneficial to integrate them into a single figure with three subplots, labeled (a) (b) (c).
    - Experimental settings (e.g., hyperparameters, update frequency) could be summarized in a table for clarity.
- The paper lacks some learning curve comparisons, which could better illustrate the differences in performance and learning speed in O2O scenarios than tables alone.
- The experimental analysis would be more informative if the coefficient $p(s,a)$ were visualized to provide insights into the learning dynamics.
- In offline and O2O RL, [1] mixes in-sample and out-of-sample maximum Q-values in the Bellman target to control generalization. Since the idea of mixing two different Q-values in the target is conceptually similar, it would be beneficial to include a discussion in the paper.

[1] Doubly mild generalization for offline reinforcement learning, NeurIPS 2024.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SAMG, which uses a pre-trained offline critic model for Q-value updates in online reinforcement learning, removing the need for an offline dataset. Essentially, SAMG introduces an intrinsic reward term proportional to the difference between the offline and online Q-values. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
- The proposed method is clearly present and easy to follow.
- The experiments are fairly extensive.

### Weaknesses
- Using a Conditional VAE to estimate $p(s, a)$ plays a crucial role in the algorithm's performance, but the paper lacks sufficient evidence that the Conditional VAE can accurately measure the degree of OOD in a data point. The authors should provide more experimental results to support this claim.
- The intrinsic reward $r^{in}(s,a) = \gamma p(s,a) Q^{off}(s',a') - Q(s',a')$ seems questionable. If the online Q is initialized from the offline Q, then within the offline data distribution $Q^{off}(s',a')$ and $Q(s',a')$ will be close, making $r^{in}(s,a)$ nearly zero. Outside the offline distribution, $p(s,a)$ is small, so $r^{in}(s,a)$ would also approach zero. Overall, it seems that $r^{in}(s,a)$ may have little practical effect. Intuitively, this intrinsic reward may not work as intended.

Overall, I consider this paper borderline. If the authors can clearly address my concerns, I’d be happy to raise my score.

### Questions
The intrinsic reward $r^{in}(s,a)$ essentially introduces Q-value information into the reward, allowing the reward to encode some long-horizon signal. This might be the reason why the proposed method achieves better sample efficiency. I'd like to hear the authors’ thoughts on this perspective.

### Soundness
3

### Presentation
3

### Contribution
2
