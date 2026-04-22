# ADM-v2: Pursuing Full-Horizon Roll-out in Dynamics Models for Offline Policy Learning and Evaluation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Model-based methods for offline Reinforcement Learning transfer extensive policy exploration and evaluation to data-driven dynamics models, effectively saving real-world samples in the offline setting. We expect the dynamics model to allow the policy to roll out full-horizon episodes, which is crucial for ensuring sufficient exploration and reliable evaluation. However, many previous dynamics models exhibit limited capability in long-horizon prediction. This work follows the paradigm of the Any-step Dynamics Model (ADM) that improves future predictions by reducing bootstrapping prediction to direct prediction. We structurally decouple each recurrent forward of the RNN cell from the backtracked state and propose the second version of ADM (ADM-v2), making the direct prediction more flexible. ADM-v2 not only enhances the accuracy of direct predictions for making full-horizon roll-outs but also supports parallel estimation of the any-step prediction uncertainty to improve efficiency. The results on DOPE validate the reliability of ADM-v2 for policy evaluation. Moreover, via full-horizon roll-out, ADM-v2 for policy optimization enables substantial advancements, whereas other dynamics models degrade due to long-horizon error accumulation. We are the first to achieve SOTA under the full-horizon roll-out setting on both D4RL and NeoRL. The code is available at https://github.com/LAMDA-RL/adm2.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a modification of the Any-step Dynamics model by Lin et al. Contrary to prior work, the authors propose to remove the per-step conditioning of the hidden state on the initial state in the dynamics model roll-out. They show that this leads to empirically better performance than comparable baselines, and give some theoretical insight into the performance of k-step models.

### Strengths
The modification the authors propose is a simple change to the prior work, ADM, removing a conditioning of the dynamics model latent state on the first state in the trajectory. To clarify: I count simple yet impact-full changes as a strength, I think it is a good idea to thoroughly analyze simple and clear design choices.
Experimental evidence supports the benefit of this design decision.
The authors also provide theoretical insight into the benefit of k-step model training in model-based RL that may be of independent interest to the community (see notes below).

### Weaknesses
The theoretical contribution is somewhat disjoint from the modification presented here. While it is reasonable to prove that a k-step model can lead to better performance than a one-step model, this proof applies to both the prior work and this work. I think some more targeted analysis of the specific design decision proposed here would have made the paper much stronger. This weakness is currently my main reason for recommending rejection: I do not think that the theoretical contribution serves to justify the design decision proposed here.

Building on this, I overall do not think the paper provides a strong enough intuition for why removing the conditioning on the state helps empirically. I think some more targeted investigation (theoretical or empirical) would greatly strengthen the paper. For example, the analyses presented in Figure 4 do not show ADM-v2 performing significantly better than ADM, yet it leads to farther higher reward. Therefore, investigating e.g. the generalization properties of the method in more detail seem highly relevant to clarify what the modification is doing to the community. Similarly, the analysis in Figure 5 is not tied to performance in the offline RL task, making it hard to assess the impact of an improved correlation.

Some results do not align between prior work and this work. For example, the ADM paper reports a Correlation of 0.98 on the hopper-medium task, while this paper reports 0.871. The score reported in prior work outperforms the performance reported here, and I think it is important for the authors to comment on this. Is this explained by the horizon of the rollout?

### Questions
The theoretical performs bound indeed shows a tighter dependency of the error on the roll-out horizon by a factor of $(1-\gamma^m)$. However, this assumes that the m-step error can be controlled to be the same value as the one step error. While this seems to empirically be true in the (very low noise) environments addressed here, I don't see a principled reason for why this should always be the case. Could the authors comment on this? Effectively, is there there a reason to believe that $max_k \delta^k$ does not depend unfavourably on $k$. Basically, is it always reasonable to assume that we can get the k-step error as small as the 1-step error.

While the authors propose full horizon rollouts, they do not test their model as a short horizon model. However, the baseline ADM model performs better as a short horizon than as a long horizon model. Does ADM-v2 actually provide a benefit from doing full horizon rollouts, or would the returns increase even more if it was used at a short horizon? This is a vital experiment that would benefit the narrative of the paper.

### Soundness
3

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
3

### Summary
This paper proposes ADM-v2 (encoding the initial state (s_0) as a hidden state (h_0), combining any-step prediction with PARoll parallel roll-out, and using any-step uncertainty as a conservative penalty). The approach is direct and feasible in engineering, demonstrating significant advantages in full-horizon generation and evaluation of offline RL. The paper has practical value, but currently lacks several key ablation techniques, details, and more rigorous robustness experiments.

### Strengths
1.Focusing on error accumulation and parallelization bottlenecks in long horizon roll-out, which are core pain points in offline MBRL.
2.Encoding (s_0) as (h_0) and removing backtracking, the parallelization approach (PARoll) is easily implemented in engineering.
3.The any-step supervised design significantly improves sample utilization, which is a reasonable path to enhance long-horizon capabilities.
4. Includes roll-out error curves, throughput (samples/sec), offline evaluation (DOPE), and downstream policy performance (D4RL/NeoRL), providing multi-dimensional evidence.
5. Using any-step variance as a conservative penalty is practically meaningful for offline learning security.

### Weaknesses
1.The design of compressing (s_0) into (h_0) is highly dependent on encoder selection and capacity. The paper lacks sufficient ablation and quantitative analysis of this (e.g., the impact of different encoder capacities, aux losses, and skip connections).

2.Relies on ensembles, but lacks horizon-wise calibration evaluation (the relationship between ensemble variance and true error should be shown more systematically); the computational/memory overhead and trade-offs of ensembles are not adequately discussed.

3.The paper claims a significant increase in throughput, but lacks reproducible PARoll pseudocode, batch organization examples, and memory consumption analysis; throughput comparisons may also be affected by implementation optimizations (a fairer baseline setting explanation is needed).

4.Most metrics are based on open-loop any-step evaluation, but the final policy will be implemented under closed-loop conditions; the paper should supplement closed-loop rollout results or use hybrid training (scheduled sampling / one-step + any-step) to validate stability.

5.Lacks robustness testing for insufficient data coverage or policy distribution drift (e.g., perturbations to initial states or actions, domain-shift testing).

6. The paper needs to provide a more complete list of hyperparameters (ensemble size, k-sampling strategy, encoder/GRU structure, number of training steps, seed), as well as a commitment to release the code/training script.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work creates an extension of the Any-step Dynamics Model (ADM, ICLR2025). This is a model-based RL method that creates a dynamics model that can directly predict m steps forward by feeding in a sequence of the next m actions (there existed some previous works that did this, but the ADM paper made it more practical). The ADM model has a GRU recurrent network core that has a transition decoder to predict the states from the GRU internal state. ADMv2 modifies this by adding a State encoder before the GRU, so that the state can be embedded beforehand into a better representation. This is trained end-to-end. Another addition is the PARoll prediction mechanism to make multiple forward predictions more practical compared to the method used in the original ADM. To give a bit of background: as ADM allows predicting for any number of steps forward (less than m steps forward), we can construct multiple different predictions for the next state, based on from how many steps backward we will aim to predict the next state. ADM uses these multiple predictions to form an uncertainty estimate of for the next state prediction akin to ensemble methods. However, the original ADM performed this ensemble-like prediction in an inefficient manner, so the current work developed a PARoll framework, that essentially keeps m separate prediction streams for the different forward prediction lengths and runs these in parallel.

For offline policy learning, the uncertainty estimate (specifically the variance) is added to the rewards in the q-updates as a penalty with $-\beta$ multiplier. They also give some theoretical performance bound.

Experimentally, the work evaluates on D4RL, NeoRL offline RL benchmarks, and on the DOPE off-policy evaluation benchmark. The method achieves new state of the art results on D4RL and NeoRL and also had the top results on DOPE. They achieve these results while doing full episode rollouts with their model (i.e. 1000 step rollouts, starting from sampled states from the buffer).

In addition, they tested points such as the model prediction accuracy (which improved), rollout throughput (which improved greatly compared to ADM, but was lower than the ensemble-based method), and also checked correlation between model errors and the uncertainty quantification (there was a good correlation for their method).

### Strengths
- The empirical performance appears very strong.

- The writing was good.

- Many metrics were tested, e.g., accuracy, uncertainty quantification, multiple benchmarks.

### Weaknesses
- Novelty is that not high as it mostly an extension of ADMv1.

### Questions
One way to strengthen the paper would be to also evaluate on OGBench:
https://arxiv.org/pdf/2410.20092

Regarding multi-step predictions, there are also other earlier related works:

Learning Awareness Models, Brandon Amos et al. (ICLR2018)
https://arxiv.org/pdf/1804.06318

Learning Latent Dynamics for Planning from Pixels, Hafner et al. (ICML2019)
https://arxiv.org/pdf/1811.04551

On the other hand, you already cited the works of Asadi (2018). But I think it would be better to cite such a work in the introduction, rather than in the appendix.

The original ADM paper also evaluated such correlations, and in their case, the ensemble model correlations are better than the ones reported in this paper. Where did the discrepancy come from?

### Soundness
3

### Presentation
4

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
The authors propose ADM-v2, a novel dynamics model architecture that builds upon the "any-step" direct prediction paradigm of the original ADM. The key innovation is decoupling the initial state from the recurrent processing of the action sequence. ADM-v2 allows full-horizon roll-outs and achieves new state-of-the-art performance on both D4RL and NeoRL benchmarks.

### Strengths
[S1] ADM-v2 achieves high empirical performance while maintaining consistent uncertainty estimation and fast dynamics rollout. It also supports full-horizon expansion, which opens up new possibilities for learning long-horizon tasks.

### Weaknesses
[W1] The evaluation only consists of MuJoCo locomotion tasks, which are mostly reactive. Model-free and model-based methods with short-horizon prediction also achieves high performance on MuJoCo. It is unknown that if full-horizon rollout is required.

[W2] This work is an incremental (albeit highly effective and important) improvement focused on a refined architecture and a new roll-out algorithm.

### Questions
[Q1] How does ADM2PO (short, limited horizon rollout of ADM2) perform compared to the reported ADM2PO-fh (full horizon rollout)?

[Q2] How does ADM2PO perform in tasks that truly require long-horizon rollout and planning?

### Soundness
4

### Presentation
4

### Contribution
3
