# Real-Time Robot Execution with Masked Action Chunking

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Real-time execution is essential for cyber-physical systems such as robots. These systems operate in dynamic real-world environments where even small delays can undermine responsiveness and compromise performance. Asynchronous inference has recently emerged as a system-level paradigm for real-time robot manipulation, enabling the next action chunk to be predicted while the current one is being executed. While this approach achieves real-time responsiveness, naive integration often results in execution failure. 
Previous methods attributed this failure to inter-chunk discontinuity and developed test-time algorithms to smooth chunk boundaries. In contrast, we identify another critical yet overlooked factor: intra-chunk inconsistency, where the robot’s executed action chunk partially misaligns with its current perception. To address this, we propose REMAC, which learns corrective adjustments on the pretrained policy through masked action chunking, enabling the policy to remain resilient under mismatches between intended actions and actual execution during asynchronous inference. In addition, we introduce a prefix-preserved sampling procedure to reinforce inter-chunk continuity.
Overall, our method delivers more reliable policies without incurring additional latency. Extensive experiments in both simulation and real-world settings demonstrate that our method enables faster task execution, maintains robustness across varying delays, and consistently achieves higher completion rates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a method for finetuning a pretrained VLA policy toward residual modifications i.e., adjusting an action chunk from the pretrained policy by taking into account delay due to latency. To do so, the paper introduces a curriculum learning method in which the policy first learns to imitate the expert and then learns to do residual corrections conditioned on a wide range of delay values. For sampling at test-time, the model conditions on the actions being executed from the previous chunk (due to delay) and the sampling procedure enables the model to adjust the trajectory towards more optimal actions.

### Strengths
1. The proposed method has been tested on a wide range of tasks in both simulation and in the real world. The paper offers interesting analysis at a low-level such as the kinematics of the robot under various methods (see figure 5). 
2. The paper compares the proposed method with all the appropriate baselines. 
3. The paper carries out several important ablations and offers interesting insights into how baseline methods can further uplift the performance of REMAC.

### Weaknesses
1. Lack of clarity in the problem formulation, the definition of inter-chunk discontinuity is very unclear (line 153). It is not clear why two trajectories that diverge must not have actions that are different i.e. if the trajectories are similar only up to time $t$ and then the two trajectories diverge to different observations, their action chunks at time $t + h$ should be different? Furthermore, it is not clear what this discontinuity means and what the expected behavior is if there is no discontinuity. Similarly, intra-chunk inconsistency is unclear. While the first $d$ actions are taken from the previous action chunk $\textnormal{A}_{t-h}$, I do not understand what the perception-action mismatch is. 
2. Lack of clarity in the method discussion. Generally, since there are various choices of implementing flow matching, it would be helpful to either have a preliminaries or an appendix section where you define both the predicted flow and the ground truth flow and, in your notation, specify which policy each is being sampled from. I understand that abstracting these modeling choices helps put the emphasis more on the generality of your method but, at least, some concrete examples will help the reader develop a more concrete understanding of your method. Similarly, the switch to using $\textnormal{x}_\mathrm{p}$ to denote the action seems strange. I also think section 4.2 is extremely dense and unclear, from notation to definitions (for e.g., the definition of $f$ in equation 6). 
3. While the method discusses handling temporal consistency, it is unclear whether this method handles the latency issues as discussed in the RTC paper (Black et. al. 2025). It seems like the delay-aware policy would also have high latency, so is it that your method relies on the conditioning on the delay to take care of that? 

As is clear, most of my complaints are with respect to the clarity of the discussion of the problem formulation and the method. I would be happy to raise my score if these issues with clarity are adequately answered. Apart from them, I have some questions about the scalability of the method. 

4. The method seems quite data expensive in fine-tuning. For example, for the real world tasks, you collected 200 trajectories. It would be useful to see how the performance changes as we change the size of the dataset. 
5. I am not sure that this method would generalize and I suspect that this might harm the robustness of the pretrained policy. Since you are finetuning offline, it is clear that the REMAC would do well on tasks/environments seen during finetuning – the model knows what the target expert actions are and the model knows what the pretrained policy outputs at these states in the offline dataset. This raises the question of whether or not this fine-tuned model would reliably adapt the pretrained policy’s behavior in unseen states. For example, one issue that might arise is that, at unseen states, the policy might be more uncertain leading to higher entropy of actions from the pretrained policy, and as such your finetuned policy needs to learn how to modify a large number of actions to match the optimal one (which the model does not know yet since this is at test time). Since you are already testing this on $\pi_0$ which demonstrates some generalization, it would be good to evaluate on some tasks/environments not seen during fine-tuning.

[1] Kevin Black, Manuel Y. Galliker, Sergey Levine. Real-Time Execution of Action Chunking Flow Policies. 
[2] Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling

### Questions
See weaknesses. For me, the most important questions are with regards to clarity and handling latency (see points 1-3 in Weaknesses section).

### Soundness
3

### Presentation
1

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
The paper proposes REMAC for asynchronous inference in real-world robotics, handling chunk discontinuities by finetuning learned policies via flow matching. The key is to condition the policy on a ground-truth action prefix during training and use flow matching to predict the remaining actions. By randomly sampling delay d and adjusting the flow matching curriculum, REMAC trains the policy to adapt to varying inference delays. Experiments on both simulation and real-world benchmarks show that REMAC outperforms baselines under high inference delay.

### Strengths
1. REMAC can adapt to various inference delays d with a single training process, without needing to retrain for each delay. The use of LoRA modules maintains model performance while reducing training overhead.  
2. Flow matching enables the model to learn fine-grained continuity between action prefixes and optimal future actions, capturing the dependency between earlier and later actions.  
3. REMAC achieves strong results across both simulated and real-world benchmarks.

### Weaknesses
1. REMAC has structural requirements on the dataset, which needs to contain diverse future action sequences from the same observation. With this requirement, the model can learn to correct deviating action prefixes resulting from inference delay. Without local adjustment samples in the dataset, REMAC may perform no better than behavioral cloning.  
2. During inference, the agent must know how long inference takes in order to select an appropriate prefix length d from the previous action chunk. In real-world settings with variable communication delays, accurately estimating this delay may be challenging.  
3. REMAC improves policies' robustness to inference delays only for flow-matching-based policies. If the bottleneck policy is not trained with flow matching, REMAC requires training from scratch.

### Questions
1. I'm confused why REMAC outperforms the pretrained policy (Naive) even when d = 0, according to Figure 2. Is it because REMAC uses additional expert data compared with Naive? If the baselines (Naive, RTC, BID) also access the additional expert data, will they outperform REMAC?  
2. Can we apply the same dataset used in the pretrained policy (i.e., Naive) to REMAC, so that REMAC does not use additional expert data?  
3. Can you further explain the Residual Alignment loss Eq. 4? What's the difference between Eq. 4 and the Prefix Masking loss Eq. 2?  
4. In the simualtion tasks, how is the expert dataset D used for REMAC constructed?

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
3

### Summary
This paper studies real-time robot control with chunk-based VLA policies under asynchronous inference, where inference latency causes the robot to execute stale actions while new ones are being generated. The authors highlight a neglected failure mode—intra-chunk inconsistency—in addition to the known inter-chunk discontinuity issue. They propose REMAC, a training-time strategy that applies prefix masking to simulate partial stale-action execution, a self-conditioned curriculum to mitigate exposure bias, and residual LoRA finetuning to correct the base policy. The method increases robustness under varying latencies without adding inference overhead. Experiments on Kinetix and a real Franka arm show improved success rates and smoother execution across delay conditions.

### Strengths
- Identifies and formalizes intra-chunk inconsistency as a distinct source of degradation in asynchronous execution.
- Simple and computationally efficient method that is compatible with existing VLA architectures.
- Strong empirical evidence across simulation and real-robot settings, including latency sweeps and ablation studies.
- No additional inference latency, in contrast to recent test-time smoothing approaches (e.g., RTC).
- Parameter-efficient finetuning strategy that preserves the backbone’s capability.
- Compatible with BID/RTC test-time methods, demonstrating good composability with existing techniques.

### Weaknesses
- The method assumes access to accurate and bounded delay estimates. It is unclear how robust the approach is when latency measurements are noisy, rapidly fluctuating, or adversarially spiky.
- Training samples delays uniformly, but real-world latency tends to be bursty and temporally correlated. Additional evaluation under realistic network- and compute-induced delay profiles would strengthen the claims.
- Finetuning with masked actions may shift the behavior of the underlying VLA model. The paper does not report out-of-distribution or language generalization results, making it difficult to assess whether broader generalization and grounding capabilities are preserved.
- The method is demonstrated only on flow-matching VLA architectures. Since REMAC integrates with the sampling process and residual flow fields, it is unclear whether the approach directly applies to non-flow controllers (e.g., autoregressive VLA policies or transformer-based continuous action models). A discussion or preliminary evidence on generality across policy parameterizations would strengthen the paper.

### Questions
1. How robust is REMAC to inaccurate delay estimates or rapidly varying latency?
2. Does delay-aware finetuning impact generalization to unseen tasks/objects or language inputs?
3. Would a continuous or learned delay embedding outperform discrete integer conditioning?
4. How does the method behave when real latency exceeds the maximum trained delay?
5. Can the approach be extended to handle coupled observation latency, not just action delay?
6. The method uses a fixed chunk length, but in practice the optimal horizon may vary with latency and task demands. Can the approach be extended to dynamically adjust chunk length based on system delay or task complexity?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses real-time robot execution with action chunks. The authors identify _intra-chunk inconsistency_ as a previously overlooked issue. They propose a method for learning a delay-aware policy by masking the portion of the prefix that restricts supervision to only the executable slice of each chunk. The proposed method incurs no additional latency.

### Strengths
- The paper makes solid contributions to an important practical problem. Asynchronous inference is a natural system-level solution for real-time execution. Identifying intra-chunk inconsistency as a distinct failure mode adds value.
- The paper does a thorough experimental analysis, comparing with the state-of-the-art method RTC, and doing ablation studies on each component of the proposed method.
- Figure 1(c) provides a great concrete example of the problem the paper is aiming to solve.

### Weaknesses
- The paper cites intra-chunk inconsistency as a core motivation for the proposed method, but doesn't provide any direct evidence that this is a major issue.
- Is there a way to apply this idea to policy classes other than flow-matching?
- I don't understand the residual alignment term. Don't the two $\tilde{u}$ terms in eq (4) cancel out, making it equivalent to (2)? Is there a typo somewhere, or am I missing something?
- The method requires specifying d_max as a hyperparameter during training. Do you have a sense of what happens when the model encounters delays longer than d_max? Does performance degrade gracefully or fail catastrophically?

### Questions
Please see the weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
2
