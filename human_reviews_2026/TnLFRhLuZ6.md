# Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Diffusion-based models for robotic control, including vision-language-action (VLA) and vision-action (VA) policies, have demonstrated significant capabilities. Yet their advancement is constrained by the high cost of acquiring large-scale interaction datasets. This work introduces an alternative paradigm for enhancing policy performance ***without additional model training***. Perhaps surprisingly, we demonstrate that the composed policies can exceed the performance of either parent policy. Our contribution is threefold. First, we establish a theoretical foundation showing that the convex composition of distributional scores from multiple diffusion models can yield a superior one-step functional objective compared to any individual score. A Grönwall-type bound is then used to show that this single-step improvement propagates through entire generation trajectories, leading to systemic performance gains. Second, motivated by these results, we propose General Policy Composition (GPC), a training-free method that enhances performance by combining the distributional scores of multiple pre-trained policies via a convex combination and test-time search. GPC is versatile, allowing for the plug-and-play composition of heterogeneous policies, including VA and VLA models, as well as those based on diffusion or flow-matching, irrespective of their input visual modalities. Third, we provide extensive empirical validation. Experiments on Robomimic, PushT, and RoboTwin benchmarks, alongside real-world robotic evaluations, confirm that GPC consistently improves performance and adaptability across a diverse set of tasks. Further analysis of alternative composition operators and weighting strategies offers insights into the mechanisms underlying the success of GPC. These results establish GPC as a simple yet effective method for improving control performance by leveraging existing policies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper propose a policy composition methods in imporving the policy performance in the inference time without any data collection or training. This paper conducts comprehensive experimentis in several benchmarks and real world experiments with various base policies, together with detailed theoretical analysis.

### Strengths
1. This paper proposed a novel framework in composing different policies during inference time without collecting more data or training.

2. The proposed method can handle different policies, datasets, and modalities for a task under the same environments.

3. This paper conducts comprehensive experiments and theortical analysis accross several benchmarks and tasks with various different policies.

### Weaknesses
1. Combining different policies and modalities will increase the computation cost during inference time.

2. The output actions should be the same. But in general, for different kinds of policies like DP, VLA, and flow-based policy might be good at using different length of action chunkings. It might be a limitation.

3. The ideas, results, and analysis are interesting, but not quite sure this method is pratical or not... To compose them and get performance, I need to train different policies for a same task, and even collect different data with different modalities. With the same cost of time, collecting more data can also increase the preformance.

### Questions
No questions

### Soundness
3

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
This paper introduced a training-free method for boosting diffusion-based policies performance via test-time policy composition. Theoretical analysis is provided to support the claim that convex combination of different policy distribution can improve functional objective. The proposed General Policy Composition (GPC) has been validated on various simulation benchmarks and real-world robot system.

### Strengths
1. The idea of training-free test-time policy improvement with policy composition is novel and interesting
2. The theoretical proof provides good mathematical foundation for the proposed method (I have not thoroughly read the proof in the Appendix, may require extra verification)
3. Detailed analysis and ablations are conducted in both simulation and real-world experiments.
4. The paper is well-written in general and easy to follow.

### Weaknesses
1. Based on the results, it appears that the proposed method yields a noticeable performance improvement only when both base policies perform well. As noted in Section 6.3, if either policy performs poorly to begin with, the proposed method tends to struggle and may even degrade overall performance.
2. The performance of GPC is highly sensitive to weight tuning, which limits its practical generalizability.
3. GPC’s effectiveness relies on the diversity of modalities offered by the base policies; however, there is no guarantee that such diversity exists, nor is there any further analysis on how varying modalities influence the final performance.

### Questions
1. GPC has an implicit assumption that two base policies should have different behavior distributions, how to make sure this is the case in practice?
2. How much extra inference cost is required for GPC?
3. In case of VA+VLA, how to guarantee the VA modality correctly follow the language instruction of VLA? If VA's modality is opposite to the language instruction, will GPC in fact harm the final performance?

### Soundness
3

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
This work presents General Policy Composition (GPC), a training-free framework designed to enhance the performance of robotic control policies, particularly those based on diffusion or flow-matching models, by combining existing pre-trained policies. GPC addresses the high cost and data requirements typically associated with scaling up or fine-tuning control policies. The core mechanism involves composing the distributional scores of two policies during inference, via a convex combination, utilizing a test-time search to identify optimal weighting coefficients. There is a theoretical foundation to show that a convex combination of scores yields a provably superior one-step functional objective compared to individual scores, and this improvement propagates throughout the entire generation trajectory.

Empirical validation across simulation benchmarks (Robomimic, PushT, RoboTwin) and real-world tasks shows that GPC improves performance, achieving average success rate increases of up to +7.55% in sim and +10% in real-world tasks compared to single-policy baselines. The author's analysis suggests that GPC generates more coherent and concentrated distributions throughout execution, contributing to stability. Although effective, the framework is currently limited by reliance on a fixed discretization for test-time weight search and potential computational increases when scaling beyond dual-policy composition.

### Strengths
1. Simple, effective, and theoretically-motivated idea that is reminiscent of other techniques in ML that use multiple models to get the best outcome (e.g. MoE, boosting).
1. Idea of utilizing test-time search is sound; it would help the authors to add references talking about the per-compute-unit performance benefits of using test-time compute vs. train-time compute. (e.g. arxiv.org/pdf/2408.03314).
1. The proposed GPC framework enables composition across both VA and VLA models.

### Weaknesses
1. Potentially very large inference time due to weight search. Already, robotics policies struggle with inference that is not fast enough to keep up with physics.
1. No major comparison of how much test-time compute is used against the compute used for the base policies that were composed.
1. No discussion about a potential increased test-time inference latency.

### Questions
1. How would you address giant compute overheads for running a VA/VLA model while trying to meet the real-time requirements of running policies on a real robot?
1. Fig 1: Explain the desired task here in the caption. Preferably give the task prompt or description that is input to the policies.
1. Line 361 (100-200 rollouts for evals): I’m trying to understand how often this test-time compute is calculated during rollout. What’s the obs horizon over which these trajs are created? Is it one DP inference call per episode?
1. Sec 6.2 and table 1: What is the reason for some combinations to exhibit a higher performance increase? Please give your analyses and hypotheses, if any. How much of this improvement would you attribute to causes such as choice of task, choice of underlying models, choice of modalitiies for each model, randomness during evals, and any other cause you would like to highlight.

### Soundness
4

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
The paper proposes General Policy Composition (GPC), a training-free method that improves robotic policies by composing distributional scores from multiple pre-trained diffusion/flow policies at test time. GPC performs a test-time weight search over convex coefficients of two score estimators, then samples trajectories using the composed score. Experiments on a simulated manipulation benchmark and on multiple real-robot tasks show consistent improvements over individual policies.

### Strengths
- The method provides a unified framework for combining diffusion and flow models trained on different modalities, with a flexibility to support multiple prediction types (noise, score, velocity, etc.).

- The actual algorithm is pretty simple and clear without requiring any additional training. 
- Experimental results on Robomimic, PushT, RoboTwin, and four real-robot tasks show consistent gains over each individual policy.

### Weaknesses
- My primary concern is the cost of test-time weight search. It involves querying each diffusion/flow policy several times and will likely add considerable inference overhead. How much additional time does this require in simulation and in real-world experiments? I recommend the authors include an analysis regarding this.

- Implementation details for each policy are missing. What is the revised pi0? What are the model architectures and training setups for the diffusion and flow policies? I recommend the authors include these details.

- If I understand correctly, another limitation is that policies being combined must use the same action-chunk length and the same number of diffusion steps. This requirement substantially limits GPC’s applicability to open-source generalist VLAs, which are often trained with different settings.

### Questions
- Have the authors tried to combine 3 or more policies? Curious if it can further improve the performance.
- Is there any way to combine different policy classes, such as a flow policy and a diffusion policy with this framework?

I’d be happy to reconsider my score once these points are addressed.

### Soundness
3

### Presentation
3

### Contribution
2
