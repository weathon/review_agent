# Taming Curvature: Architecture Warm-up for Stable Transformer Training

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Training billion-parameter Transformers is often brittle, with transient loss spikes and divergence that waste compute. Even though the recently developed Edge of Stability (EoS) theory provides a powerful tool to understand and control the stability of optimization methods via the (preconditioned) curvature, these curvature-controlling methods are not popular in large-scale Transformer training due to the complexity of curvature estimation. To this end, we first introduce a fast online estimator of the largest (preconditioned) Hessian eigenvalue (i.e., curvature) based on a warm-started variant for power iteration with Hessian–vector products. We show theoretically, and verify empirically, that the proposed method makes per-iteration curvature tracking feasible at billion-parameter scale while being more accurate. Using this tool, we find that training instabilities coincide with surges in preconditioned curvature and that curvature grows with depth. Motivated by these observations, we propose architecture warm-up: progressively growing network depth to carefully control the preconditioned Hessian and stabilize training. Experiments on large Transformers validate that our approach enables efficient curvature tracking and reduces instabilities compared to existing state-of-the-art stabilization techniques without slowing down convergence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work the authors develop a technique called "architecture warmup" to control curvature. In short, they initialize many layers to 0, and slowly bring them online into training. Due to the residual connections the initial reduced network trains well, and the new layers are brought online continuously. The authors use a warmstarted power iteration to measure sharpness and show that this method initializes at lower sharpness, and ends the procedure at higher sharpness. They show that this can allow for training at larger initial learning rates, and that it can obtain similar training curves to traditional warmup.

### Strengths
The idea of architecture warmup is simple and, to my knowledge, novel; I admit that I am not familiar with the literature on growing networks so I will defer to other reviewers on this point. Overall the simplicity of the idea and its general compatibility with common training schemes is a big strength. The experiments seem sound to the best of my ability to understand them.

The idea of warmstarting the power iterations is also simple yet effective, and I appreciate the authors for running experiments demonstrating its utility as a measurement tool.

### Weaknesses
The main question is whether or not the proposed architecture warmup provides benefits that are better than learning rate warmup. Of particular interest is the case of the large transformer models where depth and width are often scaled simultaneously. Indeed the resulting learning curves are quite similar to transformers trained with warmup.

There is a broader point that simply allowing learning with higher learning rates is not in and of itself a win; for example if the gradient norms are smaller than the effect of a larger learning rate can be cancelled out. This suggests that a deeper investigation of the method is needed to understand why it might be useful.

One potential issue with the curvature measurements: typically experimentalists do power iteration on a small batch of data to compute an estimate of the large Hessian eigenvalues. However, this does not give an unbiased estimator of the full batch Hessian, and also may give a noisy estimator of the batch-Hessian maximum eigenvalue. This makes interpreting the curvature measurements in the paper more difficult.

### Questions
Do the minibatch curvature measurements well-reflect the full batch curvature statistics? Can evidence be provided using e.g. HVP aggregation over multiple batches?

Is there any way to test whether or not networks trained with architecture warmup have benefits in the fine-tuning setting, where the curvature of the pretrained model may be more predictive of downstream success?

Suggestion: for the warm started curvature measurement, it might be better to spend less time justifying the procedure, and more time describing the experiments.

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
4

### Summary
This paper addresses training instabilities in large-scale Transformers through curvature control. The authors propose: (1) an efficient online estimator for the largest (preconditioned) Hessian eigenvalue using warm-started power iteration, reducing computational cost from >20 to <5 HVPs; (2) an "architecture warm-up" strategy that progressively increases network depth to maintain the stability criterion η·λmax(Gt) within safe bounds. Experiments on models up to 3B parameters demonstrate improved stability compared to QK-Norm, Softcap, and QK-Clip.

### Strengths
1.	Solid Theory: Provides formal analysis connecting Edge of Stability theory with practical deep learning, with non-trivial theoretical bounds
2.	Practical and Simple: Easy integration into existing pipelines without architectural surgery or complex hyperparameter tuning
3.	Comprehensive Experiments: Multiple datasets (FineWeb, DCLM, OLMo-Mix), various model sizes (640M-3B), and informative ablations
4.	Empirical Validation: Figures 1-2 effectively validate theoretical predictions with proper error bars
5.	Reproducible: Sufficient implementation details with publicly available datasets

### Weaknesses
1.	Limited Scale: Maximum 3B parameters tested; no validation on models >10B where instabilities are most severe. The "billion-parameter scale feasibility" claim needs larger-scale demonstration.

2.	Lack of Statistical Rigor: Most experiments show single runs without error bars or significance testing. Table 1 improvements need statistical validation.

3.	Limited Evaluation: Only validation perplexity reported; no downstream task performance to verify that stability improvements translate to better capabilities.

### Questions
1.	Computational Cost: Can you provide concrete wall-clock time measurements and memory overhead for 1B and 3B models? What percentage overhead compared to standard training?
2.	Gradient Clipping: How does your method compare against well-tuned gradient clipping? Can they be combined?
3.	Initial Depth Selection: How sensitive are results to starting depth (1/3 vs 1/2 vs 2/3)? Can you provide a principled selection rule?
4.	HVP Iterations: How does the required number of HVPs vary across training stages and model sizes? Should this be adjusted dynamically?
5.	RMSNorm Initialization: What specifically goes wrong with RMSNorm from zero initialization? Did you try alternatives?
6.	Failure Analysis: In Table 1, can you analyze the curvature behavior for methods that diverged (*)?
7.	Compute-Optimal: Can you provide compute-optimal comparisons with all baselines (not just QK-Norm)?
8.	LR Warm-up Replacement: Is Figure 6's result generalizable across model sizes and datasets, or specific to this setting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper makes two contributions. One is a to warm-up power iteration schemes to track the largest eigenvalue of the Hessian. The second is to propose architecture warm-up: unlocking more and more layers across training to account for natural increase in sharpness.
The warm-up scheme is quickly tested and we see that the largest eigenvalue of the Hessian evolves slow enough for the warmups to be efficient. The architecture warm-up appears to be an effective technique to improve stability during training. In particular the authors argue that it could replace the usual warm-up.

### Strengths
- The proposed tool is imple but could be of use. We hope that the authors will do their best to open source the code in all relevant platforms.
- The architecture warm-up is interesting. It is reminiscent of optimization techniques for deep learning that learned one layer at a time.
- Experiments are done at a relatively large scale. 
- The improvements in stability are somewhat surprising. They also contribute to a very different approach to improve stability in training.

### Weaknesses
- The theoretical justifications are somewhat off: the spectrum is bounded above by something that depends on the number of layers. It is not bounded below. So these justifications are unclear.
- The experiments are strangely presented (see questions below). Many baselines diverge. One culprit seems to be the numver of warmup steps (2000) that does not look standard for the baselines.
- The authors claim that architecture warm-up could replace standard warm-up. Such a claim requires detailed experimental setup: how were all hyperparameters tuned (so learning rate, weight decay, warm-up steps).
- The authors claim that spacing of unlocking does not matter for the architecture warm-up but there must be extreme cases where it does. Unlocking every step would not work for example. 
- The largest eigenvalue fo the Hessian generally drives dynamics mostly in the full batch case. The stochastic case may be governed by different mechanisms [1]. So even though architecture warm-up appears to work. It is unclear whether the authors have revealed why yet.
- Aggressive warm-up or even learning rate tuners can also directly enforce a smaller curvature [2]. So, if curvature is the leading reason for improved training curves, there could be other ways to drive the network towards more stable regions.

[1] Agarwala, A. and Pennington, J., 2024. High dimensional analysis reveals conservative sharpening and a stochastic edge of stability. arXiv preprint arXiv:2404.19261.
[2] Roulet, V., Agarwala, A., Grill, J.B., Swirszcz, G., Blondel, M. and Pedregosa, F., 2024. Stepping on the edge: Curvature aware learning rate tuners. Advances in Neural Information Processing Systems, 37, pp.47708-47740.

### Questions
- Why do the lambda max plots have a y-axis from 0 to 1? They could be larger than 1. And the plots should be in log-scale. If the plots were for example in log-scale or better centered, could we see to what value the preconditioned hessian converges (in the architecture warmup case)?
- Most importantly, can the authors provide a plot of the largest eigenvalue of the preconditioned hessian with architecture warm-up? We cannot see its evolution in the plots for now (too small). It would be great to see that evolution with the exact times when the architecture is unlocked and to which scale each time. It would be great to see whether progressive sharpening increases/decreases with the increased depth. 
- The authors say line 366, "progressively unlock additional layers only when the effective curvature remains below within the stability margin". What does that mean? As long as the edge of stability is not reached, we could unlock the architecture. The sentence above does not reflect a trigger or a "only when".
- As said in the weakness section, it would be great to have the detailed experimental setup for figure 6: how were hyperparameters tuned, how robust is this observation, could there actual be limitations to architecture warm-up etc...

### Soundness
2

### Presentation
3

### Contribution
3
