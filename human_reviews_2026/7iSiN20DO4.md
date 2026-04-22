# SeWA: Selective Weight Average via Probabilistic Masking

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Weight averaging has become a standard technique for enhancing model performance. However, methods such as Stochastic Weight Averaging (SWA) and Latest Weight Averaging (LAWA) rely on manually designed checkpoint selection rules, which struggle under unstable training dynamics. To minimize human bias, this paper proposes Selective Weight Averaging (SeWA), which adaptively selects checkpoints during the final stages of training for averaging. Both theoretically and empirically, we show that SeWA achieves a better generalization. From an algorithm implementation perspective, SeWA can be formulated as a discrete subset selection problem, which is inherently challenging to solve. To address this, we transform it into a continuous probabilistic optimization framework and employ the Gumbel-Softmax estimator to learn the non-differentiable mask for each checkpoint. Theoretically, we first prove that SeWA converges to a critical point with flatter curvature, thereby explaining its underlying mechanism. We further derive stability-based generalization bounds for SeWA, which are sharper than those of SGD under both convex and non-convex assumptions, thus providing formal guarantees of improved generalization. Finally, extensive empirical evaluations across diverse domains, including behavior cloning, image classification, and text classification, demonstrate the robustness and effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
Weight averaging is known to stabilize the training and gives smoother weights. This has reportedly improved model generalization. 
This paper proposes selective weight averaging to minimize the bias introduced by existing state-of-the-art methods that rely on manually designed checkpoint selection rules for stochastic weight averaging and latent weight averaging. 
The idea is to adaptive selects checkpoints during the final stage of training for averaging.

### Strengths
Strengths
- Targetting a difficult problem
- Avoids bias introduced due to manually selecting checkpoints
- Reduces the need for extensive hyper parameter tuning
- Mitigates performance degradation caused by redundant weight selection
- Formulated adaptive selection of checkpoints as a discrete subset selection problem by transforming the problem into a continuous probabilistic optimization framework. 
- Derive stability-based generalization bounds which are sharper than SGD in both convex and non-convex optimization

### Weaknesses
Oveall, I find the idea quite interesting. I have few comments:

- I see the idea behind Figure 1, but it did not help me  understand the fact that SeWA reaches a flatter curvature (stable minima)
- Improvements reported in Figure 3 & 4 are showing marginal improvements even compared to Random

The experimentation overall seems not super convincing. But maybe theoretical guarantees claimed in the paper might be more promising. Unfortunately, this work is not in my expertise which is why I could not closely check the mathematical details. 

I would need to rely on the fellow reviewers to verify the math. Experimentally, I am not fully convinced of the benefits of the proposed approach.

### Questions
- The difference between SeWA vs SWA is not clear in the introduction. If my understanding is correct, SWA is using previous K  epochs as well. How is SeWA different in that aspect?
- Is the analysis still valid if a model is fine-tuned after being trained with SeWA?

### Soundness
3

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
3

### Summary
The paper proposes the SeWA algorithm, which is a post-training checkpoint averaging method that learns which $K>0$ of the last $k$ checkpoints to include in the final average. It relaxes the discrete selection with Gumbel-Softmax and optimizes it using Monte Carlo (iteratively). The paper provides theoretical results for evaluating generalization error in both convex and non-convex settings. It also provides empirical results over diverse domains, including behavior cloning (D4RL), CIFAR-100, and AG News.

### Strengths
* The paper provides tighter theoretical results for convex and non-convex settings (the proofs were not checked).
* The paper uses standard assumptions of smoothness and Lipschitz. 
* The experiments were tested on diverse domains: image, text, and locomotion trajectories.

### Weaknesses
* In practice, if you select the final model on validation at multiple checkpoints, you would need to run SeWA at each validation point (or at least repeatedly near the end), which can be expensive compared to the other approaches that do not require this procedure and can be used directly. The paper would benefit from reporting wall-clock time per SeWA run, relative to one training epoch, for the chosen $K$, $k$, $M$, and max_iterations in each experiment. Also, the paper should specify the number of SeWA optimization iterations (max_iteration) and the value of $M$ used per experiment. Currently, only a D4RL ablation in the appendix touches on this, but it's still hard to capture the full cost of the proposed approach.
* The paper also should discuss the storage overhead of additional $k$ checkpoints versus the other approaches, which keep a single running aggregate
* In Figures 3 and 4, the results for SeWA and LAWA appear identical across different values of $K$. Also, SGD takes longer to converge on CIFAR-100; the authors may consider more complex architectures better suited to this task. For example, in [1] (TWA), they used ResNet and VGG and achieved more stable convergence and better accuracy on TWA than SGD (around 80%).
* The results in Figures 3 and 4 do not show a significant improvement. Given the costly SWeA optimization process, averaging $k$ checkpoints at random yields results similar to those from the complete optimization and requires neither extensive hyperparameter tuning nor a costly optimization procedure. Also, EMA and SWA should be added to these CIFAR-100/AG News experiments, not only D4RL.
* The authors did not provide any supplementary material. Since no code for the empirical evaluation is available, the community cannot verify or assess the proposed approach's contribution.


[1] Li, Tao, et al. "Trainable weight averaging: Efficient training by optimizing historical solutions." The Eleventh International Conference on Learning Representations. 2022.

### Questions
1. Regarding the experiments in Figure 2, wouldn’t a fairer comparison be to average the SWA with $K$ checkpoints over the last $k$ iterations, rather than every $K$ iterations? The same for EMA: a fairer comparison is to use a weight decay of $1-K$, which effectively averages the last $K$ checkpoints.
2. Did the authors try a trainable approach as done with TWA [1]? Why not add TWA to the comparison? 

[1] Li, Tao, et al. "Trainable weight averaging: Efficient training by optimizing historical solutions." The Eleventh International Conference on Learning Representations. 2022.

### Soundness
2

### Presentation
2

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
This work addresses an adaptive weight selecting method for weight averaging method of training DNNs at the training tail stage. Specifically, it formulates a discrete scheme of determining the binary variables for choosing the checkpoints, and then leverage GS estimator to continualize the optimization to fit the training framework. Then, some analytical results are presented, so are the experimental evaluations.

### Strengths
The continuation of the weight averaging is a good idea on the class of weight averaging methods for improving the generalization ability of DNNs.

### Weaknesses
1. The results are insufficient: except Table 2, the results (e.g.g, Figure 3 and 4 for image class-action and text classification) do not show distinctive improvements so  the competitiveness  lacks convincing supports; otherwise, the performance improvement look increment in general setups.
 Besides, the evaluated network architectures are datasets can be extended for comprehensiveness.
2. As it claims its particular effectiveness on RL that may have more unstable training. Would it be possible to have particular theoretical analysis focusing on such task and settings?
3. Similar to the insight in the above bullet, at the early training stage “more unstable training” can be expected, what if SeWA is applied in such scenario? (TWA claims its advantage also at the early training stage, as it also somehow optimizes the weights. By the way, why not compare with TWA in the experiments?).

### Questions
See the weakness.

### Soundness
3

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
This paper introduces SeWA, Selective Weight Averaging. Instead of averaging the last k checkpoints by rule, SeWA learns which checkpoints to average by casting selection as a probabilistic masking problem.
In addition, this paper uses large text space for theory analysis.
SeWA is evaluated on D4RL behavior cloning, CIFAR-100, and AG News.

### Strengths
1. This paper is well motivated and has strong theoretical backing. The paper provides both flatness convergence arguments and stability-based generalization bounds, comparing favorably with SGD, SWA, and LAWA under convex and non-convex assumptions.
2. The experiments span three domains (RL, vision, and text), showing consistent improvements across varied architectures and data distributions.

### Weaknesses
1. while the paper emphasizes SeWA’s simplicity, its sample-based optimization introduces additional forward passes and probability updates that are not accounted for in the baseline comparisons. A fair comparison should including additional computational consumption on these.
2. Except for the RL experiments, most performance curves show nearly overlapping trajectories in the figures. The visual and quantitative margins are subtle in the curve plot only form.
3. The sample-based optimization uses the training objective F as the criterion (as stated in sec. 2) for selecting checkpoints. This implicitly assumes that the training and test loss landscapes are sufficiently aligned, which rarely holds in realistic non-convex regimes. Consequently, SeWA might overfit to training dynamics when the learned selection probabilities are tuned purely on training losses.
4. Since SeWA relies on sampling and averaging across multiple candidate checkpoints, it requires storing a substantial portion (or even all) of the recent checkpoints in storage consumption. This can become expensive for large-scale or long-horizon training (e.g., vision transformers, LLMs), where checkpoints are heavy and I/O-bound.

### Questions
All my concerns/questions are listed in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
