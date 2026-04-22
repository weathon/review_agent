# Towards Policy-Aware World Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
World models have received significant attention from the robotics and computer vision community, both of whom have started scaling to networks comprising billions of parameters in the hope of unlocking new robot skills. In this paradigm, models are pre-trained on internet-scale data and then fine-tuned on robot data to learn policies. However, it is still unclear what makes a good world model for downstream policy learning. We show that world model prediction loss is in many instances uncorrelated with policy performance, forcing practitioners to train models to completion for correct evaluation. This results in slow, costly iterations of model training and policy evaluation. In this work, we demonstrate that the expected signal-to-noise ratio (ESNR) of policy gradients provides a reliable training-time metric for downstream policy performance. This provides a handle on the world model's policy awareness, which denotes how well a policy can learn from a model. We show that ESNR can be used to understand (1) when world models are sufficiently pre-trained, (2) how architecture changes affect downstream performance and (3) what is the best policy learning method for a given world model. Crucially, ESNR can be computed on-the-fly with minimal overhead and without a trained policy. We validate our metric on traditional architectures and tasks as well as large pretrained world models, demonstrating the practical utility of ESNR for practitioners who wish to train or finetune such models for robot applications. Visualizations and code available here: https://policy-aware.github.io/paper-anon.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Expected Signal-to-Noise Ratio (ESNR) as a training-time metric for evaluating world models with respect to downstream policy learning. ESNR measures how well a policy can learn from a world model by quantifying the signal-to-noise ratio of policy gradients. Originally proposed for first-order policy gradients in model-based RL from scratch, ESNR is here extended to arbitrary policy gradient estimators and to pretrained world models. The paper evaluates ESNR across various settings: different policy gradient methods, world model architectures, and pretrained models.

### Strengths
- Problem importance: The work targets an increasingly relevant problem: how to assess world models efficiently for MBRL as scaling costs rise. A predictive, policy-aware metric is indeed needed.
- Clarity of the core idea: The main motivation and intuition behind ESNR are straightforward.
- Practical utility: ESNR can be computed without real-environment rollouts or a trained policy, making it computationally efficient and potentially valuable for large world models.
- Empirical breadth: The experiments span multiple policy extraction classes (Zeroth-, First-order gradients, MPC-based methods) and world model types (from DreamerV3 to large pretrained visual encoders), showing effective results.

### Weaknesses
1. **Limited novelty over prior ESNR work:**  
   The conceptual novelty is modest. Prior works, TPX (Parmas et al., 2023b) and PWM (Georgiev et al., 2025), already proposed and used ESNR to analyze and improve policy gradient estimators in RL from scratch. The core contribution here lies mainly in extending the usage of ESNR to pretrained world models and other policy optimization methods, which is incremental rather than conceptual.
   
2. **Mathematical clarity and notation issues:**  These disrupt readability and please see my questions below.

3. **Lack of theoretical justification for cross-method comparisons:**  
   It remains unclear whether ESNR can validly compare different *policy optimization methods* (e.g., FoG vs. ZoG vs. MPPI), given their qualitatively different gradient behaviors. The “toy example” (Fig. 2) shows FoG and MPPI achieving similar final returns but very different ESNR values, which undermines interpretability. Some theoretical insight into why ESNR should generalize across such gradient estimators will be very helpful.

4. **Missing baselines against model-only metrics:**  
   When comparing world models trained with the same policy method, the paper should benchmark ESNR against traditional model quality metrics, e.g., pixel-based (MSE, PSNR) or feature-based (LPIPS), to demonstrate ESNR’s unique predictive power. As shown in Fig. 3, model loss can rankly correlate with policy success rate better than ESNR. 

5. **Unclear or contradictory empirical results:**  
   - **Biased region and negative trends:** The U-shape ESNR curve and its “biased region” (Fig. 3 and 5) suggest that ESNR may mislead early in training. This undermines its reliability as a general training metric.  
   - **Inconsistent intra-method correlation:** In Fig. 4, ESNR correlates across policy classes, but within a single method (e.g., ZOG or DreamerV3) the trend appears weak or even negative. Reporting quantitative correlations (Pearson/Spearman) would clarify these claims.  

6. **Lack of discussion on out-of-distribution (OOD) behavior:**  
   ESNR estimation could be inflated by gradients on OOD actions from offline dataset, making it unreliable for deployment. The paper does not address this limitation.

### Questions
**Mathematical:**
- What is $\mathcal{B}$?  
- In Eq. 1, the decoder should output $z_{t+1}$, not $s_{t+1}$; may mark “optional” to decoder for reconstruction-free setups.  
- In Eqs. 1–2, clarify whether $s_t$ should be $o_t$; what is $\hat{s}_{t+1}$?  
- Eq. 1 only has $\phi$, while Eqs. 2–3 introduce $\psi$, $\theta$.  Please define all parameters consistently.  
- Reconstruction-based methods often include reward/value prediction and latent consistency; Eq. 2 should reflect this.  
- Eq. 4: should $r$ be $R_\phi$?  
- It is better to clarify whether the policy is trained using *only* the world model or a mixture of model/offline/online rollouts in the background section. 
- How are initial states sampled? What is the distribution of $z_t$? The planning horizon cannot be infinite.  
- Why is Sutton et al. (1999) described as first-order?
- Eq. 6 has nested expectations over $a_t$.  
- Eq. 9 overloads $J(\cdot)$ for both policy and action-sequence objectives.  
- Eq. 10: since $\nabla_\theta J(\theta)$ is vector-valued, how is ESNR reduced to a scalar?  
- Code 1: why does `grad_f` return an additional dimension of $N$? Where does stochasticity arise if dynamics are deterministic?  
- L307: should a constant gradient estimator yield ESNR = 0 (due to the small epsilon)?  

**Conceptual:**
- Please justify why replacing LayerNorm + ReLU with Mish + SimNorm reduces regularization in modified TD-MPC2.  
- Clarify what “performance” refers to—policy success in real environments or within the learned model? Both should be compared to ESNR.

Minor: 
- Abbreviations like PVR, PWM, and SNR are not introduced clearly.  
- Any citations supporting the claim that first-order gradients have lower variance? If this is true, does this make it unfair to compare FoG with ZoG for ESNR as there is a systematic trend?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to understand world model's policy training readiness with expected signal to noise ratio, i.e., the expected ratio of expectation squared and variance of policy gradients. It justifies this metric's usefulness by showing the correlation of ESNR with the average rewards.

### Strengths
- The problem this paper tackles is indeed a known problem and a surrogate solution is justified. Evaluating world models with policy training is wasteful and slow. Having a way to know how "trainable" a world model could be helpful. 
- I appreciate the toy example mentioned in the paper in Figure 2. It was useful to understand the problem from the authors' perspective.
- The metric is easy to calculate.

### Weaknesses
- I am not convinced that the ESNR metric is _actually_ predicting what we want to know from the world model, i.e., whether a trained policy on the world model would be useful. I feel it is still correlated with the accuracy of world model. Would the authors be able to show the clear example where the world model accuracy is fixed (**not** by training epochs) but the loss landscape is different and ESNR still correlates with rewards?

- Equation 2: Reconstruction based world models also learn from minimizing the rewards, e.g., dreamer. Authors include in equation 3 for reconstruction-less world models

- Equation 2 seems sloppy. What is $\hat{s_{t+1}}$? I think author meant to write $\hat{o_{t+1}}$ for reconstructed image? Also, by authors' definition, encoders take $o$, not $s$, but the authors use $E(s_t)$ in the equation.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper explores the question of how to determine whether a world model is good enough to be used for "policy extraction" - learning a policy using the world model through a number of possible model-based reinforcement learning methods. A sure fire way to answer this question is to train a policy with the given model, and evaluate the policy's performance. However, training and evaluating an RL policy end-to-end is prohibitively expensive, especially considering the increasing costs of training large world models on internet-scale data. The paper argues that the expected noise to signal ratio of the policy gradient induced by a world model is a reliable indicator of its effectiveness downstream for policy learning. Given its relatively low compute cost, the author's perform many experiments in simulated robotics environments and note how the ESNR of the policy gradient correlates with downstream policy performance across different world models and policy extraction methods.

### Strengths
The paper presents its results and method very clearly, and the structure of the paper is easy to follow. The authors uses a diverse set of world models, environments, and policy extraction methods to make a point, which is necessary due to the experimental nature of the paper, which relies on observations based on correlations. The idea of using the ESNR of a policy gradient induced by a world model is an interesting one that has not been extensively studied to evaluate policy-aware world models, especially when it comes to large pre-trained world models.

### Weaknesses
* It is generally well-understood that low gradient variance is imperative to effective downstream learning in machine learning (https://arxiv.org/pdf/2007.04532) and particularly so in reinforcement learning (https://arxiv.org/pdf/2011.09464, https://arxiv.org/pdf/1912.02503) and model-based RL (https://proceedings.mlr.press/v202/parmas23a/parmas23a.pdf) as mentioned in the paper. In fact, the return landscape in RL has also been explored (https://arxiv.org/pdf/2309.14597) as a metric for effective policy updates. The claim that "a general, applicable metric that predicts downstream policy performance across heterogenous world model architectures and policy extraction methods as been lacking" (line 119) is too strong, as the above references show that examining gradient variances (or ESNR as a form of normalized variance) is already an existing metric for evaluation. However, this paper does provide additional empirical evidence for this to be an effective metric for policy-aware world models, it does not suggest a novel metric. 
* While the reasoning and intuition behind their method is well motivated, the absence of theoretical evidence requires more robust statistical results. There is no clear evidence why their proposed metric is more effective at predicting downstream performance than existing metrics. See questions for additional details.
* It would be nice to see further comments and discussions around the downstream performance of different world models, and policy extraction methods to better understand why certain methods or models exhibit better or worse ESNRs. For example, in line 364, the authors note that "the gradients [of FoG] will have lower variance and greater expected norm when the objective is well defined". Is this true in the experiments? Why speculate when the data should be readily available? CEM is also noted to have the best performance without further explanations.

### Questions
* What makes the ESNR in particular a good indicator of performance? Since the ESNR is the ratio of the scale of the gradient and the variance of the gradient, would either the numerator (scale of policy gradient) or denominator (variance of policy gradient) alone also correlate with the downstream performance? What about the model predictive loss? Figure 3 seems to suggest that the downstream policy success rate is actually also well-correlated with the world model loss. It would be interesting to juxtapose the model loss with downstream performance in all results where possible, since that is how most practitioners evaluate a world model's effectiveness before using it for policy extraction.
* In line 411, the authors note that their experiments "underscore the importance of objective landscapes induced by world models" and that "ESNR mirrors this gap". It would be interesting to expand upon this observation, and perhaps show additional evidence for this mirroring. 
* The observation of the U-shape of the ESNR during world model training is interesting, but problematic for the message of this paper. It suggests that the ESNR alone is sufficient to predict downstream performance, since lower ESNR actually results in better performance when in the biased region of the training. Is there a way to know in what region we're in if we only have a single snapshot of a pre-trained world model? The results in Figure 5 are also contradictory to the hypothesis of the paper -- the lowest ESNR actually resulted in the best policies, even in the case of TD-MPC2 which seems to have crossed beyond the boundary of the biased region. 

### Minor comments
- Typo in line 95: the generative **capabilities** of [...]
- Confusion in notation in section 3.1: what is $s_t$ and $\hat{s}_t$ are not properly defined. 
- Some of the formalism introduced in section 3 is not necessary for the main paper, and introduces unnecessary complexity. The exact mathematical formulations of loses, world model components, different policy gradients and online planning methods is not referenced or discussed again in the results. Consider simplifying and relegating details to the Appendix unless the content is necessary to discuss results. 
- The ESNR metric used for experiments, as provided by the pseudo-code does not match the actual ESNR of interest in equation 10. The pseudocode evaluates $\nabla_aJ$, while the actual ESNR of interest should be w.r.t. the policy parameters $\nabla_\theta J$.

### Soundness
1

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
3

### Summary
This paper introduces the Expected Signal-to-Noise Ratio (ESNR) of policy gradients as a novel, efficient, training-time metric for predicting the downstream policy performance of world models in robotics. The central problem addressed is the current costly and slow iterative process of training a world model and then fully evaluating its quality via policy learning and environment rollouts, which can take days or weeks. The authors define the concept of Policy Awareness as how well a policy can learn from a model, and ESNR is proposed as an on-the-fly, policy-agnostic proxy for this metric.

### Strengths
1. ESNR can be computed with minimal overhead without a trained policy or environment rollouts.

2. The manuscript provides a much-needed policy-centric lens for evaluating world models, which goes beyond standard prediction quality metrics.

### Weaknesses
1. This is an incremental work, as shown in the manuscript, the ESNR metric has been proposed by [1].

2. The authors state that they have not yet evaluated ESNR extensively on large-scale world model tasks with billions of parameters. Since the advantage claimed in the paper is speed, experiments should be conducted on larger world models, and we cannot be certain that small-scale conclusions can be extended to the world models of larger world models.

[1] Paavo Parmas, Takuma Seno, and Yuma Aoki. Model-based reinforcement learning with scalable composite policy gradient estimators. In International Conference on Machine Learning, pp. 27346–27377. PMLR, 2023a.

### Questions
Is there a model-agnostic or task-agnostic heuristic that can be used to computationally identify the transition point out of the "biased region" (i.e., the ESNR minimum) without needing to wait for a full policy evaluation at a later checkpoint? Since ESNR is meant to save evaluation time, relying on the full U-shaped curve defeats the purpose.

### Soundness
2

### Presentation
2

### Contribution
2
