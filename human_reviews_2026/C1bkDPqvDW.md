# Flow Map Learning Via Non-Gradient Vector Flow

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Diffusion and flow-based models benefit from simple regression losses, but inference (i.e, producing samples) incurs significant computational overhead because it requires integration. Consistency models address this overhead by directly learning the flow maps along the ODE trajectory, revealing a design space for the learning problem between one-step and many-step approaches. However, existing consistency training methods feature computational challenges such as requiring model inverses or backpropagation through iterated model calls, and do not always prove that the desired ODE flow map is a solution to the loss. We introduce SGFlow, an approach for learning flow maps that bypasses explicit invertibility constraints and expensive differentiation through model iteration. SGFlow trains a model to compute both the ODE solutions and the implied velocity from scratch by following non-conservative dynamics with a stationary point at the desired flow map. On the CIFAR image benchmark, SGFlow attains a favorable relationship of FID to step count, relative to flow matching, MeanFlow, and several other flow map learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes GameFlow, an approach for learning flow maps that bypasses explicit invertibility constraints
and expensive differentiation through model iteration. It introduces a new training objective of flow-based models with theoretical guarantee of optimality in certain sense.
Numerical experiments are conducted to verify the results.

### Strengths
The paper is theoretically grounded and proves that the proposed loss (7) is consistent with the true loss function in the sense that they share same functional stationary points. This is a non-trivial improvement upon MeanFlow. And the theories are also verified by experiments, where the solution to MeanFlow is not the ground truth in contrast to GameFlow. In addition, GameFlow outperforms MeanFlow and other methods on real-data experiments like CIFAR-10 generation.

### Weaknesses
1. Although the training objective is theoretically sound, it involves much higher computational cost than MeanFlow due to the backpropagation of JVP, especially when scaling to large models. This is the most severe issue of the proposed method and is inevitable in order to make the objective mathematically correct. Could the authors conduct larger scale experiments if possible to see the scaling ability of the proposed method?

2. I don't see any connection between the proposed framework with game and the interpretation as two-player game is quite far-fetched. The authors may need to choose a more approriate title.

### Questions
Please see weakness part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper derives a new loss for training Flow Maps models, i.e. models that can approximate ODEs in 1 step. They derive their objective function from first principles and prove that it shares same minimizers than the ideal loss. Then, they apply their loss to toy data (multivariate Gaussian) and image generation benchmarks. In toy data setting, the demonstrate that Mean Flows objective can converge to incorrect optimum.

### Strengths
* Principled derivation from first principles of a novel training objective for learning Flow Maps. This is an important results, given the attention received by Flow Maps by the research community, and given their importance to reduce inference cost of diffusion models. 

* Proved that the tractable training objective has same optimum than the target objective. 

* Interesting analysis on the Multivariate Gaussian setting, showing that Mean Flows does not converge to the correct optimum.

### Weaknesses
* Experimental results seem weak and far from results in the literature. How can the authors explain this?  Indeed, 1-step models are now around 2/3 of FID on CIFAR10 (see for example "Consistency Models Made Easy" from Geng et al.). Comparing to a FID of 29 reached on 10-step sampling, this is very low. 

* Weight decay is rarely used in diffusion/consistency models. Why do the authors choose to use it? Can it explain lower FID than in the rest of the literature?

### Questions
* Going from Equation (5) to Equation (6) is done by differentiating with regards to t. Why differentiating by t and not by u, as done in Mean Flows? Could you compare both approaches? 

* You propose replacing the velocity field in (6) by sg(f). Actually you could also rely on a pre-trained neural network to approximate the velocity field. This could give another option to train your model in a distillation mode. Have you considered this?

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
This paper proposes a method, GameFlow, to learn the flow function of the PF-ODE. Unlike other flow learning alternatives, GameFlow's optimum is proven to be the true flow map, and GameFlow does not necessitate computationally expensive or restrictive operations like invertible neural networks or nested differentiation through JVPs. To this end, GameFlow leverages an objective involving only the reverse flow and the true velocity field, the latter being equivalently approximated by the stop gradient of the optimized network itself. Experiments on toy data and CIFAR-10 against other flow models illustrate the soundness and benefits of GameFlow in terms of generative performance and memory footprint.

### Strengths
1. GameFlow is well motivated to address the shortcomings of other flow models. Although this is not new, the paper highlights and addresses the theoretical issues of MeanFlow. Further, GameFlow alleviates the computational and design burden of otherwise sound alternatives. To my knowledge, GameFlow relies on novel derivations in the field. For these reasons, I believe that GameFlow has the potential to influence future works in this research area.
2. The proposed method seems, up to my limited assessment of the appendix, sound. Empirical results do confirm its advantages within the limits of the chosen experimental setting (cf. weaknesses).
3. The paper is overall clearly written, except for a few issues described below.

### Weaknesses
### Questions on main claims

1. The interpretation of GameFlow as a game is far-fetched. It solely relies on the presence of a stop gradient in the objective, making the second player trivial. Besides, while this does change the optimization landscape, this is the case of many other methods in or outside the domain (such as many the papers cited in the submission).
2. Better intuition and presentation for the main result (Theorem 1) would be appreciated. Without reviewing the appendix in detail, it is challenging the understand the rationale behind the result -- although I understand this is a challenging task. My main question regards the fact that repeated arguments (l. 215 in the main paper, l. 1375 and 1406 in the appendix) rely on the loss values to conclude on the loss gradients, which is usually inconclusive when a stop gradient operator is involved.

### Questionable experimental setting

3. GameFlow is tested on limited benchmarks. While this is understandable depending on compute constraints and not eliminatory, it would have beneficiated from a more extensive empirical study involving standard datasets like ImageNet.
4. More importantly, the FID values on CIFAR-10 are unusually high (compared to e.g. MeanFlow and other papers in the literature). This raises doubts on the realism of the experimental setting, and questions the assertion that the training configuration is "common" (l. 320).
5. Reproducibility is limited as only part of the codebase is available within the submission PDF itself, and the paper shares few details. The promise to release the source code is duly noted.

### Advantage w.r.t. consistency models

6. The paper mentions consistency models as related methods but ignores them when discussing the method's advantages. To my understanding and based on Table 1, consistency models are sound and scalable; the only difference is that they do not model the entire flow (i.e. they are not multistep as stated in the paper). The implications of this difference in practice are unclear in the paper, especially as consistency models are left out of the experimental section.

### Other issues

7. Two minor claims need to be further clarified.
    - The benefits of using split batch JVP are not explained or shown experimentally.
    - The authors state the that gradients of stop-grad' loss "are not the gradients of any one mathematical objective", but there is no proof for this. It may be that these gradients do correspond to a mathematical objective.
8. There are a number of formatting issues and typos (some of them spotted below). I would advise the authors to proofread their manuscript.
    - Table 1 would be more readable with symbols such as ✓ and ✗.
    - Parentheses are missing for the reference l. 75.
    - "covvariance" should be "covariance" (l. 176).
    - "as" should be "is" (l. 195).
    - The second $\hat{f}$ should be $\tilde{f}$ l. 215, following l. 185.
    - Figure 1 should be integrated using vector graphics.
    - There is an extra period l. 303 and an extra space l. 345.

### Questions
The contributions of this paper appear to be valuable for the community but are hindered by presentation issues and a questionable experimental setting. Therefore, I do not recommend acceptance in the current state of the submission. Still, depending on the authors' response to the above weaknesses, I am willing to change my assessment.

Most importantly, I suggest the authors to address the following points.
1. Remove from the method name and framing as well as the paper title the notion of game.
2. Provide further intuition on the rationale behind Theorem 1.
3. Explain the experimental difference with other papers.
4. Clarify the advantage of the method compared to consistency models.

### Soundness
2

### Presentation
2

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
This paper proposes GameFlow, an objective for training flow map models for fast sampling. The method simultaneously trains the model to predict both the flow map and its underlying velocity, using a self-correcting stop-gradient target. This approach avoids costly model inversions and large Jacobians. The method is supported by a theoretical proof of correctness and empirical results on cifar-10.

### Strengths
1. Theorem 1 rigorously proves that the true flow map is a stationary point of the GameFlow objective. The comprehensive proofs are technically solid.
2. Provides reasoning and numerical evidence (Figure 1) that MeanFlow does not preserve the true flow map as an optimum.
3. The paper is well-written and clearly motivated. It systematically identifies and addresses key challenges in the field.

### Weaknesses
1. Only cifar-10 is evaluated with no higher-resolution datasets. According to line 308, the cifar-10 experiments are conditional, yet MeanFlow at 100 steps achieves FID 4.58, which is worse than the 2.92 reported in Table 3 of the MeanFlow paper[1] (Unconditional cifar-10, NFE=1). This discrepancy needs clarification.
2. Missing comparisons with consistency model variants (Consistency Models[2], simplified Consistency Models[3], etc.).
3. Missing training curves (loss, FID vs. steps) to verify training stability and convergence behavior.
4. What is the empirical impact of the corrective term in the loss? An ablation study is needed.

[1] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

[2] Song, Yang, et al. "Consistency models." (2023).

[3] Lu, Cheng, and Yang Song. "Simplifying, stabilizing and scaling continuous-time consistency models." arXiv preprint arXiv:2410.11081 (2024).

### Questions
1. Can you provide a theoretical comparison between GameFlow and Consistency Trajectory Models[1]?
2. What specifically causes the 3× memory overhead versus MeanFlow when both methods use JVPs?

[1] Kim, Dongjun, et al. "Consistency trajectory models: Learning probability flow ode trajectory of diffusion." arXiv preprint arXiv:2310.02279 (2023).

### Soundness
2

### Presentation
3

### Contribution
3
