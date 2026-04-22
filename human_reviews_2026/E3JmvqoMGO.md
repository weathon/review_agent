# Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Flow matching has recently emerged as a powerful alternative to diffusion models, providing a continuous-time formulation for generative modeling and representation learning. Yet, we show that this framework suffers from a fundamental instability in the low-noise regime. As noise levels approach zero, arbitrarily small perturbations in the input can induce large variations in the velocity target, causing the condition number of the learning problem to diverge. This ill-conditioning not only slows optimization but also forces the encoder to reallocate its limited Jacobian capacity toward noise directions, thereby degrading semantic representations. We provide the first theoretical analysis of this phenomenon, which we term the low-noise pathology, establishing its intrinsic link to the structure of the flow matching objective. Building on these insights, we propose Local Contrastive Flow (LCF), a hybrid training protocol that replaces direct velocity regression with contrastive feature alignment at small noise levels, while retaining standard flow matching at moderate and high noise. Empirically, LCF not only improves convergence speed but also stabilizes representation quality. Our findings highlight the critical importance of addressing low-noise pathologies to unlock the full potential of flow matching for both generation and representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper claims flow matching models break down in the low-noise regime, where small input changes cause large, unstable target velocity shifts, resulting in ill-conditioned training and degraded representations. The authors prove this and show empirical failures on CIFAR-10 and Tiny-ImageNet with DiT models. They propose a hybrid fix: for moderate/high noise ($t \geq T_\text{min}$), use standard flow matching; for low noise ($t < T_\text{min}$), switch to contrastive learning that aligns low-noise features with stable moderate-noise anchors and repels negatives. The paper positions the refined flow matching as a tool for better representation learning in generative models and ties to a much broader scope like large multimodal foundation models and unified models, arguing that generative supervision should balance synthesis and understanding, but the "low-noise issues" disrupt this.

### Strengths
1. Identification of Low-Noise Pathology: The paper clearly identifies and formalizes critical instability in flow matching at low noise levels, linking it to divergent condition numbers and representation degradation, addressing an underexplored issue with direct relevance to generative and representation learning.
2. Theoretical Analysis: Provides strong theoretical foundations, including proofs of condition number divergence, slow convergence, and Jacobian reallocation, offering a principled explanation that extends beyond empirical observations.
3. Practical Remedy: Introduces Local Contrastive Flow, a simple hybrid objective that mitigates pathologies, with empirical results on CIFAR-10 and Tiny-ImageNet showing faster convergence, improved FID, and stabilized representations.

### Weaknesses
1. Evaluations are confined to small-scale datasets (CIFAR-10, Tiny-ImageNet), lacking experiments on high-resolution (e.g., ImageNet 256x256) or multimodal benchmarks, which undermines claims of broad applicability (unifying generation and understanding) and ignores scalability concerns in real-world generative modeling.
2. The theoretical analysis hinges on idealized conditions like bounded Jacobians, potentially failing to reflect the nonlinear dynamics of transformer-based models in practice.

### Questions
You claim low-noise pathology is an "inherent consequence of the generative modeling objective," but your analysis depends on FM's interpolation ($ x_t = \alpha_t x_0 + \beta_t \epsilon $). AR models and most MLLMs (e.g., Janus[1], NTP-based) don't use explicit noise schedules. How does this pathology apply to AR settings with no ($ t \to 0 $) regime? Your Background draws parallels to AR models but provides no concrete evidence or analogy. Isn't this overreach? Please provide a concrete example of this issue arising in a popular MLLM, or clarify that it's FM-specific. Otherwise, why should the MLLM community focus on unified generation and understanding care?

Proposition 5 assumes the input space decomposes cleanly into semantic and noise subspaces with orthogonal projectors, leading to the Frobenius norm decomposition in Equation (51):  $ |J_g|F^2 = |J_g|{S_{\mathrm{sem}}}|F^2 + |J_g|{S_{\mathrm{noise}}}|_F^2 $. However, in high-dimensional vision models like DiT, this orthogonality seems neither unique nor stable during training. Could you provide empirical validation (e.g., via PCA or SVD on learned Jacobians, or measurements of subspace angles) to show that this assumption holds even approximately? If not, how might cross-term interactions (ignored in the decomposition) affect the bound's accuracy in practice, and would incorporating them change your conclusions?

Proposition 5 also assumes a fixed Frobenius norm constraint on the encoder Jacobian ( $ |J_g(x_t)|_F \leq B $ ), independent of noise level t and training progress. Yet, in transformers like DiT, the Jacobian norm evolves dynamically across t and epochs (as hinted in your Figure 8). How does this fixed B relate to actual network capacity, and does the bound remain tight or meaningful throughout training? Could you provide an analysis (e.g., tracking B's effective value as an estimation over epochs or noise schedules in experiments) to show how it holds up, or discuss adaptations for dynamic norms? If the assumption doesn't fully capture real training dynamics, how might relaxing it impact the pathology's interpretation?

[1] C. Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation," 2024.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies ill-conditioning of flow matching models in the low-noise regime and its implications for degraded representation learning. It proposes a local contrastive loss to address the issue.

### Strengths
The writing is clear, and the representation learning in flow matching is interesting.

### Weaknesses
1. The motivation is not clear. It's not clear to me why the blowing up Lipschitz constant as the velocity in flow matching approaching the clean data is perceived as a new phenomenon, nor why it is an important issue. The blowing up Lipschitz constant/ill-conditioning is clear from the formula of the velocity field in flow matching, which involves a denominator term that is approaching 0, see e.g Equation 3.7 in [1]. Importantly, the blowing up effect is purely due to the scheduling term, and this may not be an issue as one can alternatively use data prediction/$x$-prediction to stabilize the training. Additionally, from the $x$-prediction perspective, the objective of the neural network becomes to fit an identity map on the training data, and this may explain why the representation is not good when close to clean data. This analysis extends to velocity prediction, as the velocity field is an affine transformation of the $x$-prediction. Therefore, to me, the low-noise pathology is not valid.
2. As for the proposed local contrastive loss, it shows some effect in improving the representation learning for time near $0$ (clean data), but the representation learning is still suboptimal. If a good representation learning is the goal, why not simply use the representation at some later time instead of the clean data? Additionally, the FID score in Figure 5 is above 10, which is far from an adequate level for obtaining meaningful empirical findings.
3. Propositions 2 and 3 suggest there is some lower bound on the convergence rate, but the proof is not convincing. It is not clear to me why a conditional value lower bound can lead to a convergence lower bound, as there could exist some subset or some direction where the convergence is still quick. 

[1]Gao, Yuan, Jian Huang, and Yuling Jiao. "Gaussian interpolation flows." Journal of Machine Learning Research 25.253 (2024): 1-52.

### Questions
1. Can you explain why the training is slow in Figure 1? Note that the flow matching loss inherently does not have zero loss as a minimum, so simply comparing the loss values between different time steps may not be meaningful.
2. Can you provide more comparison with other simple baselines, like not training for a small time or using $x$-prediction to stabilize the training?
3. Can you identify where I should find the statement of Proposition 2 in the cited reference?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the flow matching training in the regime of low noise ($t\rightarrow 0$). The authors point out that when the noise level approaches zero, the learning problem becomes ill-conditioned: very small input perturbations produce large variation in the velocity target, driving the condition number to blow up. As a result, optimization becomes difficult (slow convergence) and the encoder (or representation model) ends up allocating capacity to “noise directions” rather than preserving semantic structure. The authors call this phenomenon the low‐noise pathology. They provide a theoretical analysis linking this pathology to the structure of the flow matching objective. To address it, they propose a hybrid training protocol called Local Contrastive Flow (LCF): for small noise levels, instead of regressing the velocity field directly, they add a contrastive feature‐alignment objective to the original FM loss; for moderate and high noise levels they retain standard flow‐matching. Empirical results show that LCF improves convergence and stabilizes representation quality. The authors argue that addressing the low‐noise pathology is important for unlocking flow matching’s full potential for both generation and representation learning.

### Strengths
1. The paper is well written.  The problem statement is clear and the proposed remedy is easy to follow. 

2. The observation that in the low‐noise regime flow matching becomes ill‐conditioned is insightful. In addition, the authors provide a theoretical analysis connecting the pathology with the structure of the objective and the Jacobian capacity of the encoder. This adds rigor and helps ground the intuition. By linking the condition number divergence and representation degradation to the low‐noise region, they give a basis for the remedy.

3. The proposed LCF is reasonable and simple to implement.  It decomposes training into two stages: at low noise stage the original FM training loss is appended with an additional contrastive loss term. At higher noise stage, the training loss is identical to the classic FM. 

4. Empirically, it is evident that the LCF improves the generation quality and the representation learning.

### Weaknesses
1. The experiments may be limited (e.g., CIFAR-10, and ) and might not cover large‐scale settings or varied domains. It’s not clear how the remedy scales to high‐resolution generation or very complex datasets.

### Questions
1. Do the noise scheduling choices mitigate or exacerbate the low‐noise pathology in the original FM? 

2. Could you add additional results on higher-resolution images, for example, FFHQ or full ImageNet 256x256, to show that the proposed method also scales well to higher dimensional data?

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
3

### Summary
The paper identifies a critical problem in flow matching models called “low noise pathology,” in which learning becomes unstable and learned representations degrade as the noise level approaches zero. The authors provide a rigorous theoretical analysis proving that in this regime, the conditional number of the learning problem diverges, leading to slow convergence and a fundamental trade-off that forces the model to redistribute its representational capacity from the semantic direction to the noise-related direction. To mitigate this problem, they propose Local Contrastive Flow (LCF), a hybrid method that uses standard flow matching for moderate/high noise levels and a contrastive alignment task for low noise levels, utilizing the characteristics of a stable anchor point with moderate noise levels.

### Strengths
- The paper clearly defines and formulates a non-obvious but fundamental problem (“low-noise pathology”). The empirical observation that cleaner data leads to worse learning results contradicts intuition and makes this contribution compelling.

- Theoretical analysis. The relationship between the noise graph ($\beta'_t/\beta_t$), the conditional number, the convergence rate of optimization (proposition 3), and the deterioration of representation (propositions 4 and 5) is rigorous and profound. In particular, Propositions 4 and 5 offer an elegant geometric explanation for why representation quality deteriorates, framing it as a necessary redistribution of the Jacobian under a limited capacity budget. 

- The experiments are well-designed, evaluating both generative performance (FID) and representation quality (linear probing) on several benchmarks.

### Weaknesses
- The main weakness is incomplete empirical comparison with existing low-noise stabilization techniques:
     * The paper convincingly identifies the low-noise pathology but fails to adequately situate its proposed LCF method against a several key existing strategy for mitigating this very issue.
     * It is well-established in the diffusion/flow matching literature [1] that predicting the velocity field $v$ (defined as $v=\alpha'_tx_0+\beta'_t\epsilon$) can lead to more stable training near $t=0$ compared to standard $\epsilon$-prediction, as the target $v$ tends to a deterministic limit. By comparing LCF only against the unstable $\epsilon$-prediction baseline, the authors present an incomplete picture. A comparison with a model trained using $v$-prediction across the entire time interval is crucial to determine whether the complexity of LCF (with its contrastive loss and threshold $T_min$) is justified, or if a simpler change in parameterization achieves comparable stability and performance. The absence of this strong and relevant baseline makes it difficult to fully assess the novelty and practical necessity of the proposed approach.
     * Another relevant article [2] rigorously proves that near zero $t=0$, learning can be improved (variance reduced) by using an exact formula for the vector field and estimating the integrals included in it using Monte Carlo-like methods. This reference, as well as a numerical comparison with this method, is also omitted in the paper under consideration.
- Although the results for CIFAR-10 and Tiny-ImageNet are convincing, these datasets have relatively low resolution. The community focuses primarily on high-resolution generation (e.g., ImageNet 256x256 and above). Proof of the existence of this pathology and the effectiveness of LCF in eliminating it on a larger-scale test would greatly enhance the impact and practical significance of the paper.

- The identified pathology is conceptually related to problems observed in diffusion models, where a "dispersion explosion" near $t=0$ is also known to cause learning instability. Although the formulation of the flow comparison differs, a more detailed discussion of the similarities and differences with these related phenomena in the diffusion literature will provide better context and more clearly position the contribution in the broader field.

- The method introduces additional complexity, requiring the computation of anchor representations ($x_{T_{min}}$) for a subset of the batch. The paper should include a brief discussion or analysis of the associated computational and memory overhead compared to standard flow matching. Is it negligible, or is there a measurable cost?

[1] Tim Salimans & Jonathan Ho, PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS, ICLR-2022, (https://openreview.net/pdf?id=TIdIXIpzhoI)

[2] G. Ryzhakov, S. Pavlova, E. Sevriugov, I. Oseledets, EXPLICIT FLOW MATCHING: ON THE THEORY OFFLOW MATCHING ALGORITHMS WITH APPLICATIONS,  ICOMP-2024 (https://openreview.net/pdf?id=thE8EmPVW8)

### Questions
- Have you observed the low-noise pathology on larger-scale datasets (e.g., ImageNet-1k)? If so, does LCF scale effectively and provide similar benefits in that setting?
- The choice of $T_{min}$ appears to be a hyperparameter. Did you find it to be sensitive, and is there a way to select it based on the theoretical analysis of the condition number?
- Proposition 3 relies on the Gauss-Newton approximation holding for the transformer-based DiT architecture. Could you comment on the validity of this assumption for modern architectures, for example, with extensive use of attention?

### Soundness
3

### Presentation
3

### Contribution
2
