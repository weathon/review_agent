# Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Offline reinforcement learning (RL) faces a critical challenge of overestimating the value of out-of-distribution (OOD) actions. 
Existing methods mitigate this issue by penalizing unseen samples, yet they fail to accurately identify OOD actions and may suppress beneficial exploration beyond the behavioral support.
Although several methods have been proposed to differentiate OOD samples with distinct properties, they typically rely on restrictive assumptions about the data distribution and remain limited in discrimination ability.
To address this problem, we propose $\textbf{DOSER}$ ($\textbf{D}$iffusion-based $\textbf{O}$OD Detection and $\textbf{SE}$lective $\textbf{R}$egularization), a novel framework that goes beyond uniform penalization. 
DOSER trains two diffusion models to capture the behavior policy and state distribution, using single-step denoising reconstruction error as a reliable OOD indicator. 
During policy optimization, it further distinguishes between beneficial and detrimental OOD actions by evaluating predicted transitions, selectively suppressing risky actions while encouraging exploration of high-potential ones.
Theoretically, we prove that DOSER is a $\gamma$-contraction and therefore admits a unique fixed point with bounded value estimates. 
We further provide an asymptotic performance guarantee relative to the optimal policy under model approximation and OOD detection errors.
Across extensive offline RL benchmarks, DOSER consistently attains superior performance to prior methods, especially on suboptimal datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed DOSER framework addresses limitations in offline reinforcement learning by using diffusion models to detect out-of-distribution (OOD) actions based on reconstruction error. It then selectively regularizes these actions, suppressing detrimental ones while encouraging those deemed beneficial for exploration.

### Strengths
1. The paper is clearly written and well-structured, making its central contributions easy to understand.
2. The proposed method achieves competitive performance on Gym domain.

### Weaknesses
1. The method's design is notably complex, incorporating multiple auxiliary models beyond the standard actor-critic setup (e.g., diffusion model for state, diffusion model for action, ensembled dynamics models, implicit value model, implicit target value model, Two extra critic networks). This design challenges the method's practicality and scalability for real-world applications.
2. The method introduces a large set of algorithm-specific hyperparameters ($ \tau_s, \tau_a, \tau, \beta, \lambda, \eta, \delta_{\mathrm{V}}, N, Q_{\min} $). The sensitivity and impact of most of these parameters on the final performance are not adequately analyzed, leaving concerns about the robustness and reproducibility of the results. For instance, the effect of perturbations in \( Q_{\min} \) remains unclear.
3. While the core idea is intuitive, the empirical performance gains appear limited according to the ablation study, with consistent improvements observed on only two tasks (halfCheetah-m and halfCheetah-mr). The proposed method is quite similar to SVR. By comparing their performances, one can draw the conclusion that the benefits are limited for other tasks as well.
4. Could you provide the curves of the selective ratio for ID, OOD-beneficial, and OOD-detrimental actions throughout the training process?
5. There seems to be a potential flaw in the derivation. The claim that $ \eta Q_{\min} \geq Q_{\min} $ is contradictory given the stated parameter range $ \eta \in [0,1) $. 
6. The provided performance bound $ Q_{\min} \leq Q^{\pi} \leq Q_{\in}^{\pi} + \eta \delta_{\mathrm{V}} $ is arguably trivial, as it essentially states that the value is bounded by some minimum and maximum. A non-trivial bound with asymptotic analysis, characterizing the error relative to the optimal policy, is needed for a meaningful theoretical contribution.
7. The use of four critic networks, compared to the two commonly used in general methods, may confer an unfair advantage. An ablation study demonstrating performance with only two critics is necessary to ensure an equitable comparison and to isolate the contribution of the core algorithmic innovations.
8. The evaluation would be more comprehensive with the inclusion of the AntMaze domain.

### Questions
Please see Weakness

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
This paper proposes the DOSER method, a novel framework that goes beyond uniform penalization. DOSER uses two diffusion models to learn the behavior policy and state distribution. Meanwhile, it uses a single-step denoising reconstruction error to detach the OOD action.

### Strengths
* DOSER further distinguishes between beneficial and detrimental OOD actions by evaluating predicted transitions, selectively suppressing risky actions while encouraging exploration of high-potential ones.
* DOSER establishes stable convergence with bounded value estimates under $\gamma$-contraction guarantees.

### Weaknesses
* This method needs to train two diffusion models, which may bring more computational burden.
* This method also trains a dynamic model and multiple samples, which may bring more computational burden.

### Questions
* Similar to the motivation in this paper, there are numerous other methods for achieving efficient policy optimization through action discrimination. For instance, from the perspective of policy constraints, the latest offline RL method A2PR[1] employs advantage functions to select actions that guide policy constraints toward optimization. Could you elaborate on the advantages of DOSER? Additionally, could you include experimental comparisons and discussions with A2PR?

* I'm curious about the trend in the proportion of beneficial OOD actions versus detrimental OOD actions during training for different tasks. Does this significantly impact the training results?

* There are too few main experiments. Could we add some navigation tasks, such as Antmate?



Reference:

[1]Liu, T., Li, Y., Lan, Y., Gao, H., Pan, W., & Xu, X.. Adaptive Advantage-Guided Policy Regularization for Offline Reinforcement Learning. In International Conference on Machine Learning (pp. 31406-31424). PMLR.

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
4

### Summary
This paper tackles a key challenge in offline reinforcement learning: the overestimation of out-of-distribution (OOD) actions. The authors argue that existing methods, which uniformly penalize all OOD actions, are overly pessimistic and suppress potentially "beneficial" explorations.
The paper introduces DOSER (Diffusion-based OOD Detection and SElective Regularization), a novel framework that moves beyond uniform penalization. DOSER's contributions are twofold:
1. Diffusion-based OOD Detection: It employs two diffusion models, trained on in-distribution actions $\hat{\pi}_{\beta}(a|s)$ and states $d(s)$, respectively. The single-step denoising reconstruction error ($\mathcal{E}_a$ and $\mathcal{E}_s$) is used as a high-fidelity metric to detect OOD samples, which the authors argue is superior to VAEs for capturing complex, multi-modal behavior.
2. Selective Regularization: When the learning policy generates an OOD action $a_{ood}$, DOSER does not immediately penalize it. Instead, it uses a learned dynamics model to predict the next state. It then classifies $a_{ood}$ as detrimental and beneficial. While the detrimental actions are heavily penalized in the Q-function loss, the beneficial actions are compensated in the Q-function loss with a value-difference bonus.

The authors provide a theoretical analysis showing the DOSER operator is a $\gamma$-contraction and results in a value estimate that is bounded by the in-sample value plus the beneficial bonus ($\le Q_{In}^{\pi}(s,a_{id}^{*})+\eta\delta_{V}$). Empirically, DOSER achieves state-of-the-art (SOTA) performance on D4RL benchmarks, particularly on suboptimal "medium" and "medium-replay" datasets.

### Strengths
1. **Conceptual Novelty**: The primary strength is the original idea of "selective regularization." By classifying OOD actions into beneficial ($\mathcal{A}_{ood}^{+}$) and detrimental ($\mathcal{A}_{ood}^{-}$) sets, the paper moves beyond the field's preoccupation with uniform pessimism. This is a significant and valuable conceptual contribution.
2. **Modern OOD Detection**: The use of diffusion model reconstruction error as an OOD metric is a strong, modern approach. The argument that it is better suited for multi-modal behavior distributions than VAE-based methods is compelling and well-motivated.
3. **Strong Empirical Results**: The algorithm achieves SOTA performance, especially on the challenging and more realistic "medium" and "medium-replay" datasets. This suggests the method is particularly effective at improving upon suboptimal, heterogeneous data.
4. **Clear Ablations**: The ablation study in Table 2 provides clear evidence that both the Action Classification (AC) and Value Compensation (VC) components are critical to the algorithm's success. This validates the paper's core design.

### Weaknesses
1. **The Extrapolation Paradox of the Dynamics Model**: The framework's core classifier (Definition 1) relies on a learned dynamics model to predict the consequence of an OOD action. This is a critical flaw. The model is trained on in-distribution data and has no ground-truth supervision for OOD actions. Therefore, its prediction is an extrapolation that is just as likely to be erroneous as the $Q$-function's. The algorithm relies on an unreliable, "hallucinating" component to make its most critical safety decision.
2. **Extreme Hyperparameter Sensitivity**: The algorithm is not a general, robust method but a "brittle" framework that requires massive, task-specific tuning. Appendix Table 4 reveals that key hypermeters (OOD thresholds $\tau_a, \tau_s$, penalty $\beta$, expectile $\tau$) are different for nearly every single environment-dataset pair. For example, $\tau_a$ is 99th percentile for halfcheetah-medium but 80th for halfcheetah-medium-expert. This lack of generality is a severe practical limitation and makes the SOTA claims less impressive, as they are a product of extensive tuning.

### Questions
1. **On the Dynamics Model's Reliability**: Could the authors comment on the "Extrapolation Paradox"? How can the algorithm trust the predicted next state $s'$ (and its resulting $V(s')$ and $\mathcal{E}(s')$) when it comes from an OOD action $a_{ood}$ that the dynamics model $p_{\psi}$ has never seen? Have the authors considered using uncertainty estimates from $p_{\psi}$ (e.g., via an ensemble) to gate this classification?
2. **On Practical Hyperparameter Setting**: The reliance on per-task OOD thresholds (80th vs 99th percentile) shown in Table 4 is a major concern. How would a practitioner set $\tau_a$ and $\tau_s$ for a new, unseen offline dataset without a massive tuning budget? Is there a more robust or adaptive way to set this threshold?

### Soundness
2

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
This paper proposes DOSER (Diffusion-based OOD Detection and Selective Regularization), an offline reinforcement learning (RL) framework that aims to mitigate overestimation caused by out-of-distribution (OOD) actions.
The method trains two diffusion models to model the behavior policy and the state distribution, respectively, and uses the single-step denoising reconstruction error as an OOD indicator.
Then, during policy optimization, DOSER further classifies OOD actions into “beneficial” and “detrimental” using a learned dynamics model and value comparison, selectively penalizing or rewarding them.
The authors provide theoretical analysis (γ-contraction and bounded convergence) and empirical results on D4RL MuJoCo benchmarks, claiming consistent improvements over existing VAE-based and diffusion-based offline RL baselines.

### Strengths
Interesting integration of diffusion models into OOD detection. The idea of using denoising reconstruction error as a likelihood-free OOD signal is conceptually appealing and avoids Gaussian assumptions used in VAE-based behavior modeling.

Selective regularization is an intuitive extension to uniform penalization — encouraging “good” OOD exploration rather than blindly suppressing all deviations.

Clarity and structure: the paper is generally well written and easy to follow, with clear diagrams (e.g., Fig. 1–2) and algorithmic pseudocode.

Comprehensive empirical comparison: the authors evaluate against both conventional (CQL, IQL, ACL-QL) and diffusion-based (DQL, SRPO, DTQL, etc.) baselines, and perform ablation and sensitivity studies.

### Weaknesses
Questionable reliability of OOD classification.
The core contribution—distinguishing beneficial from detrimental OOD actions—relies heavily on a learned dynamics model and single-step value predictions.
In offline RL, both are notoriously unreliable outside the data distribution, making this classification potentially unstable or self-reinforcing. The paper lacks evidence (e.g., accuracy, calibration, or failure cases) showing that such classification can be trusted.

Circular dependency between detection and evaluation.
The method uses diffusion-based reconstruction error to decide OODness and then uses a dynamics model (also trained on the same data) to judge whether the OOD transition is beneficial.
Without ground truth labels or uncertainty calibration, this process risks systematic bias amplification rather than true OOD discrimination.

Limited novelty beyond combining existing ideas.
Diffusion-based behavior modeling (e.g., DQL, IDQL, SRPO) and reconstruction-based OOD detection (e.g., Graham et al., CVPR 2023) are established.
The proposed framework largely combines these with a hand-crafted “bonus/penalty” heuristic; the theoretical part simply restates a standard contraction argument.

Experimental validation does not fully support the claims.
The performance gains on D4RL are moderate and sometimes within variance ranges; ablations do not directly demonstrate that the OOD classification truly identifies beneficial cases.
No visualization, quantitative detection metrics (e.g., AUROC on synthetic OOD test), or robustness analysis are provided to substantiate the detection accuracy.

### Questions
How stable is the beneficial vs. detrimental classification in practice? Can the authors report the proportion of actions identified as each type and the accuracy of this classification on a synthetic dataset where ground-truth OOD actions are known?

Does the diffusion reconstruction error correlate quantitatively with true behavior support likelihood (e.g., log-likelihood from a reference model)?

How sensitive is DOSER to errors in the learned dynamics model? Would performance degrade sharply if the model mispredicts next states for unseen regions?

Could the authors show qualitative visualizations (e.g., state-action heatmaps) to verify that “beneficial” OOD actions indeed lead to higher-value regions?

Have the authors tested on harder datasets (Adroit, AntMaze) to demonstrate generalization beyond MuJoCo locomotion?

### Soundness
3

### Presentation
4

### Contribution
3
