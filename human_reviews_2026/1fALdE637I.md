# Constrained Diffusion Policy Optimization for Offline Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 8, 2

## Abstract
In this paper, we propose the two-fold improved diffusion policy (TDP) for offline reinforcement learning. We first propose the constrained diffusion policy optimization (CDPO) framework, which unifies existing diffusion-based policy constraint methods. TDP harnesses the full potential of CDPO by initializing with the closed-form solution of a constrained optimization problem and then applying another constrained policy optimization for further refinement. We establish the theoretical properties of TDP, including expected policy improvement, in-distribution property, and approximate gains over existing diffusion policies. We also propose a design method for estimating the desired policy in the TDP loss function to achieve the aforementioned performance improvements. Empirical results on the D4RL benchmark show that TDP outperforms most existing offline reinforcement learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TDP, a diffusion policy-based offline RL algorithm that combines the so-called implicit and explicit policy constraint methods in offline RL. During policy improvement, TDP uses 1) a Q value term that encourages the diffusion policy to generate high Q-valued actions; and 2) a regularization term that aligns the score of the diffusion policy with the score of behavior policies and the gradient of Q-values. Several relaxations are made in the practical implementation for tractable and efficient computation. Finally, the proposed TDP is evaluated using tasks from D4RL, and it outperforms baseline methods by a large margin.

### Strengths
Empirical performance is strong. Ablation studies with respect to certain design choices, such as the noise-free estimation, are comprehensive.

### Weaknesses
The novelty is limited. For the algorithm design, it combines the techniques from DAC and the Q-maximizing objective from Diffusion-QL and EDP for policy improvement, which is straightforward. The theoretical analysis appears to follow [1] and AWAC, with an extension to diffusion divergence. 

Many relaxations to the theory are made in practice, e.g., the noise-free estimation, and the approximate inference used in EDP, making the overall algorithms biased from their theoretical characterization.

Besides, TPO seems to require extensive hyperparameter tuning efforts (Tables 8,9,10). 

Finally, several related papers should be discussed in the related work section, including DiffCPS [3] and BDPO [4], which all employ the policy constraint framework for diffusion policy optimization. 

[1] Offline RL with no ood actions: In-sample learning via implicit value regularization. ICLR 2023. 

[2] DiffCPS: Diffusion Model-based Constrained Policy Search for Offline Reinforcement Learning. 

[3] Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning. ICML 2025.

### Questions
In lines 156-159, the authors mentioned that implicit methods such as DAC suffer from a restricted function class and limited suppression of suboptimal regions. However, since we are working with policies that are parameterized as diffusion models, the function class is flexible enough and does not suffer from the mode-averaging issue of simple policy classes like Gaussians.  Besides, this claim seems to be related to the bandit experiments in Figure 1, where the implicit method generates points in suboptimal clusters. However, this actually makes sense to me since the generation results of Implicit methods conform to the Boltzmann distribution of the clusters. Adding an extra Q-maximization loss seems to encourage mode-seeking behaviors; however, this can also be achieved by adjusting the temperature for the implicit methods. In short, I would appreciate it if the authors could provide more explanation about this claim and, therefore, better motivate TDP.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces CDPO, a framework that unifies explicit and implicit policy-constraint methods for offline RL under diffusion policies, and proposes TDP (Two-fold Improved Diffusion Policy): (i) initialize from the reverse-KL-constrained closed-form “anchor” policy πη∗ (implicit step), then (ii) further improve it via a constrained optimization with a nonzero Q-loss (explicit step). Theoretically, the authors argue that the diffusion loss induces a pathwise forward KL, enabling strong duality and a policy enhancement theorem (monotone improvement for the first step and expected-Q improvement thereafter); they also state an approximate in-distribution property. Practically, they introduce a noise-free estimate to implement the anchor loss without pushing the critic into the noise domain, and add stabilizers (LCB Q-ensembles,
delayed actor updates, and an EDP one-step approximation). Experiments on D4RL locomotion, AntMaze, Kitchen, plus a 2D bandit visualization and a flow-policy variant (TFP), suggest TDP outperforms strong baselines and generalizes to flow policies.

### Strengths
The paper provides a well-motivated unified view (CDPO) that generalizes both explicit and implicit policy constraint methods.

### Weaknesses
1. The notions are complicated, and the writing, especially the methodological expression, is rather chaotic.
2. In my opinion, this paper has limited innovation. Compared to DAC, it merely uses Tweedie's formula to estimate the action a in order to mitigate the extrapolation error in calculating the gradient of the Q function in x_t.
3. This article's explanation of the two-fold policy for TDP is unclear, which greatly diminishes its importance.
4. The experimental baselines are limited, and in recent years, many new methods for diffusion policy and flow policy have emerged in the offline RL field.
5. Do all the baselines in Tables 1–3 use the same critic structure? While the paper emphasizes the importance of the LCB objective in stable Q-value estimation, it lacks persuasiveness. I believe more ablation studies should be conducted, such as experiments combining IQL and TDP, or LCB and IDQL.
6. f in (9) requires further explanation and analysis.

### Questions
See weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This manuscript proposes the  constrained diffusion policy optimization (CDPO) framework and a Two-fold improved Diffusion Policy for offline RL. The CDPO combines the explicit and implicit policy constraint methods, while TDP even harnesses the full potential of CDPO. The authors show the properties of TDP theoretically and provide practical method design to implement TDP. Experiments on D4RL benchmark demonstrates the superiority of TDP.

### Strengths
* The idea of CDPO is interesting. It unifies explicit and implicit  policy constraints, and serves as foundation for combining diffusion-based and constraint policy optimization methods. 

* The method of TDP is quite novel, starting from $\pi_{\eta}^*$ is appealing while avoiding exploring in OOD areas. In addition, it is appreciated that the authors provide rigorous prove for policy improvement. 

* The generlization ability of TDP looks pretty good. It can be easily generlized to other variants such as TFP, which also serves as a competitive alternative.

### Weaknesses
1. Some related references are missing, and it is suggested to consider the related work in the manuscript. 

* https://arxiv.org/abs/2303.15810

* https://arxiv.org/abs/2202.06239

* https://arxiv.org/abs/1911.11361

* https://arxiv.org/abs/2301.12130

* https://arxiv.org/abs/2405.16173


2. Though the manuscript is novel, the reviewer does not prefer the way of statements in the abstract. The readers may be mislead that the CDPO is a conventional method, while the authors only propose TDP. The authors are suggested to refine the statements in abstract.

3. From the experiments, it seems the TDP delivers the best performance in only 3 out of 9 tasks in D4RL benchmark from Table 1 (Similar in Table 2). The effectiveness of TDP is not fully shown.  It is suggested the authors provide further explanations on this issue. 

4. For Actor and Critic Update, if the method selects the Q-ensembles, will it lead to extra cost such as inference multiple times or extra training cost? Have the authors evaluated this issue?

5. Another question is about the starting policy $\pi_{\eta}^*$,  how could we easily get this initial policy in practice? Or could we try any other initial policies for better exploration?

### Questions
See the weakness above

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Constrained Diffusion Policy Optimization (CDPO), a unified framework that generalizes both explicit and implicit policy constraint methods in offline reinforcement learning (RL). Building on this framework, it proposes the Two-Fold Improved Diffusion Policy (TDP), a practical algorithm that leverages the advantages of both approaches. Specifically, TDP initializes from the implicit reverse KL–constrained policy $\pi^*_\eta$, which surpasses the behavior policy while remaining in-distribution, and subsequently refines it through Q-guided optimization under the generalized CDPO loss. Theoretically, the paper establishes a Policy Enhancement Theorem that guarantees monotonic policy improvement under mild assumptions, and further provides a bounded in-distribution property for diffusion-based policies.

### Strengths
CDPO unifies explicit and implicit policy constraint methods within a single formulation by generalizing constrained policy optimization (CPO) to incorporate a flexible anchor policy $\pi_0 \neq \pi_\beta$. The two-fold improvement strategy—anchoring to $\pi^*_\eta$ and refining via the Q-loss—is both conceptually elegant and empirically effective. Moreover, the paper establishes formal policy enhancement and in-distribution theorems for diffusion policies, providing theoretical grounding for results that were previously heuristic. Empirically, TDP attains state-of-the-art performance across the D4RL benchmark.

### Weaknesses
1.	The formulation of the surrogate loss involves several theoretically unsound steps. For instance, the “noise-free estimation” replaces a theoretically exact expression with an ad-hoc gradient approximation ($f \approx \frac{1}{\eta} Q_\phi(s,a_0)$), introducing bias that is not rigorously analyzed.
2.	Although the “noise-free estimation” technique is proposed, the subsequent issues of exploding or vanishing gradients in the guidance process are neither resolved nor thoroughly discussed.
3.	The theoretical results are limited to expected monotonic improvement and approximate dominance, without providing formal convergence or finite-sample guarantees. Moreover, the theoretical gains assume that $\pi^*_\eta$ is a “strong” initialization, yet its practical computation still relies on accurate Q-value estimation—introducing potential circularity. This limitation may restrict the applicability of the proposed algorithm, especially when dealing with medium- or low-quality datasets.
4.	While the paper claims to present a unified framework generalizing existing explicit and implicit policy constraint methods, several important diffusion-policy baselines [1,2,3,4] are omitted.
5.	Despite employing computational optimizations (e.g., delayed updates, half-batch gradients), the diffusion-based actor remains computationally heavy, and the discussion on training efficiency is relatively brief.

[1] Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning (Wang et al., 2022)

[2] Efficient Diffusion Policies for Offline Reinforcement Learning (Kang et al., 2023)

[3] IDQL: Implicit Q Learning as an Actor Critic Method with Diffusion Policies (Hansen-Estruch et al., 2023)

[4] Preferred Action Optimized Diffusion Policies for Offline Reinforcement Learning (Zhang et al., 2024)

### Questions
1.	Does the iterative refinement of TDP converge to a fixed point or optimal constrained policy? Are there stability guarantees under function approximation?
2.	How large is the bias introduced by approximating $f(a_k,k,s) \approx \frac{1}{\eta} Q(s,a_0)$? Does this bias accumulate during denoising steps?
3.	Diffusion models are expressive but unstable. How sensitive is TDP to diffusion step count $K$ or noise schedule choices?

### Soundness
2

### Presentation
2

### Contribution
2
