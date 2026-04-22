# OAT-FM: Optimal Acceleration Transport for Improved Flow Matching

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
As a powerful technique in generative modeling, Flow Matching (FM) aims to learn velocity fields from noise to data, which is often explained and implemented as solving Optimal Transport (OT) problems. In this study, we bridge FM and the recent theory of Optimal Acceleration Transport (OAT), developing an improved FM method called OAT-FM and exploring its benefits in both theory and practice. 
In particular, we demonstrate that the straightening objective hidden in existing OT-based FM methods is mathematically equivalent to minimizing the physical action associated with acceleration defined by OAT.  Accordingly, instead of enforcing constant velocity, OAT-FM optimizes the acceleration transport in the product space of sample and velocity, whose objective corresponds to a necessary and sufficient condition of flow straightness. An efficient algorithm is designed to achieve OAT-FM with low complexity. OAT-FM motivates a new two-phase FM paradigm: Given a generative model trained by an arbitrary FM method, whose velocity information has been relatively reliable, we can fine-tune and improve it via OAT-FM. This paradigm eliminates the risk of data distribution drift and the need to generate a large number of noise data pairs, which consistently improves model performance in various generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OAT-FM, a new Flow Matching (FM) method inspired by the theory of Optimal Acceleration Transport (OAT). Standard Optimal Transport (OT) based FM methods aim to straighten flows by minimizing kinetic energy (velocity), which is a sufficient but not necessary condition for straightness. The authors argue that a more fundamental condition for straightness is minimizing acceleration.

The proposed OAT-FM lifts the transport problem from the sample space $\mathcal{X}$ to the product space of sample and velocity, $\mathcal{X} \times \mathcal{V}$. The objective is to find a coupling that minimizes the total squared acceleration, which the authors show is equivalent to a necessary and sufficient condition for flow straightness.

Because this new objective requires reliable velocity estimates at the endpoints (noise and data), which are not available at the start of training, OAT-FM is proposed as a **two-phase paradigm**. Phase 1 involves training any standard flow/diffusion model to obtain "relatively reliable" velocity information. Phase 2 then fine-tunes this pretrained model using the OAT-FM objective, which refines the flow trajectories by minimizing acceleration. The authors provide an efficient algorithm that reduces the complex OAT coupling in the product space to a standard OT problem in the sample space, giving it the same asymptotic complexity as OT-CFM.

Experiments on 2D benchmarks, CIFAR-10, and ImageNet $256^2$ show that this OAT-FM fine-tuning phase consistently improves the performance (e.g., FID) of various strong pretrained models (including EDM and SiT-XL) with very little computational overhead.

### Strengths
1.  **Novel Theoretical Framework:** The paper successfully connects Flow Matching with the theory of Optimal Acceleration Transport (OAT), providing a new, physically-grounded objective for straightening generative flows by minimizing acceleration.
2.  **Practical & Efficient Refinement:** The proposed two-phase paradigm is highly practical. It provides a way to improve existing, strong generative models with a very small additional compute budget. The empirical results (e.g., improving SiT-XL on ImageNet with only +5 epochs) are compelling.
3.  **Strong Empirical Validation:** The method is shown to consistently improve a wide array of baseline models (FM, I-CFM, OT-CFM, EDM, SiT-XL) across diverse tasks (2D OT, CIFAR-10, ImageNet 256).
4.  **Computationally Feasible:** The authors provide a crucial simplification (Section 2.3) that reduces the complex OAT coupling problem in $\mathcal{X} \times \mathcal{V}$ to a standard OT problem in $\mathcal{X}$, making its complexity on par with existing OT-CFM methods.

### Weaknesses
**Computational Scaling:** While the asymptotic complexity is the same as OT-CFM, the lower-level problem still requires solving a minibatch OT problem, which scales quadratically with the batch size $B$ (e.g., $\mathcal{O}(B^2 \log B)$ with Sinkhorn), as noted in the limitations. This could be a practical bottleneck for scaling to very large batch sizes.

### Questions
1. The cost function $l_{\mathcal{A}}$ (Eq 6) has a hyperparameter $\alpha$. The proof of Theorem 3 suggests $\alpha=2/3$ maximizes the theoretical bound, but the text also mentions $\alpha=12/13$ when relating Eq 6 to Eq 5. What value of $\alpha$ was used in the experiments, and how sensitive is the final performance (e.g., FID) to this choice?
2.  There seems to be a slight discrepancy in the target network updates mentioned in the appendix. Appendix D.2 (2D task) mentions a "hard copy... every 500 batches," while Appendix D.3 (CIFAR-10) mentions an "EMA... with a decay of 0.9999." Which strategy was used for the ImageNet (SiT-XL) experiments? Is the method sensitive to this choice (hard vs. soft update)?

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
The authors propose a novel fine-tuning method for flow models inspired by Optimal Acceleration Transport (OAT). They first introduce the theoretical framework of OAT and derive a practical optimization objective based on minimizing acceleration. The effectiveness of the method is then validated on both low- and high-dimensional datasets, including ablation studies that demonstrate the importance of the proposed loss components.

### Strengths
- The authors explore a promising direction in optimal transport by advancing from first-order to second-order formulations.
- The proposed method maintains computational complexity comparable to first-order approaches and does not require explicit modeling of acceleration fields.
- The comparison to prior second-order methods is clear and highlights the key differences effectively.

### Weaknesses
**Theory**. Reducing acceleration may locally straighten trajectories, but the paper does not explicitly discuss how this affects preservation of the transport map from $\mu_0$ to $\mu_1$. Without enforcing a Vlasov constraint, it is unclear whether mass conservation and global flow validity are maintained, i.e., if the method (final loss 7) is theoretically valid.

**Clarity.** The previous concern is also closely related to the overall insufficient clarity of the paper. Although the authors seem to discuss the main required theory and derivations, they are still hard to parse and understand their relations, even for a reader experienced in optimal transport. Furthermore, the paper’s theory relies on a recent paper (Brigati et al., 2025) which, to my knowledge, is still not reviewed or published.

**Practice**. 
- Based on the theoretical concern above, it is essential to evaluate how the refinement behaves as fine-tuning iterations increase, for example by tracking changes in quality metrics and trajectory straightness, similar to the analyses shown in Figures 8b and 9 of [1]. This could reveal whether the transport map remains valid or gradually deteriorates under more aggressive application of OAT-FM. A plot of FID as a function of the training (fine-tunning) epoch should be provided.
- The proposed methods do not show substantial performance improvements, particularly in higher-dimensional settings (Section 4.2). This is an important weakness given the heuristical nature of the method. I am not convinced that the improvements are statistically significant (experiments in 4.1 are toys and can be considered only for illustration purposes).
- The rationale for selecting only a specific subset of baseline methods is not explained. For example, although OFM [2] is discussed in the paper, it is not included in the empirical comparison.

[1] Liu, Xingchao, Chengyue Gong, and Qiang Liu. "Flow straight and fast: Learning to generate and transfer data with rectified flow."

[2] Kornilov, Nikita, et al. "Optimal flow matching: Learning straight trajectories in just one step."

**Typos, etc.**
- In line 135 a dot is missing “...respectively Note…”.
- In line 480 a wrong capitailisation “...so Currently…”.
- Figure 1a does not clearly convey the main idea of the method. A more intuitive and visually expressive illustration would help readers better understand the key concepts.

### Questions
- In line 906 it is not clear why the highlighted term is irrelevant w.r.t. $\pi_x$. If I am not mistaken, this terms contains a cross product $v_1^{T}v_0$ which directly depends on the coupling for x because $v_0$ and $v_1$ are determined using them.
- For other comments and questions, see the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper under consideration introduces a framework, called OAT (Optimal Acceleration Transport) aimed at straightening the trajectories of a learned flow model with a novel OT-based technique. The core idea is to leverage the straightening problem in the joint phase space (coordinates times velocities) using a specific OT problem between probability distributions supported on product ($x \times v$) space. In the paper, the authors introduce the OAT problem itself; then adapt the OAT to the task of straightening flow-based models; and finally demonstrate the capabilities of the proposed methodology in a number of low and high-dimensional (e.g., deal with flow- and diffusion- based models on ImageNet 256x256) practical benchmarks.

### Strengths
- I found the narrative of the paper logical and easy-to-follow

- The adaptation and application of Optimal Acceleration Transport (which seems to be by itself a recently introduced theory) in the context of diffusion- and flow- based models is a novel and fresh (to my knowledge) idea in the community.

- The experiments demonstrate the competitiveness of the proposed framework in practice.

### Weaknesses
In a nutshell, the authors take the Optimal Acceleration Transport framework and adapt this framework to the problem of straightening flow-based models. My main concern and question regarding the paper is how this adaptation is achieved. 

- At first, the optimized loss $\ell_{\mathcal{A}}$, eq. (6) is connected with OAT problem **rather indirectly**. To be honest, I do not understand the origin of the procedure described in lines 196-206, and subsequent OAT-FM problem formulation in eq. (7): we sample a trajectory from the original flow, consider an interpolation of the end points, then compute combination of $c_{\mathcal{A}}(z_0, z_t)$ and $c_{\mathcal{A}}(z_t, z_1)$ costs, and then solve an optimization problem with restrictions on top of this, eq. (7). While it is shown that this OAT-FM objective is indeed somehow related to OAT problem (Theorem 3), from my point this relation is rather weak and does not answer the question:  why the authors chose this particular formulation of the problem (eq. 7) rather than some other.

- Secondly, the implication, eq. (9) seems to be incorrect. I checked the reasoning in Appendix C.1 - it seems there is an error in the equations: the authors wrongly ignore velocities components in their deductions: it could not be done, even if you reduce the problem from $\pi$ to $\pi_x$ (lines 901 - 907), because the velocities are hardly connected with coordinates ($v = v(x)$) $\Rightarrow$ if we optimize a coupling between coordinates, we should treat terms like $\Vert v_1 - v_0 \Vert = \Vert v_1(x_1) - v_0 (x_0) \Vert$.

In light of this, I think that the proposed method (which seems to work competitively well in practice) requires better theoretical introduction and explanation.

### Questions
1. Theoretical question, line 162: We have Vlasov equation $\partial_t \mu + v \cdot \nabla_x \mu + \nabla_v \cdot (a \mu) = 0$. How $\nabla_x$ and $\nabla_v$ are defined? And what is $v$ in this equation (I think this is velocity. However, in equation 4, an auxiliary velocity field does not appear explicitly; it only appears implicitly as a sampled quantity from $\mu$.)?

2. From the practical results, e.g., Table 2, I found that compared to OT-CFM, OAT-FM typically leads to less straight trajectories, but better captures the marginal distribution. I found this strange, because OAT-FM (similar to ReFlow) operates with trained trajectories of an FM model, not with the original data, and, therefore, not expected to *improve* the marginals capturing. Could the authors comment on this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
### **Summary of Contribution**

This paper introduces OAT-FM, a novel method for improving existing Flow Matching (FM) generative models. The core idea is to re-frame the flow-straightening objective through the lens of Optimal Acceleration Transport (OAT), which operates in the product space of samples and velocities ($\mathcal{X} \times \mathcal{V}$) rather than just the sample space ($\mathcal{X}$).

The authors argue that standard Optimal Transport (OT) based FM methods enforce a constant-velocity flow ($a=0$), which is a *sufficient* but not *necessary* condition for a straight trajectory. OAT-FM instead aims to satisfy a necessary and sufficient condition (Proposition 1: acceleration parallel to velocity) by minimizing the total acceleration.

The method is proposed as a "two-phase" paradigm:

1.  **Phase 1:** Train any standard FM or diffusion model to obtain "relatively reliable" velocity estimates.
2.  **Phase 2:** Fine-tune this model using the OAT-FM objective. This objective is a bi-level optimization problem where:
    * The **lower-level** computes an optimal coupling in the product space. Critically, this is shown to be decomposable and simplifies to a standard OT problem with a modified cost function (Eq. 9) that incorporates the endpoint velocities ($v_0, v_1$) from the Phase 1 model.
    * The **upper-level** minimizes an acceleration-based loss ($\mathcal{L}_{OAT}$) over this new coupling.

Experiments demonstrate that this Phase 2 fine-tuning consistently improves the performance (e.g., FID) of strong baselines (including FM, EDM, and the SOTA SiT-XL) with a very small number of additional training steps.

### Strengths
1.  **Strong Theoretical Motivation:** The paper provides a clear and compelling theoretical motivation. The distinction between the *sufficient* (constant velocity) and the *necessary and sufficient* (acceleration parallel to velocity) conditions for straight flows is a sharp insight. Grounding the method in the physics of OAT and the Vlasov equation (as noted in Table 1) provides a solid theoretical foundation that is distinct from prior art.

2.  **Pragmatic and Novel Formulation:** The two-phase paradigm is a very clever solution to the "chicken-and-egg" problem. The authors admit OAT-FM is "fragile" from scratch (a key limitation) but deftly turn this into a strength by proposing it as a fine-tuning method that *leverages* the reliable velocity estimates from a pre-trained model.

3.  **Computationally Tractable:** The primary technical hurdle—solving an OT problem in the high-dimensional product space $\mathcal{X} \times \mathcal{V}$—is elegantly overcome. By using the pre-trained model $v_\theta$ to deterministically define $v_0$ and $v_1$, the authors successfully decompose the coupling and reduce the lower-level problem to a tractable OT problem in $\mathcal{X}$ with a modified cost (Eq. 9). This makes the method computationally feasible.

4.  **Strong Empirical Validation:**
    * **SOTA Improvement:** The method demonstrates consistent, measurable improvements on top of very strong baselines. Improving EDM on CIFAR-10 (FID 1.96 $\rightarrow$ 1.93) and SiT-XL on ImageNet (FID 2.11 $\rightarrow$ 2.05) with minimal fine-tuning (12K batches and 5 epochs, respectively) is a significant and practical achievement.
    * **Excellent Ablation Study:** Table 4 provides a strong defense of the method's design. It clearly shows that *both* the new $\mathcal A_2^2$ based coupling (lower-level) and the $\mathcal{L}_{OAT}$ objective (upper-level) are essential. The finding that the standard $\mathcal W_2^2$ coupling *destroys* the pre-trained EDM model (FID 1.96 $\rightarrow$ 8.77) is a powerful piece of evidence for the authors' hypothesis.

### Weaknesses
1.  **Limited Generality:** The method's reliance on a "relatively reliable" pre-trained model is its primary weakness. It is an *improver*, not a standalone generative modeling algorithm. The paper is transparent about this, but it fundamentally limits the scope of the contribution.

2.  **Scalability Bottleneck:** The method inherits the $\mathcal{O}(B^2)$ mini-batch coupling complexity from OT-CFM. This is a known scalability bottleneck that is not solved here. As models and batch sizes (B) grow, this quadratic cost becomes a significant practical barrier. This is mentioned as future work but remains a present weakness.

3.  **Unexplored Sensitivity:** The entire method's success hinges on the quality of the Phase 1 model's velocity estimates. The paper does not investigate the sensitivity to this quality. What happens if the Phase 1 model is mediocre? Is there a performance threshold below which OAT-FM fine-tuning provides no benefit, or even leads to degradation? This is a crucial practical question that is left unanswered.

4.  **Marginal Gains on Large-Scale Tasks:** While the improvements on SiT-XL are consistent, they are marginal (e.g., FID 2.11 $\rightarrow$ 2.05). A skeptic could argue this is a minor tweak for a large computational cost (solving a mini-batch OT problem at every step). The visual improvements in Figures 4-6 are extremely subtle, reinforcing this point.

### Questions
1.  **On Sensitivity to Phase 1 Quality:** Could you expand on the "fragility" of training OAT-FM from scratch? More importantly, can you quantify the relationship between the quality of the Phase 1 model and the efficacy of the Phase 2 fine-tuning? For example, if you take FM models from various checkpoints (e.g., 100K, 200K, 400K batches), does OAT-FM provide more, less, or equal improvement? This would help users understand when to "apply" OAT-FM.

2.  **Analysis of Coupling Failure:** The result in Table 4 where the $\mathcal{W}_2^2$ (standard OT) coupling catastrophically degrades the EDM model is fascinating. What is your hypothesis for why this happens? Does the standard OT plan create pairs $(x_0, x_1)$ that are so "unnatural" for the model's learned velocity field that it breaks the dynamics? Does the OAT-FM coupling (Eq. 9), by incorporating $\bar{v}$, act as a regularizer that keeps the couplings "closer" to the model's existing expectations?

3.  **Practical Cost vs. Benefit:** For the ImageNet experiments, what was the practical wall-clock training time for the 5 epochs of OAT-FM fine-tuning? How does this compare to the time for 5 epochs of *continued* standard training on the original SiT-XL objective? This is essential for evaluating if the marginal FID gain is worth the computational cost of the mini-batch OT.

4.  **Hyperparameter** $\alpha$**:** What value of the hyperparameter $\alpha$ (from Eq. 6) was used in the experiments? Was it fixed to $\alpha=2/3$ as motivated by Theorem 3, or was it tuned as a hyperparameter? How sensitive are the results to this choice?

### Soundness
3

### Presentation
3

### Contribution
3
