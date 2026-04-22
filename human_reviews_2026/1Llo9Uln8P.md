# Overclocking Electrostatic Generative Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Electrostatic generative models such as PFGM++ have recently emerged as a powerful framework, achieving state-of-the-art performance in image synthesis. PFGM++ operates in an extended data space with auxiliary dimensionality $D$, recovering the diffusion model framework as $D\to\infty$, while yielding superior empirical results for finite $D$.
Like diffusion models, PFGM++ relies on expensive ODE simulations to generate samples, making it computationally costly. To address this, we propose Inverse Poisson Flow Matching (IPFM), a novel distillation framework that accelerates electrostatic generative models across all values of $D$. Our IPFM reformulates distillation as an inverse problem: learning a generator whose induced electrostatic field matches that of the teacher. We derive a tractable training objective for this problem and show that, as $D \to \infty$, our IPFM closely recovers Score Identity Distillation (SiD), a recent method for distilling diffusion models.
Empirically, our IPFM produces distilled generators that achieve near-teacher or even superior sample quality using only a few function evaluations. Moreover, we observe that distillation converges faster for finite $D$ than in the $D \to \infty$ (diffusion) limit, which is consistent with prior findings that finite-$D$ PFGM++ models exhibit more favorable optimization and sampling properties.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Inverse Poisson Flow Matching (IPFM), a distillation framework designed to train a single-step generator from a teacher Poisson Flow Generative Model. The method optimizes a generator such that the electrostatic field induced by its output distribution matches that of the real data, a concept closely tied to variational formulations of diffusion distillation. By analyzing the infinite-dimensional limit of Poisson flow, the authors establish a theoretical connection to Score Identity Distillation (SiD) [1], a known diffusion distillation approach. This insight motivates a regularization strategy that further enhances performance. Experimental results on CIFAR-10 and FFHQ demonstrate that IPFM can effectively distill a single-step generator from a teacher model, achieving competitive results.

### Strengths
1. **Sound Theoretical Formulation**  
   The proposed idea of aligning the electrostatic field of the generator’s output distribution with the ground-truth field is simple and intuitive.  

2. **Novel Connection to Diffusion Distillation**  
   The connection between Poisson flow matching and SiD [1] is both original and insightful, providing a bridge between finite- and infinite-dimensional flow formulations.  

3. **Clear Writing and Presentation**  
   The paper is clear and well written

### Weaknesses
1. **Lack of Theoretical Guarantees for the Objective**  
   While the introduction of the inverse Poisson flow matching loss is intuitive, the paper lacks formal proofs showing that minimizing the loss reduces the divergence of the generator distribution to the target distribution. Additionally, a proof demonstrating that the divergence becomes zero when the loss is zero would strengthen the theoretical soundness and rule out degenerate solutions.  

2. **Limited Experimental Evaluation**  
   The experiments are restricted to relatively simple datasets such as CIFAR-10 and FFHQ. To validate IPFM’s scalability and robustness, results on more challenging benchmarks, such as **ImageNet**, are needed, particularly since these datasets are common in diffusion and distillation research.  

3. **Insufficient Ablation Studies**  
   Ablation experiments are limited to only two values of the regularization coefficient ($\alpha$). A broader range of values should be explored, especially the critical case ($\alpha = 1/2$), where the method reduces to the original IPFM without the new regularizer. This would help clarify the contribution of the proposed regularization term.

### Questions
1. **Provide Theoretical Proofs for the Objective Function**  
   Formal theoretical analysis would help establish guarantees on distributional divergence and justify the use of the proposed objective. Specifically, proofs bounding the divergence between the learned and target distributions, and demonstrating that the divergence is zero when the loss is zero, would improve the paper’s rigor.  

2. **Expand Experimental Scope and Ablations**  
   Future versions of this work should include larger-scale experiments (e.g., ImageNet) and a more thorough study of the impact of the regularization parameter ($\alpha$). This would validate the method’s generalizability and robustness.  

3. **Clarify the Advantage of Finite-Dimensional Poisson Flow Distillation**  
   The paper should better articulate the practical advantage of distilling finite-dimensional Poisson flows compared to their infinite-dimensional counterparts. From the reported results, both seem to yield similar outcomes, so further discussion is needed to justify why the finite-dimensional distillation is valuable in practice.  

4. **Comparison with Naïve Distillation Baselines**  
   Since Poisson flows can also be represented as ODE systems, it would be beneficial to compare IPFM with straightforward trajectory distillation methods such as **Consistency Training Models (CTM)** [2] and **Consistency Models (CM)** [3]. Evaluating these baselines would contextualize the improvements introduced by IPFM and demonstrate its advantages beyond standard ODE-based distillation.

---

## References  
[1] **Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation.**  
[2] **Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion.**  
[3] **Consistency Models.**

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
The paper proposes Inverse Poisson Flow Matching, a distillation framework for electrostatic generative models, specifically PFGM++, to reduce the number of function evaluations needed at sampling time. The key idea is to view distillation as an inverse problem: given a teacher electrostatic field (learned on real data via PFGM++), learn a generator whose induced electrostatic field matches the teacher’s field. The authors show that the original constrained problem is equivalent to a tractable minimax objective and further express it in a denoising form. They also prove that, in the diffusion limit $D \to \infty$, their objective is closely related to Score Identity Distillation (SiD), which "justifies" reusing regularization style and hyperparameters.

### Strengths
- The paper gives a clear and principled formulation of distilling electrostatic models as an inverse flow matching problem.
- The connection to SiD in the $D \to \infty$ limit is well motivated.
- In the experiments presented, the method shows clear acceleration.

### Weaknesses
- Despite the elegant reformulation, the final training problem is still a minimax / alternating optimization over the generator and student denoiser. To me, this seems unstable and (as acknowledged by the authors) much more expensive than training the teacher model.
- The claim that one can directly transfer SiD hyperparameters to the general (finite-D) IPFM setting is only a heuristic rather than a theorem. The paper itself reports architecture-dependent divergence for $\alpha=1.2$, which doesn't really support the universality of this transfer.
- The role of the weighting function is underexplored. Authors just inherit EDM/SiD choices and reparameterize via $r = \sigma \sqrt D$. There’s no ablation to show robustness to \lambda, nor results for other data types or samplers.
- The experimental section is narrow: two unconditional image datasets, no higher-resolution or conditional generation, and no direct comparison to other fast distillation methods on the same teacher.
- IPFM applies only to electrostatic generative models, which remain niche compared to mainstream diffusion or flow models—so its practical impact is currently limited.

### Questions
1. Theorem 3.2 turns the constrained problem into a minimax one, but in practice, this still means alternating updates to the student and generator. Do the authors observe GAN-like instabilities (cycling, student overpowering the generator, or vice versa), and do they need multiple student steps per generator step? Some clarification on the actual training schedule would help assess feasibility.

2. The paper argues that the D \to \infty case “recovers” SiD and thus hyperparameters can be transferred. But for finite D, the perturbation kernel is heavier-tailed and the geometry is different. Do the authors have evidence that hyperparameters from SiD fail only because of architecture mismatch (ncsn++ vs ddpm++), or are there D-specific failure modes too?

3. How sensitive is IPFM to the choice of $\lambda(r)$? In particular, if the generator early in training produces very off-manifold samples, does the EDM-style $\lambda$ still give a stable learning signal to the student, or would a schedule that upweights large $r$ be preferable?

4. Are there ways to freeze the student for several iterations or to use something like the EMA students to reduce the 3x memory/cost burden?

Minor:
- The related work section is over two pages out of nine. Although quite informative, it would be more valuable to have more experiments/important discussions about the method instead.
- Please, in the main text after Theorem 3.2, refer the reader to its proof in the appendix.

I believe that these questions are important to answer and also that, if the authors (i) add a subsection on $\lambda$ choices and sensitivity, (ii) include at least one head-to-head with SiD (same dataset, same budget), and (iii) slightly expand the experiments to more datasets or a conditional setting, the paper could be stronger.

### Soundness
3

### Presentation
3

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
The paper introduces Inverse Poisson Flow Matching (IPFM), a novel method derived from first principles for distilling Poisson Flow Generative Models (PFGM++). The authors establish a theoretical connection between IPFM and Score Identity Distillation (SiD), a technique used for diffusion models. Leveraging this connection, they successfully adapt a regularizer from SiD, which is shown to empirically benefit IPFM.

Empirically, the distilled models achieve performance competitive with the original teacher model, and in some cases, even appear to surpass it. A significant advantage of the proposed method is that it is data-free, requiring no access to the original training dataset for the distillation process.

### Strengths
While model distillation is a well-studied field for standard diffusion models, its application to Poisson Flow Generative Models (PFGM++) remains largely unexplored. The primary strength of this paper is its rigorous, first-principles derivation of a distillation method specifically tailored for PFGM++.

Instead of merely adapting existing diffusion model techniques, the authors provide a principled validation for their approach. This theoretical grounding builds confidence in the method's correctness and suitability for the PFGM++ framework. Although PFGM++ is a more specialized approach compared to mainstream diffusion models, this work represents a significant contribution by addressing an important and under-researched area in generative model distillation.

### Weaknesses
The paper's primary weakness is the omission of a crucial and highly relevant prior work, [1], which adapts Consistency Distillation for PFGM++. While the authors do not explicitly claim to be the first to distill PFGM++, the manuscript is framed in a way that implies this, and the lack of discussion or comparison to [1] is a significant oversight.

For a comprehensive evaluation, an empirical comparison against the method proposed in [1] is essential. At a minimum, the authors must contextualize their work by thoroughly discussing this existing approach. Without this comparison and discussion, it is difficult to assess the relative contribution and performance of IPFM, which hinders a positive evaluation of the submission.

Minor Comments:
- The empirical validation would be strengthened by including experiments on larger-scale datasets to further assess the method's scalability and robustness.
- The repetition of Equation (21) as Equation (22) with minimal separation appears redundant and could be consolidated.

[1] Hein, Dennis, et al. "PFCM: Poisson flow consistency models for low-dose CT image denoising." IEEE Transactions on Medical Imaging (2025).

### Questions
1. The proposed method is named "Inverse Poisson Flow Matching" (IPFM). Why did the authors choose not to feature this name in the paper's title, given that it is the central contribution?
2. Regarding the notation on Line 99: should the term x′ be x (without the prime)? Please clarify this potential typo.
3. In Table 1, the distilled models are compared to the teacher model at 35 NFE (CIFAR-10) and 79 NFE (FFHQ). The authors claim "even superior sample quality" for the distilled model. However, it is unclear if the teacher model's performance has saturated at this NFE. Could the authors provide results for the teacher model at a higher NFE to confirm that this is a fair comparison and that the distilled model is not just outperforming a sub-optimal, non-converged teacher?

### Soundness
4

### Presentation
1

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
The authors proposed  a distillation method for PFGM++ a model which is somewhat tangential to diffusion models (and is identical in the limit of $D \to \infty$). Like in prior work the authors observe that small values of $D$ result in faster convergence for the distilled models.

### Strengths
* Explores distillation for a novel class of generative models (PFGM and PFGM++)
* Non-trival results
* Promising results on faster convergence with small $D$
* Highly interesting topic

### Weaknesses
## Primary Concerns
I like this paper and would be inclined to give it a higher score, if the explanation of the core concepts are more intuitively illustrated.

It is very hard to visualize what PGFM++ does for generative modeling and consequently what the contributions are. What role does $\mathbf z \in \mathbb R^D$ play, why is it necessary. What is the physically meaning of $\mathbf z$? My rough understanding is that as $D \to \infty$, $r \to \sigma$ or $t$?

Most of my questions regarding clarity reside with the questions section so I will refer the authors to that section.

Really my only primary concern is the clarity in communicating the concepts well so the reader can understand it.

### Questions
1. It isn't clear to me that in line 150 $f_\phi: R^{N+1} \to \mathbb R^{N+1}$. Wouldn't it be $f_\phi: R^{N+D} \to \mathbb R^{N+D}$?
2. How do you pick $r_max$? What is it physical meaning.
3. The claim that PFGM++ is diffusion in the limit of $D \to \infty$ is not obvious, why is this? What is the physical meaning?
4. What does integrating along $r$ mean?
5. In Figure 1 and Section 3.1 the authors state the goal is to learn $G_\theta$, but I don't see it in any of the equations in Section 3.1 or 3.2. Where is it and what does it do? From Figure 1 I suppose the signature is $G_\theta: \mathbb R^D \to \mathbb R^{N+D}$ but this is never explicitly stated. What is it?
6. Isn't the goal to learn $f_\psi$ not $G_\theta$ if this understanding is wrong, why? **It is not clear from the text.**
7. In Table 1 the authors state that $D = \infty$, but this doesn't seem tractable. Again clarity.

### Soundness
3

### Presentation
1

### Contribution
3
