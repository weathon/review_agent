# Diffusion & Adversarial Schrödinger Bridges via Iterative Proportional Markovian Fitting

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
The Iterative Markovian Fitting (IMF) procedure, which iteratively projects onto the space of Markov processes and the reciprocal class, successfully solves the Schrödinger Bridge (SB) problem. However, an efficient practical implementation requires a heuristic modification-alternating between fitting forward and backward time diffusion at each iteration. This modification is crucial for stabilizing training and achieving reliable results in applications such as unpaired domain translation. Our work reveals a close connection between the modified version of IMF and the Iterative Proportional Fitting (IPF) procedure-a foundational method for the SB problem, also known as Sinkhorn’s algorithm. Specifically, we demonstrate that the heuristic modification of the IMF effectively integrates both IMF and IPF procedures. We refer to this combined approach as the Iterative Proportional Markovian Fitting (IPMF) procedure. Through theoretical and empirical analysis, we establish the convergence of the IPMF procedure under various settings, contributing to developing a unified framework for solving SB problems. Moreover, from a practical standpoint, the IPMF procedure enables a flexible trade-off between image similarity and generation quality, offering a new mechanism for tailoring models to specific tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows that a heuristic bidirectional modification of the Iterative Markovian Fitting (IMF) procedure such as Diffusion Schrodinger Bridge Matching or Adversarial Schrodinger Bridge Matching is also a valid algorithm for solving the Schrodinger Bridge Problem (SBP). The authors prove that this heuristic modification of IMF, dubbed Iterative Proportional Markovian Fitting (IPMF), converges exponentially to the solution when the marginals are Gaussian. IPMF is also shown to converge to the solution under a weaker condition, when the marginals have bounded supports. In contrast to IPF or IMF, IPMF can be initialized with any starting process. The authors show the benefits of flexible initialization with experiments on Gaussians, SB benchmarks, unpaired image-to-image translation, etc.

### Strengths
- **[S1] This paper is significant in the aspect that it bridges a gap between theory and practice.** Specifically, the paper provides several proofs regarding the convergence of a heuristic modification of the IMF procedure, while proofs in previous works [1,2] show the convergence of the original IMF procedure.

- **[S2] This paper is original in the aspect that it provides a novel insight into better training of SBs.** The authors also show that unlike IMF or IPF, IPMF admits any initial starting process. Hence, one may potentially achieve faster training of SBs via IPMF by using a well-designed starting process.

[1] Diffusion Schrödinger Bridge Matching

[2] Adversarial Schrödinger Bridge Matching

### Weaknesses
- **[W1] Experimental results only weakly support the benefits of using arbitrary starting processes.** Experiments in Section 4 do not really show the strength of using arbitrary starting processes in terms of both scale and performance. In terms of scale, the only non-trivial task is male to female on CelebA-64. In terms of performance, non-IMF initializations such as SD SDEdit, DDPM SDEdit, or Identity all suffer from FID degradation at the cost of smaller MSE between $x_0$ and $\widehat{x}_1$. Previous works on SB such as DSBM or ASBM demonstrate consistent performance improvements on more difficult data with resolution $\geq 128$, so it would be nice to see results on similar data in this paper as well.

### Questions
- **[Q1] In Table 2, can the authors provide FIDs for initial processes as well?** I believe this FID is necessary in order to judge whether there are performance gains after running IPMF on the given initial process such as SDEdit or Identity.

- **[Q2] In Table 2, can the authors provide LPIPS between $x_0$ and $\widehat{x}_1$?**

- **[Q3] Can the authors provide results with larger $\epsilon$ when starting with the identity process?** I am curious whether this improves FID by increasing diversity in the outputs.

- **[Q4] Do observations made on CelebA 64x64 scale to higher resolution images?**

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
5

### Summary
Schr\"odinger Bridge techniques have been widely used to address unpaired data-to-data translation tasks. This paper focuses on a practical heuristic implementation of the Iterative Markovian Fitting (IMF) procedure that has been proposed in the literature because it improves empirical performance. The central observation in this paper is that the ``bidirectional'' implementations of IMF used in practice (namely Diffusion Schr\"odinger Bridge Matching (DSBM) and Adversarial Schr\"odinger Bridge Matching (ASBM) implicitly interleave both IMF and Iterative Proportional Fitting (IPF) updates. The authors formalize this approach as IPMF and analyze its convergence properties. Conceptually, IPMF alternates (1) reciprocal/Markovian projections (IMF), with (2) projections that enforce boundary distributions (IPF).

In the Gaussian case, the paper introduces an optimality matrix $A(q)$ for couplings and shows that any two-variable Gaussian coupling with D-dimensional marginals plan is the entropic Optimal Transport (OT) solution for a suitable bilinear cost $c_A(x_0,x_1)=-x_1^\top A x_0$ (Theorem 3.1). It then proves exponential convergence (Theorem 3.2;) for various cases : $D=1$ (discrete/continuous, any $\varepsilon>0$) and $D>1$ (discrete time, sufficiently large $\varepsilon$)} of IPMF to the SB (1D in continuous/discrete time; $D>1$ in discrete time under a large-$\varepsilon$ regime). Beyond the Gaussian settings, it establishes weak convergence under boundness assumptions (Theorem 3.3). 

Finally, the paper provides numerical experiments: convergence on high-dimensional Gaussians, a 2D toy problem, quantitative results on the SB benchmark (Table~1) , and qualitative unpaired image-to-image translation on Colored MNIST and CelebA. Beyond unification, the paper shows that selecting different initial couplings yields a controllable trade-off between input-output similarity and generative fidelity.

### Strengths
Clarity: The paper is well-written and the arguments are presented with excellent clarity. Theoretical concepts are well-motivated and well illustrate; e.g. Figure1 illustrates neatly how IPF and IMF projections combine within the IPMF framework.

Originality: As clearly acknowledged by the authors, the proposed method is not new but was presented as some heuristic implementation of IMF in the literature. The originality of the paper is to formalize it as an explicit technique alternating IMF and IPF.  This is significant as it connects two separate approaches in the literature, i.e. Sinkhorn/IPF and IMF/flow matching-type ideas, into a unified procedure. In particular, it helps understanding why bidirectionality helps mitigate the "prior forgetting" behaviour of the diffusion implementation of  IPF alone) and error accumulation of the naive implementation of IMF alone (which motivated the introduction of the bidirectional scheme). This will be of interest to people working on unpaired data-to-data translation.

Quality: Beyond formalizing the bidirectional method as a rigorous combination of IPF/IMF, the paper presents some interesting theoretical results. In particular, it provides (to the best of my knowledge) the first theoretical analysis of the bidirectional variant used in practice. While limited in scope, the analysis of the Gaussian case is neat and rigorous. The empirical section is fairly complete, including high-dimensional Gaussians, a 2D toy example, the SB benchmark, and real images (Colored MNIST, CelebA) . It also includes convergence diagnostics (KL forward/reverse etc.). 
The literature in this domain is plagued with over-the-top claims, this paper does not make any such claim and remains very factual which I really appreciate.

Significance and Practical Impact: The experiments convincingly illustrate that  different couplings induce a principled, tunable trade-off between input-output similarity and generative fidelity. This is useful for practitioners working on unpaired translation.

### Weaknesses
Theory results: They remain limited. Exponential convergence is only proved in the Gaussian case for (a) $D=1$ continuous/discrete settings and (b) $D>1$ in discrete time under a fairly restrictive large-$\varepsilon$ condition. For applications, we care about $D$ large and non-Gaussian distributions. In this case, exponential convergence is only conjectured.  The authors should try to clarify the practical implications of the large-$\varepsilon$ assumption. It  would be helpful to quantify how large $\varepsilon$ must be for the proof to hold in typical image dimensions, and whether this aligns with the values used in practical DSBM implementations.

Continuous-Time IPMF: The continuous-time version of IPMF is given in (22)-(23) but theoretical guarantees for that case are not provided beyond $D=1$ for the Gaussian case. The authors should discuss the specific challenges in extending the contraction arguments to the continuous-time and high-dimensional setting.

Benchmarking & Ablations: On the SB benchmark, performance is sometimes on par with prior matching solvers and sometimes worse (e.g., DSBM variants at higher $D$/$\varepsilon$ show large $\mathrm{cBW}^2_2$ errors). The paper argues that different initializations converge to similar outcomes within each solver, but stronger head-to-head comparisons would make the empirical case more compelling.

### Questions
Tightness and Practicality of the Large-$\varepsilon$, Can you quantify how large $\varepsilon$ needs to be in the discrete $D>1$ proof for typical image resolutions (i.e., dimension $D$), and how this compares to $\varepsilon$ used in DSBM practice? 

Continuous-Time Guarantees. Is there any way you could extendthe Gaussian contraction argument to continuous time beyond $D=1$ (e.g., by controlling the Markovian projection via stability of SDE discretizations)?

Number of IPMF Rounds and Stopping Criteria. How sensitive are results to the number of  IPMF iterations? Can you think of a reliable practical stopping criterion?  A small experiment (quality/similarity vs.\ rounds) on the CelebA dataset would be good.

Trade-off via Starting Couplings. The Identity and SDEdit couplings improve similarity, sometimes at the expense of FID. Could you relate the similarity/quality outcome to properties of the initial coupling (e.g. entropy or cost)?

Relation to Rectified Flows. It is correctly suggested that bidirectional IPMF could mitigate error accumulation in rectified flow. Could you provide some results on a small dataset (e.g. CIFAR10) showing that IPMF (with \epsilon=0) stabilizes Rectified Flow training? I conjecture using \epsilon decreasing to zero across iterations might even work better.

Positioning. Given the novelty relative to prior bidirectional heuristics (DSBM/ASBM), I think the paper would benefit from a short table in the appendix clarifying the  precise technical advances of IPMF (e.g., convergence rates, new unifying formulation, starting coupling analysis) over these related works.

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
This paper proposes a new Schrodinger Bridge training method for combining IMF and IPF.

### Strengths
It combines the two processes of IMF and IPF, with alternating method.
It theoretically combined the IPF and IMF with thorough analysis and also proposed to apply proposed method to both of DSBM and ASBM.

### Weaknesses
1. Although it proved convergence in gaussian and bounded support, it does not contain any generalization or guarantee on other general distrubutions. 

2. To apply this proposed method, it requires alternating process for IMF and IPF. It takes heavier computation burden compared to previous processes. Although the paper proposes faster convergence, it does not contain any comparison between previous methods in terms of computation efficiency and time complexity.

3. The performance heavily rely on initial hyperparameter setting such as initial coupling choice.

4. The most important and critiral issues is the practical usage of proposed method. Although the theoretical analysis and proposed methods is quite new, but the shown experiments are only limited on simple dataset such as toy data , colored MNIST and CelebA. Also the used dataset has relatively low resolution. To prove the practical generalizability of proposed method, the paper must include experiments on more dataset with higher resolution.

### Questions
See weakness

### Soundness
3

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
4

### Summary
The paper introduces **Iterative Proportional Markovian Fitting (IPMF)**, a unified framework connecting **Iterative Proportional Fitting (IPF)** and **Iterative Moment Fitting (IMF)** in stochastic bridge learning.  
The authors show that practical bidirectional IMF procedures implicitly perform IPF-like proportional updates, and formalize this equivalence into IPMF with convergence guarantees in Gaussian settings.  
Experiments on synthetic and low-resolution image domains demonstrate stable coupling and controllable trade-offs, situating existing DSBM and ASBM methods as special cases.

### Strengths
- Provides a **theoretical unification** between IPF and IMF with clear mathematical derivation.  
- Clarifies the conceptual foundation of recent bridge-based diffusion methods.  
- Demonstrates improved stability under diverse initial couplings.

### Weaknesses
*Limited novelty:** IPMF mainly reinterprets existing IMF practices under an IPF perspective; lacks a genuinely new algorithm.  
- **Narrow empirical scope:** Evaluations are confined to Gaussian and 64×64 image settings without large-scale or continuous-time experiments.  
- **Unclear practical benefit:** No evidence of faster convergence or lower compute cost compared to IMF or DSBM.

## Minor
- Heavy notation reduces accessibility.  
- Missing comparison with recent OT/consistency bridge baselines.

### Questions
1. Beyond theoretical unification, does IPMF yield measurable training or sampling improvements over IMF or DSBM?  
2. Is the convergence guarantee valid for non-Gaussian or continuous-time bridges?  
3. How sensitive is IPMF to the choice of starting coupling (e.g., SDEdit vs. identity)?  
4. Can IPMF incorporate stochastic regularization (e.g., adversarial or entropy terms) without breaking convergence?  
5. Are there cases where separate IPF or IMF updates remain preferable in practice?

### Soundness
3

### Presentation
3

### Contribution
2
