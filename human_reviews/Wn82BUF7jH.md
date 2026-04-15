# On Accelerating Diffusion-based Molecular Conformation Generation in SE(3)-invariant Space

- Decision: Reject
- Scores: 5, 3, 3

## Abstract
Diffusion-based generative models in SE(3)-invariant space have demonstrated promising performance in molecular conformation generation, but typically require solving stochastic differential equations (SDEs) with thousands of update steps. 
Till now, it remains unclear how to effectively accelerate this procedure explicitly in SE(3)-invariant space, which greatly hinders its wide application in the real world.
In this paper, we systematically study the diffusion mechanism in SE(3)-invariant space via the lens of approximate errors induced by existing methods. Thereby, we develop more precise approximate in SE(3) in the context of projected differential equations. Theoretical analysis is further provided as well as empirical proof relating hyper-parameters with such errors. Altogether, we propose a novel acceleration scheme for generating molecular conformations in SE(3)-invariant space.
Experimentally, our scheme can generate high-quality conformations with 50x--100x speedup compared to existing methods.
Code is open-sourced at https://anonymous.4open.science/r/Fast-Sampling-41A6.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an acceleration method for diffusion-based molecular conformation generation models. The proposed method involves modifying the score-based model in coordinate space and applying multiplier to the discretization of the SDE. The first modification is motivated by an analysis of the relationship between scores in coordinate and distance spaces, while the second addresses score approximation errors. The proposed sampling technique is applied to two SE(3)-equivariant diffusion models, GeoDiff (Xu et al., 2022) and SDDiff (Zhou et al., 2023), and is compared to the annealed Langevin dynamics sampler. While the proposed sampler shows empirical advantages, the paper falls short in terms of justifying the modifications adequately, and the mathematical details are both lacking in rigor and difficult to follow. Without further elaboration on these details and their underlying motivation, the paper has yet to provide compelling evidence to demonstrate a substantial contribution.

### Strengths
Benchmarks on two conformer datasets (GEOM-Drugs and GEOM-QM9) demonstrate that the proposed method generates conformers of comparable quality and coverage to the reference method, but with a reduced computational time, requiring 50x–100x fewer time steps.

### Weaknesses
The argument presented in this paper relies on the assumption that the set of the valid distance matrices can be endowed with a manifold structure, as stated in Section 3.3: “The manifold of valid distance matrices is a proper sub-manifold of R_+^(n×n).” Subsequently, the pushforward map dφ is employed to map the score functions from distance space to coordinate space. However, the construction of a manifold structure for distance matrices is not trivial, considering that not all “distance matrices” are valid; they must satisfy conditions such as the triangle inequality. The authors should provide relevant references or mathematical construction to support the claims made in the paper.
	The differential geometry arguments in the paper lack clarity. For example, the use of dφ in this paper is more of a “proposal” than a well-defined mathematical construct. Additionally, while d ̃∈M and ∇_d ̃   log⁡〖p_σ (d ̃│d)∈T_d ̃  M〗, it is repeatedly “assumed” in the paper that d ̃+∇_d ̃   log⁡〖p_σ (d ̃│d)∈T_d ̃  M〗, even though the addition between elements of a manifold and its tangent space is not clearly defined.
	The rationale behind the factor of 1/2(n-1) in eq 9 and the unnumbered equation on page 14 is not clear. Since this factor is directly related to the construction of the first modification (eqs 5 and 11), it should be thoroughly explained and justified. It would be better to clarify whether this factor is based on the “rescaling” approximation illustrated in Figure 6.
	The implications of the error bound analysis in Theorem 1 are not clear. At least, the analysis does not support the claim that the proposed solution in eq 9 is “optimal”, as a naive choice of C ̂=C ̃ (the original coordinates) would yield a tighter upper bound of f(d ̂ )=δ^2<(2n^2+n-1)/(2(n-1)^2 ) δ^2 if n>1.
	In Section 4.2, the authors assume that the prediction error of the score-based model is a random error leading to an incomplete time step. However, the model error incurred during the score matching training procedure is more likely to be systematic rather than random, i.e., the model error would depend on the input distances and time steps. Since the following analysis in Section 5.1 assumes a random noise scheme, the authors would need to discuss how the results generalize to the trained score-based models with systematic errors.
	While the method generates conformers of comparable quality to the 5000-step LD sampler, it would be beneficial to demonstrate that the quality is superior to the LD sampler with equivalent time steps. It would be helpful to compare the results with the LD sampler with a similar (~100) or somewhat larger (500 or 1000?) number of time steps in Table 1.

### Questions
In Section 2, FrameDiff (Yim et al., 2023) performs diffusion on frame translations and frame orientations (rotations) rather than torsion angles as mentioned in the paper.
	On the top of page 5, symbols α_ij and e_ij are introduced without explanation. Also, it would be nice to provide a brief explanation or reference to the generalized multidimensional scaling.
	In the sentence just before eq 11, the meaning of the phrase “the pairwise distance matrix is sparse” is not clear. It might be better to reformulate the optimization problem in eq 8 by replacing the summation over all i < j pairs with connected (i, j) pairs.
	In Appendix G, the MAT-R metric should be corrected as follows:

MAT=1/|S_r |  ∑_(C∈S_r)▒min┬(C^'∈S_g )⁡RMSD(C,C^' )

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, two modifications to the existing sampling methods of the diffusion model are proposed to improve the sampling rate for generating molecular conformations in SE(3) invariant spaces.

### Strengths
- This paper is well-organized.
- The paper focuses on accelerating the sampling process of the diffusion model for conformation generation, which is an interesting topic.

### Weaknesses
1. The paper is not well-motivated. The proposed sampling method is specifically designed for the task of molecular conformation generation in SE(3)-invariant space.  However, the SOTA method of conformation generation like Torsional Diffusion which operates on the hypertorus achieves much better performance than GeoDIFF which operates on Euclidean space. Moreover, the sampling process of Torsional Diffusion only needs 20 steps. Therefore, focusing on accelerating the sampling process of GeoDIFF/SDDiff seems meaningless.
2. The contribution of the introduced two minor modifications is limited. The operations just introduced two coefficients into existing sampling methods[1] to compensate for the poor performance on the conformation generation task by using [1] directly. From the experiments, the proposed methods don't get surprising results.
3. The paper is over-clams. The paper doesn’t provide the theoretical results about the statement “We analyze current modeling methods in SE(3) (Shi et al., 2021; Xu et al., 2022; Zhou et al., 2023) and theoretically pose crucial mistakes shared in these methods, which inevitably bring about the failure of acceleration” and don’t give the reasons why the modification solve the theoretical problems. If the proposed modifications can solve the problems, at least it can be directly used to improve the existing sampling methods applied in (Shi et al., 2021; Xu et al., 2022; Zhou et al., 2023).
4. The paper is not well-written. The paper hasn’t explained clearly why the introduced two modifications can accelerate the sampling process significantly. For modification one, the introduced $degree_i$ can make the sparse conformation into a fully connected conformation. Why connected conformation can outperform sparse one? 
5. The proposed sampling method mainly improves the recall but gains worse performance in precision. Therefore, this method, which is only used for conformation generation, is not very practical.
6. The experimental results of baseline GeoDiff are significantly lower than the results reported in the GeoDiff paper.

[1] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

### Questions
See section weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes novel strategies to improve diffusion-based molecular conformation generation models to achieve more accurate probabilistic modeling and faster generation. The proposed strategies include use average operation in coordinate score calculation, and multiplying a hyperparameter "scale" in generation. The proposed methods improves GeoDiff and SDDiff in some metrics of molecular conformation generation tasks.

### Strengths
Originality: The proposed strategies in this paper is novel.  
Quality: Experiments on benchmark datasets show the porposed strategies can improve recall rates of GeoDiff and SDDiff in molecular conformation generation.  
Clarity: No.  
Significance: The proposed strategies can be useful and enlightening for developping score-based diffusion models for molecular conformation generation.

### Weaknesses
Major:  
(1) Section 4.1 is not well-organized or logically smooth. It is hard to understand the overall process to obtain Equation (9) due to lack of clarification of many mathematical notations and motivations. Authors are encouraged to rewrite this section and give more clarification and explanation to the following questions:
- What are the meanings of TM, TN, $\pi_M, \pi_N$ in Figure 1?
- Why $f=\pi_N\circ d\phi = \phi \circ \pi_M$? What are the processes described by $\pi_N\circ d\phi$ and $\phi \circ \pi_M$?
- Why $\pi_M$ is chosen to be an identical mapping, $\pi_N$ is chosen to be a GMD?
- What does $e_{ij}$ mean in the formula of $\hat{d}$, and how the formula of $\pi_{M,\tilde{d}}(\hat{d})_{ij}$ and subsequently Equation (8) is derived?
- In Appendix D.2 (proof of theorem 1), what is $\lambda_{uv}$? Why does the second $\le$ (from line 30b-30c to line 30d-30g) hold?

(2) In Section  4.2, authors are recommended to clarify what is $\tilde{C}_{t+\lambda(s-t)}$? 

Also, it would be better to discuss how much error will be introduced by the approximation $k_{s_\theta}(d_s, s, t) \approx k_{s_\theta}(p_{data})$?  
(3) It will make the experimental results stronger if authors do ablation study to verify that every single strategy can improve the performance. Also, authors are recommended to make an in-depth discussion about why the proposed method achieves poor performance in precision metric.

Minor:  
Typos: In abstract, "develop more precise approximate" --> "develop more precise approximation"  
In Section 2 first paragraph, "so that the lengths of atom bounds" --> "such as the lengths of atom bonds"

### Questions
The proposed strategies are mainly used for diffusion models that calculate coordinate scores from distance scores. However, current state-of-the-art molecular conformation generation method, Torsional Diffusion [1], focusing on generating torsion angles. Can GeoDiff or SDDiff compete with Torsional Diffusion in generation speed when the proposed strategies are used? Can the proposed strategies or similar strategies be also applied to Torsional Diffusion?

[1] Torsional Diffusion for Molecular Conformer Generation. NeurIPS 2022.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
