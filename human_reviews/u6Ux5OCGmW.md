# RSAM: Learning on Manifolds with Riemannian Sharpness-Aware Minimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 5

## Abstract
Nowadays, understanding the geometry of the loss landscape shows promise in enhancing a model's generalization ability. In this work, we draw upon prior works that apply geometric principles to optimization and present a novel approach to improve robustness and generalization ability for constrained optimization problems. Indeed, this paper aims to generalize the Sharpness-Aware Minimization (SAM) optimizer to Riemannian manifolds. In doing so, we first extend the concept of sharpness and introduce a novel notion of sharpness on manifolds. To support this notion of sharpness, we present a theoretical analysis characterizing generalization capabilities with respect to manifold sharpness, which demonstrates a tighter bound on the generalization gap, a result not known before. Motivated by this analysis, we introduce our algorithm, Riemannian Sharpness-Aware Minimization (RSAM). To demonstrate RSAM's ability to enhance generalization ability, we evaluate and contrast our algorithm on a broad set of problems, such as image classification and contrastive learning across different datasets, including CIFAR100, CIFAR10, and FGVCAircraft. Our code is publicly available at \url{https://t.ly/RiemannianSAM}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work introduces the novel notion of Sharpness on Riemannian manifolds and proposes a tighter upper bound. The authors also introduce RSAM, which considers the parameter space’s intrinsic geometry and seeks regions with flat surfaces on Riemannian manifolds. They also provide empirical results to show the effectiveness of RSAM.

### Strengths
The paper is the first to consider the sharpness of parameters lying on a manifold, which has potential to be an interesting branch of SAM. The empirical results are supportive and reasonable.

### Weaknesses
1. There is no equation number in section 3.3. Also, what is the RHS of the second equation in section 3.3? 
2. The code link provided has expired, so I can't reproduce your results.
3. The error bars in Table. 1 and 2 are missing.

### Questions
1. How did you get eq 2 from the last row of the eq above eq 2? Why omit $\mathcal{L}_{\mathcal{S}}$?  (please add equation numbers)
2. Could you explain how $\mathcal{R}_\theta$ works in practice? Is it like a projection to manifold space?
3. Why in the last line of Alg 1 there is another $\mathcal{R}_{\theta_t}$? Could you explain?
4. Just curious, what if $\mathbf{D}$ has more degree of freedom, like a function of all $\theta$?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes employment of SAM on Riemannian manifolds. The proposed methods were explored on several image classification datasets with Resnet architectures.

### Strengths
The proposed RSAM boosts accuracy of SAM in a few image classification tasks.

### Weaknesses
There are two major problems with the paper.

First, several statements used to describe the proposed method and its implementation in the paper are not clear.

Second, the experimental analyses are limited. The proposed RSAM should be examined on additional DNN architectures, datasets and larger category of Riemannian manifolds in comparison with the other Riemannian optimizers and SAM optimizers.

### Questions
-	In the paper, it is stated that “we imposed orthogonality on a single convolutional layer in the middle of the architecture in all settings”. This statement is not clear. How did you define the “single convolution layer” more precisely? Did you just add orthogonality to one layer?

-	It is stated that “Since U is constrained to lies on the Stiefel manifold, we will optimize it with RSAM, and the rest of the parameters, including the backbone and the diagonal matrix S, will be learned via traditional optimizers such as SAM or SGD”. The S can be optimized using RSAM as well, since it is a diagonal matrix residing on a Riemannian manifold. How does the accuracy change when it is optimized by SAM, RSAM, SGD?

-	Can you provide the results obtained using additional optimizers such as Riemannian SGD, Adam, Riemannian Adam, and AdamW?

-	A similar work was recently published in the Neurips; Yun and Yang, Riemannian SAM: Sharpness-Aware Minimization on Riemannian Manifolds. A direct comparison with this work may not be possible since their code/paper is not completely available. However, as they mentioned in the abstract, such a work on SAM should be compared with the other SAM methods such as Fisher SAM on a more general category of Riemannian manifolds.

### Soundness
1 poor

### Presentation
1 poor

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
This paper proposes Riemannian sharpness-aware minimization (RSAM), which extends the original SAM algorithm to the case of parameters residing within Riemannian manifolds. The authors first establish the notion of sharpness for loss landscapes defined on Riemannian manifolds. Subsequently, they provide a theoretical analysis relating the sharpness to the generalization gap and propose the RSAM algorithm, designed to minimize the sharpness augmented loss. To demonstrate the effectiveness of the RSAM algorithm, experiments on image classification and constrastive learning tasks are performed, focusing on the parameters defined on Stiefel manifolds.

### Strengths
- This paper provides an efficient extension of the SAM algorithm for constrained parameter spaces, accompanied by a theoretical analysis of the generalization gap.
- Experimental results indicate some performance improvements.

### Weaknesses
- There seems to be an inconsistency in defining neighborhoods in Section 3.1 and the choice of Riemannian metric in the experiments, which can confuse the readers significantly. The Riemannian metric $D_\theta$ seems to be the ambient space metric. If this is the case, the norm $||\cdot||$ should be defined using $D_\theta$, but all derivations in Section 3 are based on assuming $D_\theta = I$ (as per the proofs in Appendix A.1), implying the Euclidean ambient space. However, experiments employ $D_\theta$ different from the identity, of which the choice seems arbitrary.
- The claim of providing a tighter bound than SAM should be more carefully nuanced. The parameter spaces possessing a manifold structure are of little concern in the original SAM paper. Therefore, it would be more accurate to state that the provided bound is tighter ‘when the parameter spaces have much smaller dimensionality than the ambient space’ rather than making a general comparison to SAM.
- RSAM seems to be a straightforward generalization of SAM to Riemannian manifolds, which might be considered a minor contribution unless the paper includes case studies applying the proposed algorithm to a range of Riemannian manifolds. While the Stiefel manifold considered in the paper is a relevant example, including application examples on other Riemannian manifolds would be beneficial.
- Even though the experimental results suggest some performance advantages of using RSAM, the analysis is not sufficiently thorough. The primary reason for the improvement appears to be the use of the R-Stiefel layer, and the comparison of RSAM with SGD and SAM without the R-Stiefel layer may not be fair. For a more precise analysis of the generalization benefit of RSAM, further experimental studies are needed, such as comparing it to Riemannian SGD, which also employs the R-Stiefel layer.
- The paper would benefit from clearer writing, particularly in Section 4.

### Questions
- How do the choices of hyperparameters, such as $\rho$ for RSAM and SAM, influence the results in Section 5, and how were these hyperparameters selected?
- When obtaining the Hessian spectral in Appendix A.2.2, shouldn’t the geometry, e.g., Riemannian metric $D_\theta$, be considered?
- The concept of retraction is used frequently without a precise definition. How is retraction defined?

[Typos]
- At the beginning of Section 3.2, it should read: $\mathcal{M} \subseteq \mathbb{R}^k$.
- In Section 3.3, the omission of $\mathcal{L}_\mathcal{S}$ in deriving the objective function and in equation (3) should be corrected.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper extends the Sharpness-Aware Minimization (SAM) approach to the Riemannian manifolds, e.g. when the learned models should satisfy certain constraints. Theoretically, the paper demonstrates that the generalization gap on manifolds scales with $\mathcal{O}(\sqrt{d})$ where $d$ is the dimension of the manifold and could be much smaller than $k$, that is the dimension of the ambient space. The paper provides experimental evaluations and compares the proposed method RSAM with other benchmarks such as SAM and SGD for supervised and self-supervised learning tasks.

### Strengths
I think the motivation and the idea of RSAM are valid and interesting. The result of Theorem 1 in which the $\mathcal{O}(\sqrt{k})$ factor in SAM's generalization gap reduces to $\mathcal{O}(\sqrt{d})$ on manifold seems quite interesting.

### Weaknesses
The paper is fairly difficult to follow in some parts. I suggest to elaborate more on the prior work on "learning on manifolds" and its technical literature. For instance, it seems that the proof of Theorem 1 relies substancially on results from (Boumal et al., 2018) and Lemma which are only touched on without sufficient discussion. Moreover, I could find several typos in math and inexact statements thoughout the paper. Pleasse my comments below.

### Questions
- Proof of Theorem 1 states that "Since the loss function L is $K$-Lipschitz, we have..." while the Lipschitz assumption is not mentioned in the theorem's statement or elsewhere. Could the authors clarify this?
- In proof of Theorem 1, what does $\tilde{{\theta}}$ denote? And what is a "logarithm map"?
- What is $v_{\theta}$ in Proposition 1? I assume it should be $u_{\theta}$?
- Section 3.1 would be easier to follow if the authors could add more elaboration on the retraction operator $R_{\theta}$ before going to Section 3.2.
- In experiments, the $\rho$ parameters for SAM and RSAM are different. I wonder if this is a fair comparison given that now the geometry of the manifold determines the robustness of RSAM as well. Could the authors elaborate on the effect of $\rho$ on the accuracies?
- What is the retraction operator considered in Lemma 1?

Minor comments:

- In Section 3.3, the second and third equations seem to miss $\mathcal{L}$ in the maximization objective.
- Equation (2) seems to be missing $\mathcal{L}_S(\theta)$ in the objective (compared to the previous derivation before eq. (2)).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
