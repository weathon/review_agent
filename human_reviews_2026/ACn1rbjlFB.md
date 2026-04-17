# Adversarial Attack on Tensor Ring Decomposition

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Tensor ring (TR) decomposition, a powerful tool for handling high-dimensional data, has been widely applied in various fields such as computer vision and recommender systems. However, the vulnerability of TR decomposition to adversarial perturbations has not been systematically studied, and it remains unclear how adversarial perturbations affect its low-rank approximation performance. To tackle this problem, we introduce a novel adversarial attack approach on tensor ring decomposition (AdaTR), formulated as an asymmetric max–min objective. Specifically, we aim to find the optimal perturbation that maximizes the reconstruction error of the low-TR-rank approximation. Furthermore, to alleviate the memory and computational overhead caused by iterative dependency during attacks, we propose a novel faster approximate gradient attack model (FAG-AdaTR) that avoids step-by-step perturbation tensor tracking while maintaining high attack effectiveness. Subsequently, we develop a gradient descent algorithm with theoretical convergence guarantees. Numerical experiments on tensor decomposition, completion, and recommender systems using color images and videos validate the attack effectiveness of the proposed methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AdaTR, an adversarial attack algorithm for Tensor Ring(TR) decomposition. It also proposes a faster variant to reduce iterative dependency called FAG-AdaTR. It first shows that directly extending the max-min formulation from ATNMF to form a baseline ATTR is not an effective attack. It shows this both empirically and theoretically. Moreover, it shows that ATR can improve performance for small perturbations. It then proposes the asymmetic max-min optimization whose objective is to directly maximize the reconstruction error of the TR decomposition. To reduce the gradient complexity and compurational cost, it proposes FAG-AdaTR. At last, it performs extensive experiments on various workloads comparing the proposesd attack with the baseline method with different defending algorithms.

### Strengths
The paper provides a clear theoretical analysis revealing that existing adversarial training formulations (ATTR, which is directly applied from ATNMF) can paradoxically improve tensor decomposition performance, thereby motivating the need for a stronger attack formulation.

It proposes a conceptually sound asymmetric adversarial objective that better aligns with the notion of maximizing reconstruction error and demonstrates this design through empirical results. The experiments cover diverse tasks and clearly show that the proposes attacks are substantially more destructive than exisiting baselines.

### Weaknesses
The paper’s main limitation is that it provides no theoretical guarantee or analysis for the proposed AdaTR and FAG-AdaTR algorithms. The only formal result concerns ATTR’s weakness, while the new methods are presented as heuristic formulations without convergence or optimality proofs. As a result, the contribution feels unbalanced.

Conceptually, AdaTR is a natural extension rather than a fundamentally new idea, and the “fast” variant is mainly an engineering improvement. The link between the theoretical critique of ATTR and the proposed algorithm is intuitive but not rigorously established.

In terms of presentation, the definition of “vulnerability” in the introduction is vague, and the distinction between adversarial training (ATTR) and attack (AdaTR) is not made clear until later. The experiments, while broad, rely only on reconstruction metrics and lack comparisons with general adversarial baselines or analysis of statistical robustness.

Typo: in line 45, it misses a citation for the ANMF paper.

### Questions
NA, see weakness

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
This paper studies the adversarial attacks on tensor ring decomposition. It is first observed that the classical symmetric min-max method (ATTR) may even improve the performance of the model under certain conditions. Therefore, it motivates to development of the asymmetric method based on the bilevel optimization. Since the proposed method requires complicated gradient computation with backpropagation, a simplified version of the algorithm is proposed by deactivating some variables’ gradients (FAG-AdaTR). Experiments on color image decomposition attacks, video decomposition attacks, tensor completion, and recommendation systems are presented to show the effectiveness of the proposed method.

### Strengths
(1) The writing is clear, and readers can easily follow.

(2) The experiments are comprehensive. It covers various tensor data, including color images, videos, recommendation systems, and general tensor data.

### Weaknesses
(1) I do not quite agree that the max-min problem (5) is symmetric and the bilevel form (10) is asymmetric. In minimax optimization, if we do not assume the Nash equilibrium (or some other conditions such as a strongly convex and strongly concave objective), the order of min and max cannot be changed. Therefore, (5) is asymmetric. I mean (5) is exactly equivalent to (10) if the order of min and max cannot be changed. In (5), given E, G is selected to minimize the objective. Therefore, the main motivation and claims in this paper are not correct from a minimax optimization perspective.

(2) In your algorithm, you deactivate many variables’ gradients (w.r.t. E) for simplifying the computation. However, does it still guarantee the convergence of the algorithm? Is the simplified gradient still a descent direction? Is it possible that after you mask some gradients, the simplified gradient is not valid for the problem? I am suspecting the effectiveness of the simplified algorithm, at least theoretically. 

(3) The paper lacks a theoretical analysis of the algorithm (FAG-AdaTR). Is your algorithm convergent? If yes, what kind of point does it converge to?

### Questions
see weaknesses

### Soundness
2

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
The paper studies adversarial attacks on TR decomposition. The authors argue that the conventional minmax ATTR objective (maximize w.r.t. perturbation, then minimize w.r.t. TR factors) can unintentionally improve low-rank reconstruction under small budgets and therefore isn't a true "attack" on TR (Thm 1). To address this, the authors propose an asymmetric bilevel formulation (AdaTR): minimize TR factors on the perturbed tensor but maximize the reconstruction error on the original tensor in Eq. (10). The perturbation is obtained through the ALS updates. The paper also introduces a faster version in which a closed-form approximate gradient for each mode is derived. Experiments on images, videos, and a recommender show larger degradation vs. ATTR across several TR-based defenses.

### Strengths
+ The paper identifies a failure mode of ATTR for small budgets.
+ The bilevel objective is an interesting formulation aligning with the attack goal. 
+ Fast approximate variant (FAG-AdaTR) with explicit gradients.
+ Good experimental evaluation (images, videos, completion, recommender).

### Weaknesses
- The problem lacks clear motivation and lacks clarity on the threat model. The attack norm is Frobenius on the full tensor (global energy budget). For vision tasks this may not be aligned with perceptual threat models. For recommendation it's nontrivial what perturbing all entries means. The paper would benefit from a precise threat model per application domain (who controls what, where noise is injected, etc).
- While Theorem 1 shows ATTR's potential to help at small $\varepsilon$, the intuition behind how the proposed formulation in Eq. (10) fixes this is lacking. It is more procedural than structural. A theorem or a lemma explaining why the AdaTR objective directly targets the final error (and can't collapse as with ATTR) would strengthen the story. You can contrast the two objectives' gradients w.r.t E to make the fix more convincing. Right now the argument is mostly empirical (Fig. 1). 
- If I understood correctly, the FAG-AdaTR efficiency comes from decoupling E from some iterates trading bias for speed. However, the approximation error (how far from the true gradient ascent) is not quantified, and there is no theoretical attack optimality.
- Beyond the approximation shortcut, there is no complexity analysis unless I missed that. 
- The paper mainly compares to ATTR and Gaussian noise, plus defense methods designed for completion/denoising, not attack methods on TR. A comparison to projected gradient attacks on the low-rank objective or to adversarial subspace attacks adapted from matrices would be important and show the gains. 
- For recommendation, it is unclear whether attacks respect the typical sparsity. Perturbing the dense rating tensor can be unrealistic. The setup should align with feasible manipulations (such as limited user/item edits).

The paper makes a worthwhile point (ATTR can help rather than harm for small $\varepsilon$) and proposes a nice bilevel objective with a fast approximation. However, there are many weaknesses that need to be addressed to make this a solid paper. For ICLR, I'd want to see a crisper formal contrast between ATTR and AdaTR, compute/scaling and approximation-error analysis for FAG, models aligned per application. If strengthened along these lines, the paper could be compelling.

### Questions
1) What is the adversary's capability per domain (images/videos vs recommender, etc)? Why Frobenius norm on the entire tensor, and how would results change with other realistic constraints?
2) Can you provide a theoretical statement showing that the AdaTR gradient aligns with maximizing the final reconstruction error, whereas ATTR can degenerate to maximizing $\Delta E$ as claimed? 
3) Can you bound or study the bias introduced by the FAG? When does it deviate most from AdaTR?
4) What are the peak memory and time vs. tensor size, rank, and inner ALS iterations for AdaTR and what speedup does FAG deliver in practice? 
5) Comparisons to other attack methods?

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
The paper investigates a previously unexplored question: are tensor decompositions vulnerable to adversarial perturbations? Focusing on Tensor-Ring (TR) decomposition, the authors formulate a dedicated adversarial attack on TR and derive a convergent gradient-based solver for the attacker.

### Strengths
1. This work proposes the first adversarial attack tailored to tensor decomposition; prior work (ATNMF, LaFa) targets matrix factorisation or poisons data, not the decomposition operator itself. 
2. Additionally, it proposes a novel asymmetric bilevel formulation that flips the usual min-max adversarial-training order, directly optimising the attacker’s goal (max reconstruction error).
3. The experiments are conducted in various applications and show the universality of the method.

### Weaknesses
1. This paper claims “tensor decomposition” vulnerability, but only TR-ALS is attacked; it is unclear whether fragility extends to CP, Tucker, TT, or SVD-based methods. The authors may run the same asymmetric objective on Tucker-ALS and TT-SVD (only requires swapping the composition operator). 
2. This paper should compare more baselines. ATTR is a natural extension of ATNMF; however, data-poisoning attacks or subspace-rotation attacks are also relevant but omitted. The authors should include a subspace-rotation attack baseline that maximises principal-angle deviation; this checks whether TR fragility is simply due to low-rank bias rather than the proposed bilevel formulation.
3. The experiments don't contain simple defenses. This work does not investigate whether adversarial training or input denoising can mitigate perturbations, leaving practitioners without effective countermeasures. The authors should add a defensive experiment: wrap TR-ALS with adversarial training using the proposed attack.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
