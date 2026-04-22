# DiBS-MTL: Transformation-Invariant Multitask Learning with Direction Oracles

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Multitask learning (MTL) algorithms typically rely on schemes that combine different task losses or their gradients through weighted averaging. These methods aim to find Pareto stationary points by using heuristics that require access to task loss values, gradients, or both. In doing so, a central challenge arises because task losses can be arbitrarily, nonaffinely scaled relative to one another, causing certain tasks to dominate training and degrade overall performance. A recent advance in cooperative bargaining theory, the Direction-based Bargaining Solution ($\texttt{DiBS}$), yields Pareto stationary solutions immune to task domination because of its invariance to monotonic nonaffine task loss transformations. However, the convergence behavior of $\texttt{DiBS}$ in nonconvex MTL settings is currently not understood. To this end, we prove that under standard assumptions, a subsequence of $\texttt{DiBS}$ iterates converges to a Pareto stationary point when task losses are possibly nonconvex, and propose $\texttt{DiBS-MTL}$, a computationally efficient adaptation of $\texttt{DiBS}$ to the MTL setting. Finally, we validate $\texttt{DiBS-MTL}$ empirically on standard MTL benchmarks, showing that it achieves competitive performance with state-of-the-art methods while maintaining robustness to nonaffine monotonic transformations that significantly degrade the performance of existing approaches, including prior bargaining-inspired MTL methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DiBS-MTL, a new multitask learning (MTL) method based on the Direction-based Bargaining Solution (DiBS). The method is invariant to monotonic non-affine transformations of task losses, differently from previous works on MTL. The authors extend DiBS to nonconvex MTL settings and theoretically prove that a subsequence of its iterates converges to a Pareto stationary point. They then propose a practical and efficient approximation that updates parameters according to the normalized sum of task gradients. Empirical results are provided for the NYU-v2 and MT10 benchmarks. DiBS-MTL matches the performance of baselines. The paper also presents synthetic and engineered examples to demonstrate the method’s invariance to monotone transformations.

### Strengths
- The paper is well-written and easy to read.
- The paper provides a theoretical contribution by proving the convergence of DiBS under nonconvex losses.
- The proposed approximation (DiBS-MTL) is fairly efficient and simple, while maintaining the invariance property.

### Weaknesses
I generally like the paper, it is well written, theoretically grounded, and quite elegant. However, I have three main issues:
1. Approximation vs. theoretical derivation: The proposed single-step DiBS-MTL approximation appears too strong and far from the exact theoretical derivation of DiBS. It is unclear whether this approximation maintains the proven convergence properties. The paper provides no theoretical justification or formal link between the full Multi-step DiBS-MTL and its approximation, beyond an empirical comparison on a low-dimensional synthetic example. This illustrative example does not reflect realistic MTL behavior. As a result, it remains uncertain how the theoretical motivation and convergence guarantees carry over to the practical version actually used in experiments. Is it possible to provide a comparison on a larger-scale benchmark (even with a small number of DiBS steps)?
1. Motivation for invariance to monotone transformations: The motivation for focusing on invariance to monotone non-affine transformations is interesting and well-articulated, but it should be better supported with real-world evidence. Currently, the usefulness of this property is demonstrated only on synthetic and engineered examples (modified MT10), rather than on natural or widely used benchmarks where such transformations occur in practice.
1. Empirical evaluation is too limited and weak: At its current state, the empirical evidence for the effectiveness of DiBS-MTL is insufficient.
    - The paper reports results on only two datasets (NYU-v2 and MT10). I suggest adding additional common MTL benchmarks, such as QM9, Cityscapes, or CelebA, to strengthen the empirical scope.
    - The baseline coverage is limited. For NYU-v2, only NashMTL (2022) and FAMO (2023) are recent, strong baselines; others are relatively older and less strong. The authors should include more modern MTL approaches such as FairGrad, CAGrad, IMGrad, or DB-MTL. The MT10 evaluation is even more limited, with only a single recent strong baseline (which also achieves better performance compared to the proposed approach).
    - Finally, the reported results show that DiBS-MTL performs comparably to existing methods but does not significantly improve over them on any benchmark. Combined with the limited experimental coverage and limited baselines, this makes the empirical evidence inconclusive regarding the claimed advantages and effectiveness of the approach.

### Questions
- Could you provide a more detailed analysis of the computational complexity of DiBS-MTL, especially its scaling with the number of tasks? How does its runtime compare to recent gradient-based approaches beyond the small-scale runtime figure reported?
- The update rule depends directly on the step-size parameter $\epsilon$, but its value and tuning strategy are not reported. How was it selected in your experiments, and how sensitive is performance to this choice?

### Soundness
2

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
3

### Summary
The authors propose to adapt a recently proposed method named DiBS for MTL in convex settings to general MTL problems. The main motivation for that is that DiBS is invariant to monotonic non-affine transformation, making it less susceptible to arbitrary scaling of the loss values. The authors first show a convergence guarantee to a Pareto stationary point for DiBS under standard SGD conditions. Then they apply DiBS to MTL under a bargaining formulation and present an efficient version of their method.

### Strengths
* Showing convergence to a Pareto stationary point asymptotically for the proposed approach under minimal and reasonable assumptions.
* Present an efficient extension of DiBS to MTL.
* Good empirical results on the NYU dataset.
* Unlike the compared approaches, the proposed approach was shown to be resilient to non-affine scalings.

### Weaknesses
* Regarding the method:
  - The novelty of this paper is limited. The main novel contribution of this paper, if I understand correctly, is the adaptation of DiBS to general MTL settings, namely DiBS-MTL.
  - If I understand correctly, the shared update direction is found by summation over the normalized gradients. However, while it is invariant to scaling, it loses information about the magnitude, which can be important at different stages of the learning procedure. Furthermore, it is a special case of Nash-MTL in which one assumes that all the gradients are orthogonal to each other. Perhaps I am missing something here, and I will be happy if the authors can clarify that point.
  - Related to the previous bullet, suppose we have two tasks with collinear gradients in opposite directions. The shared update direction will probably not be useful for any of them and may cause slow convergence. Taking inspiration from [1] where uncertainty was used, how does the authors suggest to handle that?
* The experimental comparisons and grounding in the MTL literature are severely lacking:
  - Missing references and *comparisons* to recent MTL studies, non-exhaustive list [1-4]. These papers should be addressed and compared to where appropriate. 
  - The method was also not compared to recent and leading methods on common MTL benchmarks, such as CityScapes, QM9, and UTKFace.
  - Missing a proper comparison (on a non-syntactic benchmark) between Multi-step DiBS-MTL and DiBS-MTL to evaluate the strengths and limitations of each approach.
  - The dynamics of the loss weights across training are not clear. A proper analysis (even an empirical one) is missing.
  - Implementation details are missing. What are the hyperparameters of the method besides $\epsilon$? How sensitive is the method to these hyperparameters? Is there a special schedule for the learning rate?
* In my opinion, an algorithm of the method is missing to help better understand the update rule.

[1] [1] Achituve, I., Diamant, I., Netzer, A., Chechik, G., & Fetaya, E. (2024, July). Bayesian uncertainty for gradient aggregation in multi-task learning. In Proceedings of the 41st International Conference on Machine Learning (pp. 117-134).  
[2] Senushkin, D., Patakin, N., Kuznetsov, A., & Konushin, A. (2023). Independent component alignment for multi-task learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20083-20093).  
[3] Dai, Y., Fei, N., & Lu, Z. (2023, July). Improvable gap balancing for multi-task learning. In Uncertainty in Artificial Intelligence (pp. 496-506). PMLR.  
[4] Xiao, P., Dong, C., Zou, S., & Ji, K. (2025). LDC-MTL: Balancing Multi-Task Learning through Scalable Loss Discrepancy Control. arXiv preprint arXiv:2502.08585.

### Questions
- Regarding DiBS-MTL, it seems that $\Delta \theta$ grows with the number of tasks. How do you control its magnitude? Would it be more sensible to take the mean instead of the sum? If so, how does that affect the theory?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DiBS-MTL, a method that computes a bargaining solution based solely on gradient direction. It effectively addresses the issue of varying loss scales, which can negatively influence the performance of current MTL methods. The paper also provides a convergence analysis in the nonconvex MTL setting, showing that the proposed method converges to a Pareto stationary point. Experiments on common MTL benchmarks demonstrate the effectiveness and robustness.

### Strengths
1. The comparisons of transformed loss functions are interesting (Figure 2, Figure 3); they clearly show that DiBS-MTL is more robust than other baselines.
2. Theoretical analysis is provided to demonstrate that DiBS-MTL converges to a Pareto stationary point. Additionally, extensive experiments have been conducted to show that DiBS-MTL performs comparably to existing baselines.

### Weaknesses
1. The theoretical analysis is based on the multi-step DiBS-MTL (Line 229-233), but in practice, the DiBS-MTL uses a single-step approximation. There should be a gap. It would be helpful if the theoretical analysis could account for the approximation error.

2. The practical DiBS-MTL reduces to the average of normalized task gradients, which is not particularly novel. The comparison between multi-step DiBS-MTL and single-step DiBS-MTL (Figure 2 vs Figure 5) is not sufficient. I would like to see more comparisons on real MTL benchmarks, such as Cityscapes, NYU-v2, and MT10, covering both performance and running time. 
From the current result, it is difficult to determine whether the observed improvement comes from only gradient normalization or the DiBS framework. Additionally, it would be informative to evaluate how the baselines perform when using normalized gradients.

3. For DiBS-MTL (Line 237-240), there is a term $\epsilon$. How to set its value in practice? Are there any sensitivity analyses? Or is it absorbed into the learning rate?

4. The results shown in Table 1 are not promising enough. The baselines are too old, and many recent methods are not included.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
The paper presents DiBS-MTL, a method for multitask learning (MTL) designed to be robust to monotonic, non-affine transformations of task loss functions. Existing challenges in MTL arise because different task losses can be scaled differently, which may cause one task to dominate the others. DiBS-MTL uses only normalized gradients and a Direction-based Bargaining (DiBS) approach, enabling the identification of Pareto stationary points while maintaining invariance to changes in loss scaling.

Additionally, the paper shows that DiBS-MTL asymptotically converges to a Pareto stationary point even when task losses are non-convex, and it presents the mathematical mechanism of the method. The work includes experiments on two-dimensional illustrative examples to demonstrate balanced objectives, as well as applications in computer vision and multitask reinforcement learning benchmarks. Moreover, the paper details the adaptation of DiBS to existing bargaining frameworks and efficient computation of normalized gradients.

### Strengths
The paper introduces robustness to monotonic non-affine task loss transformations, representing a key and distinctive contribution.
The paper establishes convergence guarantees for DiBS in non-convex settings.
The paper demonstrates how DiBS can be naturally integrated with multitask learning.
The paper proposes a single-step approximation of DiBS-MTL, offering a computationally practical solution.

### Weaknesses
The paper tackles a relevant problem. However, the novelty of the problem itself appears somewhat incremental, which may limit the overall impact of the contribution for the community.
Some experiments are missing, as also acknowledged by the authors.

### Questions
Please address the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4
