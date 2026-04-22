# Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We investigate why deep neural networks suffer from loss of plasticity in deep continual learning, failing to learn new tasks without reinitializing parameters. We show that this failure is preceded by Hessian spectral collapse at new-task initialization, where meaningful curvature directions vanish and gradient descent becomes ineffective. To characterize the necessary condition for successful training, we introduce the notion of $\tau$-trainability and show that current plasticity preserving algorithms can be unified under this framework.
Targeting spectral collapse directly, we then discuss the Kronecker factored approximation of the Hessian, which motivates two regularization enhancements: maintaining high effective feature rank and applying L2 penalties. Experiments on continual supervised and reinforcement learning tasks confirm that combining these two regularizers effectively preserves plasticity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the loss of plasticity in continual learning from the perspective of the Hessian eigenspectrum and proposes a principled necessary condition for successful continual training.  The paper proposes that neural networks lose the ability to learn new tasks because their Hessian spectra collapse at new-task initialization: most curvature directions vanish, and gradient descent becomes ineffective in a continual learning setup. The paper formalizes this by introducing τ-trainability (a task-dependent requirement on Hessian rank) and proving that inactive (“dead”) neurons reduce Hessian rank and thus limit trainability — in particular, Theorem 6.2 relates the number of dead neurons to an upper bound on Hessian rank, and Theorem 6.5 gives a probabilistic bound showing that if many neurons are likely to persist dead across a task shift then the probability the next task is trainable is small.

Methodologically, the paper combines (i) spectral diagnostics, (ii) analytic bounds, and (iii) a simple, practical mitigation. The Hessian spectrum is estimated using stochastic Lanczos quadrature to empirically track spectral collapse across supervised and RL continual benchmarks; these experiments reveal a clear association between low ε-rank (collapsed spectrum) and failed training. Building on a Kronecker-factored view of curvature, the paper proposes a regularizer (L2-ER method) that maintains high effective feature rank and preserves curvature in the directions that matter. Empirically, L2-ER (and related curvature-preserving interventions) prevent spectral collapse and retain plasticity across Permuted MNIST, Incremental CIFAR, Continual ImageNet, and RL tasks, thereby confirming the theoretical connection to practice.

### Strengths
The paper investigates the still poorly understood problem of plasticity loss in neural networks and proposes an interesting and principled view on the relation between Hessian effective rank and network trainability. 
In particular, I find the following strengths of the paper:

1. The paper rigorously connects “dead” ReLU neurons to reductions in Hessian rank (e.g., Theorem 6.2) rather than relying on hand-wavy intuition. This provides a concrete mechanism explaining why spectral collapse hinders optimization: inactive neurons eliminate curvature directions, thereby reducing the effective second-order information that gradient descent can utilize.
2. Lemma 6.3 (and the associated per-neuron margin condition) turns the abstract idea of dead neuron persistence into a simple geometric inequality that links margin, weight norm, and worst-case dataset shift (Hausdorff). That makes the theory falsifiable and directly measurable in practice
3. The proposed method, the L2-ER method, seems to be well principled (yet it should be evaluated more rigorously - see Weaknesses).

### Weaknesses
1. The paper should differentiate the contributions from the previous work [1, 2], with emphasis on [1], which also investigates the impact of the network's Hessian rank on the trainability in a continual setup. In the current form, the novelty of the paper is hard to assess. 
2. The number of baseline methods used in the paper and the evaluation setup is unsatisfactory. I recommend
a) investigating how the proposed method L2-ER compares to Spectral Regularization [2] and one more such as Shrink and Perturb [4] or ReDO [3],
b) provide an additional investigation in reinforcement learning setups, where the plasticity issues often are more severe than in supervised ones. 
3. The problem illustration in the introduction is confusing because there is no mention of how curvature-regularized methods work. Nevertheless, they are used in Figure 1 with no proper description.

[1] Lewandowski, A., Tanaka, H., Schuurmans, D., & Machado, M. C. (2023, November 30). Curvature Explains Loss of Plasticity. http://arxiv.org/abs/2312.00246
[2] Alex Lewandowski et al. (2024, June 10). Learning Continually by Spectral Regularization. http://arxiv.org/abs/2406.06811
[3] Sokar, G., Agarwal, R., Castro, P. S., & Evci, U. (2023, June 13). The Dormant Neuron Phenomenon in Deep Reinforcement Learning. http://arxiv.org/abs/2302.12902
[4] Ash, J. T., & Adams, R. P. (2020, December 31). On Warm-Starting Neural Network Training. http://arxiv.org/abs/1910.08475

### Questions
I don't understand the right plot of Fig. 1, in particular, why the bottom right plot looks like it has not converged?

### Soundness
3

### Presentation
2

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
The paper argues that loss of plasticity in continual learning is driven by Hessian Spectral Collapse at new-task initialization. It introduces $\tau$-trainability and upper-bounds its probability by the $\epsilon$-rank and further by the number of dead neurons in a 2-layer ReLU MLP setting. The authors propose a regularization which combines an effective feature rank penalty with L2 weight decay.

### Strengths
The paper addresses a topic of continually growing importance. The problem of loss of plasticity in Continual Learning is well-motivated and clearly formulated, and Sections 1–5 are very well written. The paper frames prior work on loss of plasticity through the lens of Hessian spectral collapse. The authors also evaluate on standard Continual Learning benchmarks and conduct a substantial hyper-parameter sweep.

### Weaknesses
Section 7, on mitigating spectral collapse, is poorly written. Key quantities are not introduced or defined, and the approximations are insufficiently justified. Equation 5 contains typos and the approximation error in equation 6 is particularly unclear.

The experiments also lack comparisons to related work on mitigating loss of plasticity (e.g., [1]); they evaluate the proposed method only against a single Continual Learning baseline and standard backprop. Additional potential weaknesses are noted in the questions.


[1] Alex Lewandowski, MichaĹ Bortkiewicz, Saurabh Kumar, Dale Schuurmans, Mateusz Ostaszewski,
Marlos C Machado, et al. Learning continually by spectral regularization. arXiv preprint
arXiv:2406.06811, 2024a.

### Questions
- Why restrict the theoretical analysis to a two-layer MLP? What are your empirical observations on how wider/deeper networks affect trainability?
- Why is the analysis conducted using the notion of $\epsilon$-rank rather than the effective rank used in the regularization term? How are the two related?

I would be happy to raise my score if the authors address the fundamental issues noted in the weaknesses.

### Soundness
2

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
4

### Summary
This paper aims to explain the loss of plasticity through the collapse of the Hessian eigenspectrum and thereby theoretically connects known drivers of plasticity loss, dormant neurons and effective rank. Based on this theoretical understanding, the paper proposes L2-ER regularization, which enables the model to maintain a stable Hessian spectrum during training.

### Strengths
The paper provides useful theoretical understanding on loss of trainability. The theory connects dormant neurons, effective rank, and loss curvature, which enrich understanding of plasticity loss.

### Weaknesses
* The method lacks novelty. The proposed L2-ER regularization is a simple combination of existing techniques — effective rank regularization [1] and L2 regularization.
* The theoretical results are incremental. It is known that loss curvature is deeply connected to loss plasticity [2, 3]. I acknowledge its novelty, but it is still incremental.
* The theoretical analysis could be further elaborated using observations from previous work. For example, [2] found that the sharpness of loss curvature correlates with the loss of plasticity, especially in the absence of dormant neurons.
* The experiments are insufficient to validate the effectiveness of the proposed method. Additional reinforcement learning experiments could be conducted in more environments.
* The idea that Figure 1 aims to convey is ambiguous. The authors mention that GD exhibits unstable learning dynamics (zig-zagging), while curvature-regularized GD remains stable. However, it is unclear which factor causes this difference. Is it due to different learning algorithms (GD vs. curvature GD), or to different starting points (valley vs. bowl)? In Task 2, the orange and blue lines start from different points and are minimized using different algorithms. Therefore, it is unclear which factor is responsible for the observed optimization stability. In addition, the starting point of GD is much closer to the valley, while that of curvature GD is much closer to the bowl. This might affect the optimization. Would the result remain the same if the loss landscape of Task 2 were vertically flipped?

[1] Kumar, Aviral, et al. "Implicit under-parameterization inhibits data-efficient deep reinforcement learning." arXiv preprint arXiv:2010.14498 (2020).
[2] Lyle, Clare, et al. "Understanding plasticity in neural networks." International Conference on Machine Learning. PMLR, 2023.
[3] Lewandowski, Alex, et al. "Directions of curvature as an explanation for loss of plasticity." arXiv preprint arXiv:2312.00246 (2023).

### Questions
This point is described in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission studies the trainability of deep networks from the perspective of the Hessian and how curvature affects neural network optimization dynamics. The paper provides evidence that spectral collapse underlies some of the deep network's loss of plasticity/trainability effect. They use these insights to propose a regularizer that operates on the hidden units of the network, and find that this proves to be effective in combination with l2 regularization.

While the paper is interesting and provides some insight, there are some weaknesses in the experiments that I am not sure a rebuttal can address. That is, the experiments seem fairly disconnected from the rest of the paper, and somewhat undermines the larger thesis.

### Strengths
- Overall, the paper is well written. The main narrative presented is largely compelling, and the main thesis that spectral collapse underlies loss of plasticity is well supported by previous work. 
- Theoretical analysis of trainability appears novel and relevant. It differs from previous work that focus on a particular perspective, such as weight resetting or regularizing. The authors also translate these insights into a regularizer.

### Weaknesses
- The empirical results are lacking in a few respects. First, there are some important baselines left out (such as various regularizers, and neural network architectures including activation functions and normalization layers)
- Connected to the experiments, there is little connecting the theoretical results to the experiments besides the proposed method. While there was substantial discussion regarding spectral collapse, it was never actually measured empirically.
- There are some clarity issues surrounding parts of the definitions and theorems (see below)

### Questions
- How do you account for curvature and Hessian changes due to changes in the data distribution on the new task?
- line 45: This introduction leaves out some previous work that has looked at second order effects on plasticity and found mixed results. While the related work goes into some detail, the introduction positions the submission as if it is the first to tackle this direction.
- I am confused by why the bowls and saddles in figure 1 and in appendix A are associated with spectral collapse. Is it right to interpret this low dimensional projection as evidence that the high dimensional manifold is actually flat and suffering from spectral collapse? Furthermore, I would be curious to know how these landscapes look at the end of the previous task. Does l2-er reach a local minimum?
- Definition 5.1 doesn't seem like a practical definition, because in practice the set of arg-maximizers can be a set of measure zero. Why not loosen this definition to include an error tolerance?
- Definition of trainable as "initialized with fewer than $m$ dead neurons" doesn't seem correct. For one thing, it doesnt respect architecture (where are the dead neurons located?) nor size (what proportion of units are dead?). For example, a very large architecture with $500$ dead neurons will likely be trainable, whereas a small network may not be.
- Definition 5.2: This definition does not seem operationalizable without specifying a reasonable threshold for each task. How does the threshold depend on the task, and on the network/optimizer?
- Theorem 6.2: Isnt it the case that $\rho_\tau = \text{rank}(H_\tau^{(0)})$? If so, then the inequality is always satisfied, and you have the vacuous bound $P(T_\tau = 1) < 1$
- line 380: Im confused as to why l2 regularization would increase the rank, since L2 regularization would regularize the eigenvalues towards zero.
- Missing related work: parseval regularization seems to capture a very similar same term, by orthogonalizing the weights.
- Why isnt the hessian rank measured in the experiments?
- Doesn't this claim: "Single metrics are insufficient." undermine the narrative surrounding spectral collapse as the unifying explanation?
- Missing layer normalization baseline: it is now common to use normalization, and this has also been found to greatly reduce loss of plasticity. 
- The neural network architectures here are rather small and basic, lacking normalization and skip connections. There is also comparatively little analysis of how the optimizer may contribute to curvature, as compared to previous work.

### Minor Comments
- line 168: $\theta^*$ is missing a subscript

### Soundness
2

### Presentation
3

### Contribution
2
