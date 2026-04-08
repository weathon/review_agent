## Human Reviewer 1

### Summary
The authors of this submission investigate the generalization ability of learned optimizers in zero-shot-like scenarios, where an optimizer trained on one architecture can be directly applied to unseen architectures. To achieve this, models are initialized following the $\mu$P (Maximal Update Parameterization) principle. The authors leverage their hyperparameter reuse property, which allows optimal parameters discovered on one architecture to be transferred to architectures with the same depth but different widths via a well-defined mapping. Under this framework, learned optimizers are treated as a form of hyperparameter, enabling transfer to new architectures. The proposed approach is evaluated across various neural network training tasks and demonstrates promising results.

### Strengths
1. The motivation is interesting and well-grounded. The idea of viewing optimizers as hyperparameters that can generalize across $\mu$P-initialized networks of varying widths is both novel and conceptually appealing.

2. The learned optimizers show encouraging results in zero-shot-like transfer scenarios.

### Weaknesses
1. Figure 5 indicates the meta overfitting with the trained optimiser performing well on the meta-train tasks while failing to generalise well on the unseen tasks. 

2. From all the experiments, only the learning curve of training stages is illustrated, with the question of whether the learned optimizer leads to advanced generalization ability not answered. 

3. Limited novelty: the Mup parameterization proposed in this submission is very close to a direct application of the original $\mu$P paper. In addition, the propositions in submission are similar to the theoretical results from the $\mu$P papers.

### Questions
1. Figure 5 suggests meta-overfitting: the learned optimizer performs well on meta-training tasks but fails to generalize effectively to unseen tasks.

2. Across the experiments, only training curves are presented. It remains unclear whether the learned optimizer contributes to improved generalization performance beyond faster or more stable training.

3. The level of novelty appears limited. The proposed $\mu$P parameterization closely follows the original $\mu$P paper, and several propositions in the submission echo theoretical results already established in prior work.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper derives muP for the small_fc_lopt and VeLO learned optimizer architectures. The authors then test the resulting metageneralization, meta-training on ImageNet classification with MLPs and applying the optimizers to other instances of that as well as using transformers for that and language modelling. They empirically find that this achieves hyperparameter transfer not only for width, but also to a significant degree for depth and training duration, outperforming other learned optimizers and also competing with task-tuned standard/hand-designed ones like AdamW and μAdam. There is also a short subsection on pre-activation stability, and similar investigations in the appendix.

### Strengths
While I think applying muP to this domain was inevitable, the authors make a good case that the failure to meta-generalize is the major blocker for LOs, and that they make substantial improvements there.

I thought the experiments were reasonable, namely the baselines, and the paper was clear throughout, including the maths (though I haven't gone line by line in the proofs).

### Weaknesses
I think the findings of ["Scaling Exponents Across Parameterizations and Optimizers"](https://arxiv.org/abs/2407.05872) should've made an appearance somewhere, since they indicate that standard parametrization can also achieve hyperparameter transfer. They also show that (in larger problem instances than here) that epsilon should be tuned in Adam.

Also, since depth scaling is mentioned (albeit as a bonus), I think the related works and perhaps some experiments would ideally address more heuristic methods for scaling with depth (and width, or rollout length, if the LO community has investigated this at all). In NeurIPS 2025 (i.e., contemporary, ignoring arXiv) work, [ComputeP](https://arxiv.org/abs/2505.01618) is quite relevant

Minor fix:
* §B.1 "When using a schedule, we always use linear warmup and cosine annealing with" just cuts off

### Questions
How much compute went into the hyperparameter sweeps for the baseline optimizers? The values swept looked reasonable, but I think it'd be useful context to compare with the meta-training figures, to see how much the LOs need to be used to make up for that (mainly thinking in seemingly underparametrized settings where that may be the right frame, like language modelling).

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper addresses the challenge of learned optimizers (LOs) struggling to optimize unseen tasks, particularly when dealing with networks larger than those encountered during meta-training. The authors introduce the Maximal Update Parametrization (P) for two state-of-the-art LO architectures and propose a new meta-training recipe for parameterized LOs. Empirical results show that LOs trained with this recipe significantly improve meta-generalization to larger unseen tasks, outperforming standard parametrization (SP) LOs within the same compute budget. Additionally, the paper observes enhanced meta-generalization for deeper networks and surprisingly effective generalization for longer training horizons compared to SP LOs.

### Strengths
1. This paper introduces the Maximal Update Parametrization to address the meta-generalization problem in learned optimizers. The idea is novel and interesting.

2. The experimental results are thorough and effectively demonstrate the method’s validity.

### Weaknesses
1. I recommend adding experiments with convolutional neural networks (CNNs). Although this limitation is mentioned, I believe CNNs are currently mainstream in neural network research, and testing the method on them would further strengthen the validity of the findings.

I am not very familiar with this field, so I will rely on the feedback from other reviewers for the final score.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper introduces \mu-parameterized Learned Optimizers to address the meta-generalization limitations of existing learned optimizers when optimizing neural networks wider than those encountered during meta-training. 

Authors derive the Maximal Update Parametrization  for some architectures and propose a straightforward meta-training recipe. They find that proposed \mu-P parameterization significantly improves meta-generalization to wider, deeper, and longer unseen tasks compared to Standard Parameterization.

### Strengths
The paper has following strengths in my opinion:
- tackles the fundamental challenge of meta-generalization in learned optimizers, which is a major hurdle for their practical adoption.
- derivation of µP for LOs provides a solid theoretical basis for the proposed approach
- results show substantial improvements in meta-generalization across various axes (width, depth, training horizons) compared to  Standard Parametrization
- authors emphasize that the benefits of µLOs come at zero extra computational cost during meta-training compared to SP LOs, which is a crucial practical advantage
- paper provides a clear description of the meta-training recipe, evaluation setup, and baselines, making the work reproducible
- good suite of evaluation tasks and architectures

### Weaknesses
My main concerns are following:
- While the improved generalization to deeper networks and longer training horizons is a strong empirical result, the explanations for these benefits are currently speculative e.g. mentions of pre-activation stability. Wondering if this could be made more precise.

### Questions
I also have some qns for the authors:
- Given that µLOs exhibit improved generalization to deeper networks and longer training horizons, and these improvements are currently hypothesized to be due to activation stability, do the authors plan to conduct further theoretical analysis to formalize these observations?
- The µLOs were meta-trained only on MLP tasks. How do the authors anticipate the performance and meta-training cost of µLOs would change if they were meta-trained on a more diverse set of architectures, including ViTs and Transformers?
- While the paper acknowledges the absence of an oracle SP AdamW baseline, could the authors discuss any ongoing efforts or future plans to conduct such a computationally intensive comparison to provide an even stronger benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4