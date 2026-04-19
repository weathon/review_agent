# Emergence of Equivariance in Deep Ensembles

- Decision: Reject
- Scores: 5, 8, 6, 5

## Abstract
We demonstrate that a generic deep ensemble is emergently equivariant under data augmentation in the large width limit. Specifically, the ensemble is equivariant at any training step for any choice of architecture, provided that data augmentation is used. This equivariance also holds off-manifold and is emergent in the sense that predictions of individual ensemble members are not equivariant but their collective prediction is. As such, the deep ensemble is indistinguishable from a manifestly equivariant predictor. We prove this theoretically using neural tangent kernel theory and verify our theoretical insights using detailed numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proves that an infinite ensemble of neural networks becomes equivariant with data augmentation under mild assumptions. They use neural tangent kernels to show the equivariance. This property is also empirically evaluated with three tasks such as rotated image classification.

### Strengths
The paper theoretically shows that the equivariance emerges by model ensemble without hand-crafted architecture design. This direction, obtaining equivariance while we can freely choose networks, is important in practical usage. 

Although the paper is theory-flavored, it is easy to read and follow.

### Weaknesses
The main finding (emergence of equivariance in deep ensembles) is not very surprising. Data augmentation imposes a bias on a model toward invariance/equivariance, and for me, it's natural to see the averaged model archives that property. I mean, if we have an infinite number of data instances and the model capacity is large enough, the model trained for many steps would be equivariant. The neural tangent approach is of course different from this asymptotic approach, but the main idea should be the same. 

The experiments have room for improvement.
1. Instead of equivariance, invariance is evaluated.
2. Only a cyclic group is considered so it is not clear what kind of consequence can we get for more complex groups such as SO(2), SO(3), or SE(3). 
3. No comparison with equivariant networks such as steerable CNNs.

### Questions
In Equation (13) you assume that we can get the index permutation. However, for continuous groups such as SO(2) we cannot do this. Can you generalize the entire theory to avoid this issue?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors prove that when trained with data augmentation, deep ensembles are equivariant during training. They use the theory of NTKs and explicitly prove that the deep ensembles are equivariant regardless of the training step or data. However, this is limited by the fact that ensembles are finite, networks are not infinitely wide, and there is a limit to data augmentation for continuous groups. The authors further provide error bounds considering these limitations.

### Strengths
Although I am not very familiar with neural tangent kernels, the authors presented the work in a way that was easy to follow. Theorem 4 in particular seems like a very strong result. The authors further consider practical and very relevant limitations such as the finite ensemble, continuous groups, and finite width and prove error bounds. The experiments support the theory.

### Weaknesses
The type of data augmentation considered seems perhaps a little strong. By using all elements of the group orbit, it naturally lends itself to rewriting the group action as permutations, which seems to be critical in the proof. However, many common data augmentation strategies involve loss of information (e.g. random crops, random non-circular shifts, etc.). If the authors could provide any insights or foreseeable limitations of this work for other data augmentation types, that would be very helpful.

### Questions
See weaknesses

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work considers deep ensembles in the infinite width limit / NTK regime. For a deep ensemble on a dataset with equivariant data augmentation to a symmetry group, the work shows that the deep ensemble is equivariant at all points in its training evolution. Bounds are given on the behavior of different approximations to this, in the cases of: finite ensembles and finite subgroups for data augmentation. Empirical results show that numerically trained ensembles approach equivariance as width or number of models in the ensemble increase.

### Strengths
1. Well-written and well-organized Section 5: the proof sketch is nice.
2. Interesting empirical results that support the theory. The ensembles become more equivariant as width or number of models increases.

### Weaknesses
1. Do these results prescribe any particular practical methods, or does it give particular insights on models trained in practice? There does not seem to be much discussion on this. For instance, do people often train ensembles on equivariant data, and how does this compare to single models?
2. Could use more details on the critical assumption on the input layer, see question 1 below.

### Questions
1. Does your assumption on the networks depending on input through $w^{(k)}x$ on Page 5 really hold for CNNs? CNNs have their filter coefficients initialized via centered Gaussians, but the underlying matrix is not (because of weight sharing). Thus, there are orthogonal transformations on the input that may change the output (e.g. permute top left with top right pixel).
2. Intuitively, what does the deep ensemble output look like at initialization? I am trying to intuit why it is equivariant then.
3. Could you give more explanation or intuition about $C(x)$ in Lemma 6 in the main text?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper shows how, in a large width limit and with the inclusion of data augmentation, a generic deep ensemble becomes inherently equivariant. This equivariance is observed at each training step, irrespective of the chosen architecture, contingent upon the utilization of data augmentation. Notably, this equivariance extends beyond the observed data manifold and emerges through the collective predictions of the ensemble, even though individual ensemble member is not equivariant. It provides both theoretical proof, utilizing neural tangent kernel theory, and experiments to support and validate these observations.

### Strengths
1) This paper presents a very interesting idea of the emergence of equivariance with data augmentation and model ensembles.

2) This paper is generally well-written.

3) The theoretical claims in the paper are sound.

### Weaknesses
1) This paper lacks a proper comparison with other methods that can bring equivariance without any constraint on the architecture like [1, 2, 3, 4, 5]. When showing the out-of-distribution transformation results it'll be great to compare with those methods. The current results in the paper are more like ablations of the proposed augmentation and ensembling technique. It is not clear where it stands with other architecture-agnostic equivariance methods. (even if the proposed method does poorly compared to those it'll be good to have those results)

2) The author claims data augmentation is the only alternate method to bring equivariance in a non-equivariant model. I'll refer these papers [1,5] to the authors where they show that equivariance can be achieved using symmetrization and canonicalization. It'll be nice to include those as well in the paper. Especially symmetrization is closely related to the idea of ensembling because you pass different transformations of the same image throughout the same network before you average. My intuition is that symmetrization keeps the architecture the same and transforms the input, whereas the current work keeps the input the same and learns a transformer version of weights or each of the networks learning to process different transformations of the input. It'll be great if the authors can shed some light on the connection and discuss architecture agnostic body of work.


[1] Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E. J., & Lipman, Y. (2021). Frame averaging for invariant and equivariant network design. arXiv preprint arXiv:2110.03336.

[2] Mondal, A. K., Panigrahi, S. S., Kaba, S. O., Rajeswar, S., & Ravanbakhsh, S. (2023). Equivariant Adaptation of Large Pre-Trained Models. arXiv preprint arXiv:2310.01647.

[3] Basu, S., Sattigeri, P., Ramamurthy, K. N., Chenthamarakshan, V., Varshney, K. R., Varshney, L. R., & Das, P. (2023, June). Equi-tuning: Group equivariant fine-tuning of pretrained models. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 6788-6796).

[4] Basu, Sourya, et al. "Equivariant Few-Shot Learning from Pretrained Models." arXiv preprint arXiv:2305.09900 (2023).

[5] Kaba, S. O., Mondal, A. K., Zhang, Y., Bengio, Y., & Ravanbakhsh, S. (2023, July). Equivariance with learned canonicalization functions. In International Conference on Machine Learning (pp. 15546-15566). PMLR.

### Questions
See the weaknesses above

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
