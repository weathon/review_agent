# A Function Centric Perspective on Flat and Sharp Minima

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Flat minima are widely believed to correlate with improved generalisation in deep neural networks. However, this connection has proven more nuanced in recent studies, with both theoretical counterexamples and empirical exceptions emerging in the literature. In this paper, we revisit the role of sharpness in model performance, proposing that sharpness is better understood as a function-dependent property rather than a reliable indicator of poor generalisation. We conduct extensive empirical studies, from single-objective optimisation to modern image classification tasks, showing that sharper minima often emerge when models are regularised (e.g., via SAM, weight decay, or data augmentation), and that these sharp minima can coincide with better generalisation, calibration, robustness, and functional consistency. Across a range of models and datasets, we find that baselines without regularisation tend to converge to flatter minima yet often perform worse across all safety metrics. Our findings demonstrate that function complexity, rather than flatness alone, governs the geometry of solutions, and that sharper minima can reflect more appropriate inductive biases (especially under regularisation), calling for a function-centric reappraisal of loss landscape geometry.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the relationship between flatness and generalization from a function-complexity perspective. For that, it empirically analyzes SAM-sharpness, Fisher-Rao-Norm, and Relative Flatness for various networks trained either with standard training, using weight-decay, or data augmentation, as well as using the SAM optimization objective with each of the previous options. The experiments show that when looking at all networks together, there is no correlation between flatness measures and generalization. The paper then argues that flatness should be interpreted as a function-dependent property, not as a universal proxy for generalization.

### Strengths
- Investigating how regularization techniques impact flatness measures is sound and interesting.
- The fact that these regularization techniques increase sharpness is insightful.
- Since flatness remains one of the main candidates for explaining why deep networks generalize, this empirical study makes a meaningful contribution to an ongoing and relevant discussion in the field.

### Weaknesses
- The paper argues that sharper minima correspond to more complex functions, but this is not formally proven. The example given in section 4 is a good illustration, but it is unclear how it relates to deep learning. 
- It is unclear whether flatness metrics computed with and without data augmentation are comparable. Changing the dataset on which flatness is measured changes the entire loss surface. Similarly, using weight decay affects weight norms, which changes the flatness measures, since both Fisher-Rao-Norm and Relative Flatness are norm-based measures. 
- Figure 1 (c) could be an instance of Simpson's paradox: while there is a negative correlation over the entire population, there are positive correlations for individual stratums. That is, the correlation inverses as soon as you condition on the training method. Following Judea Pearl's logic, the training method cannot be a mediator, since it is not influenced by flatness. Instead, it must be a confounder, and thus we need to stratify on it. While this could still suggest that regularization causes both flatness and generalization in each stratum, and therefore flatness is _not_ a cause of generalization, as the paper suggests. This needs to be properly tested, though.
- Petzka, et al. 2021 [3] is not an empirical study (line 34), it provides a theoretical explanation about why and when flatness can explain generalization.
- Andriushenko, et al. [2], and Wen et al [5] already established that SAM does not produce flat solutions (line 416).
- The counter-example in Sec. 4 does not work for relative flatness [Petzka]: It requires locally constant labels and this function is not locally constant. Therefore it is not surprising that flatness does not correlate with generalization, here.

Overall, I see this as a flawed but thought-provoking paper that opens the right questions about flatness and generalization but does not yet answer them rigorously.

### Questions
- For the results in Tab. 3, do you use the sum loss or mean loss? The large value of relative flatness and Fisher-Rao norm for data augmentation are fairly unusual and could be explained by using the sum loss. In that case, the Hessian over the dataset is the sum of Hessians of each individual sample. Then, data augmentation clearly increases the scale of the value.
- Apart from SAM, weight decay, and data augmentation (line 62), one could also regularize to obtain flatness directly (e.g., [1]). Is there a reason this was not used?
- The relationship between relative flatness and adversarial robustness was recently analyzed by Walter et al. [4]. How do these results relate to this functional view on robustness?
- Walter et al. [4] also uncovered the relation between network confidence and relative flatness. Since minimizing the CE loss leads to high confidence on training examples, this would directly explain Fig. 1 (b).
- Table 1 caption: it should probably read "4 _local_ minima", right?

[1] Adilova, Linara, et al. "FAM: Relative Flatness Aware Minimization." Topological, Algebraic and Geometric Learning Workshops 2023. PMLR, 2023.

[2] Andriushchenko, Maksym, and Nicolas Flammarion. "Towards understanding sharpness-aware minimization." International conference on machine learning. PMLR, 2022.

[3] Petzka, Henning, et al. "Relative flatness and generalization." Advances in neural information processing systems 34 (2021): 18420-18432.

[4] Walter, Nils Philipp, et al. "When Flatness Does (Not) Guarantee Adversarial Robustness." arXiv preprint arXiv:2510.14231 (2025).

[5] Wen, Kaiyue, Tengyu Ma, and Zhiyuan Li. "How Does Sharpness-Aware Minimization Minimizes Sharpness?." OPT 2022: Optimization for Machine Learning (NeurIPS 2022 Workshop).

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper offers an alternative perspective on the flatness of the loss surface, a property long regarded as an indicator of good generalization. Specifically, it shows that the flatness of the obtained solution decreases when the learned function becomes simpler, and conversely, that more complex functions correspond to sharper minima. The authors argue that regularization techniques that improve model performance tend to produce sharper solutions, thereby providing empirical evidence that models representing more complex functions exhibit greater sharpness across different levels of regularization.

### Strengths
The paper addresses the entanglement between flatness\sharpness of loss surface and generalization of neural networks from an interesting perspective of the function complexity. The empirical results advocate for considering that more complex functions learned by networks can result in sharper solutions, without declining generalization, meaning that flatness has to be considered depending on the hardness of the task.

### Weaknesses
I find the first part of the empirical experiments somewhat misleading. In lines 54–58, these experiments are introduced as showing that the complexity of the objective determines how sharp the resulting solution will be. However, it is important to distinguish that, in this group of experiments, the known functions define the target function that the network is supposed to learn but not the objective (loss) function being optimized. In other words, the optimization does not occur directly on the surface of these proposed functions but rather on the loss surface defined by the mean squared error (MSE) in a regression setting. This distinction is not clearly conveyed, and the same confusion appears in the description of the experiments in Section 4.

Furthermore, one of the key contributions of [1] is the assumption of locally constant labels, which must hold for flatness to be a meaningful measure of generalization. This assumption is reasonable in classification tasks, but not necessarily in regression tasks, especially when the target functions exhibit high curvature. Consequently, this assumption breaks down in the first group of experiments using these functions.

In the introduction, the experiments are described as being designed to evaluate adversarial robustness, yet the presented empirical results concern natural noise robustness. This inconsistency should be clarified.

While I agree that sharpness measurements should not be taken as an absolute indicator of generalization, I disagree with the claim that this can be demonstrated by comparing sharpness values across different tasks or setups. For instance, data augmentation substantially alters the geometry of the loss landscape, making direct comparison of sharpness between augmented and non-augmented training runs questionable. Within a single setup, sharper solutions might indeed generalize worse, but comparing sharpness across heterogeneous settings risks introducing misleading effects, like Simpson’s paradox, as the paper itself cautions. This issue also makes the results involving SAM regularization appear contradictory: the experiments show higher SAM-sharpness even when SAM is explicitly designed to regularize and reduce it.

[1] Petzka, Henning, et al. “Relative Flatness and Generalization.” NeurIPS 34 (2021): 18420–18432.

### Questions
1 - In classical machine learning regularization is supposed to restrict the class of learned functions, usually to a simpler functions subset (like in case of L2 regularization for example). Your work claims that regularization increases complexity of the learned functions. Can you please elaborate on this contradiction?

2 - Your main contribution is stated as demonstrating that geometry of minima reflects complexity of the learned function, but you do not have any definition of the complexity - can you propose any?

3 - In the captures to the visualizations you have a mention of "51 random directions". What do you mean by it?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors focus on the relationship between model sharpness and performance. They revisit the common belief that flatter minima lead to better generalization and instead propose a function-centric perspective: the geometry of the solution (e.g., sharpness) reflects the complexity of the learned function rather than directly determining performance. The authors conduct both toy experiments and real data experiments (on CIFAR-10, CIFAR-100, and TinyImageNet), comparing the performance of several regularization techniques such as weight decay, data augmentation, and Sharpness-Aware Minimization (SAM).

### Strengths
-	Understanding the relationship between sharpness and a model’s performance and complexity is an interesting research question.
-	The paper is well-structured overall and easy to follow.

### Weaknesses
-	In several places, the writing and presentation of the paper could be improved. See the questions section below for details.
-	I’m not sure about the motivation of Section 4. The comparisons between different objectives don’t make much sense to me. These problems are quite different from each other, so it’s not surprising that the flatness of their minima varies. This seems inconsistent with the motivation in the context of deep learning optimization, where the training objective is the same (or at least similar) but different algorithms are used to optimize it.
-	The precise meaning of function space complexity is unclear. From the discussion in Section 5, it seems to refer to several evaluation metrics (not only just test accuracy) such as accuracy and calibration error (ECE). If that’s the case, the experiments simply suggest that adding more regularization techniques (e.g., weight decay, data augmentation, SAM) improves performance.
-	See more in the questions section below.

### Questions
-	There are two contributions listed from the bottom of page 2 to the top of page 3, and they seem quite similar. Is this a repetition? It might be good to check.
-	Regarding the sharpness metric, I wonder why basic measures such as the largest eigenvalue of the Hessian or the trace of the Hessian are not used. The current metrics are fine, but including the most common ones could make the analysis more complete.
-	Related to the above, the original SAM paper shows that SAM tends to converge to points with smaller largest eigenvalues of the Hessian. This seems to contradict the message conveyed in the current paper. I would appreciate it if the authors could comment on this.
-	Many metrics mentioned in Appendices B and C are not formally defined as stated in the main paper. It would be helpful to include formal definitions rather than only referring to other papers or code, for clarity and completeness.
-	I’m also curious about the setup of the baseline. For example, with ResNet-18, the test accuracy is typically above 90% on CIFAR-10 and above 70% on CIFAR-100.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The core message here—flatness by itself is not a reliable predictor of generalization—isn’t new. The paper reframes this  as a “function‑centric view” and backs it up with a broad set of experiments (toy objectives, CIFAR‑10/100, Tiny‑ImageNet; ResNet/VGG/ViT) and a few safety‑relevant metrics (ECE, corruption accuracy, “prediction disagreement”). The coverage of datasets is solid, but the headline observation has been known since the reparameterization critique from Bengio's group  (and follow‑ups) made the “flat vs. sharp” narrative much more nuanced -- see e.g. an older paper from 2021 -Why flatness does and does not correlate with generalization for deep neural networks  https://arxiv.org/abs/2103.06219 -- for example 

So what’s left is not that new. 

Secondary mistakes:
They wrote “decisecond boundaries then base models” twice, It should me decision boundaries than base models. (Appendix E.2). Shows lack of proofreading.
The reference entry for Li et al., 2018 (Loss Landscape visualization) points to the Verified Uncertainty Calibration NeurIPS URL hash (Kumar et al., 2019). The link text and the target don’t match.
Booth function is misstated as (x+2y−y)^2+(2x+y−5)^2. It should be (x+2y−7)^2+(2x+y−5)^2

### Strengths
The coverage of datasets is solid, 

The results are largely correct.

### Weaknesses
The paper repeats many things that are already pretty well known.  I suspect that they have not fully read or digested the literature on this topic.

### Questions
what is truly novel in this work?

### Soundness
3

### Presentation
2

### Contribution
2
