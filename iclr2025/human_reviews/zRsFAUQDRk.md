## Human Reviewer 1

### Summary
The paper studies the class-generalization error bound as opposed to the more traditional case of expected generalization over the whole data distribution. The authors provide a new class-specific generalization definition and give multiple information-theoretic bounds based on the KL divergence (more classic), and the super-sample technique (maybe somewhat newer). They analyze the tightness of their bounds in some experiments on CIFAR10 and CIFAR100 showing the failure of the general generalization bounds and the success of their class-specific approach. Finally, the authors provide some examples of the applications in which the provided bounds can be useful and add new insights.

### Strengths
The paper is well-written, and easy to follow. The authors have motivated their work nicely both at the beginning of the paper and the end by providing specific examples of some scenarios in which the class-wise generalization bound can be insightful like the sub-task problem as a specific case of distribution shift. The idea of caring about each class independently is new and in some cases tightens the pre-existing bounds.

### Weaknesses
The technical contribution of the paper is limited. While the definition of class-wise generalization bound is new, the results are heavily based on the previous works [Xu and Raginksy 2017, Zhou et al 2022, Harutyunyan 2021, Steinke and Zakynthinou 2020, Clerico et al 2022, Wang and Mao 2023], and follow the same flow of proof, without any particular novelty. It would be useful if the authors mention the technical challenges their definition might propose to the problem and how it differs from the challenges one needs to overcome while taking the general generalization bounds.
The insight of some bounds being tighter than others again is well-studied in previous works. However, I agree with the authors that the tightness of class-wise generalization as opposed to general generalization bound in Corollary 1 is new.

### Questions
- In the example of "truck", the authors show that their bound captures the behavior of the noisy case. Is it possible to explain the increase in the error w.r.t samples using the proposed bound?
- In Remark 3, the authors mention that their bound is discrepancy-independent. However, the reason for this is not their bound, but the change of the definition from $L_E(W, S)$ to $L_{E_Q}(W, S)$. I would suggest the authors rephrase this part and perhaps break it down into two factors, one due to the change in the definition (still dependent on the discrepancy), and one the bound (discrepancy-independent) to avoid being misleading.
- In [1], the authors provide a general framework for information-theoretic bounds which saves a lot of proof as long as the authors can show some regularity of the distributions. Since the authors' technique of proof follows the same techniques in the general generalization bounds, have authors considered using this framework?


[1] Chained Generalisation Bounds, Clerico et al. COLT 2022

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
In this paper, the generalization performance of learning algorithms are studied on a classwise level—i.e., bounds are provided between the population loss for samples on a class and the training loss for samples from that class. These bounds are obtained using information-theoretic tools, i.e. the KL divergence and conditional mutual information. Empirical evaluations are performed to verify that classwise discrepancies exist in real applications, and that the proposed bounds correlate with them. Several extensions and connections to related problems (subtask, sensitive attributes) are presented.

### Strengths
The paper begins by motivating the problem at hand in a very nice way, illustrating that classwise error discrepancies occur and do not behave in ways that are immediately obvious. The presented results are evaluated thoroughly, and interesting extensions are discussed. While the core of the derivations rely on standard techniques, there are some technical intricacies to be dealt with, and the present application is new.

### Weaknesses
Some more discussion on the topic of imbalanced data sets and how this affects classwise generalization error would be of interest. For instance, explicitly stating that the classes in CIFAR10 are balanced could be useful for the reader.

The discussion in Appendix C.2 does not seem to be accurate. Unless I’m mistaken, the following appears to be perfectly valid:

$$ E_{\hat Z_{[2n]}, U, W}[ \frac{1}{n} \sum_i \ell(\hat Z_i^{-U_i}, W)  - \ell(\hat Z_i^{U_i}, W)] = \frac{1}{n} \sum_i  \big( E_{\hat Z_{i}, U, W}[ \ell(\hat Z_i^{-U_i}, W)]  - E_{\hat Z_{i}, U, W}[ \ell(\hat Z_i^{U_i}, W)]  \big) $$
$$ = \big( E_{\hat Z_{1}, U, W}[ \ell(\hat Z_1^{-U_1}, W)]  - E_{\hat Z_{1}, U, W}[ \ell(\hat Z_1^{U_1}, W)]  \big) =  \big( E_{\bar Z, \bar W}[ \ell(\bar Z, \bar W)]  - E_{Z, W}[ \ell(Z, W)]  \big) $$

which is the generalization gap. First step was linearity of expectation, second used the symmetry of the algorithm + iid (despite the altered process, the sample pairs are iid), and the last one used that $WW$ and $hat Z_1^{-U_1}$ are independent. The fact that the generalization gap is linear in the test/train losses mean that the $y$-specificity of the pairs is marginalized out.

Now, if one were to consider bounds where the left-hand-side was given in terms of a function of train/test loss (e.g., the binary KL divergence as in the Maurer-Langford-Seeger bound), this would no longer hold.

Minor:

Should $n_y <n $ on line 136 be $\leq$?

“$n$ super-samples” on line 193: in the original paper from Steinke and Zakynthinou, the term “supersample” was used to describe the full set of $2n$ samples.

Corollary 2 and Theorem 5 (e.g.): Would be nice to adjust the size of brackets.

Line 1070: “requieres”

### Questions
— Could you address the comment above regarding Appendix C.2? I acknowledge that I may have missed something.

— How does the approach with class-specific gradient noise in Appendix D.7 compare with an approach that adds gradient noise for every class?

— There are some results in the PAC-Bayesian literature on generalization bounds for the confusion matrix (e.g. Morvant et al, “PAC-Bayesian Generalization Bound on Confusion Matrix for Multi-Class Classification”). How does your work relate to this literature?

— Lemma 2: Does $V$ depend only on $W$ here or on $W$ and $z_{[2n]}$?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
In order to address the issue that the existing generalization boundaries cannot capture the varying generalization performance across different classes, the authors introduce the concept of "class-generalization error," and derive bounds based on KL-divergence and super-sample technique, validating their bounds in different networks.

Their theoretical tools extend to: (i) Deriving class-dependent generalization error bounds affecting standard generalization and tightening existing bounds; (ii) Providing tight bounds for subtask problems with test data subsets; (iii) Deriving bounds for learning with sensitive attributes.

### Strengths
Originality: The authors introduce a new concept - "class-generalization error", to address the issue that the existing generalization boundaries cannot capture the varying generalization performance across different classes.
Quality: The authors have provided detailed proofs for each conclusion and conducted comprehensive numerical experiments.
Significance: The authors not only extend their theoretical findings to derive bounds for diverse scenarios but also shed light on label-dependent generalization bounds.

### Weaknesses
The notation in the formulas is not concise enough. For instance, in Eq 23., the parameters of $\ell$ are in the order of w, x, y, whereas in Eq 54, they have become z, w. In addition, some formulas are not aesthetically pleasing, such as [] in Eq11.

Some questions about the proof are discussed in Question part.

### Questions
1.Motivation: Even though the authors reveal label-dependency of generalization bound, in addition to mentioned in the Sec 4.1, can you discuss how different labels affect the standard generalization error? Or can you explain the scenarios where class generalization must be considered?
2.Essentiality: The derivation of Theorem 1 and Lemma 2 starts from KL divergence (or CMI) and does not exceed this scope. Do these conclusions reveal any more fundamental facts?
3.Proof Details: (i) Regarding Equation 30, is there an inversion of the positive and negative signs? (ii) In response to the issue mentioned in Remark 2, the authors introduce a novel loss random variable. However, the bound presented in Equation 47 is also applicable to the expression preceding Equation 31. Consequently, in Theorem 3, the term$\max\left(\mathbb{1}_{\left\{y_{i}^{-}=y\right\}},\mathbb{1}_{\left\{y_{i}^{+}=y\right\}}\right) $can be omitted, similar to the approach taken in Theorem 4. Do these facts have any impact on your findings?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes characterizing classification generalization errors of each class separately. The motivation is that neural networks do not generalize equally for all classes. The paper demonstrates this phenomenon empirically on CIFAR10 and CIFAR100, and also notes that class-wise generalization depends on factors beyond the class itself. Main contributions are several variants of per-class information-theoretic generalization bounds. The paper notes that the mutual information between the model and the class data can be used to characterize (or upper bound) the class generalization error.

### Strengths
This paper is theoretical in nature. The idea of pursuing generalization errors for each class is claimed to be new. If so (to be verified with other reviewers), then this research direction is original, The proposed information theoretical bounds appear to generalize the bounds studied in  Bu et al., 2020, Xu & Raginsky, 2017, Zhou et al., 2020 and other works, which mostly study the standard generalization bounds (i.e., averaged over all classes, and are not class-wise).

Empirically it is verified that the results in Theorem 3 and Theorem 4 provide valid class-wise generalization bounds in both standard CIFAR 10 and a noisy variant of CIFAR 10. This demonstrates that the contributes bounds are widely applicable to several settings.

### Weaknesses
(Recommended actions are clearly listed in the Questions section.)


*Presentation:*

Writing can be much improved. In particular, many theoretical results are not accompanied with a justification, or a practical implication. Writing appears to be tailored to specialists of this subject area. Sections and writing flow in the paper can also be improved. For instance, Theorem 1 is introduced only to be superseded by Sec 2.2 which, at the beginning of the section, states a limitation of Theorem 1. Def 2 (a definition of class-wise generalization errors) is introduced only to be superseded by Def 3 (see reason at L210). It is hard for me to form a coherent story. Same for Theorem 2 (class-CMI) which is introduced only to be commented on before introducing Theorem 3 that the bound in Theorem 2 is hard to evaluate (see L275). 

*Novelty:*

The contributed results, while generalizing bounds in existing results, are also heavily based on them. This raises a concern about the novelty. For instance, Def 1 looks like an adaptation of Eq 3 in Bu et al., 2020.

### Questions
Major questions/comments:

1. Definition 1 looks like an adaptation of Eq 3 in Bu et al., 2020. In particular, the difference of two terms given by the population loss and the empirical loss remains. The difference is that Def 1 in the present work considers conditioning on a specific class y. Beyond this generalization to per-class analysis, **could the authors please highlight contributions compared to Bu et al., 2020 (and also Xu & Raginsky, 2017, Zhou et al., 2020)?** I am asking as a non-specialist reader.

2. Theorem 1 in the present work is similar to Theorem 1 of Xu & Raginsky, 2017, except that Xu & Raginsky, 2017 considers no conditioning on a specific class y. Could the authors comment on the difference? 

3. The idea of “supersample” used in Def 3 seems to be inspired by Steinke & Zakynthinou 2020 (see Sec 1.2 therein). Could the authors comment on the new parts in the present work wrt to the original supersample idea?


4. The notation $P(\mathbf{W}|\mathbf{S})$ is used in the paper to denote the distribution of learned weights $\mathbf{W}$ given a training sample $S$. In Def 1, the defined class-wise generalization error involves $P_{\mathbf{W}, \mathbf{S}_y}$. This presumably means  $P(\mathbf{W} | \mathbf{S}_y)  P(\mathbf{S}_y)$. What does $P(\mathbf{W} | \mathbf{S}_y)  $ mean? How can a classifier learn from only data from one class $y$?

5. L210. How does Def 2 depend on $P(y)$? Likewise how does Def 3 *not* depend on $P(y)$?

6. L272: “if model parameters W memorize the random selection U, the CMI and the class-generalization error will be large”. I thought that U is a vector of Rademacher random variables introduced solely for analysis. Do you actually empirically sample U and use it to train W?


Minor:
* L1454 in the appendix. Log is missing when expanding the KL divergence.

* Minor suggestion. I think CIFAR10 can be considered a solved problem (much like MNIST). I am aware that the paper has experimental results on CIFAR100 in the appendix. Test accuracy on CIFAR 100 of state-of-the-art models can be almost 100% these days. On the empirical side, it would be good to consider one more experimental setting where the model lacks capacity to tackle the problem. CIFAR 10 is too easy for a model class like ResNet 50. For instance, considering a ResNet 8 on ImageNet would be a good setting where the model struggles to learn well due to the lack of capacity. It would be good to check whether the proposed bounds hold in this setting as well. 
(Note that this is not a request for more experiments for the purposes of evaluating the paper. It is a suggestion to help strengthen this work. The authors need not try to conduct this experiment during the rebuttal. It’s not a major request.)

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
6

### Confidence
3