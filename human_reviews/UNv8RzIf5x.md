# Class-Wise Generalization Error: An Information-Theoretic Analysis

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
Existing generalization theories of supervised learning typically take a holistic approach and provide bounds for the expected generalization over the whole data distribution, which implicitly assumes that the model generalizes uniformly for all the classes. In practice, however, there are significant variations in generalization performance among different classes, which cannot be captured by the existing generalization bounds. In this work, we tackle this problem by theoretically studying the class-generalization error, which quantifies the generalization performance of each individual class. We first derive a novel information-theoretic bound for class-generalization error using the KL divergence, and we further obtain several tighter bounds using the conditional mutual information (CMI), which are significantly easier to estimate in practice. We empirically validate our proposed bounds in different neural networks and show that they capture the class-generalization error behavior closely. Moreover, we show that the theoretical tools developed in this paper are useful beyond this context and can be applied in several other applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the concept of "class-generalization error," which measures the generalization performance of the algorithm for each individual class. It further applies previous information-theoretic tools to derive various bounds. Empirical studies validate the proposed bounds in different neural networks. Finally, the authors discussed some possible applications of these bounds.

### Strengths
1. The motivation to investigate the class-wise generalization error is reasonable.
2. Although not in-depth enough, this paper has related the class-generalization error to several applications.

### Weaknesses
### Major Concerns
1. "...provide bounds for the expected generalization over the whole data distribution, which implicitly assumes that the model generalizes uniformly for all the classes." 
   * I am afraid that I cannot agree with this statement. Considering the expected generalization error over the whole data distribution means measuring the model **on average** over all the classes, not **uniformly**, especially when the classes are imbalanced.
2. The number of training samples in Figure 1 is the whole sample number of all the classes.  
   * To interpret the reason for the different generalization gaps over classes, we need first to consider whether the classes are balanced.
      - In the website of the CIFAR 10 dataset: "...The training batches contain the remaining images in random order, but some training batches **may contain more images from one class than another**. Between them, the training batches contain exactly 5000 images from each class."
      -  So, could the authors also plot the class-wise gap w.r.t the sample numbers for each class?
   * Moreover, introducing label noise (5%) for each class will cause an imbalanced sample number for each class, which means the class-wise sample numbers in the right plot must be different (in terms of the given label not true label). That's why we observe a different trend.
     - How the label noise is introduced? Have you introduced the noise on both train data and test data? If only on the train data, there is a distribution shift. If on both, it's the above-mentioned class imbalance.
3. Assumption 1, Lemma 1, and Theorem 1 seem problematic. 
   * The authors assume that the learning algorithm is symmetric w.r.t each training sample. As the paper is motivated, the generalization gap is non-uniform w.r.t each class. How could two samples from different classes be symmetric to the algorithm output of the whole dataset $P_{W|S}$?
   * Could the authors provide a detailed derivation for Lemma 1? I cannot find any proof for this.
   * Based on Assumption 1 and  Lemma 1, Theorem 1 is not related to the sample number $n_y$ and indicates all the samples inside a class contribute equally to the generalization. However, this is counterintuitive, where the samples close to the decision boundary should be more important.
4. The theorems are based on tiny modifications on previous MI and CMI bounds, which are not novel to me.

### Minor Concern
The citation format is incorrect, please pay attention to the use of citet and citep.

### Questions
Please see the previous section

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Several recent works have provided information-theoretic generalization bounds, which apply in an average case over all classes. This paper takes a more fine-grained approach, and obtains bounds that hold for each specific class. Specifically, the difference between training and population loss for a given class is bounded in terms of an information measure involving training data for that specific class. The bounds are numerically evaluated. Some applications and extensions are discussed, including the sub-task problem and sensitive attributes.

### Strengths
The paper gives a clear presentation of the problem at hand, and motivates well why it is interesting to tackle. The numerical evaluation is detailed for the main problem of the paper. Related literature is mostly well-covered. The connections to the sub-task problem and sensitive attributes are intriguing.

### Weaknesses
The theoretical machinery is essentially identical to prior work (Xu & Raginsky, Steinke & Zakynthinou, Bu & Veeravalli, Harutyunyan et al, etc). The only difference is in the introduction of the definition of classwise generalization error — after this, all the steps proceed as in these earlier works (some minor differences in the current CMI derivation). It also seems to me that the results can be derived in simpler ways (see questions below). Now, this in itself is not necessarily a weakness, provided that the end product is useful. There are some steps towards establishing this in Section 4, which to me is the most interesting part of the paper, but it does not go quite far enough. For instance, evaluations on particular sub-task problems, or algorithmic design for fairness inspired by the results, could be promising directions for extending these results.

### Questions
— It seems as if all of your results follow straightforwardly by the same technique as you use for Theorem 7. Consider a distribution $P_Z$ on a space $\mathcal Z = \mathcal T \times \mathcal X$, where $\mathcal T$ is, for instance, the label space or a sensitive attribute. Then, apply standard techniques for information-theoretic bounds with the joint distribution $P_{W\vert Z}P_{Z\vert T=t}$. With this, you can avoid the machinery with indicator functions used for the CMI derivations.

In the CMI setting, I am aware that $W$ still depends on the parts of the supersample where the labels are not the specified one in the class-generalization error. But since this part of the supersample only appears through $W$, it is marginalized out. In a similar vein, $W$ can depend on any other random variable (e.g., pre-trained), which also becomes marginalized out.

One potential way to exploit this would be to actually use the non-$t$ data to construct a more informed prior. Consider the data-split method in PAC-Bayesian bounds, as in “Tighter PAC-Bayes bounds” by Ambroladze et al or “On The role of Data in PAC-Bayes Bounds” by Dziugaite et al. This can be combined with the CMI approach (or the standard MI bounds), so that the prior is allowed to depend on data from $P_{Z\vert T\neq t}$.

— For the sub-task problem, are the target classes assumed to be known?

— Regarding the worst sub-population performance: isn’t it the case that $\text{gen} = E_T \text{gen}_T \leq \max_t  \text{gen}_t$? It is not clear to me why it needs to be re-written as an information-theoretic bound to establish a statement similar to (15). Furthermore, does this really motivate that the two are closely correlated? The maximum is clearly an upper bound, but I do not see why this indicates that they should be correlated.

***
Minor comments:

— After 1, you require $n_y < n$. Isn’t $n_y\leq n$ fine?

— The term “overfit” seems a bit unclear (compare “benign overfitting”, e.g.). Does this necessarily imply a high population loss?

— I believe the term “supersample” usually refers to the entire set of $2n$ samples. It seems that you use it to only refer to a single pair of samples. Is this correct?

Assumption 1: satisfying -> satisfies; \citep when needed

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the class-dependent generalization error through using information-theoretic analysis. The authors begin by demonstrating the necessity of providing class-dependent generalization bounds, highlighting the potential variability in the generalization error across different label classes. They proceed to present class-dependent information-theoretic generalization bounds, including KL divergence-based bound, weight-based CMI bound, $f$-CMI bound and loss-difference CMI bounds. Additionally, empirical results are provided, along with extensions of their findings to other problem settings.

### Strengths
Compared with previous works, this paper has the following pros:
1) As demonstrated in its empirical results, previous information-theoretic bounds fail to capture the generalization behavior of individual label classes, while the bounds in this paper overcome such limitations;
2) The class-dependent generalization bounds introduced here can also be used to bound the standard generalization error; 3) The analytical framework of this paper is extendable to other problem settings, such as learning in the presence of sensitive attributes, a domain where previous information-theoretic bounds have not been applied.

In addition, this paper is well-written and easy to follow.

### Weaknesses
One main limitation is that the empirical results are confined to ResNet50 on CIFAR10/Noisy CIFAR10. Adding more experimental settings (e.g., SVHN) would further show the effeteness of the proposed bounds.

Furthermore, I have some technical concerns; please see the questions below.

### Questions
Major concerns:

1. Regarding Definition 2: Is the class-generalization error defined in Eq.(6) equivalent to Eq.(3) in Definition 1? If so, could you rigorously demonstrate their equivalence?

To my understanding, it is natural to see that Eq.(3) is the same as the following expression:
$$
\mathbb{E}\_{\bf Z}\left[\mathbb{E}\_{U,W|{\bf Z}}\left[\sum\_{i=1}^n\left(\frac{\mathbb{1}\_{\{Y\_i^{-U\_i}=y\}}}{\sum\_{i=1}^n \mathbb{1}\_{\{Y\_i^{-U\_i}=y\}}}\ell(W,Z^{-U\_i}\_i)-\frac{\mathbb{1}\_{\{Y\_i^{U\_i}=y\}}}{\sum\_{i=1}^n\mathbb{1}\_{\{Y\_i^{U\_i}=y\}}}\ell(W,Z^{U\_i}\_i)\right)\right]\right].
$$
By definition, we know that $\sum\_{i=1}^n\mathbb{1}\_{\{Y\_i^{-U\_i}=y\}}+\sum\_{i=1}^n\mathbb{1}\_{\{Y_i^{U_i}=y\}}=n^y_{\bf Z}$, and I think on average we also have $\mathbb{E}\_{U}\left[\sum\_{i=1}^n\mathbb{1}\_{\{Y_i^{-U_i}=y\}}\right]=\mathbb{E}\_{U}\left[\sum_{i=1}^n\mathbb{1}\_{\{Y_i^{U_i}=y\}}\right]=n^y_{\bf Z}/2$. However, this may still not be sufficient to derive Eq. (6). This also raises the question of whether we can use the class-dependent CMI bounds to bound the standard generalization error, i.e. is Corollary 2 a bound for standard generalization error or some other generalization notion?

2. Regarding Corollary 1: Is it tighter than the classic individual mutual information bound in Bu et al. (2019) when $\sigma_Y$ does not depend on $Y$ (e.g., using loss boundedness instead)? I think this might be accurate due to the following:
$$
\mathbb{E}\_{P_Y}\sqrt{D(P_{W,X|Y}||P_{W}\otimes P_{X|Y})}=\mathbb{E}\_{P_Y}\sqrt{\mathbb{E}\_{P_{W,X|Y}}\log\frac{P_{W,X|Y}P_{Y}}{P_{W}\otimes P_{X|Y}P_{Y}}}=\mathbb{E}\_{P_Y}\sqrt{\mathbb{E}\_{P_{W,X|Y}}\log\frac{P_{W,X,Y}}{P_{W}\otimes P_{X,Y}}}\leq \sqrt{I(W;Z)},
$$
where the last inequality is by Jensen's inequality.

If this is correct, you could explicitly state that the individual bound can be recovered from your Corollary 1 or Theorem 1.

3. Could you elaborate more on the last sentence in Remark 2, namely "Therefore, samples from other classes can still affect these bounds, leading to loose bounds on the generalization error of class y"?  As $\max(\mathbb{1}\_{\{Y_i^{-U_i}=y\}}, \mathbb{1}\_{\{Y_i^{U_i}=y\}})=\mathbb{1}\_{\{Y_i^{-U_i}=y \\;{\rm or}\\; Y_i^{U_i}=y\}}$. I understand the other parts in Remark 2. However, I don't understand why samples from other classes contribute to the looseness of the bounds. 

4. My previous question also raises another concern regarding the comparison between Theorem 4 and Theorem 3: I agree that $I_{\bf Z}(\Delta_y L_i; U_i)\leq I_{\bf Z}(f_W(X_i^{\pm}); U_i)$; however, I would like to point out that we also have $\max(\mathbb{1}\_{\{Y_i^{-U_i}=y\}}, \mathbb{1}\_{\{Y_i^{U_i}=y\}})I_{\bf Z}(f_W(X_i^{\pm}); U_i) \leq I_{\bf Z}(f_W(X_i^{\pm}); U_i)$ (doesn't this imply that
 $\max(\mathbb{1}\_{\{Y_i^{-U_i}=y\}}, \mathbb{1}\_{\{Y_i^{U_i}=y\}})$ makes Theorem 3 tighter instead of looser?) Therefore, it is uncertain whether Theorem 4 is tighter than Theorem 3 or not. The empirical results suggest that Theorem 4 is tighter, but I hope the authors could clarify why this is expected.

**I would be happy to increase my score if authors could adequately address my main concerns.**

Minor comments:

1. There is a related work that could be included: Hrayr Harutyunyan, et al. "Improving generalization by controlling label-noise information in neural network weights." ICML 2020. In that work, they decompose the mutual information term by chain rule: $I(W;Z)=I(W;X)+I(W;Y|X)$, and using $I(W;Y|X)$ as a regularization term. This might be the first work to incorporate label information into information-theoretic bounds.

2. Notations are not always consistent. For example, in Eq.(6), the selection random variable in the loss function is ${\bf U}_i$ but it becomes ${U}_i$ in the identity function.

3. In Theorem 8 in the Appendix, you may use the widely accepted term for the loss pair-based CMI, namely *evaluated CMI* or *e-CMI*, which was initially introduced in the original CMI paper (see Section 6.2 in the arxiv version of Steinke \& Zakynthinou (2020)).

4. After Eq. (26) and Eq. (48): "Next, Let" ---> "Next, let". There might be more similar typos.

5. When the authors or the publication are not
included in the sentence, the citation
should be in parenthesis using **\citep{}** instead of **\citet{}** or **\cite{}**. Most of citations in this paper are not in parenthesis while the authors or the publication are not part of the sentence. For example, in the first sentence of introduction, He \& Tao
(2020) should be (He \& Tao
(2020)) by using **\citep{}**.

### Soundness
2 fair

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
The paper addresses focusses on the generalization performance for individual classes rather than the whole data distribution. The authors argue that existing generalization bounds, which typically apply to the average performance across the entire data distribution, do not capture the variations in performance across different classes. To address this gap, the paper introduces a novel information-theoretic bound for class-generalization error using the KL divergence. Additionally, it proposes tighter bounds derived from the conditional mutual information (CMI). The results are supported with experiments on CIFAR dataset.

### Strengths
- This is the first work which has introduced class wise generalization bounds (as per the authors) which is an interesting and important idea. They identify that information theoretical based bounds are a natural class of bounds to use for this setting which is interesting.

- The paper is generally well written.

### Weaknesses
In my opinion, the main weakness of the paper is the incremental nature of this work. The use of KL divergence and CMI in deriving generalization bounds has been explored in prior literature. The work is essentially following the previous works with this additional conditioning. I don’t see any new technical challenges that appeared due to this conditioning. Neither have the authors mentioned that. 

So, the main contribution seems to be showing that information theoretical generalization bounds can be adapted to subsets of data easily.

### Questions
- I wanted to ask if there have been any previous works on computing class wise generalization bounds. Or, are there any works on generalization bounds for a subset of dataset with a particular attribute?
- The authors state that information theoretic bounds are natural for this setting as they depend on both the algorithm and the data. It would be useful to discuss this in more detail and discuss why it would be hard to obtain class wise generalization bounds for other types like stability based or hypothesis based. Couldn’t differing property of the subclasses be captured in some way to get different generalization bounds for different classes?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
