# Weighted Deep Ensemble Under Misspecification

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Deep neural networks are supported by the universal approximation theorem, which guarantees that sufficiently large architectures can approximate smooth functions. In practice, however, this guarantee holds only under restrictive conditions, and violations of these conditions give rise to model misspecification. We categorize such misspecification into three sources: variable misspecification, arising from insufficiently informative features; structural misspecification, stemming from the limited width and depth of networks that cannot fully capture the underlying complexity; and inherent misspecification, occurring when the true model possesses properties such as discontinuities that cannot be faithfully represented. To mitigate the impact of these forms of misspecification, ensemble methods have become a common strategy for enhancing predictive performance. However, standard ensembles composed of identically architected and equally weighted models may suffer from "collective blindness", where shared errors are amplified and lead to systematically biased predictions with high confidence. To mitigate this issue, we introduce weighted deep ensemble method that  learns the optimal weights. We prove that our method provably attains the convergence rate of the best single model in the ensemble and asymptotically achieves oracle-level predictive risk. To the best of our knowledge, this is the first work to provide rigorous theoretical guarantees for weighted deep ensemble under both well-specified and misspecified settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission is concerned with weighted ensembles of neural networks.

### Strengths
Weighted ensembles of neural networks are a relevant topic.

### Weaknesses
## A review and comparison with the state of the art is missing.

A review and comparison with the state of the art is missing.

First, weighed ensembles are in no way new. The basic idea goes back to stacking

David H. Wolpert. Stacked generalization. Neural Networks, 5(2):241–259, 1992

For a more neural network focussed paper see for example:

Anders Krogh and Peter Sollich. Statistical mechanics of ensemble learning. Physical Review E, 55(1) 1997.

Second, there are a lot of papers dealing with — theoretically well motivated — weighting of deep neural networks.
For example:

Andrés R. Masegosa. Learning under model misspecification: Applications to variational and ensemble methods.
In Advances in Neural Processing Systems (NeurIPS), volume 33, 2020  

Luis A. Ortega, Rafael Cabañas, and Andres Masegosa. Diversity and generalization in neural network ensembles.
In International Conference on Artificial Intelligence and Statistics (AISTATS),  2022

Third, there are also generalisation bounds for weighted ensembles, which are applicable to neural network ensembles:

Andrés R. Masegosa, Stephan S. Lorenzen, Christian Igel, and Yevgeny Seldin. Second order PAC-Bayesian
bounds for the weighted majority vote. In Advances in Neural Processing Systems (NeurIPS), 2020

Yi-Shan Wu, Andrés R. Masegosa, Stephan S. Lorenzen, Christian Igel, and Yevgeny Seldin. Chebyshev-Cantelli
PAC-Bayes-Bennett inequality for the weighted majority vote. In Advances in Neural Processing Systems529
(NeurIPS), 2021

Hauptvogel, Igel. On Uniform, Bayesian, and PAC-Bayesian Deep Ensembles. arXiv:2406.05469 [cs.LG], 2024

In addition, I was missing a reference to 

Lars Kai Hansen and Peter Salamon. Neural network ensembles. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 12(10):993–1001, 1990.


## The paper lack mathematical rigour.

The paper lack mathematical rigour. Examples:

Line 165: incomplete, meaningless statement. Seems like a part of the equation is missing.

Lines 180-190: Statement lack rigour: Example makes no sense without stating something about h.
In expectation over all hypothesises? 

Line 229: Should this be $\hat{f}$ on the RHS?

Condition 1: $\epsilon$ is not defined

## There are several misleading statements.

Theorem 1: The theorem only talks about n. Should there not be specific assumptions about n_train and n_val. 
E.g., does this hold for small constant n_val?

Just consider the first three sentences:

„Model misspecification in statistics arises  […]  inclusion of
irrelevant variables […]. In such cases, the best possible approximation […] still maintains a significant approximation error from the true
function“:  Could you please cite a rigorous  theoretical result that states that adding irrelevant variables must cause a significant approximation error 

„In deep learning, the neural networks are always assumed to be well-specified.“: In  general not true. I do not assume that - and I do not know anybody who does.

„To the best of our knowledge, this is the first study to offer a theoretical guarantee for weighted deep ensemble.“: Clearly wrong, see the many reference given above and references therein as starting points.

## General comments

Inherent misspecification: How much does this matter on digital computers (aka  „in practice“)?
This should be discussed.

While I think it is interesting to study weighted neural network ensembles, I have to say that I could not identify exciting novel insights in the manuscript.
Theorem 1 does not come as a surprise and is not put into relation to other (e.g., PAC-Bayesian) bounds.


## Minor comments

* „Ensemble methods is“ -> „Ensemble methods are“

* l 79: Strange references for deep ensembles. Why not not 

Lars Kai Hansen and Peter Salamon. Neural network ensembles. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 12(10):993–1001, 1990.

and the later cited 

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty
estimation using deep ensembles. In Advances in Neural Processing Systems (NeurIPS), volume 30, 2017.

* I think the discussion of mode misspecification should go along with a brief discussion of parametric vs non-parametric models.
* Line 282: Why „without loss of generality“?

### Questions
see "Weaknesses" above

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses a challenge in deep ensemble learning with model misspecification, where the universal approximation does not hold, and proposes an optimal weighted ensemble approach. Typical deep ensemble suffers from collective blindness as they reinforce shared biases, while the proposed weighted deep ensemble strategy can achieve oracle-level optimality. Comprehensive theoretical analysis shows asymptotic bounds for the estimator for regression and classification compared to the best candidate model and convergence under misspecifications. Experiments on synthetic tasks show improvement under various misspecification scenarios.

### Strengths
1. Interesting problem formulation: Systematic categorization of misspecification in deep learning with rigorous definitions.
2. Rigorous theoretical guarantees: Provides a formal analysis of the weighted deep ensembles with an asymptotic error bound matching the best candidate, and oracle optimality $R(\hat{w})/\inf_w R(w) \rightarrow 1$. Proofs are technically sound and leverage modern empirical process theory.
3. Comprehensive numerical validation: Experiments span all three misspecification types with nicely designed ablations.

### Weaknesses
1. The proposed algorithm is not new. The idea of of weighted ensemble with learned simplex weights by validation risk minimization has been explored with similar theoretical guarantees, just not on neural networks.
2. The theory only proves a "no-regret" sense of guarantee--the weighted ensemble performs at least as good as the best expert asymptotically. But the author did not investigate the ensemble gain under the weighted ensemble. This is especially important in the misspecification scenarios defined in the paper, where all models suffer from one or more sources of misspecification and are imperfect. The paper did not discuss how the weighted scheme affects the diversity or variance reduction that brings the ensemble gain under equal weight averaging. Even under misspecifications, candidate models may still have uncorrelated errors, which also explains the observed improvement in the numerical experiments.
3. Experiments are only done on synthetic datasets with shallow networks, which is good for demonstrating how different ensemble strategies perform under various misspecifications. It would be great to see how the algorithm works on standard small-scale vision benchmarks like CIFAR-10 or 100, or Tiny-ImageNet.

### Questions
1. This paper is primarily motivated by this notion of collective blindness, but it was not discussed later in the analysis. Can we somehow formalize the collective blindness of equal-weight ensembles through some sort of error decomposition and get a quantitative improvement bound for WDE compared with equal-weight ensembles?
2. In the proof of Theorem 1, the author leverages a Rademacher complexity term for the simplex with respect to both $M$ and $n$, $r\sqrt{\frac{2\log M}{n}}$. $M$ is omitted in the big-O as the number of ensemble members is finite, and the resulting bound becomes $\frac{r}{\sqrt{n}}$. But in practice, the number of ensemble members should somehow scale as a function of $n$, and you can only safely omit it if $M$ grows sub-polynomially with $n$. The author should clarify this somewhere in the paper, as you are deriving asymptotic bounds with $n$ while treating $M$ fixed.
3. The paper asserted the convexity of the VRM objective in classification because the ensemble is an affine mapping of logs. But this is only true for post-softmax probabilities. If the ensemble averages the logits instead (common practice in ensemble learning), would this break the convexity?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work conducts a theoretical study on Weighted Deep Ensembles, which assume unequal weighting coefficients across ensemble members. In this framework, the ensemble weights are learned through empirical risk minimization on a pre-held validation dataset.

### Strengths
- It appears to be well grounded in existing theoretical results for deep neural networks. In particular, Corollaries 1–3, which provide asymptotic error bounds for practical architectures such as MLPs, CNNs, and RNNs, are quite compelling. Of course, the practical usefulness of such theoretical results remains somewhat unclear, but that’s often the nature of theoretical work.

- From a quick look, the derivations seem sound, and the experimental design appears reasonably solid. I particularly like that Table 4 highlights an important comparison with In-sample and Greedy Ensembles, and Figure 2 nicely shows convergence toward the oracle weights.

### Weaknesses
- The method used in this work to determine the weighting coefficients for combining ensemble members’ predictions is a form of stacking (also known as stacked generalization, functional aggregation, and perhaps other related terms, as it has been referred to under various names in the literature). This approach has been extensively studied since the seminal works of Wolpert (1992) and Breiman (1996), with further theoretical developments by Van der Laan et al. (2007), Arsov et al. (2019), Chen et al. (2024), and others. However, this line of research is not discussed at all in the paper. The proposed weighted deep ensemble should explicitly position itself within the stacked generalization framework and clarify both the established findings in this area and its specific contributions in the context of deep neural networks.

- Corollaries 1–3 are presented in a somewhat simplified form in the main text, and although Appendix B.4 offers a slightly more detailed version, it still appears insufficient. It would be beneficial to include a fully formalized version in the appendix that explicitly incorporates the necessary conditions outlined in works such as Jiao et al. (2023). While those formulations, as far as I know, involve a number of intricate and cumbersome assumptions, this work, as a theoretical contribution building upon them, should nonetheless aim to achieve a comparable level of rigor and completeness.

- At present, there is neither empirical nor theoretical validation of the claimed “collective blindness.” The only supporting evidence is the conceptual illustration in Figure 1, which does not pertain to “deep” ensembles. While the authors claim that traditional deep ensembles may suffer from “collective blindness,” this assertion seems questionable given the experimental scale considered here, which can hardly be described as involving truly “deep” ensembles. In my experience, in synthetic setups with small MLPs, ensembles trained from different random initializations through stochastic optimization (i.e., standard recipe for constructing deep ensembles) often exhibit limited diversity, which aligns with the notion of “collective blindness.” However, as the network depth and complexity increase, the highly non-convex nature of the loss landscape tends to induce substantial diversity among ensemble members, and even simple deep ensembles can perform remarkably well, as demonstrated by Fort et al. (2019). Hence, it becomes difficult to argue that “collective blindness” remains a meaningful concern in deep ensemble settings.

- The experimental results seem rather limited to be considered a proper evaluation of a weighted “deep” ensemble. Given the computational constraints, it might be a good idea to extend the results of Wortsman et al. (2022) as a way to demonstrate larger-scale experiments. Their official codebase already provides checkpoints that can be directly used as ensemble components, so there is no need for additional training. Since they have already considered Uniform and Greedy Ensembles, it would suffice to simply add the Weighted Ensemble for comparison.

---
- Wolpert (1992), Stacked generalization.
- Breiman (1996), Stacked regressions.
- Van der Laan et al. (2007), Super learner.
- Arsov et al. (2019), Stacking and stability.
- Chen et al. (2024), Error reduction from stacked regressions.
- Fort et al. (2019), Deep ensembles: a loss landscape perspective.
- Wortsman et al. (2022), Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

### Questions
- Incomplete statement on line 164?
- How were the oracle weights in Figure 2 obtained?
- If the validation split (2 out of the 6:2:2 split) is used to “train” the weighting coefficients in WDE, then it is effectively being used as part of the ensemble “training” process. What if a standard deep ensemble is trained using the combined training and validation splits, since that data could alternatively be used to train the ensemble members rather than the ensemble weights?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper first defines three different kinds of model misspecifications, which if they occur in particular lead to traditional guarantees like universal approximation theorems for deep neural networks not holding. They furthermore argue that traditional ensembles, made up of models with identical architectures and each weighted equally, are also impacted by this issue since if every submodel is biased, this will usually lead to highly confident, systematically biased predictions of the ensemble. To address this issue, they propose and analyze the weighted deep ensemble method, which trains ensembles consisting of different models and optimizes the weights of the different ensemble members to optimize the error on the validation set. They prove that the ensemble achieves the convergence rate of the best model in the ensemble and empirically demonstrate the effectiveness of the method on synthetic datasets.

### Strengths
- The story of the paper was relatively clear and the paper was well-organized.
- By investigating the question when the assumptions of traditional machine learning approximation results fail to hold, the paper is making progress and bringing more attention to a very relevant question.
- Furthermore, by providing the weighted ensemble method, they also provide a new way of addressing the issues they point out. The analyses of the theoretical properties (i.e., showing both asymptotic error bounds and asymptotic optimality in certain cases) of the weighted ensemble method is original and insightful.

### Weaknesses
The paper claims that they are 'introduc[ing] [the] weighted deep ensemble method that learns the optimal weights'. At the same time, in the related work section, they state that 'recent studies have delved into weighted deep ensemble', but do not provide much more detail about these methods although this would be relevant to judge what exactly is novel in the paper. 
Furthermore, it would be relevant and interesting to also see the performance of their method on non-synthetic datasets (and with models trained on these tasks) to investigate whether they can also provide significant advantages in real-world settings and potentially on more not misspecified settings.

### Questions
1. Could you clarify what kind of work has been done on weighted ensembles before? What are the key differences of your work from previous work on this topic? 
2. In line 165, is $f_0(x)-g(\pi(\boldsymbol{X}))$ supposed to be $0$ almost surely? 
3. Could you make the following statement more formal or illustrate it a bit more clearly: 
> As stated before, traditional deep ensembles may suffer from "collective blindness" in the presence of variable, structural, or inherent misspecification. 
4. We use our validation data to fit the weights of the ensemble, correct? Would this in the case of many different ensemble members lead to overfitting on the validation data? Could we then still use the same validation data for hyperparameter tuning, etc.? 
5. Why do we need Conditions 1 and 2 for Theorem 3? What would lead to the Theorem not holding anymore if we relax these conditions? 
6. Why did you decide not to additionally test your methods on real-world datasets or mode widely on non-misspecified settings?

### Soundness
3

### Presentation
3

### Contribution
3
