# Rethinking Fair Representation Learning for Performance-Sensitive Tasks

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
We investigate the prominent class of fair representation learning methods for bias mitigation. Using causal reasoning to define and formalise different sources of dataset bias, we reveal important implicit assumptions inherent to these methods. We prove fundamental limitations on fair representation learning when evaluation data is drawn from the same distribution as training data and run experiments across a range of medical modalities to examine the performance of fair representation learning under distribution shifts. Our results explain apparent contradictions in the existing literature and reveal how rarely considered causal and statistical aspects of the underlying data affect the validity of fair representation learning. We raise doubts about current evaluation practices and the applicability of fair representation learning methods in performance-sensitive settings. We argue that fine-grained analysis of dataset biases should play a key role in the field moving forward.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper summarizes the field of fair representation learning into three main categories and then illustrates from a causal perspective that the underlying problem that people have been trying to solve in fair representation learning is in fact somewhat ill-defined. The paper argues that if the test and training distributions are the same, fair representation learning is "futile". In addition, the paper also argues that if there is a distribution shift at test time then there is hope as well as when the features are somewhat separable.
The paper establishes theoretical foundations and well as experimentally illustrates the theoretical findings.

### Strengths
- The paper puts into perspective the whole fair representation learning field, using causal language.
- The paper critically establishes a key problem in the field and how researchers benchmark their own methods.
- The paper also shows experimentally that their theoretical findings have practical consequences which is very much appreciated. In particular, they show that if there is indeed a test=train time distribution, fair representation learning does not help much. and vice versa.

### Weaknesses
- The paper requires somewhat of a graphical model/ causal background which might be less beginner-friendly
- The paper is very dense and requires multiple passes to fully understand the paper. I would recommend adding further intuitions to the paper.
- Lastly, i might have missed this. Do the authors have any comments on the relations between FRL to standard fairness learning schemes that are not representation-based? I would be curious to hear of these problems persist necessarily.

### Questions
see above for the questions

### Soundness
3

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
3

### Summary
The paper defines causal structures representing realistic scenarios of dataset bias and discuss how the bias mechanisms may affect the performance and fairness of predictive models. The authors then prove fundamental limitations on fair representation learning when evaluation data is drawn from the same distribution as training data.

### Strengths
1. Originality and novelty: the authors propose a novel causal framework to categorize the bias in the data.

2. Supportive theoretical results: the authors prove the limitations of the FRL in IID settings.

3. Clear structure: the writing and organization is clear, making the paper easy to follow.

### Weaknesses
1. The causality-based notion in fair ML literature has been well discussed, and there have been numerous causal structures proposed. Can the authors provide a comparative analysis with one or two to further prove the novelty?

2. Limited experiments: it seems the experiments only focus on medical imaging scenarios. The results would be more convincing if datasets from other performance-sensitive areas are considered.

### Questions
1. The same as weakness 1, is there more comparison between the causal structures proposed by the authors and other literature?

2. Are there more experiments from another dataset in performance-sensitive areas other than just medical imaging?

3. Typo: in the contribution part (page 2), shouldn't the orderings begin with 1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper makes the argument that fair representation learning (FRL) is not a useful approach in the class of cases where model accuracy (or analogous metrics) is key. They provide an overview of several paradigms of fair ML, and highlighting which are “performance-sensitive”. They then apply a causal framework to discuss the situations in which FRL might be helpful, using this to claim that FRL is not helpful to achieve performance-sensitive objectives when test data is IID. They show experiments to support this, in particular highlighting that even when test data is IID, the causal structure of the problem matters.

### Strengths
- Note: I think I may have reviewed a previous version of this paper at another conference (also double blind, no anonymity issues). I have not gone back to look at my past comments so this review is with fresh eyes, and I believe that a number of improvements have been made to this draft. However, apologies if there are overlapping comments anywhere
- Really enjoyed the framing in Sec 2 around group parity, IID performance, and generalization to unbiased OOD, I think this sets up the paper quite well and gives a nice overview of the relevant parts of the fairness literature
- does a nice job of synthetizing causal reasoning with the FRL literature. In particular, the experiments around FRL effectiveness and causal structure are quite interesting
- in general I think the main point made here is correct and correctly-scoped, and useful for a specific segment of the community interested in these things

### Weaknesses
- I’m a little confused by the X_Z and X_A framework, what about features that are impacted by both Z and A? It’s not clear if this framework allows for that overlap
- It would be good at some point to discuss whether any of this applies in an anti-causal setup - the “futility” framing is quite strong and yet I feel as though the large anti-causal class of problems are not really touched on here
- Quibble with Def 4.5 - technically, I would phrase this as “have an equal amount of information” rather than “do not discard relevant information”. The question of whether information is discarded I think gets more into implementation and I’m not sure how “discarding information” is defined technically.
- Quibble with Def 4.4/Lemma 4.6 - I find something a little odd here which I think is around the corner case where ERM representations have no sensitive information. I think the important part of Def 4.4 (the “effectiveness”) of FRL is the equality to 0 - this is whether they are truly “fair” in this definition. However, Lemma 4.6 relies heavily on the first part of the def’n (having strictly less information) - if the ERM reprs have 0 sensitive information then trivially all FRL will violate effectiveness, but this seems wrong as that is not a property of the FRL method, but rather a property of the ERM representations which are given. Anyways, this is not a massive issue with the general point but I do think there needs to be some more careful wording with how the intuition is formalized here (e.g. maybe the I(ERM, A)=0 case needs to be separated out as “trivial”)
- The results in Fig 2 on subgroup separability go in the opposite direction of what I would expect - the authors note this and give some thoughts on why this might be. However, it suggests something troubling - that at separability=0.5 (i.e. no information at all about A, trivial case), the fair representation methods will yield low-accuracy representations, when in fact this should be the easiest case (just return the representations as given). This suggests to me that the implemented methods may not be working as intended. It would be good to have a supplementary experiment showing that the methods, are in fact, working as intended - otherwise the empirical results are a little harder to parse due to the chance of experimental failure.

### Questions
- The characterization in Fig 1 is interesting but a couple contextual questions - it’s not clear to me as a reader if this is a) a complete characterization of the types of causal structures that can produce this phenomenon, or b) if this is a novel contribution on the part of this paper, or something pre-existing in the literature

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles an important challenge in explainable and reliable AI by examining fair representation learning methods focused on achieving group fairness. The authors critically explore the limitations of existing approaches. They present three main fairness paradigm: enforcing group parity, maximizing subgroup IID performance, and generalizing to unbiased distributions. Their key theoretical insight is that achieving group fairness cannot be both effective and harmless (relative to ERM) within the IID setting. Furthermore, the paper suggests that the success of fairness learning methods depends heavily on shifts in test data distribution, hypothesizing that performance is also influenced by the underlying causal structure and the amount of sensitive information available in the training dataset.

### Strengths
This paper is self-contained and easy to follow, and the notation is clear. It is a joyful experience to read a manuscript like this. The theory is simple yet insightful. The experiment supports the theoretical as well.

### Weaknesses
1. One major concern is the lack of results in the distributed shift setting. While the negative result in the IID is sound, it would be good to have some results for out-of-distribution, or connects the out-of-distribution fairness to some existed theoretical analysis results in the area of  distribuiton shift.

2. Another concern is that the trade-off between fairness and accuracy has been explored by previous methods (Zhao et al., 2022). Why and how this paper is different than previous methods on analyzing the trade-off between fairness and accuracy, 

I would like to raise my score if these concerns are addressed.

[1] Zhao, H., & Gordon, G. J. (2022). Inherent tradeoffs in learning fair representations. Journal of Machine Learning Research, 23(57), 1-26.

### Questions
1. This paper mainly talks about group fairness, what about counterfactual fairness? Is the counterfactual fairness also impossible in the iid setting?

2. This work shows that distribution shift is closely related to group fairness, and there are some results in the area of domain generalization / adaptation. Can you elaborate some results in that area? Is there any method closely related to the group fairness?

3. How does different types of distribution shift (class imbalance, covariate shift, concept shift ...) in the dataset related to the three types of disparities?

### Soundness
4

### Presentation
4

### Contribution
3
