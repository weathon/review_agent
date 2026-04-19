# RoCA: A Robust Method to Discover Causal or Anticausal Relation by Noise Injection

- Decision: Reject
- Scores: 6, 6, 3

## Abstract
Understanding whether the data generative process is causal or anticausal is important for algorithm design. It helps machine learning practitioners understand whether semi-supervised learning should be employed for real-world learning tasks. In many cases, existing causal discovery methods cannot be adaptable to this task, as they struggle with scalability and are ill-suited for high-dimensional perceptual data such as images. In this paper, we propose a method that detects whether the data generative process is causal or anticausal. Our method is robust to label errors and is designed to handle both large-scale and high-dimensional datasets effectively. Both theoretical analyses and empirical results on a variety of datasets demonstrate the effectiveness of our proposed method in determining the causal or anticausal direction of the data generative process.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a bivariate causal discovery problem where the observed labels are noisy. Observing that in the causal direction $P(X)$ does not contain information about the mechanism $P(Y|X)$, authors show that $P(\tilde Y|X)$ is a good surrogate in this setting and then propose a noise injection based method to discover the causal direction. Some theoretical results are given and experimental results validate the proposed method.

### Strengths
- The method of noise injection for causal discovery is novel and interesting. 
- Theoretical guarantees are given. 
- Illustration of the proposed method is good.

### Weaknesses
- Problem setting is not very rigorous
- No explicit identification result and no assumptions/conditions under which the proposed method can identifiy the true direction.
- Lacking some details regarding the experiment part.

### Questions
I reviewed this work in a previous venue where authors have addressed many of my concerns. For this version, I only have the following questions/suggestions:

- Compared with traditional biavariate causal discovery methods, this work and the proposed method are novel in two aspects: 1) noisy labels, and 2) high-dim features $X$ and scalar label $Y$. Thus, I think the paper should be made more clear regarding the problem setting and assumptions in Section 3.1:
  - "This mechanism, being correlated with P(Y|X), provides insights into the true class posterior."---how do you define "correlated"? and in what sense?
  - "It’s noteworthy that Pθ(Y˜ |X)  generally maintain a dependence with Pθ(Y |X)." Similarly, how do you define "dependence"  in a more accurate way (e.g., using math formulas) and in what sense?
  - "It is usually highly correlated with and informative about P(Y |X). Moreover, under a causal setting, P(X) cannot inform P(Y˜ |X), since Y˜ and Y are effects of X, and P(X) and P(Y˜ |X) follows causal factorization and are disentangled according to independent mechanisms (Peters et al., 2017b). Thus, Pθ(Y˜ |X) is an proper surrogate." -----From the causal graph in Fig. 2, I can see an edge $X->\tilde Y$  for both causal and anticausal settings, so the factorization should work for both directions. Or three should be a more accurate causal graph regarding the proposed setting?

- There seems no identification conditions  discussed in the main text? Please make identification explicit in the main text and discuss also the assumptions. This would help readers have a better understanding of the proposed method.
- Experiments: please do consider to have more details in the main text (e.g., in the camera-ready version where more pages are allowed or move some content to the appendix) For example, Table 1 seems not mentioned at all in the main text.
- A minor question: GES and some other methods work for scalar variables. How do you apply these methods to image data? I do no find such details and cannot say if the comparison with these methods are proper.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces RoCA, a method of determining whether the causal direction between features $X$ and label $Y$ is causal or anticausal. The observed labels $\tilde{Y}$ are a noisy proxy for the true labels $Y$. The approach assigns a pseudo-label $Y’$ to each datapoint based on a clustering of the feature space and then decides that the dataset is causal if $Y’$ cannot be predicted from the features $X$ and observed label $\tilde{Y}$ (implying that $Y’$ contains no information about $P(\tilde{Y} \mid X)$). This decision is done tractably by selectively adding noise to the observed labels $\tilde{Y}$ depending on the features $X$, then observing the disagreement between the pseudo-labels $Y’$ and the noisy labels at different levels of noise. The argument for this approach is that if $Y’$ is not informative of $P(\tilde{Y} \mid X)$, then adding noise would not change anything. However, if it is informative, then adding noise would make $\tilde{Y}$ increasingly unpredictable. Experimental results show that the approach is more accurate than competing approaches.

### Strengths
1. The theoretical analysis on the noise injection levels is quite interesting and insightful.

2. The hypothesis testing makes the end decision more quantifiable and systematic.

3. The experimental results are quite impressive.

### Weaknesses
1. The way that the concept of independent mechanisms is presented in this paper seems misleading.
First, the paper quotes at the bottom of page 3 that “the mechanism generating the effect from its cause does not contain any information about the mechanism generating the cause” and then concludes that “the conditional distributions of each variable, given all causal parents, are independent entities that do not share any information”. This conclusion only holds under the Markovianity assumption (i.e. no unobserved confounders). Under the presence of unobserved confounding, it is possible that the causal mechanisms can be independent, but a variable can still be dependent on some other variable given its parents. It seems that this assumption is actually key to the effectiveness of the proposed approach. However, this assumption is not stated anywhere in the paper.
Second, mathematically speaking, $P(X, Y)$ can be factorized as either Eq. 2 or Eq. 3 regardless of causal orientation. It is not clear what is the causal consequence of choosing one over the other.
Third, it is not formally explained what it means for a distribution to “inform” another, yet this seems to be key to understanding the proposed approach. Is the paper claiming to somehow infer the data-generating mechanisms from the distributions?

2. I am concerned about the soundness of the approach. The approach seems to rely on the property that within causal datasets, observed labels are evenly distributed among the clusters of $P(X)$, while they are not in anticausal datasets. This property does not seem to be related to any causal properties.

3. Assumptions are quite unclear. In addition to the assumption of Markovianity, it seems there are many more made that are not explicitly stated. In Sec. 3.1, it is discussed that $P_{\theta}(\tilde{Y} \mid X)$ can act as a surrogate for $P(Y \mid X)$, but there is no formal explanation on what this means. There is also little justification on the implications of the clustering algorithm. The results of this approach heavily depend on the outputs of the clustering algorithm, so there must be some implicit assumption that the clustering algorithm outputs something relevant to the causal structure of the dataset, which should be explicitly stated.

Given these concerns, I cannot recommend the paper for acceptance in its current form. I am open to hearing author responses in case I misunderstood something.

EDIT: Following rebuttal, I am raising my score from 3 to 6.

### Questions
1. In the introduction, it is mentioned that the causal graphs are assumed to be acyclic. However, there seems to be a cycle in Fig. 1b from $X_2 \rightarrow X_d \rightarrow Y \rightarrow X_2$. This seems to be a contradiction, could the authors clarify on this point?

2. In Sec. 3.1, it is mentioned that $P(\tilde{Y} = \tilde{y} \mid Y’ = y’, X = x)$ should equal $1 / C$ for each $x$. Does this only hold if the distribution of labels is uniform?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to discern if a data generation process leans towards being causal or anti-causal. Introducing the Robust Causal and Anticausal (RoCA) Estimator, the authors attempt to differentiate the two by investigating if the instance distribution, $P(X)$, offers pertinent details about the prediction task, $P(Y|X)$. They opted for the noisy class-posterior distribution, $P(\tilde{Y}|X)$, to act as a stand-in for $P(Y|X)$, and devised clusters using unsupervised or self-supervised techniques. Their findings suggest that in a causal scenario, there's no correlation between mismatch and noise levels, while in an anti-causal context, a correlation exists. The paper furnishes empirical evidence to support these claims.

### Strengths
1. The problem of interest is an important topic in casual discovery.

2. The method proposed overall sounds interesting and new.

3. The paper is well written.

### Weaknesses
1. The core premise of the paper, notably the logic of employing \(p(x)\) predictiveness to discern between causal and anti-causal directions for \(p(y | x)\), lacks solid substantiation. A deeper justification, supported by empirical data, would strengthen this assumption.

2. The paper does not present clear identifiability results. The claim that it's unnecessary to identify all potential causal relationships among single-dimensional variables is made without sufficient exploration. Additionally, concerns arise in an anti-causal context with a potential cyclic graph, questioning whether \(P(X)\) indeed offers valuable insight for the prediction task \(P(Y|X)\).
   
3. There seems to be a discrepancy in the paper's foundational assumptions on causal inference. While the authors state they align with the definitions in Sch¨olkopf et al. (2012), their handling of the Anticausal definition, especially regarding confounded cases, suggests otherwise. The paper needs to clarify its stance on unmeasured confounders.

4. The methodology for determining the noisy distribution and constructing \(P(\tilde{Y}|X)\) appears to lack a clear rationale. Offering detailed reasons for the "Noise Injection" approach and possibly introducing sensitivity analysis would bolster this section.

5. The method's practical relevance raises concerns. While the novel concept of integrating the causal direction into (semi-)supervised problems is compelling, its adoption in real-world applications remains questionable.

6. The paper seems to omit a comprehensive review of the related literature. Engaging more deeply with existing academic contributions would provide readers with valuable context, facilitating a better understanding of the paper's novelty and its positioning in the wider domain.

7. The overall organization and clarity of the paper need improvement. The section discussing experiments is notably intricate, making navigation challenging. A clearer structure and presentation would significantly improve the paper's readability.

### Questions
Please consider addressing the weakness I mentioned above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
