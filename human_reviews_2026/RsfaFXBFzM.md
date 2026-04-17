# Know When to Abstain: Optimal Selective Classification with Likelihood Ratios

- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
Selective classification enhances the reliability of predictive models by allowing them to abstain from making uncertain predictions. In this work, we revisit the design of optimal selection functions through the lens of the Neyman–Pearson lemma, a classical result in statistics that characterizes the optimal rejection rule as a likelihood ratio test. We show that this perspective not only unifies the behavior of several post-hoc selection baselines, but also motivates new approaches to selective classification which we propose here. A central focus of our work is the setting of covariate shift, where the input distribution at test time differs from that at training. This realistic and challenging scenario remains relatively underexplored in the context of selective classification. We evaluate our proposed methods across a range of vision and language tasks, including both supervised learning and vision-language models. Our experiments demonstrate that our Neyman-Pearson-informed methods consistently outperform existing baselines, indicating that likelihood ratio-based selection offers a robust mechanism for improving selective classification under covariate shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the problem of abstaining from making a classification decision. Abstaining can be useful when the classifier is likely to make a wrong prediction. The authors consider a formalism where abstaining decision is made by thresholding a confidence scoring function. Next, the authors review the Neyman–Pearson (NP) lemma on the optimal decision, and observe that some of the popular scoring functions can be viewed as approximations of the optimal rule. The authors then introduce two new scoring methods and describe conditions under which the methods are NP optimal. Effectiveness of the proposed scoring functions is supported by experiments on vision and language data.

### Strengths
1. The problem of abstaining rather than making incorrect predictions is an important practical problem
2. The authors offer a framework to unify previous and newly proposed confidence scoring functions. Relevance to the NP lemma is an insightful observation
3. The paper provides formal arguments (i.e., proofs) on optimality of different scores
4. Evaluation on different datasets shows usefulness of the proposed scores
5. The paper is clearly presented. There are minor issues, but overall the paper is easy to follow

### Weaknesses
1. On several occasions, justification of assumptions and theoretical constructs is not clear. First, it is not clear why p(y) should remain unchanged. It changes if relative frequencies of classes change. Also, it is not clear why exactly this assumption is required. Second, the practical implications of Lemma 2 are not clear. Third, Theorem 1 uses symbol "<<", which informally means "much smaller", but does not have any formal meaning
2. The newly introduced scores are not fundamentally new, since MDS and KNN scores have already been considered
3. On the practical side, it is not clear what amount of labelled data (e.g., relative to the amount of training data) is needed for the method to work reliably. This can be an issue, because modern classifiers can be constructed from pre-trained models with a minimum amount of training data (e.g., using few shot learning).
4. In terms of presentation, it would be useful to define AURC within the paper. Also, I'm not sure how NP Lemma implies that "thresholding this score yields the lowest possible selective risk for any given coverage level"
5. The experiments do not provide confidence intervals or p-values

### Questions
1. What amount of labelled data (e.g., relative to the amount of training data) is needed for the method to work reliably?
2. What do assumptions of Theorem 2 mean in practice?
3. What are the practical implications of Lemma 2?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the problem of selective classification, where one has to decide whether to predict or abstain. The authors leverage the Neyman-Pearson lemma which sets up the problem as a hypothesis test between H0 (the classifier makes a correct prediction) and H1 (the classifier makes an incorrect prediction). The authors can cast existing methods such as RLog, MDS, and KNN into the Neyman-Pearson framework to show optimality. In the end, the method takes a linear combination of classifier logit scores and distance, which produce strong results on image and text classification benchmarks.

### Strengths
This paper provides a unified framework based on the Neyman-Pearson lemma that captures existing methods (which are often treated as ad-hoc).
The paper is fairly well-written and uses proper mathematical notation.
The empirical results are strong.

### Weaknesses
I think the optimality of Neyman-Pearson is a bit overstated, since optimality depends crucially on the distributional assumptions being valid.

### Questions
Methods like MDS require estimating the covariance, which could be statistically expensive (require many samples) compared to RLog? The methodology in the experiments could be a bit more transparent: how many examples are required? In the end, the hybrid methods improve over existing methods, but I want to make sure I understand the resources / additional tuning that's required for each one?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work studies the problem of selective classification where a classifier is allowed to abstain from making predictions if the model is not confident enough. This work proposed new approaches to selective classification  based on the Neyman-Pearson lemma and also unifies several existing approaches to this problem. They also provide experiments to support their theoretical results and show that their method outperforms various baselines under covariate shifts where the test input distribution is different form the train input distribution.

### Strengths
- The authors using NP lemma to combine several existing baseline methods is simple and intuitive. 
- The authors proposed method - linear combination of distance based and logic based methods is simple and interesting.

### Weaknesses
- Theorem 2 relies on strong assumptions that the covariance distribution conditioned on the prediction is a gaussian. Theorem 3 relies on  k tending to infinity which is not practical. 
- The authors do not provide intuitive understanding of in which cases, their proposed method should perform well compared to the baseline.

### Questions
- The authors in lemma 2 assume that density for each distribution takes a tilted form. Could the authors please elaborate that?
- The authors mention in line 305 that knn-distance based classifiers are ineffective on high dimensions. Why do they work for this work?
- Are the values of lambda and k similar across datasets and classifiers? Or, they have to be tuned separately for each setting?
- How is g^* computed in eqn 7?
- The authors have not defined SIRC method which is used as comparison in the results.

### Soundness
3

### Presentation
2

### Contribution
2
