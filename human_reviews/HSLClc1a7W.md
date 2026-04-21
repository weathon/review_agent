# Latent Score-Based Reweighting for Robust Classification on Imbalanced Tabular Data

- Avg Score: 6.25
- Decision: Reject
- Scores: 6, 5, 8, 6

## Abstract
Machine learning models often perform well on tabular data by optimizing average prediction accuracy. However, they may underperform on specific subsets due to inherent biases and spurious correlations in the training data, such as associations with non-causal features like demographic information. These biases lead to critical robustness issues as models may inherit or amplify them, resulting in poor performance where such misleading correlations do not hold. Existing mitigation methods have significant limitations: some require prior group labels, which are often unavailable, while others focus solely on the conditional distribution $P(Y|X)$, upweighting misclassified samples without effectively balancing the overall data distribution $P(X)$. To address these shortcomings, we propose a latent score-based reweighting framework. It leverages score-based models to capture the joint data distribution $P(X, Y)$ without relying on additional prior information. By estimating sample density through the similarity of score vectors with neighboring data points, our method identifies underrepresented regions and upweights samples accordingly. This approach directly tackles inherent data imbalances, enhancing robustness by ensuring a more uniform dataset representation. Experiments on various tabular datasets under distribution shifts demonstrate that our method effectively improves performance on imbalanced data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a latent score-based reweighting framework to mitigate biases and spurious correlations in machine learning models on tabular data. The method overcomes limitations of existing techniques by capturing the joint data distribution without needing prior group labels and by focusing on balancing the overall data distribution, not just the conditional distribution. It identifies and upweights underrepresented samples based on the similarity of score vectors, thereby enhancing robustness and ensuring a more uniform dataset representation. The approach has been shown to effectively improve performance on imbalanced data across various datasets, even under distribution shifts.

### Strengths
1) This paper captures the joint data distribution without the need for prior group labels. It enhances model robustness by ensuring a more uniform dataset representation. Some cases are provided to explain their motivation.  
2) The experimental results on various tabular datasets have demonstrated the effectiveness of the proposed method.

### Weaknesses
1) Besides boundary-based methods that are mainly compared in this paper, other approaches, like stable learning, are also proposed to balance the overall data distribution. These methods should be also discussed and compared.
2) The design of the method is primarily driven by motivation. The effective generalization bounds require theoretical analysis and proof. In particular, the impact of the error from using proxies instead of density estimates on the model’s generalization capability needs to be explained, along with its limitations.
3) One of the challenges in estimating density estimates is the requirement for a large amount of data; that is, a substantial number of samples are needed to achieve accurate density estimates. It is suggested to investigate whether the proposed proxy of density is independent of the sample size. Relevant experiments, like testing the method on subsets of the datasets with varying sizes and comparing the results,  are recommended to test the effectiveness of the method from this perspective.

### Questions
Why is the method proposed in this paper limited to tabular data? Please give the reasons why it is not applicable to other data types, such as images or natural language.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper studies the distribution shift problem on tabular data. A diffusion model is used to estimate the data density. Then a sample weigthed is computed based on the density score to make the dataset more balanced. The proposed method is novel. Experiments on several tabular datasets with distribution shifts validate the effectivenss of the proposed method.

### Strengths
1. The distribution shift problem on tabular data is important.
2. Applying diffusion model to estimate the sample weight to reweight the sample is novel.
3. Experiments results on several classical tabular datasets validate the proposed method could relieve the distribution shift problem.

### Weaknesses
1. Motivation is not clear. The title includes the term "tabular data". However, the proposed model does not have any specific design on the tabular data. Moreover, why cannot the model be used in image data? There a lot of image benchmarks can be used to evaluate the problem method, such as WILDS, colored  MNIST, domainbed, waterbirds, celebA, etc.
2. Concept is confused. The manuscript that previous method mainly solves P(Y|X), then the proposed method solve P(X,Y) shift problem (i.e., both P(X) and P(Y|X) problem). To my understanding, the score is estimated only based on P(X), so it could only solve P(X) problem. How the model could handle P(Y|X) problem?
3. Related work has factual error. For the "Achieving robustness with prior information" paragraph, a lots of work do not require the prior information, such as HRM, stable learning, etc. The authors should carefully discuss the difference with them.
4. If the model could only solve the P(X) problem, the stable learning methods could also handle this problem. What is your major advantange?
5. The experimental resutls are weak. The improvement over baselines are marginal. More common used datasets on image should be tested (see W1). More related baselines should be compared, such as methods in [1].
[1] Change is Hard: A Closer Look at Subpopulation Shift

### Questions
Please reponse the questions in weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper attempts to improve robustness by modeling the joint distribution and taking a global perspective on re-weighing as opposed to the popular methods which mostly focus on rectifying the samples near the decision boundary. The observation of most current robustness methods are limited by the pre-trained classification boundaries $P(Y \mid X)$ adds a fresh perspective to the robustness literature. The paper has solid experiments and ablations to support its claims and it is well written.

### Strengths
- [S1] Well motivated paper, clearly highlighting the issue in current literature that the methodology is attempting to mitigate.
- [S2] The demo code seems well written and simple enough to run and adapt.
- [S3] I really like the idea proposed by the authors as it intuitively seems to overcome quite a few problems, the experiments and ablations provide good support to the approach.

### Weaknesses
- [W1] While I like the motivation presented in terms of the sinusoidal function and the multi-modal gaussian, it would be interesting to look at a more real world scenario where one can expect the proposed method to suit the setup better than traditional methods. That would further help strengthen the case.
- [W2] Table 1 does not include variance values in the main paper, it would be helpful to contrast the effectiveness of the method across various runs, given the general discussion about variance in other parts of the paper as well. Effectively combining Table 5 and Table 6 in the Table 1 and Table 2 or referencing the variation with some reformatting could be possible in the given space. Even moving Algorithm 1, potentially at the cost of Related Work might be a good idea.
- [W3] While the visualization in Figure 3 helps gain insight into how the method works as a proxy, it would be interesting to see some analytical reasoning behind why this might work at scale for larger datasets.

### Questions
- [Q1] I’m unsure of how, despite being 2nd best on the mean performance in 2 columns in Table 2 (Columns 11 to 19) the authors claim the method is not best in this scenario due to variations in the predictive mechanism as opposed to Table 1, where the method is not the best (or even top 2) on 3 datasets. So the story doesn’t translate as well, do the authors have justifications other than the intuition as to why they made this claim? It is not obvious how the change in the causal mechanism is a potential cause for variation in the predictive mechanism; more details on this would help clarify the statement.
- [Q2] I am also slightly curious about the formalism behind Section A.4, is there prior literature investigating or stating these are the non-causal attributes? If so are all other attributes not selected necessary causal? The selection of these attributes plays a big role in the methodology, so including correlation statistics or defining a proper heuristic for selection of these attributes would help make a stronger case.
- [Q3] Is there an intuitive way to extend this approach to multiclass classification? Would it be computationally feasible?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a latent score-based reweighting framework for improving robustness of classification accuracy under distribution shift. Score-based model (diffusion model) on latent representations of data distribution is used to estimate the joint data distribution P(X,Y), so that low density regions are upweighted to ensure a more balanced representation during training. The proposed method differs from previous methods since it does not require prior information and it models the joint data distribution without making assumptions on the pre-trained classification boundary of P(Y|X). Experiments on various tabular datasets under distribution shift demonstrate that the proposed method improves worse case classification accuracy as compared to baseline methods.

### Strengths
This paper provides a novel method that solves an important task and works well in practice by performing experiments across synthetic data and several real datasets.

### Weaknesses
1. Diffusion model is used to model latent covariates X. SimDiff addresses y shift by taking the difference between between two similarity metrics across two different classes. However, the paper claims that it directly estimates the joint data distribution P(X,Y). It would be better to make the connection clearer. 
2. Average similarity measure is proposed based on an intuition derived from a simulation example with mixtures of two Gaussians. Is the intution generalizable to more general settings? If the intuition is not satisfied in all settings, how would it affect the performance of the method?

### Questions
See weakness section above.

### Soundness
3

### Presentation
4

### Contribution
4
