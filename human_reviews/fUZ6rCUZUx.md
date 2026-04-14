# Decoupling Dependency Structures: Sklar’s theorem for explainable outlier detection

- Decision: Reject
- Scores: 5, 5, 5, 6, 3

## Abstract
Recent advances in outlier detection have been primarily driven by deep learning models, which, while powerful, have substantial drawbacks in terms of explainability. This is particularly relevant in fields that demand detailed reasoning and understanding of why observations are classified as outliers. To close the gap between state-of-the-art performance and enhanced explainability, we propose Vine Copula-Based Outlier Detection (VC-BOD). We utilize Sklar’s theorem in conjunction with vine copulas and univariate kernel density estimators to decouple marginal distributions and their dependency structure for outlier detection. Our model uses a closed-form equation for the outlier score, which allows for detailed explainability and feature attribution. VC-BOD employs a traceable criterion to determine whether a new observation is an outlier, while also identifying the specific features responsible for this classification. The proposed model further distinguishes whether these features deviate from their own distributions or from interactions with other features. Our empirical evaluations demonstrate that VC-BOD outperforms most benchmarked classical models and several deep learning approaches in terms of average rank performance while proving competitive with the best-performing models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel approach to outlier detection (OD) named Vine Copula-Based Outlier Detection (VC-BOD). VC-BOD leverages Sklar's theorem, vine copulas, and univariate kernel density estimators to separate marginal distributions from dependency structures for the purpose of detecting outliers. The model provides a closed-form equation for calculating an outlier score, enabling detailed explainability and feature attribution. It distinguishes between features that deviate from their own distributions and those affected by interactions with other features. The empirical evaluation shows that VC-BOD outperforms classical models and is competitive with deep learning models in terms of performance.

### Strengths
1. The paper builds on a strong theoretical foundation with Sklar's theorem, providing a solid basis for the proposed method.
2. VC-BOD offers a high level of explainability, which is a significant advantage over black-box deep learning models, especially in fields requiring transparent decision-making.
3. The paper demonstrates that VC-BOD achieves state-of-the-art performance, outperforming all classical models and competing well against deep learning models. The empirical evaluation is thorough, with a comprehensive analysis across 31 tabular datasets, benchmarking VC-BOD against existing methodologies.

### Weaknesses
1. While the paper claims VC-BOD can handle high-dimensional data, the computational complexity and training times could be prohibitive for very large datasets.
2. The method assumes that the data can be modeled using kernel density estimators and copulas, which may not always hold, especially for data with complex or non-smooth distributions.
3. The paper acknowledges that the theory described is not directly applicable to non-continuous features, which may limit the method's applicability in certain domains.
4. The paper does not explicitly address the potential for overfitting, especially when using the full dependency structure captured by the vine copula.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a vine-copula-based method for outlier detection. The proposed method learns how features are related in terms of a vine copula, which then enables us to see which features contribute most to the proposed outlier score. The experimental results show that the proposed method is highly competitive in practice.

### Strengths
- The high-level strategy proposed is neat and makes a lot of sense.
- The experimental results look very promising.
- The interpretability aspect is very straightforward to understand for this method, largely owing to existing vine copula theory.

### Weaknesses
- Right now, the text has a *lot* of English grammar issues/typos. Please proofread and/or use a grammar checker. Partly because of these English issues, I had trouble following Section 4, where the outlier score is defined. I would suggest cleaning up the exposition to make this section much clearer, especially as it is for your proposed method.
- I think the experimental results would benefit from trying to drill down on what specific kinds of datasets the proposed method works extremely well for, versus which ones it doesn't work as well on. While Figure 3 is indeed very informative in terms of performance at the aggregate level across datasets, I don't think it helps us explain when and why we should expect VC-BOD to work well. For example, a breakdown of how the different methods (the proposed ones truncated at different levels vs baselines) perform for datasets with low vs medium vs high number of dimensions could possibly be helpful.
- Related to my second point above, I think adding some sort of experiments that looks at how training dataset size and dataset dimensionality impact results could be helpful. My suspicion is that for the Gaussian kernel density estimator to work well, it already needs a decent number of samples, and then to estimate the dependency structure takes even more data (the first paragraph in the discussion section focuses on computation time, but here I'm talking about accuracy/how well we can even estimate the dependency structure). As an example, you could take one of the datasets that you are already using and that is sufficiently large, and intentionally train only using 5%, 10%, 15%, ..., 100% of the data, and see how performance changes as a function of how much training data is used. Obviously there's also an issue here where for a dataset that has features that are independent, the problem should actually be easier since there's no dependency to learn basically.
- The paper already mentions how the Gaussian kernel density estimator's bandwidth is chosen but the existing rules of thumb for choosing the bandwidth were not designed with outlier detection or vine copula estimation in mind (as far as I know; they were just focused on minimizing mean integrated square error in density estimation), and I'm wondering to what extent tuning the bandwidth affects results. Basically if I understand the setup here correctly, part of the issue is that the bandwidth choice has a downstream impact on learning the dependency structure (so that it could be that, for example, intentionally either over-smoothing or under-smoothing has a less negative effect on how well the dependency structure is learned in the vine copula---I don't have good intuition of whether you'd want over-smoothing or under-smoothing in this case). Adding an experiment to study this would be helpful.

Minor (I might not have caught all the typos but here are some):
- On line ~142 (definition of a copula), "uniform marginals on $[0; 1]$" should instead say "uniform marginals on $[0,1]$"
- On line 219 (second line of Section 4), "In this paragraph" should instead say "In this section"
- On lines 234, 235, and 243: the standard notation for an interval is usually of the form $[a,b]$ and not $[a;b]$ (for example, even when copulas are first defined, the notation used is correct on line ~142 and says "$C:[0,1]^d\rightarrow[0,1]$" --- notice that the domain and range use interval notation correctly)

### Questions
Please address the weakness points that I raised.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a method for outlier detection that aims to achieve both interpretability and high accuracy. While powerful outlier detection methods using MLPs exist, they often lack explainability. This work, therefore, seeks to introduce a method that combines interpretability with high accuracy.

The method is based on Sklar's theorem, which decomposes the joint distribution of features into marginal distributions and a copula capturing dependencies between features. It employs kernel density estimation to estimate the univariate marginals and uses vine copulas to model the dependencies. Finally, the method assigns a score to each observation, classifying it as either normal or an outlier.

### Strengths
The strengths of this method lie in its ability to perform comparably to MLPs while providing interpretability by identifying which feature or feature dependency flagged an observation as an outlier. It is a simple method that nonetheless demonstrates good performance in experiments.

### Weaknesses
My primary concern is that the paper primarily combines existing methods and strategies rather than introducing new ones. It utilizes Sklar's Theorem, kernel density estimation, and vine copula methods—all of which have been previously developed. In this sense, the paper lacks sufficient novelty and originality. Its main contribution is limited to combining these established methods.

### Questions
The paper appears to consider a semi-supervised setting where only the labels of normal data are provided. Can this method be extended to cases where a few labeled samples from the outlier class are available (which could be viewed as a highly imbalanced classification problem)? How might this additional information enhance the method's effectiveness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on outlier detection. The authors apply Sklar’s theorem to approximate the distribution of normal observations via marginal distributions and a copula. Based on the approximated distribution, an outlier detection method is proposed by computing the scores of marginals and pair-copulas. The proposed method is able to explain the decision of detection, the decisive features, and the type of decisive features. Experimental results show that their method surpasses other classical methods.

### Strengths
- The most significant strength of this paper is the interpretability of their method. The interpretability of this method is three-fold, in terms of explaining the reason of decision, the features that contribute to decision, and the type of decisive features, which is still a gap in deep learning-based approaches.
- Compared to other deep learning-based approaches, another benefit of the method is its efficiency. It can be applied with CPUs in seconds on most datasets, which is much faster than training a Transformer. While the scalability of this method is not clear yet, it has a low computation cost in small datasets.

### Weaknesses
- Some notations in this paper is not clear with some potential typos, and some technical details are not well explained. I recommend the author to improve the presentation and include necessary explanation to make it more readable. Please check **Questions** for more details.
- Most datasets used in experiments are in a small dimension, which limits the impact of the proposed method. In Tab. 6, it takes several hours to run on larger datasets, such as Mullcross and Forest. I think it would be better if the author can discuss the scalability of the method with larger datasets.
- This paper mainly applies the Sklar’s theorem but lacks some necessary discussions, such as an ablation study of hyperparameters and the assumptions on normal distributions. Given that there is no theoretical analysis of the influence of those hyperparameters and distribution assumptions, it would be better for the authors to discuss it thoroughly to enhance the impact and practicability of their method.

### Questions
- In line 165, $u_i$ is defined as $(F_1(x_1^i),...,F_d(x^i_d))$, while its definition in Eq. 3 is confusing to me.
- In Eq. 4, what does $c_{i,j}$ represent? I don’t understand what is $F_i(x_i|d_{i,j})$ as well. What is $F_i$ conditioned on? How does $F_i$ concatenate the transformation of the feature?
- What is the effect of hyperparameters such as $k$ and $\eta$ on the performance? The author uses a default choice of those hyperparameters but there is a lack of guidance on how to selec them.
Minor:
- In lin 143, $[0;1]$ -> $[0,1]$
- In line 317, $f_2(d)$ -> $f_2(n)$

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors introduce a copula based outlier detection method, that focus on fully utilizing the Sklar's Theorem (the density of a random vector can be written as a factor of its univariant marginals and the gradient of an appropiate copula). The authors utilizes a Vine Copula, and fit all the densities using gaussian kernels with the ROT [Sheather & Jones] bandwidth. They provide a unique way of characterizing the outlier score of an outliying sample by penalizing the non-complience of any of the factor's inlier condition (in eq. 5). Furthermore, the authors provide a way of assigning explainability of an outlier sample using the Sklar's theorem formulation. 
They tested their One-class classification performance in section 6 against several classical and DL methods using a collection of 31 datasets across 40 repetitions. The Vine Copula-Based Outlier Detection method (VC-BOD) that they presented is claimed to achieve a significant performance over the collection of classical methods, while providing a competitive performance againts the more complex Deep Learning ones.

### Strengths
1. The model's novelty is good, and the idea of fully utilizing the Sklar's theorem is sounded
2. The motivation of the paper is solid and well presented by the authors
3. The method seem to provide a proper outlierness explanation following their experiments in section B.1. Explainability of outlier detection models is crucial for industrial-level applications.
4. The experiment section utilizes a fair amount of datasets and a large collection of competitors
5. The authors provide the code for reproducibility
6. The authors also provide a theoretical complexity analysis.

### Weaknesses
1. The model lacks a proper theoretical background to explain why it should work. While it is important to remark that the explainaitions are grounded in theory, the authors do not provide any guarantee of classifying an outlier as an outlier. The authors have access to an aproximation of the joint density function (as per the Sklar's theorem), which could solve this weakness. However, they do not seem to use it. 

2. The experimental evaluation is poor (which contributes to my score in Soundness). While it is true that the authors make use of a large list of datasets with a sound experimental setting, they do not use any statistical test to assest the significance of their results. Without this, it is impossible to assest wether their claims of SotA performance are true. The importance of these tests are highlighted throught recent OD literature [Campos et.al.] [Han et.al.] [Liu et.al], and even outside of Outlier Detection, even being a course of reject for similarly-sized conferences to ICLR (see section 7 of NeurIPS submission checklist).

3.The paper has a theoretical complexity study, and yet, there is no experimental evaluation of it. 

4.There is no experimental analysis of the explaination score with real world datasets, even though it is one of the major selling points of the method.

### Questions
I ask the authors to please answer my following questions: 

1. Why did they authors implement a different scoring function that the joint density? 

2. There exists a vact collection of xAI Outlier Detection methods in the literature [Sejr and Schneider-Kamp]. Is there any reason as to why the authors did not compare themselves to other xAI competitors? 

3. Lines 298-300 contains the following claim "If the vine copula fit is close to the ground truth, then adding more trees increases accuracy by reducing the number of false negatives,(...)". Why this would happend utilizing the scoring presented in eq. (5) is not imediate to me. Additionally, it seems to be an extreamly important theoretical result for your model that should be highlighted more. 

4. Please consider introducing a multiple comparison test into the experimental evaluation. As it is currently, it is impossible to assest if the method is not worst that the competitors presented.

I will consider increasing my score after succesfully addressing all of my questions/concerns listed above. 

### References
 [Sheather & Jones] Sheather, S. J., & Jones, M. C. (1991). A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation. Journal of the Royal Statistical Society. Series B (Methodological), 53(3), 683–690. http://www.jstor.org/stable/2345597

 [Campos et.al.] Guilherme O. Campos et.al. (2016). On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Mining and Knowledge Discovery 30, 891-927

[Han et.al.] Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. In Advances in Neural Information Processing Systems (pp. 32142–32159).

[Liu et.al] Yezheng Liu et.al. (2020) Generative Adversarial Active Learning for Unsupervised Outlier Detection. IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 32, NO. 8, AUGUST 2020.

 [Sejr and Schneider-Kamp] Jonas Herskind Sejr, Anna Schneider-Kamp, Explainable outlier detection: What, for Whom and Why?, Machine Learning with Applications, Volume 6, 2021, 100172, ISSN 2666-8270, https://doi.org/10.1016/j.mlwa.2021.100172.

### Soundness
3

### Presentation
3

### Contribution
3
