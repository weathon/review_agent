# Data Imputation by Pursuing Better Classification: A Supervised Learning Approach

- Decision: Reject
- Scores: 8, 8, 5, 3

## Abstract
Data imputation, the process of filling in missing feature elements for incomplete data sets, plays a crucial role in data-driven learning. A fundamental belief is that data imputation is helpful for learning performance, and it follows that the pursuit of better classification can guide the data imputation process. While some works consider using label information to assist in this task, their simplistic utilization of labels lacks flexibility and may rely on strict assumptions. In this paper, we propose a new framework that effectively leverages supervision information to complete missing data in a manner conducive to classification. Specifically, this framework operates in two stages. Firstly, it leverages labels to supervise the optimization of similarity relationships among data, represented by the kernel matrix, with the goal of enhancing classification accuracy. To mitigate overfitting that may occur during this process, a perturbation variable is introduced to improve the robustness of the framework. Secondly, the learned kernel matrix serves as additional supervision information to guide data imputation through regression, utilizing the block coordinate descent method. The superiority of the proposed method is evaluated on four real-world data sets by comparing it with state-of-the-art imputation methods. Remarkably, our algorithm significantly outperforms other methods when the data is missing more than 60\% of the features.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The article addresses data imputation in a supervised classification setting. More precisely, the proposal frames the contribution within kernel-based approaches. By using weak assumptions on the similarity between the instances, the Gram matrix can be estimated so that the resulting classifier performs well with respect to the replacements made. Then, missing values can be identified using the obtained Gram matrix. The paper first presents the context and introduces the setting. Then, related works are briefly presented. The two-stage data imputation strategy is then detailed, before experiments are reported to study the properties of the proposal, and a short conclusion is drawn.

### Strengths
The paper is overall well written, in a clear and understandable way. The proposal is well described. 

The contribution is sound.

### Weaknesses
Some key points (in particular, how parameters can be set) are not addressed. 

Although technically sound and rational, the proposal could benefit from a deeper theoretical justification. 

The results only mildly support the claim that the approach is superior to the others: they are often (but not always) better, but by a small margin only (and the difference cannot be deemed significant), and they are reported over four datasets only.

### Questions
General comments and questions: 

The presentation of existing works is somehow a bit short: many works on learning from missing data or classification from missing data were not mentioned. There is no discussion on the nature of the missingness process here. The data seem to be considered as missing at random. 

It seems that the approach detailed in Section 3.2 somehow corresponds to an adversarial optimization of the classifier (which should perform well "on all possible outcomes within a norm sphere surrounding $\tilde{\mathbf{K}}$"): can you elaborate on that ? 

The remark regarding the results obtained for "extreme" values of $\gamma$ seem somehow obvious: the cases covered either correspond to instances being all dissimilar ($\gamma=1/32$), or similar ($\gamma=32$), hence the results. This raises the question of the sensitivity of your approach to the choice of $\gamma$—or, more generally, to the model parameters, the choice of which appears to be crucial, and actually very difficult to make without strong assumptions. Could you elaborate on that ? 

The results do not seem to be significantly different between the various imputation approaches compared, the only exception being the Australian dataset, with $m=80\%$. Do you have any insights regarding this ? 


More minor comments and questions: 

It is not clearly stated whether the missing values are in the training or test data (or both). 

Does ignoring the PSD constraint to solve the problem and then projecting back onto the space of PSD matrices have an impact on the result obtained ? 

Should Step 2 be skipped in the first phase, what would be the outcome of the proposed strategy ? It seems that this amounts to implicitly assume that the data are "perfect": can you provide any insights regarding this ? 

I do not understand why $\varepsilon^*$ should be positive definite (phase I, step 2, Equation (9), page 6). For complete consistency with Stage I, shouldn't $\varepsilon$ (or $\varepsilon^*$) be used as well in Stage II ? 


Some suggestions on writing: 

- page 1, "we typically use subsets of indices [...] with $\bfseries{x}_{\bfseries{o}_i}^i$": clearly define these notations; 
- page 1, "the importance of labels has not been fully taken into account": this statement, somehow a bit assertive, is difficult to understand; 
- page 2, "since there is $N-1$ supervising information available for each data": can you clarify ? 
- page 4, Section 3.1, paragraph "Notations": sentences should not begin with a mathematical symbol; 
- page 4, Section 3.2: matrix $\mathbf{K}_\Delta$ is not properly introduced; 
- page 5, "is a semi-definite programming": should be "is a semi-definite program"; 
- page 6, $m$ does not seem to be properly introduced; 
- page 7, "including \textit{australian}, [...]": should be "namely \textit{australian}, [...]".

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a two stage procedure for dealing with missing data targeted towards classification.  In the first stage, the authors propose a method which jointly finds a Kernel matrix K_{\Delta} and solves a dual SVM formulation.  Then, in the second stage, the authors use the Kernel matrix K_{\Delta} \odot K_o to perform data imputation via solving a non-convex optimization problem provided in Eq. (10).

### Strengths
Both stages of the proposed procedure are interesting and novel.  Moreover, the experimental results are promising.  The writing is also quite clear.

### Weaknesses
The main weakness lies in the choice of experiments and settings.  In particular, the authors consider a MCAR (missing completely at random) set up for the experiments in which they induce the missingness pattern in the data.  I do appreciate the results in Table 3 and the differentiation based on the amount of missing data, however, it would be stronger to include further validation.

### Questions
Would it be possible to include an experiment and compare the different methods on a dataset with missing data in which the missingness is not induced artificially? For instance, taking a dataset with missing entries but for which there are sufficiently many labels to train your method and the other methods to which you compare.  This would help to understand the method you propose beyond the artificial MCAR setting, which would be important.  I would be willing to raise my score if an experiment of this sort were to be included.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new kernel-based method for dealing with data missing at random. Their basic idea is to estimate them through the learned imputed kernel matrix. Their main novelty compared to previous papers is in making use of unknown features as well as observed ones while learning this kernel. Once the imputed kernel matrix is obtained, the missing values are estimated numerically by minimizing
minimize the discrepancy between the imputed kernel matrix calculated from the imputed data. The proposed method is compared with 4 other strong baselines on 4 benchmark data sets. The results indicate that the proposed approach could be better than the baselines particularly when the proportion of the missing data is large.

### Strengths
+ The paper is generally well written and easy to follow
+ The proposed approach is well justified and explained clearly enough
+ The experimental results indicate that the proposed approach might be better than the baselines

### Weaknesses
- This paper is mostly an incremental contribution compared to the state of the art
- There is a long history of research on imputation of data missing at random. Thus, comparing to only 4 other baselines on 4 small data sets (both in number of examples and features) might not be comprehensive enough. It is not clear why those particular 4 data sets were selected (other than being very small).
- Looking at the error bars, for most of the results the improvement does not seem to be statistically significant

### Questions
In addition to the previous comments, it would be useful to show the computational cost for the performed experiments. The largest data set used in the experiments has only 1000 examples. Is this because the proposed method is too expensive to run on larger data?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors tackle supervised learning with missing values, using support vector machines (SVMs).

To this end, they design a new way to learn a kernel matrix of an incomplete data set, by optimising the SVM loss (what is called "Stage I" in their algorithm). After this kernel has been learned, they use it to impute the data set by minimising the squared error between the kernel of the imputed data set and the learned kernel (this is "Stage 2").

They do several experiments on four real classification data sets.

### Strengths
The two stages of the algorithms both involve quite clever ideas.

The first stage in particular is based on the idea of treating the terms of the full kernel matrix that depend on incomplete data as parameters to be optimised. This is an excellent idea that is, to the best of my knowledge, novel.

The main idea behind the second stage is less innovative but very sensible. I also appreciate the fact that this "stage II" is empirically investigated on its own in Section 4.2.1.

----- POST-REBUTTAL EDIT -------

After the discussion with the authors, and giving this quite some thought, I have decided to change my score to 3 because of the maths clarification that I asked (point 3 of the "Weaknesses") that turned out to be a mistake that makes the algorithm potentially not valid (the constraints on the matrix may not be respected).

While the additional experiments were welcome, I do not think they were completely satisfactory, since the strongest baseline remains mean imputation (and, as discussed with the authors, one of the added baselines, Neumiss, is a linear regression method, which is not really fit for nonlinear classification). Adding stronger baselines would clearly make the experiments more compelling.

The final score is a bit harsh, and I reiterate that I believe the key ideas of the paper are quite good. I just don't think the paper is ready to be published yet. If the paper is accepted, I strongly encourage the authors to try to add a strong baseline to the tables (eg gradient boosting), to acknowledge that the claimed constraints might not be respected by the algorithm, and to clarify the data-preprocessing (normalisation is not an obvious task where there are missing data).

### Weaknesses
Main concerns

1) My main concern is related to the experiments, that have several issues, in my opinion.

a) Studying only 4 small data sets ($N\leq 1,000$) is not particularly compelling, especially given that most standard deviations are quite important (in Table 3, I doubt the author's technique is statistically significantly better than mean imputation in most scenarios).

b) Mean imputation appears to be, by far, the best method (if we ignore the author's method). This is not very consistent with the literature. For instance, in the genRBF paper (Smieja et al., 2019), genRBF is on par or better than the mean on the "Australian" data set, while it is much worse than the mean in this submission. Similarly, in the GEOM paper (Chechik et al., 2008), GEOM is essentially always on par with the mean, and is generally worse in this submission.

2) The authors do not study the theoretical properties of their methods. In particular, the assumptions on the missingness mechanism are not discussed. All experiments use missing completely at random (MCAR) data, and the authors do not discuss this experimental design choice. Studying (empirically and/or theoretically) whether or not this algorithm works on non-MCAR data would be interesting.

Secondary concerns

3) The paper read generally well, but the mathematics are at times quite unclear. Several objects are not properly defined and some facts are not really proven, for instance
- I imagine $K_0 = \exp ( - \gamma \sum_{p \in o_i \cap o_j} (x^i_p - x^j_p)^2 )$, but $K_0$ is never defined,
- in Equation (3), the mathematical meaning of $ (x^i_p - *) $ is unclear
- why is $K^*_\Delta$ in Equation (6) in the proper range (between $B_l$ and $B_u$)? After clipping, you project on positive semidefinite matrices, why is it guaranteed that it would be still in the right range?

Minor things

- I find it a bit odd to call $\mathcal{E}$ a "noise", since it not something random but something that you optimize
- I also find the phrase "imputed kernel matrix", used a few times (in different forms) a bit odd: this matrix is always a complete matrix, it is the dataset used to build it that needs to be imputed
- There has been a significant amount of work on supervised learning with missing values recently. Some of these papers could be interesting to discuss, for instance:

Josse et al., On the consistency of supervised learning with missing values, arXiv:1902.06931, 2020

Le Morvan et al., What’s a good imputation to predict with missing values? NeurIPS 2021

Bertsimas et al., Beyond Impute-Then-Regress: Adapting Prediction to Missing Data, arXiv:2104.03158, 2022

### Questions
- see questions in the "Weaknesses" section, point 3

- In your experiments, I did not understand if you used as a final classifier SVM with $\tilde{K}$ as a kernel matrix, or with the kernel matrix of the imputed data set ?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
