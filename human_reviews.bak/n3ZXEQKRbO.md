# Bridging Debiasing Tasks with Sufficient Projection: A General Theoretical Framework for Vector Representations

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
Pre-trained vector representations in natural language processing often inadvertently encode undesirable social biases. Identifying and removing unwanted biased information from vector representation is an evolving and significant challenge. Our study uniquely addresses this issue from the perspective of statistical independence, proposing a framework for reducing bias by transforming vector representations to an unbiased subspace using sufficient projection. The key to our framework lies in its generality: it adeptly mitigates bias across both debiasing and fairness tasks, and across various vector representation types, including word embeddings and output representations of transformer models. Importantly, we establish the connection between debiasing and fairness, offering theoretical guarantees and elucidating our algorithm's efficacy. Through extensive evaluation of intrinsic and extrinsic metrics, our method achieves superior performance in bias reduction while maintaining high task performance, and offers superior computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper applies the dimensionality reduction method of pooled marginal slicing (PMS) to find a linear subspace of word embeddings to project away information pertaining to the sensitive attribute for debiasing, which adds to a long line of research on mitigating (gender) bias in word embedding models.  The authors demonstrate the utility of the method on several benchmark tasks/datasets, and provide a theoretical justification for the proposed method.

### Strengths
- The paper is well-written and easy-to-follow.
- The proposed method is competitive with existing methods.

### Weaknesses
1. Description of hyperparameters is missing.  How was $H$ decided, and $q$ chosen, in algorithm 1?  I expect a tradeoff between accuracy and fairness for different settings of $q$, and it would be nice to include such a plot it in the paper.

2. The theoretical justifications/contributions—which is the main contribution of the present work highlighted by the authors— are very weak.

	- The *theory* relies on the strong assumption that "there exists a full rank matrix $B \in \mathbb R^{p_1\times q}$, such that $Z\perp X \mid B^\top X$".  Is there a theoretical discussion on the scenario where this assumption is violated?  How would the proposed method perform in such cases?
	- The authors lists *independence* and *separation* as the two desiderata of their method in definitions 3.1 and 3.2.  On the other hand, it is well-known that these cannot generally be simultaneously attained, see <https://fairmlbook.org/classification.html#relationships-between-criteria>.  How is it reconciled in the proposed framework?
	- The second result in theorem 6.1 is, in my opinion, too weak and not really meaningful.  If $X\perp Y\mid Q_y X$ and $\mathrm{span}(Q_y)\subseteq \mathrm{span}(Q)$, then $\widetilde X = (I-Q)X\perp Y$, i.e., the fair representation is useless for predicting $Y$ (related to the incompatibility between *independence* and *separation*).  In this regard, the first contribution claimed in section 1 of bridging the debiasing and fairness tasks does not hold up.

3. Some design choices are not (theoretically) justified.

	- In section 4.2, when $Z$ is categorical, why is there a need of learning a classifier instead of just using the one-hot encoding?

In summary, I think the authors has demonstrated that PMS is effective at debiasing word embeddings at least when compared to existing methods, although, I have my reservations on the practical usefulness of performing linear projection for debiasing word embeddings.  In particular, for text classification, there are more powerful methods (with theoretical guarantees) at achieving EO than performing linear debiasing (see follow-ups of Hardt et al. (2016)).  Instead, I would be more convinced if the author can demonstrate improved performance on WinoBias or WinoGender NLP tasks, for which bias mitigation methods for general classification tasks would not apply.

On the other hand, the theoretical discussions, as mentioned above, seems lacking if not problematic, and this weakens the overall contribution of the present work—is there anything new besides showing that PMS can give good empirical debiasing performance on the benchmark datasets?

### Questions
- What does *evenly* mean in "We first split all the words evenly into two classes by calculating the cosine similarity between..."?

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a debiasing method (SUP) that identifies minimal subspace correlated with sensitive attribute and project vectors against them. Under the linearity assumption, authors show that the suggested method can recover fair representation consistently. Authors show that the suggested method is superior to previous methods via evaluation using intrinsic metrics (word embedding bias) and extrinsic metrics (fair text classification performance).

### Strengths
- This method is applicable to continuous sensitive variables.
- Using a traditional tool (Pooled Marginal Slicing), the suggested method can provide theoretical guarantee under some conditions.
- Using intrinsic, extrinsic metrics, evaluation is conducted in two levels, supporting the connection between debiasing and fairness.

### Weaknesses
- While the method can be applicable to any tasks that use vector representations, experiments are limited to NLP tasks.
- Most of theoretical results are trivial from the linearity assumption and method.

### Questions
- Can this result be extended to other modalities? (e.g. fair image classification) While applications to other modalities are implied in the discussion, still I wonder if it is effective in other modalities as well.
- Can the asymptotic result for PMS in this setting be provided? I am curious about the sample complexity and how the number of partitions affect. Also, it would be nice if it is possible to reveal how the estimation error propagates to bias with some quantifiable measures.
- Can additional comparison with fair post processing methods be conducted? (e.g. https://fairlearn.org/)

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides a method for representation debiasing - the task of removing sensitive information from a learned representation post-hoc. The proposed method takes a projection-based approach, leveraging two existing approaches (SIR and PMS) to learn covariance matrices and taking their eigenvectors. Empirically, they run a range of experiments on word embeddings comparing to baselines demonstrating improved performance on metrics that look at both bias in feature space and classification.

### Strengths
- clear presentation
- thorough experimental section: I appreciated the quantity of experiments and baselines. The empirical component of this is I think a pretty significant contribution, as I don't think I've seen so many debiasing methods lined up side by side like this
- empirical results are strong and if they hold in general would be a nice improvement over other approaches in the space

### Weaknesses
- I think the "Debias" and "Fairness" terminology introduced in Defns 3.1, 3.2 is misaligned with how these words are usually used: I would consider these tasks to both be debiasing and both be fairness tasks. This is a simple fix but I think a very important one - I think there are many other ways to map these concepts onto notions of fairness/bias which make more sense (e.g. they correspond to different fairness metrics - demographic parity and equalized odds - I'm not saying this is the way authors have to go but it's an option)
- I think the contribution of the empirical section is nice - however I'm unsure of exactly how far the methodological contribution goes. It seems that the core of this method is in the regression approaches SIR + PMS. It would be good to know a bit more about why these approaches were chosen and why they are superior to other projection approaches, including those that already lean on eigenvalues (e.g. PCA-based approaches)
- Additionally, given how core SIR/PMS are to this method, I think they could use some more explanation: for instance, why is Z partitioned into H intervals? why does PMS require continuous inputs? how well does it work to first train the classifier f for its probability outputs?
- In general, I'm left a little bit unclear from Algo 1 how the "debiased" and "fair" representations, as they are referred to, are computed differently. It seems like everything specified here is for the X \indep Z setting, rather than the X \indep Z \| Y setting
- I find myself a little confused by some of the experiments in Sec 5.1, it seems like some details are bit short: for instance, I don't quite understand terms like "bias-by-projection" in Correlation or "original bias" in Profession Words. How are the top 500 male/female words in Clustering chosen?
- Also missing a few details in Sec 5.4: what model is used for prediction? is Time in seconds? specify more clearly what the predicted labels are
- I found Thm 6.1 a little confusing: 1) should the last statement be (I - Q_y)X? 2) is the assumption backwards - should span(Q_y) >= span(Q)? I looked at the proof and didn't it very enlightening, I think it could use a little more detail

Smaller comments:
- Assumption 3.3 + Eq. (1): it seems like there's a contradiction here - Ass'n 3.3 says that Z is perfectly predictable from a linear function of X, then (1) says there's a noise term included as well

### Questions
- Clarification: why would we need an intersection of sufficient dimension reduction subspace? in what cases would they differ from each other?
- Should clarify: is the output of PMS still minimal? it seems like the dimension-wise aggregation might result in a space with redundancies, but I could be wrong
- In general, a number of experimental details seem to be missing: for instance, how is q chosen? what about H?
- Interested to see that SUP improves over Glove in Table 2, left - I think this merits further exploration 
- I'd also like to see a comparison in Sec 7 to a PCA-based approach since that one also seems similar and solves directly for the debiasing direction

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
