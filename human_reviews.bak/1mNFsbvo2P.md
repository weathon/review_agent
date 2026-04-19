# Domain constraints improve risk prediction when outcome data is missing

- Decision: Accept (poster)
- Scores: 8, 8, 8, 5

## Abstract
Machine learning models are often trained to predict the outcome resulting from a human decision. For example, if a doctor decides to test a patient for disease, will the patient test positive? A challenge is that historical decision-making determines whether the outcome is observed: we only observe test outcomes for patients doctors historically tested. Untested patients, for whom outcomes are unobserved, may differ from tested patients along observed and unobserved dimensions. We propose a Bayesian model class which captures this setting. The purpose of the model is to accurately estimate risk for both tested and untested patients. Estimating this model is challenging due to the wide range of possibilities for untested patients. To address this, we propose two domain constraints which are plausible in health settings: a prevalence constraint, where the overall disease prevalence is known, and an expertise constraint, where the human decision-maker deviates from purely risk-based decision-making only along a constrained feature set. We show theoretically and on synthetic data that domain constraints improve parameter inference. We apply our model to a case study of cancer risk prediction, showing that the model's inferred risk predicts cancer diagnoses, its inferred testing policy captures known public health policies, and it can identify suboptimalities in test allocation. Though our case study is in healthcare, our analysis reveals a general class of domain constraints which can improve model estimation in many settings.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the selective labels problem applied to a healthcare context. Specifically, the paper proposes a Bayesian model for the problem and analyzes a special case of this model to show why two sensible constraints (a prevalence constraint, a human expertise constraint) improves inference. The paper also provides experimental results on synthetic and real data to show the effectiveness of the proposed model.

### Strengths
- The paper is very well-written and easy to follow
- The proposed model is simple and elegant; the authors do a good job explaining why the Heckman correction model is a special case
- The theoretical result is reassuring and also helps justify why the two suggested constraints help
- The experiments are well thought-out and I found the results to be compelling

### Weaknesses
I would like to see a more detailed discussion on how the model generalizes to more complex inputs (basically I'd like a more comprehensive discussion of Section 6's last sentence), especially as I think this is a very practically relevant extension. It would be helpful to understand to what extent the theory could explain this more complex setting (and under what assumptions one might need to additionally impose). It seems like a trivial extension would also be a partially linear model where some features are captured by a linear component and the rest are captured by a neural net.

Minor:
- Page 2: The text currently reads "Throughout, we generally refer to $Y_i$ as a binary indicator, but our framework extends to non-binary $Y_i$, and we derive our theoretical results in this setting" --- I would suggest rewording the last part so that it is clear what "this setting" refers to.

### Questions
See "weaknesses".

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, the authors introduce the widespread phenomenon that the data lies within the human decision censors that tend to be biased. The authors then proposed a hierarchical Bayesian model that addresses such data distribution mismatch between what has been tested and the underlying true distribution. The authors further proposed two constraints, prevalence constraint and expertise constraint to decrease the uncertainty of parameter estimation.

### Strengths
1. The proposed hierarchical Bayesian model to address the unobservables and connect it with the actual observation to evaluate the risk score and test decision makes sense and is novel. 

2. The prevalence constraint and expertise constraint used to shrink the estimation uncertainty is novel. In practice, the two constraints are usually easy to access, making such constraints practically useful. 

3. The authors demonstrated in synthetic data that the constraints proposed can effectively reduce the confidence interval and show in real data that the proposed constrained Bayesian model yields more reasonable discovery.

### Weaknesses
1. The actual Bayesian model derived from Proposition 3.1 seems too simple in practice. Having the assumption that the unobservable always comes from an independent normal distribution can be too strong. 

2. When applying the model to UK Biobank, filtering out individuals whose age is below 45 is not convincing.

### Questions
Can you explain in more detail why, without prevalence constraint, the beta_y parameter will decrease when the age variable increases in Figure 4? You mentioned that being tested for breast cancer before age 50 is unusual, but that doesn't completely explain why you observe this trend.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a Bayesian model designed to infer risk and evaluate historical human decision-making in settings with selective labels. The authors integrate prevalence and expertise constraints, leading to enhanced parameter inference, as demonstrated both theoretically and empirically.

### Strengths
- The paper is well motivated.
- he constraints introduced are logical and reasonable.
- Both theoretical and empirical analysis show improved performance.

### Weaknesses
- The chosen Bernoulli-sigmoid model may be overly simplistic. Especially in the healthcare field, the intricate relationship between features and labels might not be fully represented by this basic model.
- The empirical tests were limited to only 7 features, raising questions about the model's scalability with a larger feature set.
- Section 5.2's results are somewhat ambiguous. For instance, in the subsection "Inferred risk predicts breast cancer diagnoses," it would be beneficial to include a specific predictive metric, such as the F1 score.
- The paper doesn't specify how the new model's diagnostic prediction performance stacks up against a model that doesn't factor in selective label issues. For instance, how would a straightforward linear model perform (1) by training solely on the tested population or (2) by treating the untested group as negative?

### Questions
- How does the model perform on the older population, where the distribution shift is less severe?
- Can you elaborate more on why the $\beta_{\Delta}$ is negative for genetic risk score?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a Bayesian model for disease risk of the patients where only the outcome of the tested patients are observed. The proposed model has linear model for risk and testing decision on the observed variables. The paper introduces two constraints: prevalence constraint -- sets expectation of outcome based on prevalence of the disease and expertise constraint --  fixes some parameters to zero based on domain knowledge . The proposed approach is tested in a synthetic and real breast cancer data.

### Strengths
1. The paper is very well-written, readers can easily follow the motivation, problem formulation and their experimental design.
2. I appreciate the experiments trying to run experiments in real breast cancer dataset. The experiments in a setting where outcomes for non-tested patients are missing is a very difficult setting. 
3. The paper addresses a significant problem where the outcomes of the patients that are tested are missing and there is distributional shift between tested and untested patients. There is a variety of applications -- which are also motivated in the paper.

### Weaknesses
1. I think the paper has limited novelty. The linear risk setting has been considered before as cited in the paper before [(Hicks, 2021)]. This paper aims to add two more constraints: prevalence constraint and expertise constraint. The expertise constraint sets one of the variables to 0 - could be easily addressed by dropping that feature in the dataset, and prevalence constraint sets the expectation of the outcome -- could be addressed by normalizing the feature space and adding a bias term. I am not convinced that these contributions are significant enough to grant acceptance. 
2. I am not sure what theoretical results bring in the paper. For example, Proposition 3.2 shows that variance on the unknown parameters are less if you condition on the fixed parameters. Isn't this expected ? I am not sure how much value this adds to the paper.

### Questions
1. The experimental setting for breast cancer patients are interesting -- you are using patient follow-up to validate the methodology ? What happens if there is no follow-up ? How accurate is it to use follow-up data ?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
