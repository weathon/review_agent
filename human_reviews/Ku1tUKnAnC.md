# Measuring the Reliability of Causal Probing Methods: Tradeoffs, Limitations, and the Plight of Nullifying Interventions

- Decision: Reject
- Scores: 6, 5, 5, 3, 5

## Abstract
Causal probing aims to analyze large language models (or other foundation models) by examining how modifying their representation of various latent properties using interventions derived from probing classifiers impacts their outputs. Recent works have cast doubt on the theoretical basis of several leading causal probing intervention methods, but it has been unclear how to systematically evaluate the effectiveness of probing interventions in practice. To address this, we formally define and quantify two key causal probing desiderata: completeness (how thoroughly the representation of the target property has been transformed) and selectivity (how little other properties have been impacted). We introduce an empirical analysis framework to measure and evaluate completeness and selectivity, allowing us to make the first direct comparisons of the reliability of different families of causal probing methods (e.g., linear vs. nonlinear or counterfactual vs. nullifying interventions). Our experimental analysis shows that: (1) there is an inherent tradeoff between completeness and selectivity, (2) no leading probing method is able to consistently satisfy both criteria at once, and (3) across the board, nullifying interventions are far less complete than counterfactual interventions, which suggests that nullifying methods may not be an effective approach to causal probing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose metrics to measure the completeness and selectivity of causal probing techniques, and conduct an empirical study with diverse causal probing techniques; including recently proposed counterfactual intervention techniques. The motivation behind the empirical study is to formally understand the challenges with causal probing, and establish trends to understand the relative advantages/disadvantages of the different causal probing techniques; counterfactual vs nullifying intervention techniques, and linear vs non-linear counterfactual intervention techniques. The authors report results with BERT on the LGD subject-verb agreement dataset, and find that counterfactual probing methods are better than nullifying probing methods in terms of completeness, and also linear counterfactual techniques fare better than non-linear counterfactual techniques in terms of reliability.

### Strengths
- The proposed evaluation framework is novel to the best of my knowledge, and the analysis of recently proposed counterfactual intervention techniques has not been done in prior works. However, I am not very familiar with the literature in causal probing, hence I am not the best judge for this.

- The experiment results are present clearly and there is a good discussion around them, which makes it easy to follow the main claims. The explanations given in the paper behind observed trends are good and justified.

- The observation of counterfactual interventions better than nullifying interventions for causal probing maybe of significance to the community, and help in development of better causal probing techniques. 

- The experiment setup is comprehensive enough as it covers a variety of causal probing techniques, and considers widely used benchmark for causal probing.

### Weaknesses
- There should be more details provided regarding the performance of (trained) oracle probe. While the authors just report that the oracle obtains high accuracy for predicting the causal variables, more details are needed regarding the distribution learnt by the oracle probe. For example, the completeness metric for nullifying interventions (e.q. 2)  is designed under the assumption that the oracle probe has a uniform  distribution after the null intervention. Is this true for the oracle probe trained by the authors in their empirical study? If this is not the case and the distribution learnt by oracle probe is not uniform, that might bias the completeness metric for nullifying interventions, leading to the question whether we can trust the observed trend that nullifying interventions are worse than counterfactual interventions for completeness. Hence, the authors should verify the oracle distribution in the case of nullifying interventions and check how similar it is to the uniform distribution. 
 
- Table 1 should report standard error/deviation as well over the different examples in the test set. Same comment for figures, but if the error bars are not too significant then they can be dropped, but they must be explicitly mentioned in the appendix. Also, it is not clear how many random seed were used for conducting the empirical study. Are the findings based on a single random seed? If that is the case then I advise the authors to conduct experiments with multiple random seeds to have robust trends.

- Overall, I have some concerns with the writing of the paper in terms of details regarding notations and metrics. For example, for the metrics (e.q. 1, 2, 3), why do the authors remove the dependence on $z_{i'}$ from $P^\star$ and $\hat P$, as they are a function of the value set after intervention. Similarly, the derivation behind the metric are not clearly stated in some cases. For example, equation 3, what is the min and max of the distributions $P^\star$ and $\hat P$? Over what argument are we taking the min/max over? More justification should be provided for the derivation of the selectivity metric. I have expanded more on this point about notations etc. in the questions section ahead.

### Questions
- Regarding the discussion on linear vs non-linear counterfactual probing methods, can the authors conduct a small experiment where the intervention happened on not the last layer? This will help to understand whether the improvement in completeness with linear counterfactual probing is due to intervention happening on the last layer, or there are some other reasons behind it. Also, can you provide an example where non-linear counterfactual probing is better than linear counterfactual probing?

- The training objective of the oracle probe is not clear to me, please provide more details on it.

- The notation of $\hat h^{l}$ and $h^{l^{\star}}$ should be more clearly defined in the text. Also, why are authors using $h^{l^{\star}}$ and not $h^{\star^l}$, was there some specific reason? Currently its a bit confusing as the notation for layers is not consistent, $l$ versus $l^{\star}$ in $\hat P$ and $P^{\star}$.


Minor comments

- It might be good to have a figure denoting the overall design of the evaluation framework.

- The contributions in the introduction can be stated more clearly; either in terms of bullet points and some of the content about the limitation of causal probing can be brought in the introduction.

- There is a typo on linear 172; it should be $c(\hat h^{l, k})$ instead of $c(\hat h^{l, i} )$.

- Line 64, it would be good to provide a clear reference to prior works that criticize the nullifying interventions; instead of referring to Section 2.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
How do you evaluate a causal probe when you can't even be sure it has representation for the right concept (instead of some correlated concept)? This paper seeks to answer this popular question by training "oracle probes" for both the target and off-target concepts, thereby increasing confidence that an intervention to the representation space is *actually* the one that was intended.

With these oracle probes in hand, the authors then propose evaluating causal probing methods according to two desiderata which they formally define: completeness and selectivity (corresponding to whether the representation captures *all* of the concept, and whether it *does affect anything else*, intuitively). 

These are then used to experimentally evaluate different types of causal probing methods, thereby retrodicting the observation that nullifying interventions are less complete than counterfactual interventions.

### Strengths
The proposed method of training "oracle probes" is simple yet versatile: it seems applicable to any attribute with a supervised dataset.
To my knowledge, such an approach is novel, as probes are often used as a source of explanation, rather than as a means to assess the effect of an intervention. The contribution of "oracle probes" thus seems analogous to the development of synthetic controls for causal inference: use a machine learning model to close the causal inference loop, under some suitable assumptions (which here are missing, see weaknesses).

This paper emphasizes that completeness and selectivity empirically induce a trade-off, and so aptly propose the harmonic mean as a way to aggregate these appropriately.

### Weaknesses
1. This approach requires enumerating all off-target concepts in order to construct the oracle probe for Z_e. However, this is a common assumption in the causal-inference-in-text literature, so not surprising.
2. A crux of this paper is the development of *good* oracle probes. However, there is very little analysis of how to evaluate the oracle probes (which are then used to evaluate causal probing methods).
3. Currently, this paper only considers binary concepts. This is a common assumption, so not very limiting.

### Questions
1. Could you elaborate on the implicit assumptions on the oracle probes? For instance, in the experiments section, you mention the assumption that they are trained in a such a way that they have no spurious correlation in their training data. Are there other theoretical assumptions on the oracle probes?

UPDATE Dec 8, 2024: I would like to thank the authors for their thorough responses to my review as well as the other reviewers, and apologies for missing the discussion period for this paper. At the AC's request I am now updating my review, having read all other reviews and rebuttals.

The authors brought up a good point in C1, that new oracles can be added independently of other probes. While this doesn't alleviate the need to enumerate the off-target concepts, it does suggest that the test for specificity might be iteratively approximated by adding more and more oracles.

However, fundamentally this approach relies on the validity of the oracle probes, which should be either theoretical or empirical (ideally both). 
- Reviewer hJpz and I both highlighted that the paper lacks any theoretical validation for the oracles; the authors acknowledged in the rebuttals that indeed no theoretical assumptions are listed. 
- As for empirical evidence of the validity of the oracle probes, the authors in their rebuttal to my review say "we simply perform standard hyperparameter grid search over possible oracle probes and select the MLP architecture and learning rate that yields the highest validation-set accuracy (see Section 4.3 and Appendix C.2)". The key issue here is that validation-set accuracy cannot falsify whether the oracle probes have errornously picked up on a spuriously-correlated concept, because the validation set is sampled from the same distribution as the training. An empirical validation for the oracles probes should demonstrate that oracle O_i generalizes out of distribution, changing Z_i and no other concepts Z_j. Validation set accuracy does not test any of these.

Due the the heavy reliance on the validity of the oracle probes, the lack of either theoretical or empirical validation, I am slightly lowering my score. I believe this approach is promising but does not yet provide clear conclusions without further oracle validation.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper introduces an empirical framework to evaluate causal probing methodologies in language models, particularly focusing on completeness and selectivity. The study highlights a trade-off between these two criteria, showing that counterfactual interventions are generally more effective in altering targeted representations without impacting unrelated properties, unlike nullifying interventions.

### Strengths
1. **Novel Evaluation Framework**: I appreciated the proposed structured framework for measuring the reliability of causal probing interventions. Such a standardized idea can be used to evaluate and compare various methods.

2. **Detailed Experimental Analysis**: The experiments are conducted with detailed analysis across multiple intervention methods and settings, offering a comprehensive evaluation of the methods' performance under varied conditions.

### Weaknesses
1. **Limited Model Scope**: The major concern is insufficient experimental verification. As the authors also claim in the paper, this evaluation is conducted only on a simple model and task (line 672), which might limit the generalizability of the findings to more complex language models and tasks. Since some ideas seem challenging to achieve in practice (line 145), it would be beneficial to see verification or discussion on how the proposed framework could adapt to more general and complex scenarios.

2. **Redundant Descriptions**: Some sections, like Sections 4 and 5, provide extensive details that could be condensed, with some specifics better suited for an appendix. Simplifying these sections (maybe at the same time introducing more settings and results) could enhance readability without compromising clarity.

### Questions
Can the authors provide any guidance or verification on applying the proposed idea to a more general setting? Although the framework appears novel, its current verification relies on several strict assumptions (e.g., full determinism, binary variables, etc.). I’m curious about how this framework could be implemented in a more universally applicable manner.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a method to evaluate interventions in a model’s representation to enforce specific properties. The authors propose to evaluate these interventions by training a model evaluator (approximated oracle probe) that can assess whether the targeted property has been set or removed, as well as the degree to which the intervention affects other properties. The evaluation is conducted on a single-task case study where the target property has two possible values.

### Strengths
The paper is well-written and organized, making the proposed method easy to follow.

### Weaknesses
The primary concern with this work lies in the overlap between the evaluator (approximated oracle probe) and the intervention generator (underlying structural probe used, for example, in the counterfactual approach). Both aim to predict the latent variable \( Z \) from the representation, meaning the evaluator and the intervention generator essentially encode the same information. This overlap likely explains why the counterfactual intervention seems to perform best, as assuming they both perform well (also, the authors use the same architecture for these models), it is almost like directly leveraging the evaluator in generating the intervention and assessing using the evaluator afterwards.

The simplicity of the experimental setup also limits the work's scope. It would be more compelling if the authors tested their method on cases with more than binary causal variables. Additionally, introducing interventions that could potentially not impact task accuracy would provide a clearer understanding of whether performance loss is due to the intervention’s effectiveness or simply the disruption caused by perturbing the representation. 

Another suggestion would be to evaluate a scenario where \( Z_c \) and \( Z_e \) are independent, \( Z_c \) only impact label. This setup could test if intervening on \( Z_e \) affects task accuracy while maintaining good selectivity or completeness.

### Questions
- The rationale behind equating nullified representations to a uniform distribution isn’t entirely clear (I have already read your detailed description in Appendix B). Why not add a distinct category representing "neither" of the options, rather than assuming a uniform distribution?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper studies about existing causal probing methods through the lens of completeness and selectivity. Here, completeness measures how effectively the model’s representations capture the target property and selectivity measures the undesirable impact of the intervention on irrelevant property. The implication is that there exist certain tradeoff between completeness and selectivity and the authors further suggest that causal probing approaches with counterfactual interventions are more effective compared to approaches with nullifying interventions.

### Strengths
- The paper studies important and timely topic, i.e., the thorough examination of existing causal probing methods, in a principled way using two measures (completeness and selectivity). These metrics seem to be intuitive and reasonable.
- The paper is well-written and well-structured. The paper is easy to follow and the authors provide detailed background which makes the reader outside of the field easy to understand the problem and their analysis.

### Weaknesses
- The paper does not provide actionable solution for improving existing approaches. The implication of the study in this paper suggests that the causal probing approaches with counterfactual interventions might be more effective compared to approaches based on nullifying interventions, but I believe the paper would further strengthened if the authors could discuss the possible directions for (potentially) improving existing probing methods.
- The framework heavily relies on the approximated oracle probe, i.e., the most expressive probe, but it is unclear whether such assumption holds or not. Specifically, the authors acknowledge that there should be no spurious correlation between $Z_c$ and $Z_e$ for training the oracle probe, but it is unclear how to measure such correlation and whether this approach is always feasible in more realistic scenarios. Finally, some theoretical guarantee on the approximated oracle would further strengthen the paper (e.g., clearly stating the assumptions, discussion on the assumption violations, estimator error, etc)

### Questions
- See above

### Soundness
2

### Presentation
4

### Contribution
3
