# Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF

- Decision: Accept (poster)
- Scores: 6, 8, 6, 5

## Abstract
In practice, preference learning from human feedback depends on incomplete data with hidden context. Hidden context refers to data that affects the feedback received, but which is not represented in the data used to train a preference model. This captures common issues of data collection, such as having human annotators with varied preferences, cognitive processes that result in seemingly irrational behavior, and combining data labeled according to different criteria. We prove that standard applications of preference learning, including reinforcement learning from human feedback (RLHF), implicitly aggregate over hidden contexts according to a well-known voting rule called *Borda count*. We show this can produce counter-intuitive results that are very different from other methods which implicitly aggregate via expected utility. Furthermore, our analysis formalizes the way that preference learning from users with diverse values tacitly implements a social choice function. A key implication of this result is that annotators have an incentive to misreport their preferences in order to influence the learned model, leading to vulnerabilities in the deployment of RLHF. As a step towards mitigating these problems, we introduce a class of methods called *distributional preference learning* (DPL). DPL methods estimate a distribution of possible score values for each alternative in order to better account for hidden context. Experimental results indicate that applying DPL to RLHF for LLM chatbots identifies hidden context in the data and significantly reduces subsequent jailbreak vulnerability.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper describes the ubiquitous problem of 'hidden context' in the setting of preference learning (among a finite number of alternatives) and  in particular for RLHF (reinforcement learning through human feedback)  when RLHF is applied to training large language models to behave well, for instance, to output maximally safe but useful information. The authors provide a number of theoretical results, for instance, when the problem reduces to learning the Borda counts as opposed to maximizing overall expected utility. This may not be desirable, and the authors present their improvements, in particular distributional preference learning (DPL), and present a few experimental special cases to showcase the utility of approaching the problem from this overall perspective as well as the potential of DPL.

### Strengths
The authors develop a theory of hidden context as it applies to preference learning, and make connections to a number of relevant domains, such as social choice theory.  The paper presents the concepts and their potential utility fairly well. It is important to be aware of the pitfalls in current preference learning approaches, and the authors make some progress and present methods that could address the issues.

### Weaknesses
The empirical results are the weaker aspect of the paper: all are in a hypothetical setting. The strength of the paper lies mostly in the conceptual and theoretical perspective.   Some issues with paper clarity.

### Questions
General comment:

- There is always much hidden context! (many factors that influence decisions, and are not observed or modeled explicitly, in terms of feature vectors in ml, etc). As I read the paper,  gradually I came to the realization that the authors mean aspects that could  substantially impact what is learned in unwanted ways (for instance, there could be multiple clusters of people, and it would be good to test for that.. the DPL approach may begin to address that issue).

Main two questions:

- isn't the assumption of a single utility function too simplistic to be useful in practice? maybe this relates to 'hidden context'.. but
 to me, it is more direct to identify and tackle the problem that different people could have different utilities and so there may not be a single preferred global ranking of all the alternative items ... it would be good to design solutions that aim to identify the rough number of utility functions in addition to what the functions are.. (the problem is somewhat akin to detecting the number of clusters, and whether one or a relatively small few that dominate, ie account for most of the data, etc.. )

- It seems that the best utility function (equation 1, page 3, section 2), mapping alternative to R, is not unique (imagine the ideal case when p_u(a, b) is 1  when u(a)> u(b), and we have two alternatives..)..   The regularization in equation 1 would prefer smaller and smaller values... Any comments on these considerations (perhaps the assumption is that the utility values have some minimal absolute magnitude?)


----------

Other observations or questions:


-- Example 1 is useful. And pretty clear through section 1. However, I am a bit skeptical that annotators can cooperate and/or not show their true beliefs, etc..


- 1st paragraph of section 2, page 3: why probabilities if the utility function is one/unique? Perhaps the subjects are "noisy".. ? A few lines later, you do talk about noisy subjects.. but would have been better to describe the  need for probability first or earlier, before presenting the probabilistic model.

- replace 'are' with 'get': "as the utilities for a and b are closer"..

- hard to parse 1, in particular, \hat{u} is both the final selected/assigned, on the left, and the variable on the right!!  

- It seems that the best utility function, mapping alternative to R, is not unique (imagine the ideal case when p_u(a, b) is 1 
when u(a)> u(b), and we have two alternatives..)..   The regularization in equation 1 would prefer smaller and smaller values... Any comments on these considerations (perhaps the assumption is the utility values have some minimal absolute magnitude?)


- isn't the assumption of a single utility function too simplistic to be useful in practice? may be this relates to 'hidden context'.. but
 to me, it is more direct to identify and tackle the problem that different people could have different utilities and so there may not be a single preferred global ranking of all the alternative items ... it would be good to design solutions that aim to identify the rough number of utility functions in addition to what the functions are..



- The example is useful, but just underscores that different groups or populations (based on demographics, race, socioeconomic background, gender, ..) have different preferences and one needs to obtain RLHF training data on a sufficiently large population reflecting what a model will be tested (deployed) on... one may also need to learn different sub-clusters (that is 'personalization' but at a group level ... )

- replace with 'aggregates' in ".. a single ordering over  alternatives that implicitly aggregating feedback over the ... "
 (page 3)

- Section 5: the  case of 'competing objectives in RLHF'  appears artificial (training on mixed data or mixed objectives). It is ok to use it to demonstrate a point, but if it is actually done by organizations in practice, it appears it is more a reflection of  poor/naive experiment design, than a real problem.

- Section 5, page 9: how was 0.01 quantile picked to lower the jail-breaks  in  ".. optimization according to the 0.01-quantile of.. " (trial and error, that is pick the quantile that yields lowest jail-break reate? or somehow it is a natural or plausible choice?)

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors raise awareness to the problem of hidden context and why preference learning can be misled into optimizing the wrong objective. The authors propose distribution preference learning as an alternative method that can explicitly account for hidden context. The key difference seems to be that DPL produces a distribution of scores for each alternative.

### Strengths
originality: Based on reading the related work, this work seems original. I am not able to tell if they missed any related work.
quality: 
* The authors gave a very good example of why this work is important.
* They then provided a way to overcome some of the limitations of just traditional preference learning using DPL
* Experiments using DPL show promising reduction in jailbreaks
clarity: clear on parts that I'm familiar with (I did not follow section#3, not that it is not clear, just that I don't have sufficient knowledge to judge)
significance: Seems reasonably significant. More than the proposal, perhaps just drawing attention to this problem and ways to measure is great first step.

### Weaknesses
The authors wrote "We show that our model can represent many challenges in preference learning, such as combining data from different
users, accounting for irrationality, and optimizing for multiple objectives." It is a bit of a stretch to say that this work accounts for all forms of irrationality.. it would be great if the authors can also discuss when this approach fails.

### Questions
please see above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the challenge of "hidden context" in preference learning, where relevant variables affect human feedback but are unseen by the learning algorithm. This arises frequently due to irrationality, population diversity, competing objectives, and other factors. Theoretical analysis demonstrates that standard preference learning methods implicitly aggregate preferences via Borda count. This contrasts with expected utility maximization and can produce counterintuitive results. To address these issues, the paper proposes "distributional preference learning" (DPL) methods that learn a distribution over utilities rather than a point estimate. Experiments on a dataset with two competing objectives show that DPL can automatically detect disagreement and mitigate harmful incentives. By using DPL to optimize lower quantiles of the learned distribution, the rate of problematic "jailbroken" responses is reduced.

### Strengths
* This paper makes an original contribution by formally characterizing and providing a theoretical analysis of the problem of "hidden context" in preference learning. Though the challenges of population diversity, competing objectives, and human irrationality have been recognized, the key insight is to model all these issues through the unified lens of missing variables that affect human feedback. 
* The technical quality is high with rigorous proofs relating preference learning to Borda count, conditions for equivalence with expected utility, and results connecting to social choice theory. The writing is also clear in explaining the hidden context setting, comparing to regression, and motivating real-world implications.
* The proposed methods for distributional preference learning seem promising for detecting the effects of hidden context. The experiments provide a nice proof-of-concept.

### Weaknesses
* Both synthetic and real experiments are done in low-dimensional and small data settings. Testing DPL when there are complex interactions between many features in a larger-scale regime would be more convincing. 
* The fact that DPL can detect competing objectives on this dataset is promising. It remains to be seen if DPL reliably detects other sources of hidden context like population diversity or irrationality. Testing on more diverse datasets could address this.
* There is no comparison to other methods for handling disagreement like modeling annotator identity or clustering. Comparisons could reveal the benefits and limitations of DPL's approach.

### Questions
* The paper focuses on theoretical results for the binary comparison setting, but most real preference datasets have ratings on a larger scale (e.g. 1-5 stars). How do the results extend to this ordinal rating setting?
* There are other approaches related to handling subjectivity and disagreement in ratings, such as models of annotator bias. How does your proposed DPL method compare to these?
* In the introduction, several different causes of hidden context are described (population diversity, irrationality, etc). Do you expect the implications of the theoretical results to be different depending on the source of the hidden context?
* The social choice theory connections provide an interesting viewpoint. Does your analysis suggest any new insights about voting rules or mechanisms given unknown/diverse preferences?

### Soundness
3 good

### Presentation
3 good

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
Reinforcement learning from human feedback, which uses human preferences as labels, is a key step in training large language models. This paper demonstrates that there may be hidden context when people label the data, which may hurt the performance of large language models, e.g., generating helpful but harmful responses. The authors proved that existing preference learning methods may not learn the true aggregated order of items (responses in the context of large language models), and proposed distributional preference learning to mitigate the problem.

### Strengths
This paper tackles an interesting and important problem in large language models. The paper is mostly well organized and easy to follow.

### Weaknesses
Theoretical results: I did not check every detail of the proofs, but I did see some issues. It's very hard to evaluate the correctness of these theorems with these issues. 
1. Proof of Theorem 3.2 on page 15. u' seems not defined, which is very confusing. The u'(a)-u'(b) equation has unmatched parentheses. 
2. Proof of Proposition A.3, the first equality. "x" variable is not defined or explicitly integrated. 

Proposed algorithm: this paper does not provide an explicit objective function to optimize for the proposed DPL method. I would not accept the paper without the objective function.

Experiments: the results are interesting but not sufficient. If I understand correctly, the authors only trained the utility function. It would be best if the authors could train a language model using the utility functions and provide comparisons of the real language models. I understand that finetuning a model with billions of parameters is very challenging, but finetuning a GPT-2 size model should be possible.

### Questions
See my comments in the weakness section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
