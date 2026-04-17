# Eliciting and evaluating generalizable explanations from large reasoning models

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Large reasoning models (LRMs) produce a textual chain of thought (CoT) in the process of solving a problem. This CoT is potentially a powerful tool to understand the problem, surfacing a human-readable, natural-language explanation. However, it is unclear whether these explanations generalize, i.e. whether they capture general patterns about the underlying problem rather than patterns which are esoteric to the LRM. This is a crucial question in understanding or discovering new concepts, e.g. in AI for science. We study this generalization question by evaluating a specific notion of generalizability: whether explanations produced by one LRM induce the same behavior when given to other LRMs. 

We find that CoT explanations do show generalization (i.e. they increase consistency between LRMs) and that this increased generalization is correlated with human preference rankings. We further analyze the conditions under which explanations do or do not yield consistent answers and propose a straightforward, sentence-level ensembling strategy that improves consistency. These results prescribe caution when using LRM explanations to yield new insights and outline a framework for characterizing LRM explanation generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work is motivated by understanding if CoT explanations capture 'general patterns' about the underlying problem rather than patterns which are esoteric to each specific LRM. Experiments are performed substituting a model's native CoT with one of 3 CoT variants (no CoT, CoT from another model, or the "best" CoT picked from an ensemble of models) to evaluate whether the same CoT trace will produce the same final answer in different LRMs. A human user study is also conducted to determine that users prefer models whose CoT explanations are universal/consistent.

### Strengths
The paper is clearly written, the setup is easily understandable from the description, and Fig 1 is helpful.

### Weaknesses
My main issue with this work is the novelty/significance of the overall goal of this work and the actionable insights it generates. Even if users demonstrate a preference for consistency in CoT, I'm not sure why CoT traces have to be universal across models to be a sound explanation/reasoning trace (after all, different people can explain the same concept in very different ways). Furthermore, since the authors note that consistency is in fact quite high, the result is not surprising nor does it generate interesting directions for future research. The result of the user study (users prefer models with consistent CoT) also is not surprising. In order for me to raise my score, I would need to see an actionable, surprising result (e.g., a systematic failure of LRMs with certain types of CoT).

Small note: The PDF document attached in OpenReview has a different title than the OpenReview submission.

### Questions
How do the authors define 'generalizable'? In this context I would probably say the work is testing whether CoT traces are 'universal' or 'consistent'; 'generalizability' is a property of a model.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies whether explanations from large reasoning models generalize: whether they induce the same behavior when provided to other LRMs. They use accuracy and consistency to evaluate. When evaluating, they use an LLM to remove the answer from the reasoning. They also evaluate empty CoT (no "thinking token"), default CoT (the thinking that is done by the same model), transfer CoT (patching one model's CoT to another model), and ensembled thoughts (have generator models generate multiple reasonings, and an evaluator model to choose different steps from differemt model generations to put together into one).  They find that CoT explanations do generalize and make outputs more consistent, even for ones that make wrong answers, and the ones that generalize (produce more consistent answers) are rated higher by users too.

### Strengths
1. The paper is well motivated, studying one aspect of a good explanation: one that should guide the user to the same conclusion. This angle is important and novel.
2. The paper shows that using the same explanation does induce more consistent behavior than just having the model generate on its own, which is evidence of how the explanations generalize. This is a new way of testing explanations.

### Weaknesses
1. The human study size is very small. Also since the participants see multiple examples, it should be paired tests instead of independent t-tests. And there is a debate between if we should treat likert scales as parametric or non-parametric. It might be good to add Wilcoxon signed-rank test as well as paired t-test.
2. Figure 5 shows that if we use the same CoT on the same model that created the CoT, then it produces more consistent answers than if we use a different model. But the paper did not show how if we just sample from the model (which might give different CoT) whether the result will be different. It is thus hard to interpret what this "CoT creator involved" results should be interpreted.
3. Some typos:

line 047 a problems -> a problem

the last line of page 1: MedCalc Bench. (add period)

line 093: duplicated "they"

### Questions
1. (W1) Could you add Wilconxon signed-rank test and paired t-test to the human study?
2. (W2) Could you add results of how a model agrees with itself just by sampline different outputs from the model and test self-consistency? This way we can show that if using CoT from the same model actuall improves from just sampling from the same model.

### Soundness
3

### Presentation
2

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
This paper proposes a method for evaluating if and how well chain-of-thought (CoT) reasoning traces can serve as general explanations. The basic idea is that, if a CoT reasoning trace is not just sequence of model-specific "thinking token"-like outputs that allow the model to spend more computation on the problem, but a series of generally applicable reasoning steps, then providing such CoT reasoning traces should be useful for (a) other models and (b) human users.
The paper operationalizes this idea in two main ways:
1. Providing CoT reasoning trace z produced by model A as input to models B_1, ..., B_n and then evaluating if this improves task accuracy with respect to ground truth, as well as consistency, i.e., does reasoning trace z elicit the same answer in models B_1, ..., B_n?
2. Collecting human ratings of LM-generated CoT reasoning traces. The paper finds a moderate positive correlation between human ratings and the accuracy and consistency scores measured in step 1.

In addition to these two main contributions, the paper also devises a method for aggregating multiple reasoning traces from different models and shows that such an ensemble yields the best explanations under the considered evaluation metrics.

### Strengths
The idea of evaluating the explanatory function of CoT reasoning traces by providing them to other models and humans is interesting and, to my knowledge, novel. The evaluating framework, which includes various combinations of LMs as well as human ratings, is well-designed.

### Weaknesses
While I think the idea and overall framework is a valuable contribution, the scale of the presented experiments does not lead to convincing conclusions.

1. Experiments are only performed on one dataset with 100 samples. I realize that human evaluation is costly, but at least for the LLM-based evaluation, larger-scale experiments both in terms of datasets/tasks and sample sizes should lead to more robust findings.

For example, the observed positive correlations between consistency/accuracy and human user ratings (Figure 3, top two figures) is based on four datapoints. It appears that excluding a single datapoint, namely the QwQ+DAPO/OSS result, would flip the sign of correlation, so I find myself having to question the robustness of this result.

2. There is a mismatch between the model and human evaluation. The model evaluation measures accuracy with respect to ground truth (measuring whether the explanation helps models perform the task better) and consistency (measuring whether explanation elicits possibly wrong, but consistent answers in different models). In contrast, the human annotators are asked to rate aspects like "clarity" and "confidence" and then the paper correlates these ratings with model evaluation metrics. Simply using the same metrics for humans and models would allow for a direct comparison. If having humans actually perform the task is too difficult, I would argue that (related to point 1.) experiment should also included a task that humans can reasonably do.

3. The analysis of model results limited. When transferring a CoT reasoning trace from one model to another, there are several possible cases which, I believe, should be analyzed to better understand when and how CoT transfer works. For example:
- Model A predicts the wrong answer without CoT but correct answer with CoT z_A. Model B predicts wrong answer without CoT and with its own CoT z_B. When transferring CoT z_A from A to B, model B now predicts the correct answer with CoT z_A. This would be a case in which CoT z_A is effective and in this sense z_A is a good CoT that "generalizes".
- Model A predicts the wrong answer with CoT z_A, Model B predicts the correct answer without Cot and with its own Cot z_B. There is now a conflict between model B's tendency to predict the correct answer and transferred CoT that goes against that tendency. It's less clear what outcome we want here, but if z_A flips model B's prediction to a wrong answer, at least we could say that z_A is "convincing".
There are several more of these possible combinations, and I would have expected some analysis along these lines to better understand the proposed method. However, related ot 1., the small sample size likely does not allow such an analysis.

### Questions
- lines 117-220 are confusing: what is the difference between an "eval" model and a "test" model? Why is the explanation z given to a "to a different LRM leval and a set of other test models L = {li, lj ...lq }" but then, if I understand "We then produce an answer given an LRM_leval" correctly, the "other test models" are not used? Looking at Eq (1), it appears that the test models are used for the consistency metric G, but this is not explictly explained.

- Eq (1): does the index j in the equation for G refer to the j in the definition of L on line 117 or this a typo for the summation index i? (Guessing the latter, since i doesn't appear anywhere else...)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates whether CoT explanations generated by LRMs are generalizable. The author define generalizability as cross-model consistency whether an explanation from a source LRM induces different receiver LRMs to produce more consistent answers. Using this evaluation framework on MedCalc Benchmark, the authors find that CoT-based hints, which is the explanation with answers removed, significantly increase the pairwise consistency of receiver models' answers compared to no-explanation baseline.

### Strengths
1. The paper is well-written and easy to follow.
2. The paper tackles the critical question of whether CoT explanations capture general which is a significant question for the community.
3. The experimental validation is thorough employing diverse set of LRMs

### Weaknesses
1. The paper title and pdf title is different.
2. The main text of the figure and table needs to provide guidance on how to interpret the results in the table (e.g., what the takeaway is) (for figure 2,5, table 3). Also the readability of the figures are poor (for figure 2,3,4).
3. The experiment is only held in one medical benchmark dataset which lacks generalizability. The author should mention 'medical domain-specific' in the title or do additional experiment across different domains.
4. The framework measures consistency of the final answers but not their correctness. There should be an analysis of correctness since it might consistent on incorrect answer.
5. The paper positions itself in the LRM reasoning space but largely bypasses the decades of xai research on explanation evaluation.
6. The paper equates high consistency with generalization. This needs formal justification or empirical validation.

### Questions
Look at the weaknesses

### Soundness
2

### Presentation
2

### Contribution
3
