# Learning to Reject with a Fixed Predictor: Application to Decontextualization

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
We study the problem of classification with a reject option for a fixed predictor, crucial to natural language processing. We introduce a new problem formulation for this scenario, and an algorithm minimizing a new surrogate loss function. We provide a complete theoretical analysis of the surrogate loss function with a strong $H$-consistency guarantee. For evaluation, we choose the \textit{decontextualization} task, and provide a manually-labelled dataset of $2\mathord,000$ examples. Our algorithm significantly outperforms the baselines considered, with a $\sim 25$% improvement in coverage when halving the error rate, which is only $\sim 3$% away from the theoretical limit.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the submission, a novel loss function parameterised by the learnable rejectors is studied to train the LLM model to handle diverse outputs for one prompt. To compute the existing loss function for tackling this problem is NP-hard. At the same time, the authors proposed a trackable surrogate loss, which is differentiable and convex, and by optimising it, the generalisation error is minimised supported by the theorem proposed as the main result in the submission. In the experiments, by training the model with the proposed loss function, the models are improved in terms of precision vs. Coverage compared with the existing method.

### Strengths
1. The derivation is clear and sound. I am not an expert on the proofing behind so I only scan the appendix. However, by following each proposition and theorem, it is easy to follow the logic step by step and reach the point that by minimising the proposed loss function the target generalisation error is minimised. 
2. Nice plot for intuitively explaining the property of the surrogate loss with the changing of the r(x).

### Weaknesses
1. The experiment setting is not well explained and the settings are not comprehensive. As the authors mentioned,  the LLM predictors are their main focus but in the experiments, there is only one type of LLM. As T5X is a family of models, there are other choices and more architectures from other families will be more comprehensive for evaluating the proposed loss. 
2. The loss function is tested on image classification but on a tiny setting and this pure classification task does not really relate to the LLM setting. 
3. To train the rejector still required to label new information, it is hard to distinguish whether the improvement is from the additional information or the loss function.

### Questions
1. According to my understanding, applying the surrogate loss requires labelling the output from the given model. Then the model is further trained by the learned surrogate loss. Thus some extra information is introduced. Can the Cross entropy loss and Maxprob have the same information? 
2. What is the format of the ejector?
3. WHy the F1 score is not applied for clear comparison?
5. For the std comparison is not very clear whether rejection loss is much better than Maxprob in Figure 4 and in Figure 5.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of learning to reject with a fixed predictor, motivated by the case when the fixed predictor is a pretrained large language model (LLM). The goal is to learn a rejector function that allows for post-hoc filtering of LLM outputs: by rejecting lower-quality outputs, the combined predictor/rejector system can have higher precision at the cost of lower coverage.
This is especially critical in high-stakes sequence-to-sequence problems in domains such as health data.

The paper designs an H-consistent surrogate loss for learning to reject with a fixed predictor using the framework of Awasthi et al. (2022). This loss is then used for the decontextualization problem, which is to rephrase a sentence into a form that's understandable without the context around the original sentence.

On the decontextualization experiment, the proposed surrogate outperforms strong baselines from the learning-to-reject and model calibration literature.

### Strengths
- The surrogate loss derived in the paper doesn't require tuning of a threshold parameter, which is a drawback of confidence-based approaches.

- Empirical evaluation that avoids a common pitfalls of other work in this area: because different loss functions are being compared, it's only fair to compare performance with the optimal learning rate, since changing the loss function changes the scale of gradient updates.

- The paper's theoretically-derived relationship between the two hyperparameters $\alpha$ and $\beta$ works well empirically.

### Weaknesses
- The method is only applied to the decontextualization problem when it actually seems to have much more broad applicability. I can think of several LLM applications where the ability to increase precision by rejecting would be useful. For example, we could try to learn a rejector that cuts down on hallucinations in summarization or text simplification. More exploration of this technique beyond the decontextualization problem would make it more impactful.

- > Additionally, to the best of our knowledge, minimizing the cross-entropy loss does not have any proven guarantee with respect to our main objective: minimizing the induced rejection loss.

    - When $\mathcal{R} = \mathcal{R}\_{all}$, can't we use a cost-sensitive surrogate loss plus usual (Bayes) consistency results? 
      I.e., I don't understand the following claim in Appendix C given that the paper's results only consider $\mathcal{R} = \mathcal{R}\_{all}$:
     >  (i) There is a lack of any H-consistency bound guarantees for cost-sensitive surrogate losses with respect to the induced rejection loss. 

        In general, I think the paper could use more presentation on why the naive approach of directly training the rejector to predict the labels $a$ using a standard cost-sensitive surrogate loss doesn't work, since the results only consider the $\mathcal{R} = \mathcal{R}\_{all}$ case. That, or some results for linear or 1-layer-NN rejectors, as in Awasthi et al. (2022), would strengthen the theoretical part of the paper.

- No ablation on $\alpha$ even though it's important in the bound (Thm 5), and different experiments use different values (e.g., 4 in main text experiments, 3.5 in appendix vision experiments)

### Questions
- Why do we care about $\mathcal{H}$-consistency even though we only consider the space $\mathcal{R}_{all}$ of all measurable functions? It would be helpful to further emphasize this earlier in the paper as a major difference / novelty / reason why more naive baselines don't work (see earlier comment under weaknesses).

- small typo in display at bottom of pg4; $\le 0$ should be in the subscript of $\mathbb{I}$

- small typo in display at bottom of pg4; not enough parentheses

- Minor semantics/notation: $r(x,f(x))$ reads like a score for rejecting, so the notation reads like $r>0 \implies$ reject, when it's actually being used in the opposite sense. This tripped me up several times. I know this notation is inherited from Cortes et al. but I think it's a bit confusing.

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
For the purpose of letting a model 'reject' or 'abstain' from classifying some samples, a technique of using a rejection loss is proposed; from which a surrogate loss is derived and used. This is in contrast to either using the 'confidence' output or training for an (n+1)-th class in the sense that the predictor can be fixed. Theoretical guarantees for this surrogate loss are provided, and this framework is evaluated on real-world decontextualization tasks, and also for image classification. The results show promise and seem to perform better than other methods studied.

### Strengths
- Very promising real-world results
- Theoretical guarantees motivated well and provided
- Good theoretical comparison to other methods, and good motivation for the need of proposed method provided
- NLP examples but possibly further extensions to other areas
- Surrogate loss performance better than other models and also quite close to theoretical limits at times.

### Weaknesses
- Although GitHub repo links and other identifying information cannot be written in a paper under review, I did not see any indication of the intention to make the code public, nor is it provided in supplementary materials
- Page 4. First equation/inequality.
	- $c$ is positive. indicator functions are either 0 or 1, so:
$\mathbb{I}\_{a\leq 0} \mathbb{I}\_{r(x)>0} + c \mathbb{I}\_{r(x)\leq 0} \geq max(\mathbb{I}\_{a\leq 0}\mathbb{I}\_{-r(x)<0}, c \mathbb{I}\_{r(x)\leq 0})$

	max of two different terms that are positive should be less than or equal to the sum.
	 - And second comparison should be equal. As the first term in the first max is saying: "Both $a$ and $-r$ should be less than zero for the indicator product to be one". And the the first term in the second max is saying "the max of both $a$ and $-r$ should be less than zero for the indicator to be one". Both of these statements imply each other and therefore the last relation should be of equality. 	 
	 - The bound becomes a lower bound, not an upper bound.
	 - I did not check the last relation. That might still hold despite this, but need to know why.
 - There is no test set. Only train and validation, where cross-validation is used so training algorithm sees all data.

### Questions
- page 5. it is said that "underlying scores are not favourable for that precision level". Why is that?
- These are possibly standard deviation bars in figures 4 and 5. How were they generated? Is it from different folds of cross-validation?
- Are Maxprob and cross-entropy trained on different models? Why is that?

**Minor Typing / Formatting / Clarity issues**
- page 4. last equation "<=0" should be in the subscript
- please recheck format for citations: some citations use et al while other list all authors.

**Comment**: I chose "good" in soundness, presentation and contribution, but "3: reject" in the overall rating. That's mainly because of the mathematical inconsistency, which I hope can be resolved.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
