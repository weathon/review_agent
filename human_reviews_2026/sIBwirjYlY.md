# The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
As AI becomes more capable, we entrust it with more general and consequential tasks. The risks from failure grow more severe with increasing task scope. It is therefore important to understand 
the ways extremely capable AI models will fail: Will they fail by systematically pursuing goals we do not intend? Or will they fail by being a hot mess, and taking nonsensical actions that do not further any goal? We operationalize this question using a bias-variance decomposition of the errors made by AI models: An AI's *error incoherence* on a task is measured over test-time randomness as the fraction of its error that stems from variance rather than bias in task outcome. Across all tasks and frontier models we measure, we find that the longer models spend reasoning and taking actions, *the more incoherent* their failures become. We observe that error incoherence changes with model scale in a way that is task and experiment dependent. However, in several settings larger, more capable models are more incoherent than smaller models.
Consequently, scale alone seems unlikely to eliminate incoherence. Instead, as more capable AIs pursue harder tasks, requiring more sequential action and thought, our results predict failures to be accompanied by more incoherent behavior. 
This suggests a future where AIs sometimes cause industrial accidents (due to unpredictable misbehavior), but are less likely to exhibit consistent pursuit of a misaligned goal. 
This increases the relative importance of alignment research targeting reward hacking or goal misspecification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper adapts a bias-variance decomposition for the expected errors of LLMs. Then, it defines “incoherence” as the variance normalised by the error. They estimate variance and bias using repeated sampling (and changing few-shot prompts) and study how incoherence evolves with respect to difficulty of question (using length of reasoning chain or number or actions as a proxy), model size, and reasoning effort dedicated. The experiments are conducted on a range of state-of-the-art LLMs.

### Strengths
- originality: the idea of decomposing model errors into bias and variance to understand how they evolve seems novel.
- clarity: overall mostly clear, robust notation and formalisation, plenty of references to Appendices. Explanation of the experimental setup is also great.
- quality: highly informative and readable figures, great and thorough experimental setup.
- significance: the findings relate to various state-of-the-art models and relate the introduced metric to various independent variables.

### Weaknesses
- Main issue: throughout the presentation, the authors seem to suggest that models become more incoherent as they become larger, or as they tackle longer tasks. However, the introduced metric for incoherence, variance/(variacne + bias), does not allow to disentangle whether the change in incoherence is due to a change in variance, or one in bias, or the relationship between the two. Indeed, if the bias decreases faster than variance, the outcome is that this incoherence metric increases even if both bias and variance decline. Most of the experiments do not report the variance separately from the bias (except the one in Fig 3, which does show a decreasing variance), and this makes it hard to understand what is actually driving the increase in incoherence metric. I believe this decreases the significance of the findings, by making them less clear. Moreover, this may lead people to misinterpret them, so I believe addressing this is necessary for publication. However, this should be a small change, as I believe the experimental setup is mostly robust and thorough, so what I am suggesting is a small change in how the results are presented and discussed.

minor issues: 

- title: there is no really talking of “misalignment” in the paper that much, as most of the focus is actually on the variance. Also, “model intelligence” is used in the title and line 52, but actually the experiments consider model size. While this is a somehow good proxy for intelligence, there is not a 1-1 correspondence
- Sec 3.5 asks the question “Does the model become an optimizer faster or slower than it converges on the right optimization objective?” I am not fully clear as to how this is related to the original questions asked.
- lines 52-53 says that one of the addressed questions is “Asymptotically, as extremely capable models perform extremely complex tasks, which class of undesired behavior will dominate?” I don’t think this question is addressed; while empirical results are reported, I don’t think these are necessary to extrapolate to an asymptotic regime. Moreover, there are no studies of how the incoherence (or variance) evolve when the two independent variables (model size and task complexity) are varied at the same time.
- lines 443-445: “for AIs to reliably acquire a broad range of capabilities, the capacity expansion requires enlarging the hypothesis space faster than any specific capability, which manifests as increased incoherence.” I don’t get this comment, can the authors please elaborate in the text?

### Questions
- Is there any particular reason why the normalised version of the variance is interesting or useful, instead of simply studying the unnormalised variance? As most of the experiments aim to answer how incoherence evolves with model size or question difficulty, it seems that the normalisatoin is not strictly needed and, actually, hides the real signal a bit.
- In line 129, it is explained how the mean term appearing in the bias-variance decomposition has an exponential and a logarithm. Is this taken into account when estimating the variance term? How does this fact affect the interpretation of the results, if at all?
- Sec 3.5: “Models learn the right objective faster than to be an optimizer over a long horizon.” Do the auhors think this indicates that we will require different training procedures from next-token prediction to get models better at long-horizon tasks?
- why does Sec 3.5 use an ill-conditioned matrix?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the bias-variance decomposition of LLM responses. The main empirical findings are:
* Variance increases with reasoning length.
* Bias may scale lower with model size, but variance may grow even larger. 
* Averaging over several roll-outs reduces variance. 
* On synthetic data, the bias of a transformer reduces faster than variance.

### Strengths
* Clarity. The paper is very well written, with a precise formalization of bias/variance and clear figures.
* Significance. It raises an important question about how errors change with task complexity and scale. It's not just average accuracy, but usefully complementing work on model self-consistency and best-of-N sampling. 
* Results span multiple benchmarks (reasoning, coding, alignment probes, synthetic optimizers), strengthening the validity.
* Actionable implications. The bias/variance framing yields concrete safety and engineering takeaways (verification, rollback, ensembling/gating) rather than purely diagnostic metrics.

### Weaknesses
* Positioning with literature. The paper would benefit from a fuller discussion of prior work on test-time exploration, self-consistency, and best-of-N/majority-vote approaches. E.g. 

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., ... & Zhou, D. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations.

Huang, A., Block, A., Foster, D. J., Rohatgi, D., Zhang, C., Simchowitz, M., ... & Krishnamurthy, A. Self-Improvement in Language Models: The Sharpening Mechanism. In The Thirteenth International Conference on Learning Representations.

* Following the previous point, it might be better to use *consistency* instead of *coherence* to align with previous work.

### Questions
My main question is about the connection to existing literature.

### Soundness
4

### Presentation
4

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
The paper introduces the concept of Bias, Variance, and incoherence in the context of AI safety. 
Bias is the tendency for a model to coherently pursue a misaligned goal; variance is closely related to the model's tendency to make incoherent choices. They find that in longer the models spend reading, the more incoherent they become. Interestingly, as model scales, they do not consistently become more coherent.

### Strengths
I like the conceptual framing of safety in terms of bias and variance. The scaling study was very surprising. In particular, the fact that for some questions the model incoherence increases with size. It was also interesting seeing the different scaling exponents for bias and variance. 

There are many empirical studies in the paper, which is a strong plus. 

Overall very thorough paper.

### Weaknesses
I’d move the related work section to the front. 
Some of the legends are really small; I’d either remove them or make them larger. 

The title is catchy, but to my knowledge, the only place it is mentioned is in Figure 1. I’d explain and define this term as used in : “Jascha Sohl-Dickstein. The hot mess theory of AI misalignment: More intelligent agents behave less”

It's not entirely clear how the reasoning bins were chosen in Figure 3. The first graph depicts the binning with the sigmoid but are the thresholds based on anything particular or chosen arbitrarily within some reasonable range. 

I'd be interested in more discussion of mitigations for bias vs variance; the current ensembling discussion is okay (see questions) 

Some of the Figures are overwhelming, ie, 4-6 panels in a figure, each with many points and lines. If it's possible, I'd consider being more selective or taking aggregate lines across models, so there are things to track.

### Questions
Is there a bias, variance tradeoff, or does the analogy break down there?

Can you conceptualize many AI Safety interventions in this bias-variance framework?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study incoherence in LLMs: the fraction of error that can be associated with variance rather than bias. Authors find that longer reasoning increases incoherence, that model scale can increase incoherence, that inference costs are dominated by the effects of natural variation (i.e., overthinking), that ensembling improves incoherence, and other related findings. Evaluations are done on SWE-bench, MMLU, and GPQA.

### Strengths
The research questions surrounding incoherence are interesting and impactful. The questions asked surrounding incoherence are interesting and the findings are valuable for the community. 

Some parts of the paper are extremely well written. Particularly, Figure 2 and 3 were extremely easy to follow and illustrate the various parts of the experiments and results very well. 

The overall efforts and comprehensiveness in the study of incoherence is also great and culminates in a coherent story. Experiments are well done as well.

### Weaknesses
While some parts of the paper are very well written, other parts of the paper are very confusing. The part in the abstract about industrial accidents has no relation to the paper whatsoever and is misleading about the paper contents. The title also doesn't seem to match the paper --- I didn't get the impression of a "hot mess" at any point, and misalignment isn't even the central topic---variation in performance and coherence is. 

This is concerning so I'm debating whether to flag it for an ethics review just to be safe---but before that, can the authors clarify why they chose the previous title, and if/how they would change it?

Small details:
typo in line 322, "questiosn"
The last sentence on page 6 is confusing.

### Questions
1. Authors mention irreducible noise in Section 2.1. I'm wondering how this interacts with some factors such as problem difficulty or reasoning length, and whether it could potentially influence the observed results?

2. In section 3.1 discussion: task complexity, authors mention that sorting questions by reasoning length implicitly selects for task difficulty. Is there a way to disentangle whether this is caused by additional reasoning length or by problem difficulty, e.g., asking models to solve easy problems with more reasoning steps to match length?

3. For the results on qwen with different model sizes, can you somehow test if 32B was the outlier, or if the trend is truly upwards? Maybe with a different model family (e.g., llama)? As presented, it's a bit shaky to say that as model size increases so does incoherence. 

4. As currently written, I don't see how the experiment in 3.5 relates closely enough to the paper. Could the authors clarify the purpose of this experiment and how it fits into the overall paper?

### Soundness
4

### Presentation
2

### Contribution
3
