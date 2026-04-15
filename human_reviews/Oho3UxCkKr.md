# SCREWS: A Modular Framework for Reasoning with Revisions

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 3

## Abstract
Large language models (LLMs) can improve their accuracy on various tasks through iteratively refining and revising their output based on feedback. We observe that these *revisions* can introduce errors, in which case it is better to roll back to a previous result. Further, revisions are typically homogeneous: they use the same reasoning method that produced the initial answer, which may not correct errors.
To enable exploration in this space, we present SCREWS, a modular framework for reasoning with revisions.
It is comprised of three main modules: *Sampling*, *Conditional Resampling*, and *Selection*, each consisting of sub-modules that can be hand-selected per task. We show that SCREWS not only unifies several previous approaches under a common framework, but also reveals several novel strategies for identifying improved reasoning chains. We evaluate our framework with state-of-the-art LLMs (ChatGPT and GPT-4) on a diverse set of reasoning tasks and uncover useful new reasoning strategies for each: arithmetic word problems, multi-hop question answering, and code analysis. 
Heterogeneous revision strategies prove to be important, as does selection between original and revised candidates.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents SCREWS, a methodology for reasoning with revisions. The pipeline includes three stages: sampling, conditional resampling, and selection. The experiments demonstrated that using different strategies for sampling and conditional resampling can boost the reasoning performance of the GPT-3.5 language model.

### Strengths
The proposed framework is general and modular, meaning various techniques can be employed in different stages of it.

### Weaknesses
1. > A student preparing for an exam may use deductive reasoning to solve problems and inductive reasoning to verify the results

    This is surely the wrong way around?

2. Figure 2 is too visually complicated to be helpful. It's better to present a simplified and more abstract pipeline than listing every component.

3. This is the main thing I am unsure about: In tables 1 and 2, the results are supposed to demonstrate the usefulness of the resampling strategy. However, in table 1, only 4 out of 9 pairings are statistically significant. Also, when using Subq (Or) as the sampling strategy, it does not seem to matter much (or have statistical significance) as to which conditional resampling strategy is used. Does this maybe suggest that there is a saturation, or utility limitation of SCREWS, if the sampling strategy is good enough?

    In table 2, although conditional sampling is cheaper, independent sampling does have significantly higher performances (and upper bounds given oracle selectors). The figure 4 provides further breakdown of the accuracy - cost relation and no strategy really beats CoT in absolute performance. This leaves the question of the overall usefulness of SCREWS uncertain.

### Questions
Could you provide ablation studies to show clearly that given the same computational cost, SCREWS can perform better than naive CoT? If so, I can be convinced of its effectiveness.

### Soundness
2 fair

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
This paper studies refinement and revision in reasoning. They propose a modular framework for improving reasoning with revisions. The proposed framework unifies several previous approaches under a common framework but also reveals several novel strategies for identifying improved reasoning chains. It consists of three main modules, Sampling, Conditional Resampling, and Selection, each consisting of sub-modules that can be hand-selected per task. The framework is then implemented with GPT-3.5-turbo and GPT-4 and is evaluated on multiple benchmarks for arithmetic reasoning, multi-hop question answering, and code analysis. The proposed strategies achieve substantial improvements over vanilla strategies. The heterogeneous sampling strategy is demonstrated useful in the experiments. They also discuss the importance of a model-based selection strategy.

### Strengths
This paper studies the problem of revisions in reasoning, including reducing errors introduced by revision and alleviating homogenous revisions, which are important research questions for current large language model reasoning. The authors propose a unified framework to address the questions. Many previous works can be viewed as an instance of the proposed framework. As a result, the framework is convenient for ablating the strategies during the pipeline. The experiments and analyses are comprehensive. The proposed strategies are effective. And the experimental findings are inspiring.

### Weaknesses
Please see the questions listed below.

### Questions
Q1: How do you choose the specific sub-modules (e.g., self-ask/tool use for conditional resampling, LLM-based selection/Rule-based selection for selection) for each of the three modules in the framework?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a framework, called SCREWS, with modular components for reasoning tasks where revisions and selection are needed. The framework contains three modules, Sampling, Conditional Resampling, and Selection. Each of the modules can then be implemented with several alternatives. The authors conducted experiments on GSM8K, StrategyQA, and Auto Debugging, using gpt-3.5-turbo, aiming to see how different combinations of modules can affect the task performance. The important observations include: 1) conditional resampling helps when it is based on a different method than the sampling, 2) a good selection is promising for improving the task performance, but the current selection method still falls short in it, and 3) enabling tools is critical for StrategyQA where additional facts are beneficial.

### Strengths
1. The paper has touched upon a popular topic of LLM reasoning, especially when iterative revisions are needed. The proposed framework summarized the typical implementation of different modules.
2. The paper conducted experiments with different combinations of module instantiations and investigated their effectiveness. The experimental results have led to several interesting takeaway messages.
3. The paper is easy to follow.

### Weaknesses
The contribution of this paper seems to be incremental, as it is mainly an empirical exploration of existing module implementations. While the experimental results led to interesting observations, these observations are mostly expected, whereas the more critical questions, such as how to improve the existing selection method, are not well addressed.

### Questions
I found the tool use experiment of StrategyQA a bit confusing.
1. I wonder if the conditional resampling can still be helpful if the LLM is configured to access the retrieved fact in its initial sampling? 
2. The setup seems to directly provide relevant facts to the conditional resampler, and the model does not actually use any Web search tool for fact retrieval. Is this the major reason for task improvement? I wonder in the more realistic case of using an external tool, if the same improvement can be observed (considering the potential noise, long retrieved passages, etc.).

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
SCREWS, a modular framework for reasoning with revisions.  The author observe that these revisions can introduce errors, in which case it is better to roll back to a previous result. So there should be a framework that decide we should accept the current revision or not. 

The proposed approach consists of three steps: 
[1] Sampling instantiate SCREWS by fixing the submodules for each module  
[2] Conditional Resampling, which decides whether to generate a revision conditioned on the initial sample, and does so if needed. 
[3] Selection: all samples and revisions are given to the Selection module, which selects the best one. 

Each of the above three modules include several existing effective methods: 
Sampling: CoT, decomposition
Condition resampling: Self ask, tool use
Selection:  self-consistency, rule-based etc

### Strengths
The paper works on an interesting problem. The paper collects a couple of well known approaches and integrate them into this framework, and provide suggestions on how to use them. The paper objectively reports results, and performs analysis and comparison. Regarding ideas, self-ask with respect to multiple steps of decomposition is quite interesting.

### Weaknesses
1 the paper is a collection of existing approaches, the contribution is a bit incremental and the novelty is a bit limited. 

2 the effectiveness of the proposed approach is not quite conclusive yet. 

- Table 1 the conclusion is sampling and conditional reasamping should use different sampling approach, i.e. CoT + Subq (QG) or Subq (QG) + CoT. However, the improvement is rather incremental (i.e. 73-> 73.99). Especially considering SOTA of GSM8K IS 90+ https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k (although we understand the foundation models are different, the effectiveness of the approach is not clear)

- Table 2 “independent sampling” combines Subq (QG) and CoT (74.90) give the best performance than “conditional sampling” (73.99 table 1), which makes me unclear of the effective of conditional reasoning (i.e. combine two samplings are easy and just do majority vote on, i.e. no need to ask LLM whether to resample or not)

- The right half of Tab. 2 shows Selection between the Sampled and Conditionally Resampled. Does that mean the selection module doesn’t bring significant gain?

### Questions
See above

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
