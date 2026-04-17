# Generalization of RLVR Using Causal Reasoning as a Testbed

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for post-training large language models (LLMs) on complex reasoning tasks. Yet, the conditions under which RLVR yields robust generalization remain underexplored. This paper provides an empirical study of RLVR generalization in the setting of probabilistic inference over causal graphical models. 
This setting offers two natural axes along which to examine generalization: (i) the level of the probabilistic query---associational, interventional, or counterfactual---and (ii) the structural complexity of the query, measured by the size of its relevant subgraph. We construct a dataset of causal graphs and queries spanning these difficulty axes and fine-tune Qwen-2.5-Instruct models using RLVR or supervised fine-tuning (SFT). We vary both the model scale (3B-32B) and the query level included in training. We find that RLVR yields stronger within-level and across-level generalization than SFT, but only for specific combinations of model size and training query level. Further analysis shows that RLVR's effectiveness depends on the model's initial reasoning competence.
With sufficient initial competence, RLVR improves an LLM's marginalization strategy and reduces errors in intermediate probability calculations, producing substantial accuracy gains, particularly on more complex queries. These results show that RLVR can improve specific causal reasoning subskills, with its benefits emerging only when the model has sufficient initial competence. Our code and data is available at https://github.com/zhichul/rlcausal.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies the generalization of reinforcement learning with verifiable rewards (RLVR). The authors selected causal reasoning as the probing task. Specifically, authors generate a benchmark of causal graphs alongside corresponding questions. With respect to the experiments, the authors make a comparison between SFT and RLVR and find that RLVR outperforms SFT on some subsets of the benchmark. Finally, the authors conclude that RLVR indeed enhances the generalization abilities of causal reasoning at the association and intervention level.

### Strengths
1. The experiments are comprehensive and sufficient to verify the authors' claims.

2. The analysis is comprehensive, authors provide in-depth analyses.

### Weaknesses
1. I think the first weakness lies in the writing of the abstract. The current version seems a bit colloquial rather than a formal academic paper. Specifically, the sentence "We choose this setting because causality is an important area that LLMs still struggle with, and because this setting ..." is too colloquial and lengthy. I suggest authors consider rewriting the abstract thoroughly and consider using shorter sentences.

2. Similarly, in the introduction, the sentence "However, we focus on identifying situations in which RLVR itself generalizes effectively
(versus not), and we focus on the causal reasoning domain." may be informal. Authors employ "we focus on" twice and "(versus not)" may be a little informal. Besides, "in which RLVR itself generalizes effectively" can be confusing, what dose RLVR itself mean here? I would encourage authors to revise the introduction throughly. For example, "our work differs from prior studies by focusing on an essential and challenging task: causal reasoning." 

3. I would suggest that authors include a discussion on the practical value of the studied formal causal reasoning. Since I believe LLMs are more targeted at the commonsense setting, and there already exist lots of formal causal tools in the area of causal inference.

4. Authors should consider assigning a formal name for their datasets (e.g., RLCausal or other names), since it is an important contribution of this work.

5. As there already exist other formal causal reasoning benchmarks (e.g., the CLADDER [1]), I would suggest that authors add an individual section on the differences between their newly proposed datasets and existing benchmarks. Why can't other benchmarks test the generalization abilities of RLVR?

6. The colors in Figure 1 are too light; the authors should consider deepening them. The text is too small, and it's unnecessary to show every detail of the sample; simplifying the content would allow for a larger font size. Besides, I do not quite get the main structure of the Task Formulation in this version. Authors should revise Figure 1 to a more abstract and summarized representation.

> [1] Jin Z, Chen Y, Leeb F, et al. Cladder: Assessing causal reasoning in language models[J]. Advances in Neural Information Processing Systems, 2023, 36: 31038-31065.

Happy to raise the score if the authors could address my concerns.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates the relationship between large language models’ (LLMs) post-training mechanisms and their ability to perform causal reasoning tasks. The author introduces a data generation process to create and evaluate causal question-answering scenarios. The study then compares models post-trained using Supervised Fine-Tuning (SFT) and Reinforcement Learning from Verbal Reward (RLVR, including GRRO and DAPO variants) across different evaluation dimensions.

### Strengths
The paper is well-structured and clearly written, with a logical flow that makes it easy to follow the author’s reasoning. The analysis sections are particularly strong: insightful, well-grounded, and supported by detailed experiments. The results are extensive and could serve as a valuable reference for future researchers studying the intersection of LLM post-training and causal reasoning.
I was initially debating between a rating of 6 and 8 and currently lean toward the former, though I remain open to adjusting this based on the rebuttal and other reviewers’ feedback.

### Weaknesses
1. The RLVR experiments use 7.5K and 2.5K samples, while the SFT model is trained on 5K samples. This discrepancy makes the quantitative comparison between models less reliable. I suggest adding an ablation study where models (or checkpoints) are trained on the same amount of data and for comparable GPU hours to mitigate this concern.

2. LLMs learn differently from humans as they rely primarily on language pattern recognition rather than true causal inference. The proposed causal reasoning tasks implicitly require two skills: (a) retrieving or identifying the correct numerical values from context, and (b) performing basic computations. It would strengthen the paper to include a detailed error analysis comparing SFT and RLVR models on these sub-tasks, in addition to the LLM-judge results that emphasize human-like problem-solving performance.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper investigates the generalization capabilities of RLVR for post-training LLMs, using probabilistic inference in causal graphical models as a testbed. The authors fine-tune Qwen-2.5-Instruct models (3B-32B parameters) using both RLVR and SFT on datasets of causal graphs and queries spanning three difficulty levels and complexity. The work contributes to understanding the conditions under which RLVR effectively generalizes, highlighting both its strengths and limitations in challenging formal reasoning tasks.

### Strengths
- The choice of probabilistic inference in causal graphical models as a testbed is genuinely innovative. Unlike prior RLVR generalization studies that focus on text/visual reasoning tasks, this formal mathematical domain enables precise control and analysis.
- The findings have practical implications: practitioners should check if their base model has sufficient reasoning capability before investing in RLVR. The identification that counterfactual reasoning remains unsolved even with RLVR and 32B models highlights a key challenge for the field.

### Weaknesses
- SFT is trained only to predict final answers while RLVR generates full reasoning chains. This creates an asymmetric comparison that conflates two factors: (1) reasoning vs. direct prediction and (2) RL vs. supervised learning. A fair strategy is to include an SFT baseline trained on optimal reasoning chains (generated by the solver or sampled from successful RLVR rollouts). This would isolate whether gains come from RL exploration or simply having reasoning chains.
- The paper observes 3B models fail to benefit from RLVR and regress to direct prediction after training, but doesn't investigate what specifically these models lack. 
- The reward is simply r = 0.8 · accuracy + 0.2 · format with threshold t=0.01. Is the 0.8/0.2 weighting optimal? Would shaped rewards (partial credit for intermediate steps) help?
- The writting is vague. For example, the description of the third finding "What did RLVR learn? " is too abstract, containing too many technical terms. They should present this result in a concrete style.
- Why not compare RLVR with other RL-based post-training paradigm, such as RLHF and RLAIF?

### Questions
- Can you provide results for SFT trained on reasoning chains (even if just for 7B/one level)?
- Can models solve trivially simple counterfactuals (e.g., 3-node graphs, no marginalization needed)?

### Soundness
3

### Presentation
3

### Contribution
3
