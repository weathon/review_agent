# Beyond Outcome Reward: Decoupling Search and Answering Improves LLM Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Enabling Large Language Models (LLMs) to utilize search tools offers a promising path to overcoming fundamental limitations such as knowledge cutoffs and hallucinations. Recent work has explored reinforcement learning (RL) for training search-augmented agents that interleave reasoning and retrieval before answering. These approaches usually rely on outcome-based rewards (e.g., exact match accuracy), implicitly assuming that optimizing for final answers will also yield effective intermediate search behaviors. Our analysis challenges this assumption: we uncover multiple systematic deficiencies in search that arise under outcome-only training and ultimately degrade final answer quality, including failure to invoke tools, invalid queries, and redundant searches. To address these shortcomings, we introduce **DeSA** (**De**coupling **S**earch-and-**A**nswering), a simple two-stage training framework that explicitly separates search optimization from answer generation. In Stage 1, agents are trained to improve search effectiveness with a recall-based reward. In Stage 2, outcome rewards are employed to optimize final answer generation. Across a variety of QA benchmarks, **DeSA**-trained agents consistently improve search behaviors, delivering substantially higher search recall and answer accuracy than outcome-only baselines. Notably, DeSA outperforms single-stage training approaches that simultaneously optimize recall and outcome rewards, underscoring the necessity of explicitly decoupling the two objectives.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Through systematic analysis, the authors demonstrate that outcome-only training leads to multiple deficient search behaviors, including agents failing to invoke tools, issuing invalid queries, and performing redundant searches, ultimately degrading final answer quality.
To remedy this, the authors propose DeSA (Decoupling Search-and-Answering), a two-stage training framework using Reinforcement Learning (RL). Stage 1 (Search Skill Acquisition) uses a recall-based reward (R recall) to explicitly train the agent to gather useful information and mitigate inefficient behaviors. Stage 2 (Outcome Optimization) fine-tunes the agent using the standard Exact Match (EM) reward to improve document denoising, evidence synthesis, and accurate answer generation.

### Strengths
- DeSA achieves superior overall accuracy compared to strong baselines across a comprehensive suite of seven question-answering benchmarks. The benchmark suite includes both General QA and complex Multi-Hop QA.
- The proposed DeSA framework is intuitive, simple, and easy to reimplement. Its success, despite its simplicity, is a significant plus.

### Weaknesses
- The R_recall reward (Eq 4) is ad hoc and appears insufficient for complex, multi-hop questions. The formulation gives a reward if any ground-truth answer is found in the context. For questions requiring multiple pieces of evidence, this reward signal is flawed—it would reward an agent for finding only one of two necessary facts, which is not enough to answer the question. This reward function also fails to distinguish between a single precise query and multiple redundant ones—yet the paper criticizes redundancy as a deficiency.
- The claim that “simple recall works better” (Section 6.3.1) is based on final EM, not search behavior quality. I feel there isn't enough ablation studies to draw to this conclusion here. A more nuanced reward might improve both.

### Questions
- How is the ground-truth answer set A (used for R_recall) defined, especially for multi-hop training data like HotpotQA? Is it just the final answer string, or does it include the intermediate supporting facts?
- The choice to switch stages at “peak EM before collapse” seems post-hoc. Is there a principled, automatic way to detect this transition point without monitoring EM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents an interesting idea that identifies and categorizes different types of failed search behaviors in Search-R1, providing useful insights for developing more robust research agents. 
The proposed method to mitigate these failures is conceptually valuable, and the writing is generally clear and easy to follow. In particular, the analysis of Search-R1’s failure cases is insightful and helps motivate the proposed approach. 

However, the paper also suffers from several notable weaknesses. First, the recall reward design appears hacky and unstable, yet the paper lacks sufficient analysis or explanation of its behavior and training dynamics. Second, although the proposed method shows some improvement, the evaluation is superficial and misses a fine-grained breakdown across different failure categories, which limits the interpretability of the results. Third, the implementation details are incomplete—important training configurations and hyperparameters are not reported, making the work difficult to reproduce and evaluate fairly.

### Strengths
- The overall idea is interesting. The paper identifies and categorizes different types of failed search behaviors, which provides useful insights for building more robust research agents. The proposed method to mitigate these failures is conceptually valuable.

- The writing is generally clear and easy to follow. In particular, the analysis of Search-R1’s failure cases is very insightful and helps readers understand the motivation behind the proposed improvements.

### Weaknesses
- The recall reward design appears somewhat hacky. As shown in Figure 5, when the recall curve increases, the answer accuracy (EM) curve decreases. The authors manually select step = 50 based on observation, since continuing training leads to a sharp drop in EM. However, the paper provides no analysis of why the EM curve collapses after this point. What are the model’s outputs beyond that step? Would using a softer evaluation metric (e.g., F1 score or an LLM-as-a-judge metric) change this trend? It is also intriguing that when EM approaches zero, the recall curve remains high — suggesting that the model still generates good retrieval queries but fails to produce correct final answers. Some explanation or insight into this phenomenon would be valuable.

- Although it is encouraging to see that DeSA reduces the deficiency-search rate from 23.36% (Search-R1) to 6.96%, the analysis lacks finer-grained breakdowns. It would strengthen the paper to include both quantitative and qualitative analyses across different failure categories (e.g., Failed to Search, DQ, and IS), ideally through a case-by-case comparison with the original Search-R1 failure examples.

- The implementation details are insufficient. Important factors such as total training steps, learning rate, and other hyperparameters of the proposed model are not clearly reported. These should be included for reproducibility.

### Questions
Q1: Analysis of the Recall–Answer Trade-off:

The recall reward appears to cause instability in training — as recall improves, EM drops sharply (Figure 5).
1. Could the authors analyze why the EM curve crashes after step = 50?
2. What do the generated answers look like beyond that step?
3. Would the same trend persist if the evaluation used a softer metric (e.g., F1 score or an LLM-as-a-judge metric)?

Q2: Granular Evaluation of Failure Categories:

The reduction of the deficiency-search rate from 23.36% to 6.96% is promising, but more fine-grained analysis is needed.
1. Can the authors provide per-category breakdowns (Failed to Search, DQ, IS) and quantify improvements for each?
2. A case-by-case comparison with Search-R1’s failure examples would help verify whether the improvements are systematic or limited to specific types of errors.


Q3: Implementation Details:

The paper omits crucial hyperparameters (e.g., learning rate, total training steps, batch size). These should be provided, ideally in an appendix, to ensure reproducibility. The authors could also report the number of training examples and their distribution across different query types.

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
5

### Summary
This paper addresses the limitations of training search-augmented LLM agents solely with outcome-based rewards (e.g., exact match). The authors propose DeSA, a two-stage training framework that first trains agents on search effectiveness using recall-based rewards, then optimizes answer generation with outcome rewards. Experiments across seven QA benchmarks show that DeSA reduces deficient search rates and improves both search recall and final-answer accuracy compared to single-stage outcome-only training.

### Strengths
* **Intuitive solution**. The two-stage decoupling approach is conceptually straightforward and well-motivated by the sequential nature of search-then-answer tasks.
* **Improved results**. Testing across seven diverse QA benchmarks provides the evidence for the method's effectiveness.

### Weaknesses
1. **Weak motivation-solution alignment**. The motivation analysis discusses issues listed in Figure 1, including Fail to Search (Skip Search), Duplicate Queries, and w/ Invalid Searches. However, Skip Search is due to knowledge being within the parameters or hallucination; Duplicate Queries may be due to reward hacking and entropy collapse; and w/ Invalid Searches is due to the agent generating an invalid search query format. These issues can be addressed by applying a retrieval reward, a repeat penalty, and a format reward. The paper claims that improving recall indirectly enhances search capabilities, yet it fails to establish clear causal links between the identified problems and the proposed remedy.
2. **Recall robustness**. In Eq. 4, the authors use string matching to calculate the recall reward. This method does not address practical challenges such as spelling variations, abbreviations, etc. These issues could significantly degrade the quality of the reward signal.
3. **Insufficient analysis of single-stage alternatives**. Comparison with single-stage training in Section 6.3.2 using combined rewards may be unfair due to suboptimal hyperparameter choices. Different weighting ratios or more sophisticated reward combination strategies should be explored before concluding that decoupling is necessary.
4. **Missing error analysis and failure modes**. The paper lacks detailed post-training error distributions and concrete failure case studies. Without understanding when and why DeSA fails, it's difficult to assess the method's limitations and potential improvements.
5. **Incomplete latest backbones**. The absence of comparisons with Qwen3 models  is notable, especially given the rapid progress in base model capabilities that might naturally address some search deficiencies.
6. **References**. Multiple citation formatting errors throughout the paper (missing conference venues and arXiv identifiers).

### Questions
Line 372 & 376: Where are Figures 3a and 3b referenced in the text?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces an extension to the Search-R1 framework by decomposing the end-to-end pipeline to a two-stage pipeline, one for search and one for answer generation. The reward for search is defined by recall of answer being presented in the retrieved documents.

### Strengths
+ The paper focuses on a timely topic and Search-R1 seems like a reasonable framework for exploring the idea.
+ The paper is easy to follow.
+ Extensive experiments are conducted on seven datasets.

### Weaknesses
- I do like the general idea of having reward models that evaluate the retrieval part of the Search-R1. However, I don't think this paper does a good execution of this idea. First, retrieval happens multiple times in Search-R1 and reward seems to be outcome-based reward. So if the model repeats the same query over and over, there is no penalty for the model. I see that the results look at this and shows the model reduces duplication however duplication is just the extreme case. Also I'd like to see more modeling contribution on that front or at least theoretical justification on why outcome-based reward would lead to this observation.

- Another thing that I don't like about the execution of the general idea in this work is that end-to-end training enables us to do much better in complex tasks. I am surprised that the authors decided to tear apart the end-to-end training pipeline of Search-R1. I think this is a poor design choice and I encourage them to focus on better reward modeling for end-to-end training. Note that I do not base my overall recommendation based on this comment, because one may disagree with my comment and this is mostly a suggestion for future work.

- I am surprised that the authors decided to run their experiments on Qwen instruct models. Multiple RL papers, including the Search-R1 paper which is the basis for this work, suggest that base models are more suitable for RL training. The Search-R1 results with Qwen 7B base is 0.431 on all seven datasets, about 4% higher than the highest results presented by DeSA in this work. 

- Statistical significance tests are required to demonstrate meaningful improvements compared to baselines (at least compared to Search-R1).

- Minor: why Search-o1 result is bold-faced in Table 2? Maybe a mistake...

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
