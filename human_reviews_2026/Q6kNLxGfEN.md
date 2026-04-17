# Searching for Difficult-to-Translate Test Examples at Scale

- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
NLP models require test data that are sufficiently challenging. The difficulty of an example is linked to the topic it originates from ("seed topic"). The relationship between the topic and the difficulty of its instances is stochastic in nature: an example about a difficult topic can happen to be easy, and vice versa. At the scale of the Internet, there are tens of thousands of potential topics, and finding the most difficult one by drawing and evaluating a large number of examples across all topics is computationally infeasible. We formalize this task and treat it as a multi-armed bandit problem. In this framework, each topic is an "arm," and pulling an arm (at a cost) involves drawing a single example, evaluating it, and measuring its difficulty. The goal is to efficiently identify the most difficult topics within a fixed computational budget. We illustrate the bandit problem setup of finding difficult examples for the task of machine translation. We find that various bandit strategies vastly outperform baseline methods like brute-force searching the most challenging topics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Interesting research for searching difficult translation examples. In Natural Language Processing (NLP), finding difficult - to - translate test examples is crucial for effectively evaluating model capabilities. Since the Internet offers a vast amount of language data but manual curation is infeasible due to scale and topic diversity, this paper formalizes the task as a multi - armed bandit problem to identify the most challenging topics within a fixed computational budget for machine translation. Here, we will explore the advantages and disadvantages of this approach to understand its implications for NLP research.

### Strengths
Research strengths include: 

- Efficient in Resource Allocation​

By formalizing the task of finding difficult - to - translate examples as a multi - armed bandit problem, the approach can efficiently identify the most difficult topics within a fixed computational budget. Each topic is treated as an “arm”, and pulling an arm involves drawing and evaluating an example. This way, it strategically explores the topic space, avoiding blind brute - force evaluation across all topics. For example, instead of exhaustively sampling from every potential topic, it focuses on the most promising ones, thus saving computational resources and time.​

- Scalable and Adaptable​

The method can generate and sample from a large amount of Internet data. It hierarchically generates topics starting from top - level ones and samples text through language model generation with the help of Google search tool calling. This process is generally inexpensive. Moreover, this formulation is compatible with any NLP task where Internet data can be a valid input and the input difficulty can be estimated. It can continuously explore new topics, adapting to different evaluation requirements in various NLP tasks, such as machine translation, question - answering systems, etc.​

- Outperforms Baseline Methods​

Experimental results show that various bandit strategies, like the ε - greedy algorithm, vastly outperform baseline methods such as brute - force searching. As shown in Figure 2, with the same budget and the cost of a single sampling being 1, the bandit - based search algorithms can achieve higher top - 1 and top - 10 difficulty scores. This indicates that these strategies can more effectively discover challenging test data for machine translation, providing better evaluation data for NLP models.

### Weaknesses
- High Computational Cost​
Each difficulty estimation of a source text involves generating a translation using a target model and then evaluating the translation’s quality to determine the text’s difficulty. For example, in the case of machine translation, this process requires significant computational resources. Since the number of potential topics is vast and each topic may need to be sampled multiple times, the overall computational cost can be extremely high. This high cost limits the scalability of the approach, making it challenging to apply on a large - scale, especially when dealing with a large number of topics and models.​

- Difficulty in Topic Selection​
Judging the actual difficulty of a topic for a model is challenging. There is a disconnect between a topic’s perceived difficulty by a human and its actual difficulty for a model. For instance, a topic that seems complex to humans, like Baroque music theory, might be easily processed by a model, while a seemingly simple topic like concrete masonry can be difficult for a model if its specialized terminology was not present in the training data. This makes it hard to accurately select the most difficult topics, and incorrect topic selection can lead to wasted resources and less - effective evaluation data.​

- Reliance on Model and Estimation Method​
The approach relies on a specific set of models for translation and a particular method for difficulty estimation. Different models may produce different translation results, and different difficulty estimation methods may yield varying difficulty scores for the same text. This means that the results obtained are highly dependent on the chosen model and estimation method. As a result, the generality and reliability of the discovered difficult - to - translate examples may be affected. If the model or the estimation method changes, the identified difficult topics and examples may also change, reducing the stability and broad applicability of the findings.

### Questions
Providing computational cost​ comparements and reducing methods , this research score will be much higher.
Topic selection is interesting, but only providing cases. More analysis will be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework for automatically discovering challenging test examples for machine translation by framing the search process as a multi-armed bandit problem. The goal is to efficiently identify the most difficult topics within a fixed computational budget. The paper demonstrates that bandit-based strategies (especially ε-greedy) significantly outperform brute-force baselines, scaling to millions of topics while uncovering test sets that are more challenging than existing benchmarks like WMT and FLORES. The approach is task-agnostic and could extend to other NLP tasks.

### Strengths
1. I like the experiment  on the scalability an efficiency, which empirically validates that ε-greedy achieves near-oracle performance with only ~6% of the budget required for brute-force evaluation, and  Demonstrates logarithmic scaling of difficulty with topic set size, supporting theoretical claims. Also the error analysis reveals terminology/accuracy errors (70% major errors), aligning with known MT weaknesses.

### Weaknesses
1. I feel the comparison with other baselines is missing, such as active learning or curriculum learning baselines, which also target difficult examples.
2. I would like to see the generalization of this frame across different tasks. While framed as task-agnostic, experiments are exclusive to MT. No validation on other NLP tasks (e.g., summarization, QA) is provided, weakening claims of broader applicability.

### Questions
see weakness for detail:  comparison with active learning or curriculum learning baselines and the generalization across different tasks.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of efficiently identifying difficult test data for language models. The authors formalize the search for challenging topics as a *multi-armed bandit* problem, where each topic represents an "arm," and sampling it (at a cost) produces an example whose difficulty is then evaluated. The objective is to identify the most difficult topics within a limited computational budget. Applied to machine translation, the proposed method significantly outperforms brute-force or greedy approaches in discovering challenging test examples.

### Strengths
The paper addresses an important practical problem: how to effectively identify the weaknesses of language models. The problem formulation and the final solution are neat and well-suited to the task. The efficacy of the proposed method is further supported by comprehensive empirical studies.

### Weaknesses
1. It would be more interesting if the paper can include one more task, such as safety or factuality. I believe the post-training stage shares similar problems, e.g., designing a prompt set for the reinforcement fine tuning step.
2. Minor comments:
    - "desiderate" &rarr; "desiderata" in line 54.
    - Avoid using the same notation to represent different concepts. For example, $t$ is used for both "translation" (line 84) and "topic (throughout the rest of the paper).
    - Hyphens in "top-k"s are incorrectly in math mode.

### Questions
See ``Weaknesses``.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a MAB framework for automatically identifying difficult topics from large-scale Internet text to construct more challenging test sets for NLP models, particularly machine translation. The authors observe that example difficulty is stochastically related to its topic, some topics tend to yield harder examples, but this relationship is noisy and costly to explore exhaustively across thousands of potential topics. To address this, the paper formalizes topic selection as a stochastic bandit problem, where each topic is treated as an arm, and pulling an arm corresponds to sampling a text from that topic, translating it, and measuring its difficulty using a model-based quality estimation metric (e.g., GEMBA). Within a fixed query budget, the goal is to identify the k most difficult topics efficiently.
The paper proposes and compares several bandit strategies (random, greedy, and epsilon-greedy, with batched variants) for this exploration-exploitation trade-off, showing that bandit methods significantly outperform uninformed or brute-force baselines in finding difficult topics.

### Strengths
* The paper's framing of difficult-topic discovery as a multi-armed bandit problem is intuitive and useful.

### Weaknesses
* The conceptual novelty is limited. The formulation is a straightforward application of standard stochastic MAB principles, with no new theoretical insights, regret analysis, or algorithmic extensions. 
* The difficulty estimation pipeline and sampling procedures rely entirely on existing methods, and the empirical results confirm expected behavior rather than reveal new dynamics. 
* Furthermore, the approach ignores the inherent structure of the topic space, missing opportunities for contextual or hierarchical modeling. Overall, the contribution is incremental, a solid engineering integration rather than a substantive methodological or theoretical advance not suitable for ICLR.

### Questions
Kindly refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
