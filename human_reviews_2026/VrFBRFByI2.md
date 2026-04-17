# AInstein: Assessing the Feasibility of AI-Generated Approaches to Research Problems

- Decision: Reject
- Scores: 2, 6, 6, 2, 4

## Abstract
language models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet it remains unclear whether such success reflects genuine reasoning or sophisticated recall.
We introduce **AInstein**, a framework for testing whether LLMs can generate valid solutions to AI research problems using only their pretrained parametric knowledge---without domain-specific fine-tuning, retrieval augmentation, or other external aids.  Our approach extracts distilled problem statements from high-quality ICLR 2025 submissions, then tasks specialized solver agents with proposing and refining technical solutions through iterative critique loops, mimicking the cycles of proposal, review, and revision central to scientific inquiry.   We evaluate AInstein on 1,214 ICLR papers stratified by acceptance tier (Oral, Spotlight, Poster), using an LLM-as-a-judge paradigm guided by a structured rubric, complemented by targeted manual checks. Performance is assessed with three metrics: Success Rate (does the solution address the problem?), Rediscovery (does it align with human-proposed methods?), and Novelty (does it yield valid, original approaches?).   Our results reveal that while LLMs can rediscover feasible solutions and occasionally propose creative alternatives, their problem-solving ability remains fragile and highly sensitive to framing. These findings provide the first large-scale evidence on the extent to which LLMs can act as autonomous scientific problem-solvers, highlighting both their latent potential and their current limitations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes AInstein, a framework for evaluating LLMs on the task of generating solutions to given research problems. AInstein focuses on evaluating LLMs in situations where they rely solely on their pretrained parametric knowledge, without using any external information. For evaluation, the authors create a benchmark consisting of problem–solution pairs extracted from the abstracts of ICLR 2025 papers. The evaluation is based on three criteria: success rate, rediscovery, and novelty, which are assessed using LLM-as-a-judge and cosine similarity of embeddings. The authors evaluate a solution-generation framework leveraging self-correction and claim that LLMs are capable of generating creative solutions rather than merely reproducing the human-generated solutions.

### Strengths
* This paper introduces a new benchmark for evaluating LLMs in generating solutions to given problem statements, which are extracted from ICLR 2025 papers. Although there are many studies on research idea generation, there is a lack of benchmarks that include reference solutions for given research questions. This dataset could serve as a useful benchmark for the research solution generation task.

### Weaknesses
* **The studied setting is not well-motivated.** This paper targets the setting of "using only their pretrained parametric knowledge," without relying on external aids. This is a reasonable baseline setting; however, it is not sufficiently justified to explore the setting without external information such as retrieval. The paper claims to "isolate scientific problem-solving ability," but the studied setting is largely affected by the amount of knowledge stored in the model parameters. If the goal is to isolate scientific problem-solving ability from other capabilities, it would be more reasonable to provide sufficient information necessary for solving the target problems.

* **Unreliable evaluation framework.** They employ cosine similarity between the problem and the generated solution to evaluate alignment. It is not sufficiently justified whether this simple metric effectively captures alignment quality. For example, demonstrating its correlation with human evaluation would strengthen the claim. Similarly, regarding LLM-as-a-judge, they state that "GPT-OSS-120B as an LLM-as-a-judge is known to correlate well with human evaluation," but they need to provide evidence of this specifically for the studied task.

---
Minor limitations

* **Findings are not surprising.** Although the settings are not exactly identical, many previous studies have shown that LLMs can generate creative and feasible research ideas [e.g., 1].

* **Novelty of the method.** The proposed framework is a straightforward application of self-correction to this task, which is also not novel. 

* **Missing baselines.** This paper does not provide any baseline scores for comparison with LLM performance, such as human performance. As a result, from the provided numbers, it is difficult to understand how well LLMs perform on the proposed task.

[1] Si et al. (ICLR 2025). Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers. https://openreview.net/forum?id=M23dTGWCZy

### Questions
I expect responses to the points I listed in the Weaknesses section.

### Soundness
1

### Presentation
2

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
This paper investigates whether large language models can serve as autonomous scientific problem-solvers. It introduces an iterative proposal–critique–revision framework that includes both internal and external critique loops to extract research problems from existing ICLR 2025 papers while minimizing information leakage and solution bias. To evaluate solution quality, the authors propose three complementary metrics: Success Rate, Rediscovery, and Novelty. They combine an LLM-as-a-judge rubric with selective human verification through a head-to-head tournament. The results show that solution quality is closely tied to the ability of the reasoning model, and that current LLMs are more adept at generating novel yet scientifically valid alternatives than at reproducing human-designed methods.

### Strengths
This paper explores a highly interesting and valuable question about whether large language models can serve as reliable autonomous problem solvers for research problems. It examines the originality and divergence of LLM-generated solutions compared with human-designed ones.

The paper presents an iterative proposal, critique, and revision framework that enables accurate problem extraction without solution leakage while improving the quality of generated solutions.

It also introduces systematic evaluation metrics, including success, rediscovery, and novelty, which allow for a fine-grained analysis to provide clear insights into the capabilities and limitations of current LLMs in scientific problem solving.

### Weaknesses
**Overclaim about question:** Although this paper claims to explore whether LLMs can solve AI research problems, the actual experiments only investigate the generation of solution proposals, rather than their detailed implementation. As we know, even if two solution descriptions appear similar, the specific implementation details can lead to drastically different outcomes. Therefore, the act of “solving a problem” should not be separated from its implementation — they should be evaluated as an integrated whole. From this perspective, the paper does not convincingly answer its central question.

**Potential Data Leakage:** While the paper states that “to prevent data leakage and ensure our evaluation tests reasoning rather than retrieval, all models used in our experiments have knowledge cutoffs that predate the ICLR 2025 submission deadline”, I believe data leakage may still exist. Many ICLR papers are resubmissions of earlier works, meaning these papers are probably already available on the internet before the ICLR 2025.  For instance, although GPT-OSS-120B has a knowledge cutoff in June 2024, many papers were already online several months earlier, especially resubmissions, making it possible that some of this information was included in the model’s training data.

### Questions
1: When computing the Elo scores based on human evaluations, were the comparisons conducted across different human evaluators? Human bias (e.g., differences in background knowledge or familiarity with specific research areas) could strongly affect the preference judgments between different solutions.

2: How does the number of iterative refinement cycles in the critique loops influence the final results, such as the success rate? Have you explored how performance varies with different numbers of refinement iterations?

3: Have you tried retrieval-augmented approaches, where the model retrieves related papers or prior works to inspire more accurate or innovative solutions? I believe incorporating retrieval could potentially improve both the quality and originality of the generated research proposals.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new framework to test whether large language models (LLMs) can act as independent scientific problem-solvers using only their internal knowledge without fine-tuning, retrieval, or external data. The proposed AInstein framework mimics the real research cycle through two stages: Problem Extraction and Solution Generation. Using 1,214 ICLR papers across Oral, Spotlight, and Poster tiers, the study measures Success Rate, Rediscovery, and Novelty with both LLM and human evaluation. Results show that powerful models (e.g., GPT-OSS-120B) can rediscover feasible ideas and even generate new, valid research directions, though their reasoning remains fragile and sensitive to prompt framing.

### Strengths
- The paper raises a novel and important research question: whether LLMs can independently solve research problems without any external knowledge.  
- The two-phase setup (Problem Extraction + Solution Generation) with internal/external review loops nicely reflects how real research and peer review work.  
- The experimental setup is solid. The dataset includes 1,214 ICLR 2025 papers across multiple acceptance tiers (Oral, Spotlight, Poster), and multiple model families are tested.

### Weaknesses
- The study focuses only on AI/ML papers, so it’s uncertain whether the findings generalize to other fields such as biology or physics.  
- Even though the models were trained before ICLR 2025, they might still have internalized common research patterns, making it difficult to fully separate reasoning from memorization.  
- The evaluation remains at the text level. There’s no actual implementation or experimental validation to confirm whether the generated solutions would work in practice.

### Questions
- Can the findings generalize beyond the AI/ML domain to other scientific domains?  
- How can we truly determine whether the model is memorizing familiar research patterns rather than demonstrating genuine reasoning?  
- How can we be sure that the generated solutions would actually work in practice?

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
5

### Summary
This paper introduces AInstein, a framework for evaluating whether LLMs can autonomously generate viable research solutions without external knowledge. The system distills problem statements from recent ICLR papers and uses nested critique loops to produce and iteratively refine methods, which are then assessed through automated scoring and human comparison.

### Strengths
1, The problem studied int this work  is timely and ambitious;

2, The proposed problem-to-solution and dual-critique framework is well designed and mimics real scientific workflows.

3, The experiments cover a broad range of topics and models, and reported multi-dimensional analyses (e.g., rediscovery, novelty, human comparison).

### Weaknesses
1, "Feasibility" may need to consider compute constraints、data acquisition practicality. 
2,  A big concern when comparing with human-written methods: LLMs or human annotators may identify human-authored solutions from academic phrasing and reward them unfairly. Incorporating style obfuscation or content-only evaluation would strengthen the fairness of human-vs-model comparisons.
3, The paper's human verification is limited to the authors evaluating their own system's outputs. This is a major methodology flaw: the risk of self-evaluation bias is high, and given the breadth of ICLR topics, the authors are unlikely to be qualified to judge novelty and technical correctness across all problem domains. Without external and diverse experts, the setup here is not acceptable.

### Questions
N/A

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose AInstein, a framework for evaluating whether LLMs can generate valid solutions to AI research problems using only their pretrained parametric knowledge. They first collect the accepted papers from ICLR 2025 and rewrite the abstract to become a question. Then, an iterative question-solving method is used to generate an answer. Finally, these results are compared with human solutions with LLM-as-a-Judge. Results show that the problem-solving ability of LLM remains fragile and highly sensitive to framing.

### Strengths
1. AI4Research is a very important topic, and it might be shaping the next generation of AI.
2. The proposed method and datasets are intuitive and easy to use.
3. The paper is relatively easy to read, and Figure 1 is helpful in understanding the logic.

### Weaknesses
1. I am trying to convince myself about the motivation. Why do we need to isolate LLM from external knowledge searching? In my opinion, this is a normal step for human researchers. The authors mention: "(a) synthesize disparate observations into coherent theories and (b) traverse the conceptual path from problem to principle to solution". However, these things can be tested with resource searching as well. Besides, LLMs are trained on most of the data provided on the Internet; there is no difference between using the external knowledge or not if the knowledge cutoff is the same.

2. Lacks human evaluation. First, all experiments are evaluated by LLM-as-a-Judge, we do not know the accuracy of LLM judge, and research points out that the LLM judge is biased towards its own kind. Second, the metrics themselve are newly proposed, and we do not know the accuracy of the metrics. Overall, at least some human annotation is needed to compute the similarity between human and LLM annotations to make the annotation useful. 

3. Knowledge cutoff is unsure. It is not known whether the model knows the provided ICLR 2025 paper and generates the answer from memory.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
