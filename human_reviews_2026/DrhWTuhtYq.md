# QuRL: Rubrics As Judge For Open-Ended Question Answering

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has significantly improved the performance of large language models (LLMs) on tasks with gold ground truth, such as code generation and mathematical reasoning. However, its application to open-ended question answering (QA) remains challenging, primarily due to the absence of reliable evaluation and verifiable reward signals. This difficulty is further compounded by the limitations of existing evaluation paradigms. Previous approaches typically rely on human feedback or LLM-as-judge strategies, which are costly, prone to reward hacking, and often fail to provide sufficiently discriminative or interpretable evaluation signals. To address these limitations, we introduce a schema for generating case-wise  rubrics that are  question-specific, content-based and stylistically sensitive, thereby evaluating both factual soundness and writing quality. Building on this schema, we propose QuRL (Open-Ended QA with Rubric-guided Reinforcement Learning), a framework that automatically mines rubrics for each question from easily accessible online sources and leverages them as reward signals. With these rubrics, QuRL employs the GRPO (Group Relative Policy Optimization) algorithm to guide the model in exploring the correct generation path. Extensive experiments show that our framework achieves significant improvements of total +17.0 points on evaluation benchmark, demonstrating the effectiveness of rubric-guided reinforcement learning for open-ended QA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose a new/novel approach to improving the performance of LLMs on Open Ended QA. This paper proposes QuRL (Open-Ended Question Answering with Rubric-guided Reinforcement Learning), a framework that transforms human-authored web sources into case-wise rubrics. 

The key contributions are:
• QuRL, a framework that leverages internet text to construct case-wise rubrics as reward signals for open-ended question answering
• With the assistance of QuRL, two new datasets have been created. QuRL-Train dataset consisting of 800 Question–Rubric pairs, along with a QuRL-Test dataset of 400 entries that underwent human verification.

### Strengths
The main strength of the paper is in proposing a novel method to obtain rubrics on a case basis and use these rubrics to help train the LLMs to give an answer that is more human like. The authors have conducted various experiments using a variety of LLMs comparing their approach to some existing benchmarks. Thorough implementation details and results are a notable strength. The results indicate that QuRL achieves strong correlation with human judgments, outperforming other existing approaches.

### Weaknesses
Nothing major however the impact of the work is difficult to understand from a practical usecase point of view. It looks like the authors have not studied the case where facts are misrepresented in an answer by the LLM and how the rubrics deal to penalize such behavior.

### Questions
1. The paper would become easier to follow if a couple of examples for rubrics generated are shown as part of the main paper.
2. A minor typo in line 90 - Deepseek

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a rubrics guided RL based framework for open-ended QA. Authors claim that the GRPO based approach with case-wise rubrics significantly improves QA performance on multiple evaluation benchmarks. The main contributions of this paper are as follows:
1.QuRL, a framework which leverages internet text to build case-wise rubrics as reward signal for open-ended QA.
2. A new QuRL-Train dataset with question-rubrics pair, along with a human verified test set for evaluation

### Strengths
The main strengths of the paper are as follows:
1. The question-rubric train and test set might be useful for researchers.
2. Rubric based reward models seems to have some novelty and it is interesting.
3. Detailed experiments and comparison with SOTA LLMs to support the claim that QuRL improves the avg performance on multiple benchmarks
4. Human verification of the dataset improved trust and increases quality.
5. Consistency analysis between automatic evaluation method and human judgements is interesting.

### Weaknesses
The main weaknesses of the paper are as follows:
1. The proposed approach used search engine returned results for initial evidences. 
2. LLMs for meta-description generation might have its own error and biases.
3. Designs of rubrics might be subjective and might have some generalizability issues.

### Questions
1. How rubric parameters are decided? is it domain specific?
2. How strong are the reward signal coming from the rubrics?
3. How does agentic QA methods like react compare against QuRL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines the automatic creation of rubrics for a given open-ended question $q$.  The idea is to induce a set of useful rubrics to score generation output so as to allow a RLVR-like mechanism to apply for such questions.  The idea proxies search engine API ranking results as a means of implicitly creating preferential document listing that applies to $q$.   Finally with the stable of metrics derived from a limited training set (also provided by the authors), the authors apply GPRO as the RL to tune output, resulting in visible improvements to 3 benchmarks.

### Strengths
* Contributes a training set for RL tuning of such open-ended questions (albeit quite small: 800 train/400 test)
* Substantial baselining against a range of models for useful input.
* Conducts a human evaluation to help assess consistency, shows that the correlation is not high (although not discussed much).
* Contributes a useful introspection of the aligned / created metrics in the **4.5 Case Study** section.

### Weaknesses
* The submission is generally ok, but suffers from a lack of proofediting: the arguments are not crisp and precise.  
* In the face of prior work (see **Questions**), I find the work incremental, as there have been directly relevant work on inferring rubrics from question.
* The work implies that the metrics are done _per question_ and as described, over _multiple runs_ (but I may be mistaken, since the training seems to be done over the entire set of rubrics for the training set.  This seems overly expensive, but the inference cost of creating such metrics is not discussed.  
* Some main text figures reference key metadata only present in the Appendices.  This makes the paper reliant on the supplemental materials and hence I would judge it not fitting in the length requirements for ICLR.

Other minor problems
* The "contributions" final paragraph of the intro overlaps significantly with the rest of the work, it'd be better to leave it out or at least de-duplicate the text with the rest of the intro.
* 090: Deepseek is misspelled.
* 121-124: your claim should mention on what data.  The gains could not be properly scoped otherwise.
* It seems you use $\LaTeX$ for typesetting.  Please use the appropriate quote structure (e.g., 301-303 caption)
* The ethics statement is mis-used.  The ethics block should not be used to describe annotation guidelines and data capture; that should be part of the normal space requirements.  
* References should be appropriately checked for correct formatting.  Many titles have capitalisation protection errors (e.g. "llms" vs. "LLMs").

### Questions
* There are a few relevant works that work on inferring metrics from tasks that don't have specific metrics, which are relevant means to induce guidelines/metrics that seem to be missing from related work.  See:

  * Minzhi Li, Zhengyuan Liu, Shumin Deng, Shafiq Joty, Nancy Chen, and Min-Yen Kan (2025) DnA-Eval: Enhancing Large Language Model Evaluation through Decomposition and Aggregation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 2277–2290, Abu Dhabi, UAE. Association for Computational Linguistics.
  * Do Xuan Long, Duong Ngoc Yen, Do Xuan Trong, Luu Anh Tuan, Kenji Kawaguchi, Shafiq Joty, Min-Yen Kan and Nancy F. Chen (2025) Beyond In-Context Learning: Aligning Long-form Generation of Large Language Models via Task-Inherent Attribute Guidelines. In Proceedings of the Findings of the Association for Computational Linguistics (ACL '25), Vienna, Austria.

* 330: how important is the cold-start SFT to the work?  While not a contribution it is important to know.
* 296: is this meant to say `w/o rlfh`?  Also: I'm not sure that these are additive ablations or whether they are single ablations.  Notation might help.
*

### Soundness
2

### Presentation
2

### Contribution
2
