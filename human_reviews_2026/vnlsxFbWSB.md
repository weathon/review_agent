# Bench-CoE: A Framework for Collaboration of Experts from Benchmark

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 2, 4, 4, 4

## Abstract
Large Language Models (LLMs) are key technologies that drive intelligent systems to handle multiple tasks. To meet the demands of various tasks, an increasing number of LLMs-driven experts with diverse capabilities have been developed, spreading from language to visual understanding and generalization, accompanied by corresponding benchmarks to evaluate their performance. This paper proposes the Bench-CoE framework, which enables Collaboration of Experts (CoE) by effectively leveraging benchmark evaluations to achieve optimal performance across various tasks. Bench-CoE consists of a set of specialized expert models, a router for assigning tasks to corresponding experts, and a benchmark dataset for training the router. Based on this framework, we first formulate Query-Level Bench-CoE that is an abstraction of existing CoE methods exploiting the benchmark dataset. We further propose Subject-Level Bench-CoE, a new method that effectively addresses the potential issues of Query-Level Bench-CoE in poor generalization and labeling costs during training the router. Experiments show that the Query-Level Bench-CoE excels in in-distribution tasks, while the Subject-Level Bench-CoE demonstrates stronger out-of-distribution generalization. Our proposed Bench-CoE achieves efficient expert collaboration with minimal training label costs, improving adaptability in multi-task and cross-domain scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Bench-CoE framework to enable effective collaboration among LLMs by leveraging benchmark evaluation results. The authors validate Bench-CoE through experiments on multimodal (MMMU, MMStar) and NLP (MMLU-Pro, BigBench-Hard) tasks, demonstrating that Subject-Level Bench-CoE outperforms individual experts and Query-Level methods.

### Strengths
- The core insight of using benchmark evaluations as "free labels" for router training effectively solves two key pain points of existing CoE methods.
- The subject-expert mapping mechanism allows dynamic updates (e.g., integrating new experts or updating leaderboard rankings) without retraining the router

### Weaknesses
- While the paper contrasts Bench-CoE with Mixture of Experts (MoE) and traditional CoE methods, it overlooks recent works that also leverage benchmarks for model selection or routing.
- The paper mentions using BERT and TinyLLaVA as classifiers but provides no details on training details.
- The experiments are weak, it only compare with single methods.
- The citation format is not correct.

### Questions
- What is the contribution of this work compared to recently proposed routing methods?
- What if multiple experts perform equally well on a subject (e.g., two models with <1% accuracy difference on MMMU’s Math subject)? How does the framework handle ambiguity in mapping?
- How to choose the subject effectively? What if I want to add new subject?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper builds a framework to route queries to different LLM-based experts to achieve high performance and generalize across queries from new tasks. It proposes to use subject level meta-data of benchmark evaluations to train a router. Assuming the subject-expert mapping (i.e., best expert for each subject) is available through benchmark evaluation, the router is trained to predict the subject of the query, to then route the query to the corresponding best expert.

### Strengths
- The problem is highly relevant with respect to efficiency, reuse, and collaborative development. 
- The paper is easy to follow 
- It provides ablations showing that subject level routing generalizes better compared to existing approaches that route at query level without utilizing subject level meta-data

### Weaknesses
- It assumes that benchmarks have clearly separated subject level meta-data in them, which might not always be true. Categorizing a benchmark into a set of distinct subjects/expertise is a challenging problem on its own.
- There are cases where even though the subject remains the same, there might be different difficulty associated with them which won’t be captured if routing is learnt at subject level. For example, GSM8k vs AIME benchmark fall under math subject, where as there might multiple experts associated with this subject and they have different performances across the datasets in a given subject. 
- Naive evaluation doesn’t make sense. You can’t have the same training and test dataset. 
- For other evaluations, please provide non-zero shot baselines like best expert in the pool or best expert per query when evaluated with every expert to get a sense of benefit of the approach. For these baselines, see (https://arxiv.org/pdf/2402.05859)

### Questions
Please see weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work follows the line of Collaboration of Experts and proposes 2 abstractions that route experts, Query-level and subject-level Bench-CoT and show interesting results that Query-level excels in in-distribution tasks and Subject-level works better in OOD tasks

### Strengths
1. The methodology seems to be novel, simple, and effective, and potentially efficient.
2. The related work section is well-written and helps with the understanding of the scope of this work

### Weaknesses
1. There is usually a suite of benchmarks people in this domain test to showcase that other benchmark performances do not drop; in this work, they do MMStar, and there are more OOD/Cross-Bench datasets like MME, etc., as well. It would be more convincing to show that the performances using this method are on par or do not drop much.
2. Usually, it is great to show a model of different sizes for ablations to show the Subject-level and query-level different advantages. Doesn't need many, but good to have.
3. L160, not sure where WGM, LGM come from
4. The citation format can use some care.

### Questions
1. Subject-level and query-level has different advantage, so what is the recommendation for practitioner

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Bench-CoE, a new framework that trains a router to select expert models for a specific query. The paper proposes 2 methods: (1) query based router which directly routes the query to an expert model (2) subject based which first maps the query to a subject, then finds the expert that excels most at this subject to answer the question. The router is trained based on models' results on benchmark. The paper then conducts experiments and shows that this framework performs better than single expert models.

### Strengths
- Proposes a new framework that trains a router to select an expert for a given query 
- Sees some performance gain over single experts.
- The paper writing is generally clear and easy to follow

### Weaknesses
- Experiments are not solid enough and lack some essential baselines. The paper only compares their method of combining multiple experts with just a single model. It is not surprising that this would perform better than a single model. The paper should conduct more detailed analysis with simple baseline methods such as majority vote, or other router training methods / other expert selection methods. 
- The selected experts are outdated and not essentially the state-of-the-art models, it would be useful to incorporate the current state-of-the-art models / combine strong models with weak models and conduct how the performance would change.
- In each experiment setting, how many expert models there should be and what the expert models should be are heuristically chosen. The paper lacks essential ablation studies on these design choices.
- The design of the subject level router is not convincing enough. Especially in OOD cases because the performance is intrinsically bounded by the subject labels.

### Questions
- For out of distribution cases (for example, when the subject labels from the training set and the test set are completely different), would the subject level router still help? It would be useful to discuss this.
- How does the method perform compared with other baseline methods as mentioned in the weakness section?
- How does the expert number / expert choices affect performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Bench-CoE, a framework for collaborating multiple expert LLMs by leveraging benchmark evaluation results for routing. The authors formalize 1) Query-Level Bench-CoE, which routes queries based on which expert performs best on each individual query, and 2) Subject-Level Bench-CoE, which classifies queries into subjects and routes to experts based on subject-level benchmark performance. Experiments on multimodal tasks and language tasks show that Query-Level excels on in-distribution data while Subject-Level achieves better cross-domain generalization.

### Strengths
- The framework is well-defined and its difference from the existing methods are clearly stated.
- The proposed methods are evaluated under both in-domain and out-of-domain scenarios which makes the evaluation comprehensive.
- Experiments show performance improvement over the existing baselines.

### Weaknesses
There are existing works that propose to route experts according to the queries or topics which are not included in the paper as baselines. The proposed methods, therefore, IMO, are not very novel and bear limited impact to the research community.

### Questions
How do the proposed methods perform when compared to the "LoRA Soups" line of works, which seek to merge LoRA weights instead of routing inputs to them [1]?

[1] Prabhakar, Akshara, et al. "Lora soups: Merging loras for practical skill composition tasks." Proceedings of the 31st International Conference on Computational Linguistics: Industry Track. 2025.

### Soundness
2

### Presentation
3

### Contribution
2
