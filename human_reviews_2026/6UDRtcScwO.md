# Harnessing the Power of Reinforcement Learning for Language-Model-Based Information Retriever via Query-Document Co-Augmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Recent advances have explored the use of large language models (LLMs) as retrievers by rewriting user queries, while other work has focused on expanding or augmenting corpus documents. However, such unidirectional augmentation, applied to queries or documents in isolation, often fails to reconcile the lexical and stylistic mismatches between them, limiting recall and overall retrieval robustness.
    To this end, we present an LLM-based retriever empowered to augment both user queries and corpus documents, with its policy fully explored via reinforcement learning (RL) and minimal human inductive bias. Notably, we find that simply training the LLM to augment queries and documents separately, even when combining both at inference, yields little benefit unless paired with our carefully designed bidirectional RL framework, which enables the LLM to simultaneously learn and collaborate on both query and document augmentation policies. A key technical challenge in realizing such a framework lies in jointly updating both policies during training, where the rewards for the two directions depend on each other, making their entangled reward intractable. Our approach addresses this by introducing a reward sampling strategy and a specifically designed RL algorithm that enables effective training with these sampled rewards. Experimental results demonstrate that our approach significantly enhances LLM-based retrieval performance in both sparse and dense settings, particularly in difficult retrieval domains, and achieves strong cross-benchmark generalization. Our code will be publicly released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a reinforcement learning framework for enhancing large language models (LLMs) in information retrieval (IR). It introduces a query–document co-augmentation method, enabling the LLM to simultaneously augment both the query and relevant documents. This bidirectional optimization is designed to improve retrieval effectiveness across sparse and dense retrieval architectures (e.g., BM25, BGE-base-en-v1.5).

### Strengths
The paper introduces a well-motivated reinforcement learning framework that jointly optimizes both query and document representations, rather than treating them as separate components.
The experiments are well-structured, spanning both sparse (BM25) and dense (BGE-base-en-v1.5) retrieval backbones.

### Weaknesses
Overall, the paper’s idea is interesting, but the presentation is vague and lacks clarity, especially in mathematical formulation and experimental description. The following issues should be addressed to improve readability and credibility.

**Clarity**
1. The paper does not provide clear mathematical notations when describing the reward function and other key components of the framework. This makes the overall process vague and hard to follow. The authors are encouraged to formally define all important equations (e.g., reward computation) to improve clarity.
2. The prompts used during RL training are not provided. This information is important because prompts determine the exploration direction of LLMs during RL training. The paper should include the exact prompts or at least representative examples.
3. (Line 294) – The authors mention “3 of the datasets” from BEIR, but do not specify which ones are used. Please clearly indicate which datasets are used for training and which for evaluation.
4. (Lines 407–410) – The computation process described here should include explicit mathematical notations. The current textual explanation is vague to understand how the calculations are performed.

**Accuracy for presentation**
1. (Lines 248–249) – The sentence “remove within-group normalization while retaining within-group centering” actually corresponds to Dr. GRPO (see Dr. GRPO, https://arxiv.org/pdf/2503.20783). The authors should cite this work accordingly.
2. (Line 370) – The statement that Qwen2.5-3B’s instruction capabilities are limited is not accurate. According to Qwen2.5 technical report, https://arxiv.org/pdf/2412.15115, its instruction-following ability is relatively strong. The suboptimal augmentation outputs are more likely due to the smaller capacity of Qwen2.5-3B, leading to a narrower exploration space compared with Qwen2.5-7B. As a result, Qwen2.5-3B cannot sample better trajectories during RL training.
3. (Figure 3) – The caption’s phrase “state-of-the-art reinforcement learning algorithms” is not accurate. It would be more accurate to say “representative reinforcement learning algorithms for optimizing LLMs,” and you should emphasize that these algorithms are representative within the LLM + RL setting, to narrow the scope of whole RL research.
4. (Line 788) – The phrase “based on reward beyond preference alignment” is inaccurate. In RL training, preference alignment is also a form of reward signal. Thus, the distinction made here is conceptually incorrect and should be rephrased for precision.

**Experimental Limitations**
1. The paper does not include comparisons with recent strong baselines such as DeepRetrieval [https://arxiv.org/pdf/2503.00223]. Although some baselines are mentioned in the related work section, experimental comparisons are necessary to demonstrate relative performance.
2. In Figure 5, it is unclear why the “original document” does not contain the term “Radioactive.”, since my understanding is this doc should be relevant to "Radioactive", and the doc should have mentioned the keywords in the document. This should be clarified to help readers interpret the qualitative results properly.

**Minors**
1. Line 863  ¡answer¿¡/answer¿ -> < answer > < /answer >, some rendering problems
2. Line 462, calculation settings described in Sec -> Sec [what?]

### Questions
See the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a reinforcement learning framework for LLM-based information retrieval that jointly augments both queries and documents. The authors introduce a bidirectional co-augmentation scheme trained via a novel within-batch reward sampling and advantage computation strategy.

### Strengths
- The paper developed an RL framework that jointly augments both queries and documents, moving beyond the typical one-sided query rewriting or document expansion approaches in LLM-based retrieval.
- The batch–unbatch alternating mechanism demonstrates thoughtful engineering that allows the framework to remain compatible with standard LLM RL pipelines.

### Weaknesses
- This method requires augmenting both the query and the document, which poses significant computational efficiency challenges in practical retrieval tasks — especially when performing full-scale document rewriting.
- There is no comparison to existing LLM-based retrievers or question rewriting works. 
- The experimental results show only marginal improvement over the base model, while introducing a much more complex training process. I notice that the performance gap between using LLM augmentation and the base retriever is also quite small, which suggests that the baseline may have been set too weak.

### Questions
- How did the authors design the prompts? The authors should provide more details about the prompts used and discuss the potential impact of the prompt design on the base model’s behavior.

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
4

### Summary
This paper proposes an LLM-based retriever that learns to augment both queries and documents through a bidirectional reinforcement-learning (RL) framework. Instead of training query and document augmentation separately, the authors introduce a joint policy with a reward-sampling mechanism to handle the coupled reward dependency between query and document spaces. Experiments on BEIR datasets demonstrate improvements over BM25 and BGE retrievers. The paper claims strong generalization across domains and provides ablations showing that joint query-document training outperforms unidirectional augmentation

### Strengths
•	Addresses an under-explored bidirectional augmentation problem for retrieval.

•	Provides thoughtful engineering to make joint RL training feasible.

•	Strong empirical gains over static query/document augmentation baselines.

•	Includes qualitative analysis showing lexical alignment between queries and documents

### Weaknesses
•	Missing comparison with DeepRetrieval (Jiang et al., 2025), which is explicitly cited but not benchmarked. That work already employs on-policy RL for retrieval optimization and serves as the most relevant baseline.

•	The improvements (e.g., +0.02–0.04 NDCG@10) are relatively small considering the complexity of the method.

•	Limited evaluation: only a few BEIR datasets, no real-engine or large-scale web-retrieval experiments.

•	The reward-sampling estimator lacks theoretical analysis or variance/bias quantification.

•	Computational cost and convergence behavior are not fully discussed.

### Questions
1. Can the authors provide variance or bias estimates for the sampled rewards?
2. How sensitive is the performance to the number of rollouts N and to batch size?
3. Would this method scale to real-engine settings (e.g., ClueWeb or MS MARCO) without prohibitive cost?

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
3

### Summary
This paper proposes a reinforcement learning (RL)-based framework for improving large language model (LLM) retrievers through query-document co-augmentation.

### Strengths
The bidirectional RL formulation gives new findings in IR by aligning query and document semantics through co-augmentation

### Weaknesses
1. The work focuses on query–document co-augmentation, yet does not compare against established query expansion or document expansion methods (e.g., Query2Doc). As a result, it remains unclear whether the proposed RL-based co-augmentation offers consistent advantages over existing augmentation paradigms.

2. The paper does not provide the full prompts used for augmentation, despite prompts being central to model behavior.

3. In Table 1, models such as “Qwen2.5-7B” are presented under the heading “Base Retriever”, while they are actually used as rewriters rather than retrievers. This mislabeling can confuse model roles in the pipeline.

4. Line 295: "select three of its datasets for model training", which three?

5. Writing Issues, e.g., Line 132, "the policy augments both the query and performs retrieval" should be revised for grammatical correctness.

Writing contains many "—" (possible AI generated writing), e.g., in Sec1, or around line 301 302, long sentence.

### Questions
How does training cost scale with corpus size? For instance, can the proposed method feasibly handle corpora on the scale of MS MARCO or larger web collections?

### Soundness
2

### Presentation
2

### Contribution
2
