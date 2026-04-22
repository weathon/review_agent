# SKILL: Structural Knowledge Injection into Large Language Models for Inductive Knowledge Graph Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Knowledge Graph Reasoning (KGR) aims to predict missing (head, relation, tail) triples by inferring new facts from existing ones within a knowledge graph. While recent methods embed entities and relations into vectors or model multi-hop paths, they predominantly rely on statistical co-occurrence patterns, yielding logically inconsistent or semantically implausible paths that degrade prediction quality. We introduce SKILL, a new framework that revolutionizes KGR by injecting structural knowledge into large language models (LLMs) through inductive reasoning, thereby optimizing the reasoning process with LLMs' semantic understanding capabilities. Our novel rule-miner module extracts and semantically validates symbolic reasoning rules from closed paths using LLM-based one-shot prompting, effectively filtering out invalid patterns. This innovative rule injection fine-tunes LLMs with explicit symbolic guidance, leading to a comprehension of KG structures required for downstream reasoning. Extensive experiments on three standard inductive benchmarks show that SKILL surpasses competing baselines by up to 5 absolute Hit@1 points, establishing a new state of the art for inductive knowledge graph reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces SKILL, a framework for inductive Knowledge Graph Reasoning (KGR) aiming to improve upon embedding/path-based methods and existing LLM integrations. The core idea is to inject validated structural knowledge into LLMs. It involves mining symbolic rules from the KG, using an LLM via one-shot prompting to semantically validate these rules, and then fine-tuning another LLM using prompts that combine the validated rules with pruned subgraph context related to the query triple. The authors argue that this provides explicit structural guidance, enhancing inductive reasoning over unseen entities. Experiments on inductive KGR benchmarks claim state-of-the-art performance, especially on Hits@1.

### Strengths
1. **Problem Statement:** Addresses the challenging and important problem of inductive KGR.
2. **String empirical results:** Achieves strong empirical results, particularly in Hits@1 accuracy, on (limited) standard inductive benchmarks.
3. **Utilization of Foundation models:** Utilizes LLMs to replace legacy KG methods by integrating symbolic rules and subgraph context into LLM prompts for fine-tuning, attempting to provide explicit structural guidance.

### Weaknesses
1. **Scalability:** Rule mining and LLM fine-tuning (even with LoRA) are computationally expensive and unlikely to scale to very large KGs. LLM-based validation adds another potentially costly step.
2. **Reliability of LLM Validation:** The LLM validation step via one-shot prompting is heuristic and probably unreliable being sensitive to prompt design/instance selection. There is also a possibility of it inheriting LLM biases. The criteria "Reasonableness" and "Usefulness" are subjective, and hard, especially for a small model like Qwen2-7B.
3. **Marginal Benefits of some components:** Ablation results suggest that the expensive LLM validation step provides only a small improvement over using raw (unvalidated) rules combined with subgraph context. This questions the practical value of the validation component. The same is with fine-tuning the model, maybe the improvement is not justified by the cost.
4. **Interpretability Claims:** While rules can be interpretable, there is no evaluation to confirm whether the LLM's reasoning process actually follows these rules or how explanations could be extracted from the fine-tuned LLM.

### Questions
1. Why are Llama 3.1-8B and Qwen2.5-7B worse than Qwen2-7B in Table 3? I find this odd, do the authors have any hypothesis for this?
2. How computationally expensive is the LLM validation step in practice? What fraction of the total pipeline time does it consume? Does this cost justify the marginal performance gain observed over using raw rules in the ablation study?
3. Table 4 shows a very high number of raw rules found by NCRL compared to SKILL's candidate rules (derived from BFS paths before validation?). Why is there such a large discrepancy in the number of initial candidate rules between methods? The authors mention redundancy in NCRL, but is there evidence to back this claim?
4. Can you provide evidence for the robustness of the one-shot LLM validation? What happens if different instances are used for the same rule, or if different prompts are used? How was the risk of the LLM simply confirming patterns aligned with its pre-trained knowledge (rather than KG structure) mitigated?
5. How does this compare with KG Foundation Models, like ULTRA [1], and more specifically, KGFMs like SEMMA [2], which leverage LLMs along with the purely structural pipeline?

---

_[1] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, & Zhaocheng Zhu (2024). Towards Foundation Models for Knowledge Graph Reasoning. In The Twelfth International Conference on Learning Representations (ICLR)._

_[2] Arvindh Arun, Sumit Kumar, Mojtaba Nayyeri, Bo Xiong, Ponnurangam Kumaraguru, Antonio Vergari, & Steffen Staab. (2025). SEMMA: A Semantic Aware Knowledge Graph Foundation Model. In The Thirtieth Conference on Empirical Methods in Natural Language Processing (EMNLP)._

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
The paper proposes SKILL, a framework for inductive knowledge graph reasoning that integrates symbolic rules into large language models (LLMs). SKILL first mines multi-hop relational paths from a knowledge graph and converts them into candidate logical rules. These rules are then filtered for semantic plausibility using an LLM-based one-shot prompt, keeping only those deemed logically valid. The validated rules are injected into the LLM through instruction-style fine-tuning, enabling it to perform structured, interpretable reasoning over unseen entities.

### Strengths
The use of an LLM to filter candidate rules (for “reasonableness” and “usefulness”) adds a self-reflective, semantic validation layer not seen in prior inductive reasoning work.

### Weaknesses
1) The paper repeatedly claims to “inject structural knowledge into LLMs” as a novel paradigm. But this is now a standard LLM-KG based reasoning method. Many papers like ChatRule (Luo et al., 2025), Think on Graphs (Sun et al 2024), RoG (Luo et al., 2024), already explore LLM-mediated rule mining or KG-guided reasoning. The novelty of this paper is very low.

2) The framework simply fine-tunes an instruction model on rule-augmented prompts, i.e., data-level conditioning. That is not structural integration; it’s dataset augmentation. Hence, the paper’s title (“Structural Knowledge Injection”) oversells what is essentially fine-tuning with extra textualized context.

3) The inductive generalization claims rely on dataset splits with disjoint entities, but the method doesn’t explicitly model inductive transfer. The model still memorizes textual co-occurrence of rule templates; there’s no architectural or representational mechanism ensuring entity-agnostic reasoning.

4) LLM validation is ungrounded. The LLM-based rule filtering is the central novelty, yet the authors never evaluate its correctness. There’s no evidence that the LLM’s “Yes/No” judgments correlate with ground truth logical validity or human reasoning.

5) LLM is good at common-sense KG, but may performs worse on domain-specific KG. The authors should also test the method on domain-specific KGs like biomedical knowledge graphs.

### Questions
1) How is SKILL fundamentally different from prior LLM rule-mining or LLM fine-tuning frameworks like ChatRule or KG-FIT?

2) How reliable is the LLM-based rule validation, did you measure accuracy or consistency?

3) How sensitive are the results to the specific prompt wording used for validation?

4) Are the same LLMs used for rule validation and reasoning, and if so, how do you avoid circular supervision?

5) What is the computational cost of the rule-mining and LLM validation stages?

6) Are the reported gains statistically significant across runs?

7) How do you ensure inductive generalization isn’t driven by lexical overlap of entity names?

8) How scalable is SKILL to larger KGs beyond the benchmark datasets?

9) How does your method perform on domain-specific knowledge graphs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Paper summary: The paper proposes SKILL, a two-stage pipeline for inductive knowledge-graph reasoning (KGR) that (1) mines closed-path symbolic rules from a KG with BFS, filters them by support/confidence and an LLM one-shot “Yes/No” semantic check, then (2) injects the validated rules into an LLM via instruction-style fine-tuning and logic-aware subgraph prompting. A soft rule–path matching and confidence-weighted pruning pick the most relevant paths/rules for each query. Using LoRA on Qwen2-7B, SKILL reports state-of-the-art Hit@1 in inductive settings on FB15k-237 and NELL-995, and competitive transductive results; few-shot variants also improve strongly.

### Strengths
This paper treats the KG as a source of relational logic (not just embeddings) is well-motivated and nicely executed via rule mining + LLM semantic validation. The pipeline is modular and interpretable. Also, LLM filtering sharply reduces noisy rules while keeping useful ones; examples are human-readable (e.g., language/film rules), aiding transparency.

In general, this is a solid application of LLM on knowledge graph reasoning. To me, it is amazing that people can achieve over 0.7 Hit@1 on FB15k-237. Back to 2020, the best models are like RotatE, ComplEx, GPFL etc, which can achieve around 0.2-0.3 Hits@1 on the same dataset. I will say that LLM and in general large autoregressive (AR) model based on transformer tremendously push forward the performance on almost all research directions.

Anyway, this paper applies LLM and produces new state of the art performance, which is solid.

### Weaknesses
The paper evaluates with 1 positive + 49 negatives per query and appears to use reduced subgraphs of standard datasets (table stats are much smaller than canonical FB15k-237/WN18RR sizes). This makes cross-paper comparability to classical KGE work (which uses filtered ranking over all entities) unclear, and very high WN18RR scores likely reflect the sampled-candidate setting. Please report both sampled and full-ranking metrics, or justify the choice and ensure baselines are re-run under the same protocol.

The match×confidence heuristic (Eqs. 8–9) is sensible but fixed; no learning of rule/path weights or uncertainty modeling is presented. This could under-perform on longer dependencies or noisy KGs.

### Questions
1. You evaluate with 1 positive + 49 negatives per query. How are negatives sampled (type-consistent? filtered for trivial heuristics), and can you also report full filtered ranking (over all entities) for comparability with KGE work?

2. The reasoning subgraph for each query includes first-order neighborhoods of h and t plus closed paths (length ≤ k). At test time, does this subgraph draw strictly from the train graph (to avoid inductive leakage), and what exact edges are visible? Please detail the construction for transductive vs inductive splits.

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
This paper presents a meaningful attempt to enhance LLM reasoning through explicit rule-based semantic verification. The restricted dataset, missing metrics, and lack of scalability discussion further reduce its impact. Addressing these issues—particularly by improving the theoretical foundation, expanding experiments to full datasets, and optimizing inference efficiency—could make the work more competitive in future iterations.

### Strengths
1. The experiments are relatively extensive, covering multiple datasets and metrics to validate the proposed framework.

2. The proposed model explicitly integrates rule-based verification into the reasoning process of large language models (LLMs), which strengthens the interpretability and semantic correctness of generated reasoning chains.

### Weaknesses
1. The proposed framework is complex and computationally expensive, causing poor scalability and practical inefficiency.

2. In Figure 2, “subgraph pruning” is incorrectly written as “subgraph proning.”

3. The experiments use only 49 negative samples, which follows an early inductive evaluation convention but lacks credibility and general applicability in modern settings. Furthermore, the datasets used are subsets of WN18RR and FB15k-237, rather than the full versions, reducing the strength of the evaluation.

3. Only MRR and Hits@1 results are presented. Additional metrics such as Hits@3, Hits@10, or runtime comparisons should be included for a more comprehensive evaluation.

4. The proposed model provides limited theoretical insight or methodological novelty. It mainly extends existing symbolic verification frameworks without introducing fundamentally new ideas or proofs.

5. The paper lacks necessary explanations of the training procedure, inference workflow, and how rule verification is integrated with the LLM generation process.

### Questions
When dealing with large-scale graphs, does the proposed model suffer from a rule explosion issue? If so, how is computational efficiency maintained or mitigated in such scenarios?

### Soundness
2

### Presentation
3

### Contribution
2
