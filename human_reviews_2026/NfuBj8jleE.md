# Explore-on-Graph: Incentivizing Autonomous Exploration of Large Language Models on Knowledge Graphs with Path-refined Reward Modeling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
The reasoning process of  Large Language Models (LLMs) is often plagued by hallucinations and missing facts in question-answering tasks.
A promising solution is to ground LLMs' answers in verifiable knowledge sources, such as Knowledge Graphs (KGs). Prevailing KG-enhanced methods typically constrained LLM reasoning either by enforcing rules during generation or by imitating paths from a fixed set of demonstrations. However, they naturally confined the reasoning patterns of LLMs within the scope of prior experience or fine-tuning data, limiting their generalizability to out-of-distribution graph reasoning problems.
To tackle this problem, in this paper, we propose Explore-on-Graph (EoG), a novel framework that encourages LLMs to autonomously explore a more diverse reasoning space on KGs.
To incentivize exploration and discovery of novel reasoning paths, we propose to introduce reinforcement learning during training, whose reward is the correctness of the reasoning paths' final answers. 
To enhance the efficiency and meaningfulness of the exploration, we propose to incorporate path information as additional reward signals to refine the exploration process and reduce futile efforts.
Extensive experiments on five KGQA benchmark datasets demonstrate that, to the best of our knowledge, our method achieves state-of-the-art performance, outperforming not only open-source but also even closed-source LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of poor generalization in existing Large Language Model (LLM) methods for Knowledge Graph Question Answering (KGQA), which often fail when faced with reasoning patterns not seen during training. The authors propose "Explore-on-Graph" (EoG), a novel framework designed to incentivize LLMs to autonomously explore diverse reasoning paths on a knowledge graph (KG).

### Strengths
The core contribution—incentivizing autonomous exploration on KGs via reinforcement learning—is a novel and important step beyond prevalent imitation-based methods. While RL has been used for KG reasoning in the past, its application to modern LLMs with the proposed two-phase reward structure (outcome + path-refinement) is original. The "path-refined reward" is an intuitive and clever mechanism to make the exploration process more efficient and meaningful, rather than just rewarding the final outcome. This directly addresses the critical problem of out-of-distribution generalization in KG reasoning.

### Weaknesses
The path-refined reward, Rpath, is a key component of the method. Its calculation (Equation 4) requires a "ground-truth reasoning path" rg. However, the paper fails to explain how these ground-truth paths are obtained for the training data across all five datasets. While some datasets (e.g., 2WikiMultihop) may provide such paths, it is not standard for others like CWQ, WebQSP, or GrailQA. This is a critical missing methodological detail. 

The proposed SFT + RL pipeline, particularly using GRPO with multiple rollouts per prompt, appears to be computationally very expensive. The paper mentions using 8xH100 GPUs but does not provide a broader discussion on the training time, total computational budget, or the trade-offs between performance gains and the significant training overhead. A comparison of the computational cost against simpler fine-tuning methods would be valuable for assessing the practical applicability of EoG.

### Questions
see weaknesses

### Soundness
3

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
4

### Summary
This paper proposes Explore-on-Graph (EoG) framework that incentivizes LLMs to autonomously explore diverse reasoning paths on Knowledge Graphs for Question Answering. The methods consist two stages: (i) SFT using long CoT by Gemini 2.5 Flash, (ii) reinforcement learning with Group Relative Policy Optimization using both outcome reward and path refined reward. The authors evaluate EoG on five KGQA benchmarks and demonstrate state-of-the-art performance, though I must say the results for the baseline are mostly missing.

### Strengths
1. This method outperforms powerful closed-source models such as GPT-5 and Gemini-2.5 Pro, which is impressive.
2. The paper is well written, with clear motivation and problem formulation, and provides a good example (Figure 1) illustrating the limitation of existing approaches and the EoG methods.
3. Smaller open-source LLM trained by EoG can compete with larger closed-source ones, which means EoG addresses some of the current compute resource limitations.
4. Well-organized experiment, ablation study effectively demonstrates the importance of each component.

### Weaknesses
1. Limited technical novelty: The core contribution combines existing techniques (SFT, GRPO, simple reward design) without significant algorithmic innovation. The path reward is particularly simplistic, using only substring matching. In the area of KG reasoning, using reinforcement learning to explore path is a general and common practice. Check MINERVA [1] and DeepPath [2].

2. The reliance on Gemini 2.5 Flash for dataset generation creates a dependency that may limit reproducibility. The paper doesn't discuss alternatives or provide the generated datasets.

3. More than half of the baseline experimental results in Table 1 are missing, and the authors do not provide any explanation in the paper. I do not think that the effectiveness of EoG can be proven based on Table 1 alone, as the baseline results are extremely limited.

4. The paper lacks a discussion of training costs, convergence time, and computational requirements for the RL stage.  As the paper says it trains EoG on 8*H100 GPUs, which is very costly. This may be a strong limitation of the proposed method. 

5. The substring-based matching in Equation 4 is brittle and may not capture semantic equivalence, paraphrases, or partial correctness in reasoning paths.


[1] Das, Rajarshi, et al. "Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning." International Conference on Learning Representations. 2018.
[2] Xiong, Wenhan, Thien Hoang, and William Yang Wang. "DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017.

### Questions
1. What are the computational costs of the RL training stage compared to standard supervised fine-tuning? How many GPU hours are required for convergence?
2. Have you considered more sophisticated path reward designs that account for semantic similarity rather than exact substring matching? For example, using graph edit distance or learned similarity metrics?
3. Can you provide the generated CoT datasets or detailed statistics about them to enable reproducibility without access to Gemini 2.5 Flash?
4. How sensitive is the method to the quality of the initial SFT stage? What happens if a weaker teacher model is used for data generation?
5. Have you analyzed failure cases where exploration leads to incorrect paths? How does the model handle ambiguous questions where multiple valid reasoning paths exist?
6. Could you provide theoretical analysis or empirical evidence about the exploration efficiency and coverage of the reasoning space? For example, to improve the content of case study?
7. Why are so many experimental results missing in Table 1? (I don’t think “the original paper did not report the experimental results of this dataset” is a good answer)

### Soundness
2

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
The paper proposes Explore on Graph (EoG), where not only the correct answer but also the reasoning path is rewarded via reinforcement learning, enabling the model to explore novel reasoning paths that fall outside the distribution of pre-defined rules or supervised fine-tuning data.

The experiments are comprehensive and show strong results on five KGQA datasets, outperforming not only open-source but also even closed-source LLMs.

### Strengths
1. The motivation is well grounded. Not only the answer but also the reasoning path can server as good reward signals.
2. The experiments show strong results. For example, table 1 show EoG outperforms not only open-source but also even closed-source LLMs, and table 4 show strong results on OOD settings. The improvement is significant.

### Weaknesses
1. Some implementation details are not clear. Are phase 1: Outcome Reward Modeling and phase 2: Path-refined Reward Modeling implemented sequentially or simultaneously (as in Equation 5)?
2. Reproducibility: The code is currently unavailable, which hinders verification, reproduction, and improvement efforts. Open-source code is crucial for these processes.

### Questions
1. Same as Weakness 1: Are phase 1: Outcome Reward Modeling and phase 2: Path-refined Reward Modeling implemented sequentially or simultaneously (as in Equation 5)?
2. For the OOD experiment (Figure 5), what do the x and y axes represent? Is the model trained on the dataset on the x-axis and then evaluated on the dataset on the y-axis, or vice versa?
3. For the OOD experiment (Figure 5), what is the performance of other models in the OOD settings? It would be useful to compare EoG not only with EoG-SFT but also with other SOTA models (as in Table 1).
4. Some things can be improved for clarity and readability. For example, in Figure 3, the meanings of the x and y axes are only available in the main body text. The reading experience would be greatly enhanced if the x and y axes were directly labeled in the figure.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes **Explore-on-Graph (EoG)**, a KG-augmented LLM reasoning framework that couples (i) supervised fine-tuning on long chain-of-thought traces and (ii) reinforcement learning with a **path-refined reward**. The core idea is to *incentivize autonomous exploration* of novel multi-hop paths on knowledge graphs, improving generalization beyond rule/imitation patterns. Concretely, the RL stage optimizes a GRPO-style objective using a final-answer **outcome reward** (entity-level F1 extracted from the `<answer>` tag) and an auxiliary **path reward** that measures how many ground-truth triples appear in the model’s `<think>` text; a weighted sum defines the joint reward. Experiments on **WebQSP, CWQ, GrailQA, QALD-10-en, and 2WikiMultihop** show consistent gains over recent KG-enhanced systems, and ablations indicate the path-refined reward materially contributes beyond SFT alone.

### Strengths
* **Clear motivation & problem framing.** The paper articulates why rule/imitation approaches struggle on OOD patterns and positions exploration as the missing capability. Figure 1 illustrates this vividly. 
* **Method is simple, modular, and reproducible in principle.** The rewards (answer F1; path triple-match ratio) are transparent and plug into a standard GRPO objective with KL control.  
* **Strong empirical results across diverse KGQA datasets** with consistent gains vs. strong baselines; ablations show each component’s contribution and explore the α trade-off between outcome and path rewards. 
* **Ablations that are decision-useful.** Removing SFT, outcome, or path rewards degrades performance in expected ways, supporting the design choices. (Table 2 & ratio analyses in the text.)

### Weaknesses
1. **Potential reward gaming / verification gap.** The **path reward** credits substring co-occurrence of `(subject, relation, object)` tokens in `<think>` text rather than **verified KG traversals**. This leaves room for *verbalization without execution* (i.e., asserting triples to earn reward). The paper should either (a) execute the predicted path against the KG to produce a structural match reward, or (b) at least audit hallucinated triples vs. KG edges. 
2. **LLM-judge reliance for qualitative criteria.** The analysis of comprehensiveness/relevance/exploration uses **GPT-4o-mini** as judge; such measures can be noisy and model-biased. Human evaluation or KG-grounded automatic proxies would strengthen claims. 
3. **Comparisons to closed-source LLMs** (Gemini 2.5, “GPT-5”) are intriguing but ambiguous: API prompting details, temperature, and decoding budgets can shift outcomes; moreover, the “GPT-5” reference seems tenuous. I recommend focusing the SOTA claim on open-source or adding stricter evaluation parity. 
4. **Statistical reporting.** Tables omit **confidence intervals/standard errors** and **significance testing**, especially important across multi-seed RL runs. This weakens the strength-of-evidence for SOTA claims. (No CIs in Table 1.) 
5. **Scope of robustness/OOD probes.** While the paper argues improved OOD generalization, it would help to (i) include **systematic stress tests** (entity aliasing, edge deletions, spurious edges), and (ii) evaluate **path length sensitivity** with explicit adversarial splits beyond the included subsets.

### Questions
1. **Path reward verification.** Can the authors compute the path reward by **parsing the predicted path** into a sequence of relations and **executing it** on the KG to verify edge existence (vs. substring matching)? If not feasible, can they report a *hallucination rate* of triples mentioned in `<think>` but absent in the KG? 
2. **Robustness to reward hacking.** Did the authors observe behaviors where the model *lists many unrelated triples* to improve match probability? Any safeguards (length penalty, entropy regularization, path-structure constraints)?
3. **Ablation on α and stability.** Figure discussing α suggests a sweet spot. Please include **training curves** and variance over **≥3 seeds** for each α to assess RL stability. (Also report GRPO hyperparameters.) 
4. **Closed-source comparison protocol.** For Gemini/GPT, please provide **identical decoding parameters**, prompt templates, and **budget parity** (n-samples, context length), or move these to an appendix with full reproducibility details. 
5. **KG execution failure modes.** In cases where KG lacks labels/edges (Figure 6-style), how often does EoG succeed via exploration vs. spurious textual correlations? Any breakdown by hop length and relation sparsity?
6. **Significance & compute.** Please add **CIs** for Table 1 and **report training compute** (SFT tokens, RL steps, batch, GPU hours) to contextualize the gains.

### Soundness
3

### Presentation
3

### Contribution
2
