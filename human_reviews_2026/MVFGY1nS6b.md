# Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4

## Abstract
Large Language Model (LLM)-based agents have emerged as a transformative approach for open-ended problem solving, with information seeking (IS) being a core capability that enables autonomous reasoning and decision-making.  While prior research has largely focused on improving retrieval depth, we observe that current IS agents often suffer from \textit{low search efficiency}, which in turn constrains overall performance. A key factor underlying this inefficiency is the sparsity of target entities in training tasks, which limits opportunities for agents to learn and generalize efficient search behaviors. To address these challenges, we propose WebLeaper, a framework for constructing high-coverage IS tasks and generating efficient solution trajectories. We formulate IS as a tree-structured reasoning problem, enabling a substantially larger set of target entities to be embedded within a constrained context. Leveraging curated Wikipedia tables, we propose three variants for synthesizing IS tasks—Basic, Union, and Reverse-Union—to systematically increase both IS efficiency and effectiveness. Finally, we curate training trajectories by retaining only those that are simultaneously accurate and efficient, ensuring that the model is optimized for both correctness and search performance. Extensive experiments conducted on five IS benchmarks—BrowserComp, GAIA, Seal-0, WideSearch, and xbench-DeepSearch—demonstrate that our method consistently achieves improvements in both effectiveness and efficiency over strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces **WebLeaper**, a framework to enhance **efficiency and efficacy** of LLM-based information-seeking (IS) agents. It reformulates IS as a **tree-structured reasoning** task, addressing entity sparsity in existing datasets. Three task variants—**Basic**, **Union**, and **Reverse-Union**—progressively increase reasoning depth and realism.
To ensure learning from high-quality data, trajectories are filtered via **Information-Seeking Rate (ISR)** and **Information-Seeking Efficiency (ISE)** metrics, balancing correctness and search economy.
Experiments on **BrowserComp, GAIA, Seal-0, WideSearch, and xbench-DeepSearch** demonstrate consistent improvements in both accuracy and efficiency .

### Strengths
* Addresses a neglected but critical dimension—efficiency;
* Clear theoretical justification and measurable impact;
* Well-designed ablation studies and visualization;
* Comprehensive benchmark evaluation with strong improvements;
* Highly reproducible (data construction and algorithmic details disclosed).

### Weaknesses
* Lack of hyperparameter sensitivity analysis for α and β;
* Dataset bias (Wikipedia-only) not discussed;
* No multilingual or real-world deployment tests;
* Missing analysis on computational training overhead.

### Questions
* Can ISR/ISE thresholds adapt dynamically during training?
* Does Reverse-Union risk overfitting due to fuzzy clue anchoring?
* Could integration with knowledge graphs further enhance reasoning coverage?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents WebLeaper, a framework aimed at enhancing the efficiency and effectiveness of information-seeking (IS) agents based on large language models (LLMs). The authors identify the problem of low search efficiency in current IS agents, attributing it to the sparsity of target entities in training tasks. To address this issue, they propose a novel approach that constructs high-coverage IS tasks using a tree-structured reasoning model, allowing for a greater number of target entities within a limited context. The framework includes three dataset variants—Basic, Union, and Reverse-Union—to systematically increase task complexity. Additionally, the authors curate training trajectories based on the Information-Seeking Rate (ISR) and Information-Seeking Efficiency (ISE) to ensure that the model is optimized for both accuracy and efficiency. Extensive experiments on five IS benchmarks demonstrate that WebLeaper consistently outperforms strong baselines, validating the effectiveness of the proposed method.

### Strengths
- The paper addresses an important issue in most current LLM-based information-seeking agents, specifically the problem of low search efficiency. The focus on efficiency is a valuable contribution to the field, as it complements existing efforts that primarily target search depth.
- The proposed tree-structured reasoning framework can support more comprehensive IS tasks through more structured trajectories, which may lead to better learning of search strategies.
- The use of ISR and ISE metrics to curate training trajectories is a valuable contribution that ensures the model is trained on high-quality data.

### Weaknesses
- The paper claims to present an information-seeking agent; however, the agent mainly focuses on finding target entities via web search given artificially constructed complex questions. This is more akin to entity mining tasks rather than general information-seeking tasks. The authors should better clarify the definition of an information-seeking agent in this work and discuss the limitations of the proposed method in broader information-seeking scenarios—particularly, how the three proposed dataset variants can help the agent seek information beyond entity finding.
- The proposed tree-structured reasoning framework is interesting, but the paper lacks a detailed analysis of how this structure is advantageous compared to other possible structures, such as graphs. The authors also overlook the potential limitations of tree structures in capturing complex relationships among entities.
- Since the proposed method mainly focuses on synthetic dataset construction and trajectory filtering, it may not generalize well to real-world applications. For example, web content is often noisy, and retrieved documents may lack crucial clues or entities. This suggests that the proposed method might not perform well in practical scenarios. The authors should discuss these limitations in more depth and provide analyses or experiments to verify the robustness of their method under more realistic conditions.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the critical inefficiency of IS agents, identifying "entity sparsity" as a key bottleneck. It proposes two metrics ISE and ISR to quantify the IS efficiency, and also shows that the variance of ISE is negative correlated to target entities. The authors propose WebLeaper  to automatically synthesize entity intensive training tasks from wiki tables. It models IS as a tree-structured reasoning problem. The framework increases task complexity through three variants: basic, union and reserse-union. WebLeaper also generates solution trajectories and curates them using ISE and ISR to ensure efficiency. Experiments over challenging QA tasks demonstrate that WebLeaper-trained agents significantly outperform open source baselines, achieving both higher accuracy and greater efficiency.

### Strengths
1. The paper highlights the low efficiency of the current IS agents, providing evidence that most of actions are often invalid. It formally defines two metrics, ISR and ISE to quantify this problem. The authors provide Proposition 1 that the variance of the ISE metric decreases as the number of target entities n grows.
2. The paper proposes an innovative tree-based pipeline, WebLeaper, to generate entity-intensive training data from Wiki tables. This method systematically increases task complexity through three variants (Basic, Union, and Reverse-Union). The framework also curates solution trajectories by filtering for high ISR and ISE, ensuring the agent learns from optimal, efficient examples.
3. WebLeaper demonstrate better performance in challenging QA tasks comparing to the open source IS agents.

### Weaknesses
1. The WebLeaper is finetuned on a single base model (Qwen3-30B-A3B-Thinking-2507). The observations may change with a different base model.
2. The ablation study could include the analysis of how the average action rounds change with different data sources (similar to table 2).

### Questions
It would be helpful to run the experiments mentioned in weakness part. 
1. finetune over a different base model
2. enrich ablation study

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes WebLeaper, a data-centric framework for improving the efficiency of web-based information-seeking agents. The authors construct dense, entity-rich tasks from Wikipedia tables (Basic, Union, Reverse-Union), generate ReAct trajectories using a strong model, and filter these trajectories with two empirically motivated metrics: Information Seeking Rate (ISR) and Information Seeking Efficiency (ISE). The filtered trajectories are used for supervised fine-tuning. Experiments on five benchmarks show solid and consistent improvements in both accuracy and efficiency, supported by ablations.

### Strengths
1. Novel task synthesis channel using structured Wikipedia tables to create dense, multi-entity information-seeking tasks.
2. Empirically grounded filtering strategy (ISR/ISE) that directly addresses observed inefficiencies in existing agents.
3. Strong empirical results across five benchmarks, with clear gains and well-supported ablations.

### Weaknesses
1. Limited insight into the underlying mechanism of why the efficiency-oriented filtering leads to such large improvements; the explanation remains at an engineering level.
2. The claim that long trajectories are inefficient conflicts with recent RL-based reasoning advances, where longer, adaptive chains often improve quality.
3. The ISE filtering may be too rigid, potentially suppressing necessary complex or difficulty-adaptive reasoning (e.g., long-form CoT or AIME-like tasks).

### Questions
1. Why does efficiency-based filtering have such a large impact?
It is unclear whether the gains come from richer sub-queries, reduced long-context degradation, or other behavioral shifts (eg, more focused planning).
2. How does this reconcile with RL models that benefit from long chains?
If shorter, cleaner trajectories are key here, why do math/reasoning systems still require RL-driven long CoT (eg, AIME-style problems)?
3. Is strict efficiency optimal for tasks requiring exploration?
The ISE constraint may suppress necessary divergent reasoning (eg, adaptive exploration), and it is unclear whether the model loses performance on complex tasks such as AIME24/25.

I would be happy to engage with the authors during the rebuttal phase regarding these concerns, and I am open to revising my score should the responses address them satisfactorily.

### Soundness
3

### Presentation
3

### Contribution
2
