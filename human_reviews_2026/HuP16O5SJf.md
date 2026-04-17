# WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
To significantly advance the capabilities of open-source web agents, we present WebSailor-V2, a complete post-training pipeline encompassing data construction, Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL). Our methodology features two key innovations: (1) On the data front, we developed SailorFog-QA-2, a novel dataset built from a densely interconnected knowledge graph that introduces a wide variety of uncertainties beyond simple obfuscation, fostering more sophisticated reasoning. (2) For training, we engineered a dual-environment RL framework, combining a high-fidelity simulator for rapid, low-cost algorithmic iteration with a robust, managed real-world environment for stable final policy training, all integrated within a symbiotic data-policy feedback loop. Trained on the Qwen3-30B-A3B model, WebSailor-V2 achieves state-of-the-art results, scoring 35.3 on BrowseComp-EN, 44.1 on BrowseComp-ZH, and 30.6 on Humanity's Last Exam (HLE). Notably, our 30B-A3B MOE agent significantly outperforms all existing open-source agents and surpasses even the 671B DeepSeek-V3.1, demonstrating performance competitive with leading proprietary systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents WebSailor-V2, a post-training framework for pre-trained LLMs designed to develop agentic capabilities by integrating reasoning and tool use.
The pipeline comprises two stages: supervised fine-tuning and reinforcement learning.
Training data are derived from a densely constructed knowledge graph, from which a novel QA dataset is sampled.
For RL training, the authors develop a simulation environment that enables efficient experimentation by approximating real-world interactions, which are inherently slow and fragile.
The final training stage is executed in real-world conditions using a custom tool-calling API.
Empirical results show that WebSailor-V2 surpasses both state-of-the-art open-source and most proprietary models on benchmarks assessing agentic reasoning.
However, the paper provides insufficient methodological details, limiting reproducibility and interpretability of the proposed techniques.
Furthermore, many claims are presented without supporting evaluations, reducing the evidential strength of the conclusions.
Overall, the work is better characterized as a technical report rather than a scientific paper, as critical details are unclear and unsupported claims abound, making replication highly challenging.

### Strengths
Next to the actual weights (if made publicly available, not mentioned in the paper), the main contributions of this paper consist of the vaguely described data-set creation and agentic research simulation environment. The paper itself is clearly written and figures are neat and polished. 
If correct and not biased, the results are very impressive. While this experimental report has nothing to do at a **scientific conference**, the effort of the authors to improve open source SOTA is of great value for any non-scientific user.

### Weaknesses
The Core Issue: The paper presents itself as an engineering or experimental report (i.e., "we built a system that achieves X performance") rather than a scientific paper that answers a fundamental research question. It lacks a clearly defined scientific problem (e.g., "To what degree does data structure, versus data scale, impact an agent's reasoning?"). This focus on the final artifact, rather than the underlying principles, makes it difficult for the scientific community to extract deep understanding, generalizable intuitions, or understand why this method is better than baselines. To address this main drawback, I advise the authors to:
* Reframe the Contribution: The paper should be reframed around a central scientific question, not just a final performance score.

* Provide Rigorous Ablation Studies: The paper proposes several new components at once (Agentic CPT, graph-based synthetic data, scalable RL). It is impossible to know which part contributed most to the performance. The paper must include ablations to isolate the impact of each component on the improvement.

Another core weakness of this paper is the vague descriptions of its key contributions. Overall, the clarity could be improved tremendously by: elaborating more detailed how the knowledge graph is built, how the data is sampled from the graph, how exactly the simulated environment works, how the reward is computed, etc.

Additionally, there are no attempts made at making the method reproducible.

Lastly, throughout the text is a number of claims that were not shown empirically, e.g
* Page 2, uncertainty in QA: “We also expand the diversity of our QA generation by incorporating a wider variety of uncertainty definitions beyond obfuscation, directly targeting the need for more sophisticated reasoning”
Where are those uncertainty definitions?
* Page 2, considering the simulated environment: “Through meticulous design, it achieves high fidelity, ensuring that the agent’s interaction dynamics, state transitions, and reward mechanisms closely mirror those of a real-world setting”
Never shown.
* Page 4, about the graph subsampling: “Ultimately, this strategy enables us to efficiently gather a sufficient quantity of non-isomorphic connected subgraphs that collectively represent the full spectrum of structural complexities, without the prohibitive cost of a brute-force search.”
Never shown, (and "non-isomorphic nodes" is never defined).
* Page 4, QA generation: “To this end, we introduce a wider array of defined uncertainties, aiming to elicit a more diverse and comprehensive suite of advanced reasoning abilities from the model” 
It’s never tested if the reasoning abilities are actually more diverse or comprehensive
* Page 5, Data Curation: “ The quality of the training data directly establishes the upper bound for the model’s ability to generalize to out-of-distribution scenarios through self-exploration.”
Has this been shown somewhere? 

Finally, while the figures are neat and the text clear. I'd advise the two following points to improve the clarity, but also improve the scientific accuracy of the paper:
Clearly include:
I. A list of contributions at the end of the introduction (often denoted with i., ii., ...) that stands out visually.
II. A list of scientific research questions at the beginning of the experimental evaluation section (often denoted RQ1, RQ2, ... etc), that are each answered in different paragraphs. For example
* (RQ1) Can our training pipeline help open source agents outperform existing research agents?
* (RQ2) Can the graph creations help external viewers to verify the integrity of the provided results?
* (RQ3) How important is component A ? (conducting an ablation study)
* ...

The figures and tables can thus help the readers understanding the addressed problem, the method, and the experimental evaluation during their first pass over the paper. They should thus be structuring the paper like, e.g.:
1. Figure 1 (describing the problem, if the problem is not obvious),
2. Figure 2 (describing the method),
3. Figures and Tables describing each important results (that answers precise scientific questions mentioned above). 

Further, each caption should make its figure a standalone component of the paper. Thus, each figure should be built in the following way:

The first sentence should highlight the main message of the Figure/Table (e.g. "Our method outperforms the existing SOTA methods on the studied problem.")

The next sentences then explain what is depicted in the Table/Figure. E.g. Mean test accuracy, on 5 seeded trainings, with std. 

Finally, details and references to e.g. appendix can be provided if necessary. E.g. Our method outperforms baseline 1 in 3 out of 4 tasks, ... etc.

### Questions
* Will you publish your code for the complete pipeline, incl. the building of the knowledge graph, sampling from it, the simulated training environment, the reward model, the tool-calling API, the training loop itself?
* How do you ensure that the simulated environment closely resembles the real world?
* Will you make the SailorFog-QA-2 dataset publicly available?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents WebSailor-V2, a complete post-training pipeline (SFT + RL) designed to bridge the performance gap between open-source and proprietary web agents.

### Strengths
1. Motivated Data Construction: The paper identifies a clear weakness in existing agent datasets (acyclic, tree-like structures and over-reliance on simple obfuscation). The proposed SailorFog-QA-V2, with its densely interconnected, cyclic graph and "wider array of uncertainties," is a strong and novel contribution that better reflects the complexity of real-world information.

2. The dual-environment RL approach is a very practical and well-executed solution to a major bottleneck in agent training. The high-fidelity simulator (using an offline Wikipedia dump) allows for rapid, low-cost algorithmic iteration, while the robust real-world environment, featuring a unified tool interface with fault tolerance and retry mechanisms, insulates the final training from the latency and failures of live APIs. This is a significant engineering contribution that enables stable RL.

3. The results are impressive and clearly demonstrate the efficacy of the proposed pipeline. The fact that the 30B WebSailor-V2 significantly outperforms all other open-source agents and even much larger models like the 671B DeepSeek-V3.1 validates the hypothesis that superior data and training strategies can overcome raw model scale.

### Weaknesses
Could the authors elaborate on the specific types of "uncertainties" (beyond obfuscation) that were introduced in the SailorFog-QA-V2 dataset? Providing a few examples would greatly clarify this contribution.

### Questions
See weakness.

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
4

### Summary
The paper introduces Sailor-Fog-QA-V2, a refreshed data generator that builds denser, cyclic knowledge graphs and samples non-isomorphic subgraphs via random-walks, plus a dual-environment RL setup (offline Wikipedia simulator + robust real-web tools stack). With SFT on synthetic trajectories from V2 and a GRPO-style RL phase, a Qwen3-30B-A3B agent delivers SOTA among open-source and competitive vs proprietary systems on BrowseComp-EN/ZH, xbench-DeepSearch, GAIA, HLE, and DeepResearch Bench (e.g., 35.3 on BrowseComp-EN, 44.1 on BrowseComp-ZH, 30.6 on HLE).

### Strengths
* Sailor-Fog-QA-V2 deliberately creates cyclic, densely connected graphs (vs tree-like) and preserves procedural metadata (queries, URLs, stats) that supports richer QA targeting across orbit nodes. Subgraphs are random-walk sampled and WL-checked for non-isomorphism. 
* A Wikipedia-based simulator enables fast, low-cost iteration, whereas the real environment adds a unified tool interface with QPS controls, caching, retry/timeout, and fallback switching, which directly addresses brittleness in web-tool rollouts. 
* The 30B-A3B MoE agent surpasses prior open-source, narrows the gap with leading proprietary systems, and even beats DeepSeek-V3.1 (671B) on HLE, showing that data and stability matter as much as model scale.

### Weaknesses
* The paper states it goes “beyond obfuscation” but never enumerates the additional uncertainty types or how they’re realized during QA generation (only a general description of obfuscation appears). This makes it hard to attribute gains to the proposed “wider variety.” 
* The dual-environment design is described, but how rollouts are scheduled/allocated between simulator and real web (by phase, by task type, by curriculum, or adaptively from reward signals) is not specified. Figure 2 shows the loop but not the selection strategy.  
* Training uses Sailor-Fog-QA (v1), V2, and IterBench, yet there’s no quantitative ablation showing the marginal benefit of each component on the benchmarks.

### Questions
1. Please list and define the uncertainty categories added beyond obfuscation, with 1–2 concrete generation templates per type. Where in the pipeline are they injected? 
2. Are RL rollouts taken in both simulator and real web? If yes, what mix over training (e.g., % rollouts or steps per phase), and do you adapt the mix based on validation reward or environment-specific success rates? 
3.  You drop some negative trajectories to avoid format collapse. Which categories are excluded beyond “no final answer due to length,” and how sensitive is performance to this filter?

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
The paper presents WebSailor-V2, an open-source web agent trained through a multi-stage post-training pipeline. The approach combines a dataset generated from a knowledge-graph-based process intended to cover diverse forms of uncertainty, supervised fine-tuning on synthetic demonstrations, and reinforcement learning within a dual-environment setup, consisting of a Wikipedia-based simulated environment and a managed real-world environment with tool-execution interfaces. The model is based on Qwen3 and uses 128k-token contexts. The RL stage employs a GRPO-style algorithm with token-level updates, leave-one-out advantages, on-policy sampling, and filtering of negative trajectories. The paper reports that this combination improves performance relative to SFT-only baselines and narrows the gap between open-source and proprietary web agents.

### Strengths
- Clear description of a complete training pipeline.
- Reported benchmark gains on BrowseComp and HLE relative to earlier open models.
- RL procedure and stability considerations are explicitly stated.
- Diagnostic plots provide visibility into training behavior.

### Weaknesses
- The paper provides no quantitative validation that the simulated environment approximates real-world web interactions.
-  The paper does not report SailorFog statistics such as number of QA pairs, node/edge counts, or category distributions.
- The effect of individual components (dataset, SFT, RL) and RL design choices is not shown beyond one aggregated comparison.
- Reported comparisons to proprietary systems use previously published numbers without matching evaluation conditions (context length, sampling, tool limits).
- Accuracy is measured with an LLM-based judge and the paper does not provide calibration or human-agreement checks.
- It is not explicitly stated whether code, data, and simulator will be released concurrently with publication.

### Questions
- What are the quantitative properties of SailorFog (size, graph statistics, uncertainty-type counts)?
- How is simulator realism assessed? Are there any metrics comparing simulated and real tool outputs?
- Which parts of the performance gain come from SFT data versus RL fine-tuning?
- Were all benchmark runs conducted under identical context lengths and tool budgets?
- How consistent is the LLM-as-judge evaluation, has human validation been performed?
- Will the dataset, simulator, and training code be publicly released?

### Soundness
3

### Presentation
3

### Contribution
3
