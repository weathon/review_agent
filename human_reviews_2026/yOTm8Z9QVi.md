# Expert-Integrated Active Learning for Optimizing LLM Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Recent advances in Large Language Models (LLMs) have created new opportunities for their application in interactive environments. However, these agentic tasks present significant challenges due to the complexity of long and specialized interaction trajectories that are underrepresented in standard training distributions. While Reinforcement Learning (RL) post-training offers a promising approach to mitigate the need for extensive human-annotated data, it faces fundamental limitations in exploration efficiency when applied to LLMs. In this paper, we introduce a novel framework that synergistically combines RL post-training with Active Learning (AL) for LLM agents. By choosing informative tasks with reward-based filter and diversity-based selection criteria, our approach enables models to not only refine their capabilities through autonomous exploration but also strategically request expert demonstrations for challenging scenarios, thereby extending their exploration boundaries. We demonstrate the efficacy of this method on the AppWorld benchmark, showing significant performance improvements with minimal expert demonstrations. We then further look into adapting our framework for different budget and examine the factors that affect the final performance. Our method highlights the potential of efficiently integrating human resources within RL pipelines to enhance LLM agents' capabilities in complex interactive environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
```
I used LLM to fix the grammar of the Official Review, but all opinions are my own
```
Collecting expert data is challenging, which is why reinforcement learning (RL) has recently become popular. However, combining large language models (LLMs) with RL introduces several problems. One issue is that LLMs operate over a massive token vocabulary, where each token corresponds to a distinct, environment-specific action. This leads to an enormous exploration space in which only a sparse subset of tokens are actually meaningful. Another problem is that pretraining constrains the model’s exploration capability. To address this, the authors propose leveraging active learning to enhance exploration efficiency: the model first explores freely, and then the system filters for high-quality samples, which are subsequently annotated by experts for further learning. The filtering focuses on selecting challenging samples. While the idea itself is interesting, the paper’s execution appears incomplete. Many parts of the methodology are unclear, and the experiments fail to convincingly demonstrate the method’s general applicability. My current inclination is to reject, though the final decision could depend on the quality of the rebuttal.

### Strengths
Combining active learning with RL to guide expert labeling is interesting and potentially valuable.

### Weaknesses
1. I am not convinced of the generalizability of the proposed approach.
2. The paper’s exposition is too brief and lacks clarity, making it difficult to understand the key implementation details.

### Questions
1. Your selection rule prioritizes difficult samples and those showing little progress, but does this only work for environments that provide intermediate rewards? For benchmarks like HLE or SWE-Bench that lack intermediate signals, how would your method still apply?

2. The “expert” component seems crucial, but the paper’s description is very vague. Are these experts humans, or stronger models? Have you tested different types of experts for comparison? Could you provide concrete examples of expert demonstrations? Without examples, it’s hard to understand the actual procedure.

3. If the experts are humans, the cost would be prohibitively high. Could the process instead leverage stronger models as experts? More generally, how could you design a multi-agent RL setup with active learning using strong model-based experts?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework that integrates AL and RL for training LLM–based agents in interactive environments. The authors identify the key limitation of standard RL post-training—inefficient exploration—and address it by introducing a mechanism through which the model can actively query for expert demonstrations. Specifically, the proposed system employs a reward-based filter and diversity-based task selection to identify challenging and informative tasks for expert annotation, which are then incorporated into training through a carefully designed mixing strategy. Experiments on the AppWorld benchmark show that this approach achieves notable improvements in task success rates over RL-only baselines, while requiring significantly fewer expert demonstrations compared to full supervision. The authors also provide comprehensive ablations on budget control, task diversity, and trajectory analysis, highlighting how expert data improves exploration efficiency and consistent behavioral patterns.

### Strengths
1. The paper presents a well-motivated approach that bridges AL with RL-based post-training. The proposed framework effectively addresses the long-standing challenge of balancing exploration efficiency and annotation cost in agentic training.
2. The framework is clearly explained, including detailed algorithmic steps for reward-based filtering, diversity-based selection, and the trajectory mixing strategy. The proposed mechanisms are conceptually sound and practically relevant.
3. The authors provide extensive experiments on AppWorld, demonstrating consistent improvements under multiple model scales (7B and 14B). The inclusion of ablation studies on similarity thresholds, early stopping, and demonstration efficiency adds strong empirical support.

### Weaknesses
1. The proposed method still requires non-trivial manual effort. Although active selection reduces the annotation cost, the framework's practicality in large-scale real-world deployment remains uncertain. It would be beneficial to discuss possible automation or self-improvement mechanisms to reduce expert reliance.
2. The experiments are conducted solely on AppWorld. While this benchmark is well-suited for interactive environments, validation on additional settings (e.g., WebShop, OSWorld, or τ-bench) would strengthen the claim of general applicability.
3. The paper uses synthetic expert data (generated by DeepSeek-V3.1) instead of real human annotations. This design choice raises questions about the robustness of the findings in true human-in-the-loop scenarios.
4. The framework involves repeated task selection, similarity computation, and trajectory mixing. The paper does not provide detailed analysis of computational overhead or memory usage, which would be important for evaluating its scalability to larger models or environments.

### Questions
Please check my comments in Weaknesses.

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
3

### Summary
This paper proposes an expert-integrated active learning (AL) framework to optimize LLM agents through GRPO-based RL post-training. The core approach involves a two-stage task selection process: (1) reward-based filtering to identify tasks with persistent failure and stagnant returns; (2) diversity-based max-min selection to eliminate redundancy. Subsequently, expert demonstrations are requested only for these tasks. Training employs a hybrid policy that blends high-quality expert trajectories with model self-sampled trajectories at a constrained ratio, leveraging expert knowledge while preserving exploration capabilities.

### Strengths
## Strengths

1. **Originality**
- The paper integrates active selection of tasks with expert demonstrations into an on-policy GRPO training loop for agents—moving beyond passive offline demo mixing; the reward-stagnation + success-rate filter is a simple, RL-appropriate proxy for uncertainty.
- The diversity-based max-min selection with a historical buffer is a practical twist that avoids repeatedly annotating near-duplicates across steps.
2. **Quality**
- Under matched budgets and consistent evaluation protocols, the method demonstrates stable and statistically significant performance gains relative to baselines, reflecting its empirical quality.
- Not only did the report show overall improvement, but it also presented detailed metrics analysis such as demonstration utilization rate and reuse frequency (efficiency), revealing the phenomenon that "a small number of high-value demonstrations are repeatedly utilized." This aligns with the expectation that active learning reduces annotation overhead.

3. **Clarity**
-  The paper is well-organized, with a coherent narrative from motivation to validation, making its technical and empirical contributions easy to follow.

4. **Significance**
- The paper addresses a real bottleneck for agent RL—inefficient exploration and annotation cost—and shows measurable gains on a recognized agentic benchmark with reduced cost


---

### Weaknesses
## Weaknesses
1. Lack of side-by-side comparison with existing AppWorld proxy methods: The experiment only compared the author's own three settings (GRPO/baseline, full demo, active learning) without providing direct numerical comparisons against representative methods publicly reported on AppWorld.
2. Figures and tables are not self-contained: beyond titles, they lack descriptive captions specifying the evaluation setup, metric definitions/units,  and significance markers. This weakens clarity and makes it hard to interpret results at a glance.



---

### Questions
## Questions
1. Would you add head-to-head comparisons against representative AppWorld agents reported in prior work (e.g., recent public baselines), under the same evaluation protocol? 
2. In the main results table, why are Test-set results for the expert model (DeepSeek-V3.1)—which supplies demonstrations—omitted?

### Soundness
4

### Presentation
4

### Contribution
4
