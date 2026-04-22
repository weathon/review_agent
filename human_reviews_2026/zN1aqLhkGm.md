# Principle Process Reward For Search Agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Large Language Models (LLMs) increasingly rely on external tools such as search engines to solve complex agentic tasks that require reasoning and external knowledge retrieval. Recently, reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in advancing capabilities of LLMs by rewarding the final answers via outcome rewards. While straightforward to supervise, outcome rewards only provide sparse signals and delayed feedback, which limits their effectiveness on long trajectories. Process rewards address this by evaluating intermediate steps, providing fine-grained supervision and encouraging grounded problem solving. However, it is notoriously hard to annotate step-wise labels, especially in non-verifiable process without "golden" answers. Furthermore, step-wise judgment requires the balance between local quality with contribution to the final outcome, as optimizing towards higher process reward may not always align with better final outcomes. To address the above challenges, we introduce Principle Process Reward (PPR), an RL approach that unifies principled step-level assessment and outcome verification. We train a principle-based reward model to improve the transparency and reliability of process evaluation, and further introduce a Reward Normalization (ReNorm) strategy to calibrate outcome and process rewards. Experiment results show that PPR achieves state-of-the-art performance across a wide range of benchmarks, demonstrating its impressive robustness and generalization. Code, data and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Principle Process Reward (PPR), a RL framework for LLM-based search agents that combines outcome and step-level supervision.
It introduces (1) a Principle-based Process Reward Model (PPRM) that evaluates intermediate steps via interpretable principles and (2) a Reward Normalization (ReNorm) method that aligns process and outcome rewards for stable training. Their method is evaluated on multiple QA and multi-hop benchmarks with Qwen2.5 models. PPR achieves up to 28% gains over non-RL baselines and 15% over RL counterparts, showing improved stability and generalization. The authors also released NVProcessBench, a benchmark for non-verifiable process evaluation.


The paper is clear, well-organized, and technically sound. It presents strong empirical results and thoughtful ablations. Its main contribution, combining principle-based process rewards with reward normalization, offers a practical and interpretable improvement to RL-based LLM training. However, the idea builds on prior PRM work. The scope is limited to search-based QA, and limitations are only discussed implicitly.

### Strengths
+ The integration of principle-based process evaluation with outcome reward normalization is conceptually clear and empirically effective.

+ Extensive comparisons against baselines show consistent and significant gains across both in-domain and out-of-domain QA tasks.

+ NVProcessBench is a meaningful step toward systematic evaluation of reward models for non-verifiable reasoning steps.

+ The paper is generally well written, with clear motivation, architecture diagrams, and quantitative evidence.

### Weaknesses
+ While ReNorm is intuitively and empirically justified, its theoretical analysis in Appendix A.9 is limited to simple assumptions and does not fully explain stability in more general cases.

+ All evaluations are in search-based QA. It remains unclear whether the approach generalizes to other non-verifiable agentic tasks such as code generation, planning, or multi-tool reasoning.

+ PPRM relies on a fine-tuned 8B model; cost and scalability implications are not deeply analyzed.

+ Methods such as reward rescaling, entropy regularization, or variance reduction are briefly mentioned but not compared much.

+ NVProcessBench is built partly using GPT-based annotation, which may introduce bias; details on annotation validation are limited. How do you address this ?

+ The figures are difficult to read due to dense layouts and low-contrast color schemes; improving visual clarity would make the results and framework easier to interpret.

+ The paper acknowledges some limitations. However, it does not explicitly discuss broader constraints, such as limited domain coverage, scalability to other agentic tasks, or computational overhead of PPRM training. Future directions is also given more implicitly.

## Points to Improve

*This is not meant for the rebuttal (see below), but general points that could strengthen the paper*

+ Extend experiments to other domains (e.g., math reasoning, code tasks) to demonstrate generality beyond search.

+ Provide more rigorous theoretical or empirical justification of why ReNorm ensures stability, e.g., via variance or gradient norm analysis during PPO training.

+ Analyze the cost and efficiency of training the PPRM relative to baseline PRMs setups.

+ Clarify bias and reproducibility aspects of NVProcessBench, especially regarding synthetic labels from GPT models.

+ A.1 as written is not accurate. LLMs were also used in benchmark annotation (see A5 and A6), so that usage should be explicitly acknowledged to maintain full transparency and reproducibility compliance. I would not say the current form is unethical or try to hide the use, it is just a little inconsistent. 

+ The figures could be improved for clarity and accessibility. Consider using colorblind-friendly palettes instead of red–green contrasts, and ensure that legends are placed outside or below the plots so they don’t overlap with the lines. Improving visual contrast and spacing would make the results easier to read. Further, each caption should make its figure a standalone component of the paper. Thus, each figure should be built in the following way:
* The first sentence should highlight the main message of the Figure/Table (e.g. "Our method outperforms the existing SOTA methods on the studied problem.")
* The next sentences then explain what is depicted in the Table/Figure. E.g. Mean test accuracy, on 5 seeded trainings, with std. 
* Finally, details and references to e.g. appendix can be provided if necessary. E.g. Our method outperforms baseline 1 in 3 out of 4 tasks, ... etc.

In your captions, the answer to the addressed research question is missing. For example, in Figure 3: The joint used of all components included in PPR allow it to provide more stable reward than the compared baselines... (could be better formulated). 
Put this sentence in bold.

Nit: Typos:
In abstract: Experiment*al* results

### Questions
1. Did you make a small ablation about how PPR performs in non-search domains? Do you have reasons to believe that it would not work with other types of tools ?

2. Can the authors provide empirical evidence that ReNorm reduces gradient variance compared to other normalization schemes?

3. How large is the training cost overhead (compute and/or wall-clock time) relative to baseline RLVR methods?

4. Does NVProcessBench include any human validation to confirm that GPT-based step labels align with human judgments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a reinforcement learning framework for training search agents, integrating both outcome rewards and process rewards. Additionally, the authors designed and trained a generative process reward model and further introduced a reward normalization strategy to combine outcome and process rewards. The framework was evaluated on multiple datasets. The authors also provided a new benchmark to assess scenarios where process rewards are non-quantifiable.

### Strengths
1. This paper models the process reward, thereby enriching the training pipeline for the search agent.
2. The authors introduce a new benchmark to evaluate scenarios where process rewards are non-verifiable.
3. The experimental design  is well-executed.

### Weaknesses
1. The proposed reward normalization method seems to deviate from the standard understanding of PPO and GAE. Reward whitening is a very common trick in PPO training. However, in the proposed "ReNorm" method, the authors simplistically transform it into subtracting 1/2 from both the outcome and process rewards. Essentially, subtracting a \textbf{constant} from the rewards \textbf{has ​no impact​ on} the calculation of GAE. Therefore, the motivation behind this approach is fundamentally flawed. 

2. Furthermore, it is rather rash to directly assume a 1:1 ratio between positive and negative advantage samples. Empirical observation shows that the proportion of positive to negative advantages is \textbf{not balanced} during the training process.

3. This paper directly employs a GenRM to output a numerator and denominator for non-verifiable process rewards, but it lacks a description of the golden-answer verification for the outcome reward. In reality, especially in search scenarios, the rewards for an agent's search process are often quantifiable (e.g., whether a tool call was successful, whether the returned text is highly relevant to the query). However, a quantifiable metric for the final answer produced by the agent is often missing.

Based on the above three points, the motivation and contributions of this paper are flawed.

### Questions
Same as Weaknesses.
1. Subtracting a \textbf{constant} from the rewards \textbf{has ​no impact​ on} the calculation of GAE. 
2. Empirical observation shows that the proportion of positive to negative advantages is \textbf{not balanced} during the training process.
3. In search scenarios, outcome rewards are often more difficult to quantify than process rewards.

### Soundness
3

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
This paper proposes a type of process reward for training LLM search agents with RL. Specifically, the authors propose of process reward they call "Principle Process Reward", which is a (prompted) reward derived from a set of prespecified set of "principles" (e.g., covering aspects like formatting, correctness, etc). Like other process reward work, this is combined with the outcome reward, but with a particular type of heuristic normalization applied (called Reward Normalization). The paper demonstrates that this method outperforms non-RL baselines and other RL methods (both outcome- and process-based) through experiments on several QA benchmarks.

### Strengths
The overall approach is sensible and well-evaluated. Using a generated reward model to provide rubric-based feedback is a practical way to get dense reward signal for a search agent. The ReNorm strategy is also simple and intuitive. Experimental setup appears sound, and it seems clear that this process reward leads to better performance.

### Weaknesses
The main weakness of this work is novelty and technical contribution. Using process rewards is well established in the literature. Generative rubric-based reward models have also been explored before, see e.g., Gunjal et al 2025, Anugraha et al 2025, or generative reasoning-based rewards more broadly, e.g., Guo et al 2025. While the resulting system works well, the paper primarily demonstrates that a well-engineered combination of known components (generative PRM + outcome reward + a simple normalization rule) is effective for training search agents. This is a useful finding, but the lack of substantial novel insight (and unclear implications as to its broader applicability, see questions below) makes it a borderline case for ICLR in my opinion.


[1] Gunjal et al 2025. https://arxiv.org/pdf/2507.17746v1

[2] Anugraha et al 2025. https://arxiv.org/pdf/2505.13388v1

[3] Guo et al 2025. https://arxiv.org/pdf/2505.14674

### Questions
- I'm having a hard time understanding what the <max_score> exactly is? An upper bound on the score? I don't see it defined anywhere.
- How was the set of principles developed and validated? It is unclear how generalizable it is, or if it would need to be re-engineered for different types tasks.
- Renorm is presented as a core contribution --- I'm curious as to how this compares to standard methods for combing rewards like some convex combination of $r_p$ and $r_o$ or using the process reward as a potential function.

### Soundness
3

### Presentation
3

### Contribution
1
