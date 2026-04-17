# A$^2$FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A$^2$FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third instant mode that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A$^2$FM achieves 13.4\% on BrowseComp, 70.4\% on AIME25, and 16.7\% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only \$0.00487 per correct answer—cutting cost by 45.2\% relative to reasoning and 33.5\% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the Adaptive Agent Foundation Model ($A^{2}FM$), a framework designed to unify reasoning-centric and agentic large language models (LLMs). The authors argue that existing models are inefficient, often overusing complex reasoning or external tools for simple queries.

The proposed solution, $A^{2}FM$, integrates three distinct execution modes within a single backbone:

For tasks requiring complex chain-of-thought processes.
For tasks that necessitate the use of external tools like web search or a code interpreter.
For simple queries that can be answered directly, minimizing computational cost.


The paper's primary contributions are:

1. $A^{2}FM$ Framework: The first model to integrate agentic, reasoning, and instant modes under a unified backbone with a self-adaptive router.
2. Adaptive Policy Optimization (APO): A novel reinforcement learning method with adaptive sampling and a cost-regularized reward system to jointly optimize for accuracy and efficiency.
3. Demonstrated Efficiency: The adaptive execution model significantly reduces the cost per correct answer by 45.2% compared to the reasoning-only mode and 33.5% compared to the agentic-only mode, while maintaining comparable accuracy

### Strengths
The core strength of $A^{2}FM$ is its integration of three distinct operational modes (reasoning, agentic, instant) under a single, self-adaptive router. While prior work focused on switching between short and long chain-of-thought, this paper is novel in its inclusion of tool-use as a third, distinct modality.

The proposed Adaptive Policy Optimization (APO) is a creative and original extension of reinforcement learning for this specific task. It builds upon existing methods but introduces a tailored rollout strategy and a cost-regularized reward that explicitly teaches the model to balance accuracy with computational efficiency, which is a significant methodological advance.

The paper goes beyond simply reporting scores. It includes a deep analysis of the model's behavior, such as the accuracy of its mode routing on human-labeled data , its performance evolution during APO training , and a clear breakdown of how mode allocation changes with task difficulty.

### Weaknesses
The paper presents a two-stage training process (SFT then APO) but only reports the final performance . A crucial missing experiment is an evaluation of the model's performance after only the supervised fine-tuning (SFT) stage, before APO is applied. This would quantify the exact performance and efficiency gains attributable to the reinforcement learning phase, which is a central contribution.

The paper's primary motivation is to improve efficiency by avoiding overthinking on simple queries via an "instant mode". However, this claim is not tested directly. An ablation study training a two-mode variant (reasoning and agentic only) using the same APO framework would be highly informative. Comparing this two-mode model's performance and cost-of-pass against the three-mode $A^{2}FM$ would provide direct evidence for the efficiency gains provided by the instant mode.

The APO methodology introduces an adaptive, cost-regularized reward to balance accuracy and efficiency . The paper would be stronger if it included an experiment where the model is trained with APO using a reward function that only considers accuracy. This would demonstrate how essential the cost penalty is for learning efficient routing behavior.

### Questions
The paper presents a complex, multi-stage framework, and its success is impressive. To better isolate the impact of each novel component, could you provide additional details or ablation studies?

The primary contribution is the Adaptive Policy Optimization (APO) stage . To quantify its specific contribution, could you report the performance and efficiency (cost-of-pass) of the model after only the supervised "route-then-align" SFT stage? This would create a clear baseline to measure the exact improvements gained from the reinforcement learning phase.

The efficiency argument hinges on the "instant mode" preventing overthinking on simple queries. To validate this, have you considered an ablation where you train a two-mode variant ($A^{2}FM_{no-instant}$) with only the reasoning and agentic modes? Comparing its cost-of-pass to the full three-mode model on a benchmark with a mix of difficulties (like SuperGPQA) would provide direct evidence of the instant mode's value.

APO uses a cost-regularized reward to balance accuracy and efficiency. How critical is this specific reward design? What happens if the model is trained using APO with a reward function that only considers accuracy ($r_{acc}$)? Does the model learn to default to more powerful (and costly) modes, or does it still achieve some level of efficiency?

The APO stage involves several key hyperparameters, such as the number of forced vs. adaptive rollouts ($\rho=3$, $\gamma=3$) and the adaptive reward scaling factor ($\alpha=2$). Could you comment on the model's sensitivity to these choices? A brief analysis would help future practitioners understand the robustness and tuning requirements of the APO method.

The accuracy reward is determined by an "LLM-as-Judge" . Could you please specify which model was used in this role? Moreover, a brief discussion on the potential biases or failure modes of this judge model would be valuable, as its reliability is critical to the quality of the reward signal during RL training.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes A²FM, a single-backbone model that routes each query to one of three execution modes—instant (short, direct answers), reasoning (explicit CoT), or agentic (tool-use + code)—and then aligns mode-conditioned trajectories via supervised fine-tuning (“route-then-align”). To further couple accuracy and efficiency, the authors introduce Adaptive Policy Optimization (APO): an RL scheme that (i) forces per-query rollouts in all modes with prefix injection to avoid mode collapse, and (ii) uses a cost-regularized, multiplicative reward that penalizes non-instant modes on “easy” queries, with correctness judged by an LLM-as-Judge. Experiments on agentic, reasoning, and general benchmarks at the 32B scale report competitive or SOTA results and a lower cost-of-pass than forcing a single mode.

### Strengths
- The method is about a unified 3-mode formulation with an explicit router under a shared backbone, representing clear contributions and problem setup. 
- The authors conducted breadth of evaluation across agentic (BrowseComp/GAIA/XBench-DS), reasoning (AIME24/25, MATH500), and general (GPQA-d, SuperGPQA, MMLU-Pro, HLE).

### Weaknesses
- Novelty relative to prior “when-to-think” / hybrid routing is under-differentiated. The core idea—self-routing among “instant vs. CoT vs. tool-use” with RL—resembles recent hybrid reasoning mode in GPT-5; the paper claims the “first adaptive agent foundation model” but should sharpen distinctions.
- APO is a minor extension on GRPO with two straightforward additions: (i) forced rollouts per mode, and (ii) adaptive reward penalizing non-instant modes on easy queries. Neither represents significant technical innovation.
- The proposed method seems to rely heavily on distillation from specialized teachers (DeepSeek R1 for reasoning, DeepSeek V3.1 for agentic/instant).

### Questions
- The trajectory rules specify that the reasoning phase must be greater than 1000 words. What is the methodological justification for enforcing such a substantial minimum length?
- The agentic mode’s parallel architecture is a unique design choice. Is there an ablation study quantifying the performance or efficiency improvements?

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
2

### Summary
The paper proposes A2FM, a single-backbone model that dynamically routes each query to one of three execution modes: instant (direct answer), reasoning (explicit CoT), and agentic (tool-use with planning). Training follows a two-stage route-then-align SFT and a reinforcement-learning phase called Adaptive Policy Optimization (APO) that rewards correctness while penalizing unnecessary complexity on “easy” queries. On diverse benchmarks, A2FM reports strong results at the 32B scale and improved cost-of-pass relative to pure reasoning or agentic execution.

### Strengths
1. APO integrates (i) forced per-mode rollouts via prefix injection and (ii) an adaptive reward that explicitly favors instant answers when sufficient—this is a thoughtful twist on GRPO-style objectives for mode selection. 

2. Method is specified with a clean problem statement (router policy + mode policies) and a staged training recipe; the trajectory schemas and masked tool-responses (to avoid memorizing tool outputs) are sensible.

3. Hybrid reasoning + tool-use is central for modern LLM systems. Showing that a single model can approximate the “best-mode oracle” while cutting cost is practically important for research and real-world deployment.

### Weaknesses
1. APO design ablations & hyperparameter sensitivity: APO drops KL, relies on on-policy sampling, and uses several knobs (forced rollouts, adaptive rollouts, “easy” threshold, exponent). Stability and sensitivity are unclear. Adding ablations and showing training curves and mode-collapse checks would be helpful.

2. Router supervision and error analysis: Routing accuracy is reported on a few datasets; but failure modes (e.g., over-confidence in instant on BrowseComp; tool over-calling on easy Qs) deserve deeper analysis. Provide a confusion matrix (GT vs. chosen mode) per dataset, plus qualitative case studies where agentic and reasoning are both plausible.

3. Generalization across backbones/scales: All results use Qwen2.5-32B-Instruct as the backbone, which may not demonstrate the generalizability. it will be good to include at least a small-scale reproduction (e.g., 7B/14B) and one alternate 32B backbone to demonstrate portability.

4. Confidence intervals and seed variation are not reported for evaluation.

### Questions
1. APO reward shaping. How sensitive is performance to τ (easy threshold) and α (penalty exponent)? Did you try adaptive τ per domain (math vs. web tasks) or curriculum schedules? Please provide more ablation like a grid or Bayesian sweep and stability metrics. 

2. KL-free on-policy choice. Removing KL can destabilize RLHF-like training. Did you observe mode collapse or reward hacking without KL? If so, what's the solution?

3. Please add head-to-head ablations or analysis vs. recent “when-to-think / routing” methods and hybrid models (e.g., LHRM (Jiang et al., 2025), BPO (Yang et al., 2025b), AdaCoT (Lou et al., 2025), Search-R1 (Jin et al., 2025)). What incremental gain comes specifically from (i) three-way routing, (ii) the adaptive penalty, and (iii) prefix-enforced exploration?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes A2FM (Adaptive Agent Foundation Model), a single-backbone model that routes each query to one of three execution modes—instant (direct answer), reasoning (explicit CoT), or agentic (tool-use)—and introduces an RL method, Adaptive Policy Optimization (APO), to jointly optimize routing and generation with a cost-aware reward. Training follows a “route-then-align” SFT stage and then APO. The agentic mode executes multiple tools in parallel and masks tool responses during SFT to avoid memorization. Reported results claim strong performance across agentic (BrowseComp, GAIA, XBench-DS), reasoning (AIME24/25, MATH500), and general (GPQA-d, SuperGPQA, MMLU-Pro, HLE) benchmarks, with improved “cost-of-pass.”

### Strengths
- Clear problem framing (when-to-think/act): The unified routing over instant/reasoning/agentic is well-motivated and ties into efficiency concerns that are very good-framing and practical.
- Methodical RL design: APO’s forced vs. adaptive rollouts via prefix injection plus an adaptive reward that penalizes non-instant modes on easy queries is a neat, concrete recipe. The multiplicative reward with format checks is pragmatically sound. 
- Routing and costs analysis: The paper measures routing accuracy (e.g., GAIA 92.2%, BrowseComp 94.0%, AIME 100%) and discusses where routing errs. This is interesting. Using cost-of-pass and showing adaptive < agentic < reasoning is useful for the community that cares about $/correct answer.

### Weaknesses
- The novelty claim would benefit from a sharper comparison to recent capability-aware routing, bimodal/bihybrid reasoning, and hybrid agent efforts. The core elements—routing head, SFT with tagged modes, RL with exploration across modes, and a length/cost-aware reward—feel like incremental synthesis rather than a conceptual leap.

- The binary judge drives APO’s reward; while practical, it risks systematic bias and false positives/negatives, especially on agentic tasks where answers can be partially correct or depend on retrieval freshness. A small-scale human evaluation or rule-based metrics where available (e.g., AIME exact answers) to validate judge reliability and quantify disagreement will be helpful.

- The ρ forced and γ adaptive rollouts are critical. Additional analyses about ρ, γ and show accuracy/efficiency trade-offs and training stability will be helpgul.

- A deeper error analysis disentangling routing mistakes (wrong mode) from within-mode generation errors (bad tool plan, crawl failures, or code mistakes), and quantify the contribution of the parallel tool plan itself will be appreciated.

- Lastly, how generalized is it to unseen tasks?

### Questions
How generalized is it to unseen tasks?

### Soundness
3

### Presentation
3

### Contribution
3
