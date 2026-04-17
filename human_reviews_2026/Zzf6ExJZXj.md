# WebGen-R1: Incentivizing LLMs to Generate Functional and Aesthetic Websites with Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Large Language Models (LLMs) have demonstrated strong capabilities in functional-level code generation, yet their performance remains limited in project-level scenarios such as generating large-scale multi-page websites. Such tasks require coherent multi-file structures, handling of intricate cross-page dependencies, and visually appealing designs. Prior works address only partial aspects of this challenge. For instance, WebDev Arena focuses exclusively on single-page static sites, while agent-based frameworks decompose tasks into subtasks coordinated through multi-turn execution, often relying on proprietary models and suffering from fragile integration, particularly in visual coherence and stylistic consistency. In this work, we introduce WebGen‑R1, pushing toward a more ambitious and practically relevant goal of training a small-scale LLM via reinforcement learning (RL) to generate the entire multi‑page websites in an end‑to‑end manner. A key obstacle lies in reward design. Unlike functional code generation where correctness can be verified by passing automated test suites, web aesthetics covering layout harmony, typographic consistency, and stylistic alignment are inherently subjective, and functional verification often requires dynamic execution across pages where rule-based reward function tend to be brittle. To address these limitations, we design a vision–language–model-based reward model that jointly optimizes functional correctness and aesthetic quality, enabling the model to produce websites that are both visually coherent and faithful to the intended task specification. Extensive experiments across real-world benchmarks demonstrate that WebGen-R1 consistently outperforms, or is comparable to, strong proprietary and open-source baselines in a multi-dimensional evaluation protocol. To facilitate future research in end-to-end multi-page website generation, we release our code and data at https://anonymous.4open.science/r/WebGen-R1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes WebGen-R1, which casts end-to-end website generation into RL framework. It integrates a VLM reward that jointly evaluates both functional correctness and visual aesthetics based on the screenshots of the rendered website. The agent model is fine-tuned using GRPO with the VLM reward. The experiment results demonstrate that WebGen-R1 significantly improves functional success rate, render reliability, and aesthetic quality, outperforming or matching larger proprietary models such as GPT-5, Claude, and Gemini.

### Strengths
1. The idea of incorporating a VLM-based reward model for website generation is straightforward and practically valuable.

2. The proposed method achieves superior performance compared to existing general-purpose LLM agents

3. The paper is clearly structured and well written

### Weaknesses
1. The VLM-based reward is limited to static screenshots, which cannot fully assess the functional correctness of interactive elements such as dropdown menus, buttons, or dynamic state transitions.

2. Most baseline models are general-purpose LLMs rather than website-specific agents. The paper lacks evaluation against alternative fine-tuning strategies tailored to website generation, such as training with a human-preference reward model or other reward models.

3. The technical depth is relatively limited. The proposed method mainly adapts existing RL methods to the website generation domain without new algorithmic components.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces WebGen-R1, a framework designed to train a small-scale Large Language Model (LLM) to generate complete, multi-page websites from natural language prompts in an end-to-end fashion. It presents a novel reinforcement learning (RL) approach that uses a Vision-Language Model (VLM) as a reward function. This VLM assesses rendered screenshots of the generated website to provide a unified reward signal for both functional correctness and aesthetic quality, overcoming the limitations of brittle, rule-based reward systems.
The methodology involves generating the entire project codebase as a single sequence, which is then processed by an automated pipeline that builds, renders, and evaluates the website. The resulting reward, which also incorporates signals for code structure and chain-of-thought reasoning, is used to fine-tune the base LLM using the Group Relative Policy Optimization (GRPO) algorithm. The evaluation uses a variety of both rule as well as LLM-based metrics, as well as human evaluation to mitigate reward hacking and performs similarly or better than the latest SOTA closed and open models.

### Strengths
- Rigorous pipeline and evaluation - the paper is a good improvement on existing works, combining an end-to-end pipeline, critical rules based evaluation steps, and also testing for human alignment, reporting remarkably high Pearson’s correlation coefficient r = 0.903
and Spearman’s rank correlation ρ = 0.888. This, together with the qualitative examples, strengthens the validity of the metrics actually measuring across the intended dimensions.

- Strong Empirical Results on Key Metrics: The dramatic improvements in Aesthetic Alignment Score (+44.32%) and Valid Render Ratio (+65.33%) are impressive. 

- Multiple datasets, good generalizability to WebDev Arena benchmark.

### Weaknesses
- Dependence of a large proprietary model for feedback
- The FSR of 29.21% is low in absolute terms and significantly lower the 57.72% achieved by Claude-3.7-Sonnet. This could be further explored.
- Small evaluation - while decently sized compared to the previous works, the evaluation sample is still a relatively small sample and might not approximate real workflows well,

### Questions
- Could you provide an ablation study on the reward function? Specifically, what is the impact on final performance if you remove the code format reward or the reasoning reward?
- What were some of the more complex samples? while informative, the qualitative study demonstrates fairly basic setups.
- Could you share more details about human study, how did human to score correlation compare to inter inter-human one?

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
The authors present WebGen-R1, an MLLM trained with GRPO using feedback signals from a reward signal generated from a model (GPT-4o) and also some rule-based signals to evaluate the visual and rendering quality of generated website code. The model, WebGen-R1-7B, is compared against several state-of-the-art closed-source models and the Qwen family of models, outperforming all baselines on the aesthetic and rendering metric (AAS).

### Strengths
- The task of generating complete websites from scratch is highly challenging and represents an ambitious and impactful direction for MLLMs.

- The proposed reward function aligns well with human preferences, and the authors demonstrate this alignment effectively.

### Weaknesses
- It is unclear why WebGen-LM models are not included in the comparison. Including them would provide a stronger and more complete baseline for evaluating progress.

The interpretation of the AAS metric appears inaccurate. The authors claim that WebGen-R1 achieves superior performance “across all 13 categories on AAS, indicating consistent improvements in both functional correctness and UI/UX quality.” However, AAS is an aesthetic metric that primarily captures visual quality, as indicated by the system prompt in the appendix. Functional correctness should instead be measured by FSR. This confusion between aesthetic and functional metrics is repeated in several parts of the paper.

- Since the model is optimized to maximize the reward model’s score—80% of which comes from AAS under the configuration λ=0.1 and γ=0.1—there is a risk of overfitting to aesthetic quality. To demonstrate that the method works more generally, the model should also show improvements on FSR or other metrics beyond AAS.

- The paper does not provide results for larger γ values (γ = 0.5 or γ = 1.0). Showing AAS and FSR results under different γ (and also λ) values would help illustrate how varying the weight between aesthetic and functional and reasoning rewards affects performance.

- It would be helpful to identify which model serves as the most reliable judge or produces outputs that best align with intended outcomes. Benchmarks such as PairBench [1] or AgentRewardBench [2] could be used to evaluate this aspect.

[1] PairBench: Are Vision-Language Models Reliable at Comparing What They See?

[2] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories

### Questions
Please refer to the issues raised in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents WebGen-R1, a reinforcement learning framework that fine-tunes a small-scale language model (Qwen2.5-Coder-7B-Instruct) for end-to-end multi-page website generation. Unlike previous works that handle only static or single-page generation or rely on fragile multi-agent decomposition, WebGen-R1 treats website generation holistically as a single policy optimization problem. The key innovation lies in the reward design: instead of brittle rule-based verification, the authors build a vision–language-model-based (VLM) reward model that evaluates both functional correctness (via executable builds) and aesthetic quality (via rendered screenshots). Training uses Group Relative Policy Optimization (GRPO), avoiding the need for an explicit value function. Experiments on WebGen-Bench and WebDev Arena show strong gains in both visual and functional metrics, especially a 65-point improvement in valid render ratio over the base model and competitive performance against proprietary models such as Gemini-2.5-Pro and Claude-3.7-Sonnet.

### Strengths
[+] Propose a VLM-based perception RL for code and design quality enhancement of the website

[+] A 7B model surpasses or matches proprietary giants, highlighting efficiency

[+] Multiple benchmarks, ablations, and human studies establish robustness

[+] The presentation of this paper is good.

### Weaknesses
[-] Reward dependence on specific VLM evaluators (GPT-4o) raises the issues of heavy cost and generalizability

[-] The scope of aesthetic evaluation is mostly page-level rather than full-site user experience

[-] Fewer new challenges are addressed, or new foundational methods are developed.

### Questions
1. Why do you use GPT-4o as a VLM evaluator rather than other models (more advanced or open-weighted options)
1. Can the VLM reliably distinguish functional correctness from mere visual completeness?
1. What is the compute cost per RL iteration, and how scalable is the approach to larger models or datasets?
1. Did the explicit reasoning traces (…) measurably improve cross-page consistency or reward quality?
1. Did you observe reward-hacking or mode collapse during extended RL training?

### Soundness
3

### Presentation
4

### Contribution
3
