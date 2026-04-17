# Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Tool-integrated reasoning has emerged as a key focus for enabling agentic applications. Among these, DeepResearch Agents have gained significant attention for their strong performance on complex, open-ended information-seeking tasks. We introduce Fathom-DeepResearch, an agentic system composed of two specialized models. The first is Fathom-Search-4B, a DeepSearch model trained from Qwen3-4B and optimized for evidence-based investigation through live web search and targeted webpage querying. Its training combines three advances: (i) DUETQA, a ∼5K-sample dataset generated via multi-agent self-play that enforces strict web-search dependence and heterogeneous source grounding; (ii) RAPO, a zero-overhead extension of GRPO that stabilizes multi-turn Reinforcement Learning with Verifiable Rewards through curriculum pruning, reward-aware advantage scaling, and per-prompt replay buffers; and (iii) a steerable step-level reward that classifies each tool call by cognitive behavior and marginal utility, enabling explicit control over search trajectory breadth, depth, and horizon. These improvements enable reliable extension of tool-calling beyond 20 calls when warranted. The second is Fathom-Synthesizer-4B, trained from Qwen3-4B, which converts multi-turn DeepSearch traces into structured, citation-dense DeepResearch Reports for comprehensive synthesis. Evaluated on DeepSearch benchmarks (SimpleQA, FRAMES, WebWalker, Seal0, MuSiQue) and DeepResearch-Bench, the system achieves state-of-the-art performance in the open-weights category while closely rivaling proprietary
closed systems, while also demonstrating strong performance in general reasoning benchmarks: HLE, AIME-25, GPQA-Diamond, and MedQA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Fathom-DeepResearch, an open-source framework for long-horizon information retrieval and data synthesis for deep research agents.

### Strengths
1.	The motivation is clear and well-grounded, directly addressing key limitations in current DeepResearch models, such as unstable multi-turn reinforcement learning and inefficient tool usage.
2.	The proposed methodologies are sound and well-justified, with each component, RAPO, DUETQA, and the Steerable Step-Level Reward, clearly contributing to overall model stability and performance.
3.	The experiments are comprehensive, spanning both DeepResearch-specific and general reasoning benchmarks, and the results convincingly demonstrate the framework’s effectiveness and competitiveness against baselines.

### Weaknesses
1. The paper points out that GRPO suffers from reward-hacking behaviors, where the model tends to overuse tools without improving reasoning quality. However, it does not clearly demonstrate that the proposed RAPO algorithm effectively mitigates this issue from the perspective of actual tool-call frequency. In Figure 4, only the response length is presented as a proxy for the number of tool calls, which is an indirect measure. A more concrete quantitative analysis or visualization of tool-call usage would strengthen the claim that RAPO truly alleviates reward-hacking behavior.
2. More challenging  benchmarks, such as BrowseComp, are expected to be included to better demonstrate the model’s deep research capabilities and generalization to new, open-ended evaluation settings.

### Questions
1. What's the meaning of "SLMs"? The title mentions SLMs, but the paper does not define this abbreviation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a two-part agentic reasoning system built from Qwen3-4B models: Fathom-Search-4B, which performs multi-turn, evidence-based web investigation, and Fathom-Synthesizer-4B, which transforms search traces into citation-dense reports. The system introduces three contributions — the synthetic DUETQA dataset for training, the RAPO algorithm for stable multi-turn reinforcement learning, and a steerable step-level reward.  This work is evaluated across multiple benchmarks, including DeepSearch and general reasoning benchmarks, on both closed-source and open-source models/agents.

### Strengths
1. This work provides synthetic training data to facilitate training the deep research agents.
2. This work proposes reward-aware policy optimization (RAPO), and the experiments show the proposed methods can achieve better performance than vanilla GRPO.
3. This work provides extensive experiments to verify the effectiveness of the method on multiple benchmarks and compare different closed-source and open-source baselines.

### Weaknesses
1. This work lacks a detailed analysis of whether the final trained model can truly work for long-horizon tool calls. When it succeeds and when it fails?

### Questions
1. How long does it take to train the model for different stages?
2. This work is claimed to build on top of RECALL (Chen et al., 20). Why not treat it as one of the baselines?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Fathom-DeepResearch, an open-source deep research agent system. The system consists of two coordinated 4B-parameter components: Fathom-Search-4B, a web-search model trained with an improved stabilized reinforcement-learning framework, and Fathom-Synthesizer-4B, a structured report generator trained under a *plan-then-write* protocol. The paper further introduces a dataset (DUETQA ) of 5k self-play-generated, live-search-dependent QA examples, and a synthetic DeepResearch-SFT corpus for training long-form synthesis. Evaluations across benchmarks show competitive performance of the proposed agent.

### Strengths
The work is well-motivated and technically complete, with reasonable contributions in this crowded space (1) a simple multi-agent framework and models (while most works explore single end-to-end model) (2) proposing a new RL training framework (3) datasets

### Weaknesses
- As this is a quickly evolving and crowded space, it's better to compare with more recent baselines, and more importantly, to compare with more recent related research on data synthesis and recent upgrades to GRPO (such as GSPO which is now widely used in this space).
- Relatedly, the components in this work, including the data synthesis and minor improvements on the training part, by far, have been largely explored, which undermines the contribution of this work.
- The RAPO relative to GRPO is still minor update per se, and the improvements in Table 3 is not very substantial.
- To thoroughly evaluate the "deep research" capabilities of the agent, the paper would be benefited from incorporating commonly used browsecomp benchmark. Some RAG benchmarks such as SimpleQA is already relatively too old and too short to be informative in this space.

### Questions
1. A typo in the Table 2: systems such Kimi-Researcher should be closed source agents?
2. Will all the models and datasets be opensourced?
3. Broader comparison: 
   1. The authors might consider summarizing how Fathom differs from other recent research agents (in dataset construction, training paradigm, and report synthesis) to contextualize its contributions within the current surge of deep-research systems.
   2. the same for RL algorithms, such as GSPO which is now widely used by search agents.

### Soundness
3

### Presentation
2

### Contribution
3
