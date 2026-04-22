# Group Verification-based Policy Optimization for Interactive Coding Agents

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Recent advancements in reinforcement learning from verifiable rewards (RLVR), particularly through Group Relative Policy Optimization (GRPO), have significantly improved the capabilities of large language models (LLMs) for interactive coding agents.
However, these methods overlook process-verifiable environment feedback (e.g., code execution failures), leading to inaccurate advantage estimation at each reasoning step and insufficient learning.
To address this issue, we propose Group Verification-based Policy Optimization (GVPO), a novel RL algorithm that introduces an advantage shaping framework integrating both outcome-verifiable and process-verifiable signals.
While outcome-verifiable rewards ensure alignment with long-term task objectives, process-verifiable feedback derived from intermediate execution traces (e.g., syntax errors, runtime exceptions) serves as corrective shaping terms at the step level.
By jointly leveraging these two forms of verifiability, GVPO achieves more accurate credit assignment, balancing short-term process guidance with long-term outcome alignment.
This unified formulation yields more stable optimization, faster convergence, and stronger generalization in complex interactive environments.
A 32B-parameter agent trained with GVPO in the AppWorld environment outperforms OpenAI’s o1 agent by 12.7\% on the more challenging Test-C split and surpasses the strongest 32B RL-trained state-of-the-art baseline by 3.7\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GVPO, a reinforcement learning method for interactive code agents. The approach integrates outcome-verifiable signals and process-verifiable signals (e.g., syntax errors, runtime exceptions, partial test passes) into a shaped advantage function, and further stabilizes training through asymmetric clipping and sequence-mean–token-mean aggregation, which improve stability and length fairness.
In the AppWorld environment, GVPO achieves consistent improvements over baselines such as GRPO, Dr.GRPO, DAPO, and LOOP, and provides several ablation studies.

### Strengths
1. The stability design is well thought out: asymmetric clipping and two-level aggregation reduce gradient explosion and sequence-length bias.

2. Experiments are relatively comprehensive within a single environment, including main tables and multiple ablations with consistent trends, and the setup appears reproducible.

### Weaknesses
- I am not an expert in this subfield, but from my perspective the conceptual novelty is limited: combining intermediate and final feedback for reward shaping is rather straightforward within the RLVR framework (Code Agent).

- The theoretical analysis is insufficien, the paper lacks quantitative discussion of bias/variance changes and convergence properties introduced by advantage shaping.

- Empirically, GVPO performs comparably to LOOP on the normal splits and shows moderate advantage on the challenge split.

- The related-work section does not provide a sufficiently comprehensive overview of prior works, making the claimed contributions difficult to clearly position.

### Questions
- How is the shaping coefficient b chosen, and how sensitive is performance to it? 


- How sensitive are the results to the chosen upper and lower bounds in asymmetric clipping?


- The authors note that the KL term is removed for GRPO (only comparing with GRPO-w/kl), but the effect of GVPO + KL has not been evaluated.

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
3

### Summary
This paper proposes Group Verification-based Policy Optimization (GVPO), a novel reinforcement learning algorithm that integrates outcome-verifiable rewards with process-verifiable feedback. By leveraging both long-term task alignment signals and intermediate corrective signals, GVPO improves credit assignment, optimization stability, and generalization in complex interactive environments. Experimental results demonstrate that a 32B-parameter agent trained with GVPO outperforms strong baselines, including OpenAI's o1 agent, particularly on challenging benchmarks like Test-C in the AppWorld environment.

### Strengths
1. The proposed GVPO framework is novel and well-motivated, with a clean and intuitive formulation that combines outcome-verifiable and process-verifiable feedback.
2. By incorporating step-level successes and failures through process-verifiable signals, GVPO effectively constrains optimization, leading to more reasonable and guided learning. The experimental results convincingly support this claim.
3. The paper presents a comprehensive experimental setup, including appropriate ablations and a sufficiently large model size (32B), which adds credibility to the findings.

### Weaknesses
1. The paper does not sufficiently discuss the computational cost of GVPO, particularly in terms of scalability to larger models or environments.
2. The experiments are limited to a 32B-parameter model, leaving open the question of how GVPO would perform with larger-scale models, especially given the trend toward scaling in reinforcement learning research.

### Questions
1. How does GVPO scale computationally when applied to larger models or more complex environments? Are there practical limitations or bottlenecks?
2. Could GVPO benefit from additional architectural modifications or optimizations specific to larger-scale models (e.g., 64B or beyond)?
3. Would GVPO’s advantage shaping framework generalize to other interactive environments beyond AppWorld? Any insights into potential domain-specific challenges?

### Soundness
2

### Presentation
2

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
This paper introduces Group Verification-based Policy Optimization (GVPO), a novel RL from Verifiable Rewards (RLVR) algorithm designed to enhance the training of interactive coding agents. The core insight is that existing RLVR algorithm rely solely on sparse, outcome-verifiable rewards (e.g., final unit test results), leading to inaccurate credit assignment across an agent's reasoning trajectory. GVPO addresses this by integrating dense, process-verifiable feedback according to the advantage estimation directly into the optimization objective. Experiments shows that GVPO outperforms existing RLVR algorithms on AppWorld, and Qwen2.5-32B-Instruct trained with RLVR outperforms than zero-shot commercial LLMs.

### Strengths
The paper is easy to follow and well-organized.

The paper is well-motivated. Constructing step-level signals are of great importance for RLVR algorithms, especially for long-term hard tasks.

Some empirical results are compelling. Achieving a new SOTA on a complex benchmark and outperforming a model like OpenAI's o1 by a significant margin (12.6% on Test-C) with a 32B model is a strong testament to the method's efficacy.

### Weaknesses
One of the main claimed contribution, process-verifiable feedback derived from intermediate execution traces (e.g., syntax errors, runtime exceptions), are not implementated in the algorithm.

The proposed shaping function for $B_{i,t}$ and the resulting $A_{i,t}$ is complex and contains several case-based rules. Designing such function for each secnario is diffcult and may cause reward hacking [1].

The experiments are limited to one agent type(CodeAct), one scenario (AppWorld), and one model (Qwen2.5-32B-Instruct), which can not prove the generalization of GVPO. No evidence is given that GVPO helps in other widely-used agent domains (e.g., WebArena, SWEBench).

All experimental results are point estimates from a single run. It is unclear whether the +3.6 % TGC gain over LOOP is statistically significant.

Reference: 
[1] Guo D, Yang D, Zhang H, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning[J].Nature, 2025.

### Questions
- Why can GVPO prevent the entropy decay in Figure 2?

- Considering that the number of training samples is only 35, I wonder how many gradient steps the training has?

- The proposed shaping function for $B_{i,t}$ and the resulting $A_{i,t}$ is complex and contains several case-based rules. Designing such function for each secnario is diffcult and may cause reward hacking.

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
This paper proposes Group Verification-Based Policy Optimization (GVPO), a new reinforcement learning (RL) algorithm designed to improve the training of large language model (LLM)–based interactive coding agents. GVPO extends prior Group Relative Policy Optimization (GRPO) by introducing an advantage shaping framework that combines two complementary reward types:

Outcome-verifiable rewards -- reflecting final task correctness (e.g., unit-test results);

Process-verifiable signals -- reflecting intermediate feedback such as syntax errors or runtime exceptions.

The method shapes per-step advantages using both signals to achieve finer credit assignment and more stable optimization. Evaluations on AppWorld, a challenging multi-turn code-execution benchmark, show that a 32B-parameter GVPO agent outperforms OpenAI’s o1 by 12.6% on the hardest test split and the strongest 32B RL baseline (LOOP) by 3.6%. Analyses indicate better exploration stability, reduced execution errors, and more cautious agent behavior.

### Strengths
1. The integration of process-verifiable signals addresses the sparsity and delay of traditional outcome-only RL signals, providing a principled improvement to credit assignment.
2. GVPO achieves state-of-the-art performance among open 32B-scale agents on AppWorld, significantly surpassing both outcome-only RL methods and prompting-based models.
3. Ablation and behavior analyses (entropy, cautiousness, failure rates) convincingly support the claimed benefits in stability and exploration.
4. Since process-verifiable feedback is rule-based, GVPO avoids additional reward-model training and can generalize to other deterministic environments.

### Weaknesses
1. Experiments focus solely on the AppWorld coding environment; it remains unclear how GVPO performs in non-deterministic or partially verifiable domains (e.g., reasoning or natural dialogue tasks).
2. Lack of comparison to LLM-based verifiers: Recent RLVR works using LLM verifiers are not empirically compared, which limits positioning relative to that line of research.
3. While the paper observes “cautious behavior,” it lacks deeper qualitative discussion of how process feedback influences reasoning patterns.
4. The method assumes reliable step-level feedback (syntax/runtime correctness), which may restrict applicability to highly structured environments.

### Questions
1. Can this method be applied to other problems beyond coding agent RL training?
2. How sensitive is GVPO to the choice of shaping coefficient b? Could adaptive or learned shaping improve stability across domains?
3. Could the asymmetric clipping mechanism interact with shaping to introduce bias in long trajectories?
4. How well does GVPO scale beyond 32B parameters or to more diverse environments such as web or embodied agents?

### Soundness
3

### Presentation
3

### Contribution
2
