# Designing Observation and Action Models for Efficient Reinforcement Learning with LLMs

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 6, 4, 4, 6

## Abstract
The design of observation and action models is a fundamental step in reinforcement learning (RL), as it defines how agents perceive and interact with their environment. Despite its importance, this design choice is often overlooked in standard benchmarks, which typically use handcrafted models. However, this choice can substantially influence both learning efficiency and final performance. To address this gap, we propose LOAM (LLM-based design of Observation and Action Models), a framework that leverages large language models (LLMs) to automate the generation of these models. LOAM extracts information about simulation variables and task objectives from the environment and uses an LLM to generate Python functions for the observation and action models, enabling seamless integration into standard RL training pipelines. When applied to the basic locomotion tasks of HumanoidBench (stand, walk, run), LOAM-designed models achieve over 3× faster learning on average with the same FastTD3 algorithm compared to the default benchmark models. Furthermore, to handle the variability of LLM outputs, we race multiple generated designs and progressively select the top performers under a fixed training budget. To our knowledge, this is the first work to propose an efficient learning method that mitigates quality diversity in LLM-designed observation and action models within the same timestep as the standard single-model training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LOAM, which uses an LLM to automatically generate the observation and action functions that define how an RL agent perceives the environment and issues commands. Instead of relying on handcrafted mappings, LOAM creates these interfaces in code and plugs them into existing training pipelines. A second component, LOAM-Race, samples multiple candidate designs, briefly trains each one, and then reallocates the fixed training budget to the most promising candidates. Experiments across 12 HumanoidBench tasks show faster convergence and, in some cases, better performance than other baselines.

### Strengths
1.1: Addresses an underexplored component in RL, which is the design of observation and action mappings. 

1.2: Strong experimental results with different environments (same domain) with multiple seeds.

### Weaknesses
2.1: No optimization in the design space. The method goes from zero-shot to train to pick. There is no iterative refinement or evolution of the generated code (obs/action functions).

2.2: LOAM-Race is not clearly described. The figure mentions selection “every ~128k steps,” but the paper does not specify the details. The results show that LOAM-Race has the same total training steps as the other baselines, so it is unclear how the total training budget is divided.

2.3: The discard rule of weakest every 128k steps can be wrong. More complicated functions require more training time to converge and may lead to better results. That should be investigated. 

2.4: The claim that LOAM-Race “handles LLM stochasticity” is misleading. The approach samples multiple candidates and reallocates training based on performance every 128K steps. 

2.5: Limited domain diversity. All experiments are on humanoid tasks; the method should be tested on other domains or harder environments to verify generality and performance beyond convergence speed.  

2.6: Section-3 is overly detailed with prompt templates that belong in the appendix. The space should instead expand on 2.2 (racing/budget details).

### Questions
3.1: Can the authors provide more information about the overall pipeline design and justify why there is no optimization or refinement over the generated code space (i.e., beyond zero-shot generation and simple selection among K candidates)?  

3.2: Can you show additional results or analysis verifying whether weaker early-performance candidates in LOAM-Race eventually lead to worse final policies, or if early performance reliably predicts long-term outcomes?  

3.3: Can you add additional results in more domains where LOAM (and/or LOAM-Race) outperforms other baselines consistently, not only in speed but in performance as well?

### Soundness
2

### Presentation
2

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
Disclosure: Claude is used to refine this review.

This paper introduces LOAM (LLM-based design of Observation and Action Models), a framework that leverages large language models to automatically generate observation and action representations for reinforcement learning tasks. Given environment specifications and task descriptions, LOAM produces Python functions that transform raw sensory inputs into compact observation vectors and map low-dimensional policy outputs to full actuator commands. The authors also propose LOAM-Race, which handles LLM output variability by racing multiple generated designs and progressively selecting top performers. Experiments on HumanoidBench demonstrate that LOAM-designed models achieve approximately 3× faster learning on basic locomotion tasks compared to handcrafted baseline models.

### Strengths
- The paper addresses an important yet understudied problem in reinforcement learning - automated design of observation and action spaces, which practitioners typically handle through manual feature engineering. The core insight that LLMs can automate this process is compelling and timely. 
- The empirical results are strong, with consistent improvements across multiple tasks in HumanoidBench, and the reach task result is particularly impressive where LOAM succeeds while all baselines fail. 
- The paper provides extensive implementation details, including full prompt templates and generated code examples in the appendices, which aids reproducibility and understanding.

### Weaknesses
- The evaluation scope is limited to a single benchmark (HumanoidBench), so it's unclear if it can generalize to different robots or task domains. 
- The prompts appear heavily engineered with domain-specific guidance, which contradicts claims of automation and suggests significant manual tuning was required. Maybe a reasonable comparison is how much human effort it saves (i.e., how many human hours are needed to match the performance of the proposed method).

### Questions
N/A

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
LOAM uses LLMs to automatically generate Python functions for observation and action models in RL, based on environment specs and tasks, enabling efficient integration into training pipelines. LOAM-Race races multiple variants to select the best under fixed budgets. Evaluated on HumanoidBench (12 locomotion/manipulation tasks) with FastTD3, achieving 3x faster learning and higher returns than baselines like handcrafted features and LESR.

### Strengths
1.Automates a key RL bottleneck (obs/action design) with LLMs, yielding compact, task-relevant models that boost sample efficiency and final performance across diverse tasks.
2.LOAM-Race effectively mitigates LLM output variability via optimistic racing, identifying strong designs in a single run without exhaustive training.
3.Structured prompts incorporate robotics priors (e.g., posture stability), enhancing model quality as shown in ablations.

### Weaknesses
1.No quantitative LLM cost analysis—racing requires multiple generations/evaluations, potentially prohibitive for larger setups.
2.Baselines (e.g., LESR) are adapted but may not be optimally tuned; lacks comparisons to non-LLM obs/action methods.
3.Results confined to simulation; real-robot gaps (noise, delays) unaddressed, limiting claims of practical utility.
4.Over-reliance on proprietary GPT-5 without testing open-source alternatives or model robustness.

### Questions
1.How does LOAM perform with open-source LLMs (e.g., Llama-3) versus GPT-5? Any degradation in model quality?
2.What are the total LLM inference costs (tokens, time) for generating and racing models per task?
3.Can LOAM handle visual or partial observations, e.g., by incorporating neural encoders?
4.Why no ablation on racing hyperparameters like K (candidates) beyond K=1-4, or confidence estimation methods?
5.Have you tested LOAM on non-MuJoCo envs or real hardware to validate beyond simulation?

### Soundness
2

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
This paper introduces LOAM (LLM-based design of Observation and Action Models), a novel framework that leverages Large Language Models (LLMs) to "automate the generation" of observation and action models for reinforcement learning agents, particularly in complex domains like humanoid robotics. The core idea is to use structured prompts—containing information about the agent's morphology , the "task description" , and "available state variables" —to guide an LLM to generate Python functions (compute_obs and compute_action). These functions serve as a "wrapper" around the original environment , creating a lower-dimensional and more informative state-action space for the RL agent. To address the "inherent stochasticity" of LLM outputs, the authors also propose LOAM-Race, a mechanism based on the "principle of optimism in the face of uncertainty (OFU)" that efficiently evaluates multiple generated models in parallel and allocates training resources to the most promising candidates. The method is evaluated on the HumanoidBench benchmark, where it demonstrates significant improvements in sample efficiency and final performance over strong baselines.

The paper's claimed contributions are exceptional. The proposed method demonstrates strong empirical performance on the challenging HumanoidBench benchmark. This suggests a true qualitative leap in capability, not merely a quantitative improvement. The qualitative analysis in Appendix B further reveals that the LLM (allegedly) generates sophisticated, domain-aware Python code that embeds complex physical priors, such as "heading-invariance" via quaternion rotations and "biomechanical priors" like contralateral coordination.

### Strengths
1. The paper tackles a well-known and significant bottleneck in applying RL to complex robotic systems: the design of observation and action spaces. The proposed approach of using LLMs to automate this process is novel, timely, and presents a compelling new direction for environment design in RL.
2. The experimental results, as presented, are impressive. Achieving "over 3x faster learning on average" on a challenging benchmark like HumanoidBench (Figure 1) is a substantial improvement. The learning curves in Figure 4 clearly demonstrate that LOAM and LOAM-Race consistently and significantly outperform strong baselines like FastTD3 and LESR across a wide variety of tasks. The qualitative win on the h1hand-reach-v0 task is particularly noteworthy, suggesting the framework can discover representations superior to human-engineered ones.  
3. The LOAM-Race mechanism is an intelligent and practical solution to the inherent stochasticity of LLM code generation. Instead of viewing variability as a weakness, the authors turn it into an opportunity for robust model selection. The method, based on the principle of optimism in the face of uncertainty, is principled and shown to be effective at finding better and more stable solutions.

### Weaknesses
1. The work primarily automates the implementation of the wrapper, not the conceptual design of the task. The LLM is effectively "automating the translation of a detailed human specification into code." The framework's success hinges on access to the pre-existing, human-engineered structure of the HumanoidBench environment and a "well-defined reward signal", a limitation the authors admit in the conclusion.  The LLM does not operate from raw physics but from a curated set of variables and, most importantly, a human-scripted reward function for each task. The LLM is given the task description and reward logic, which drastically simplifies the problem of identifying relevant features. The paper frames this as "automating design," but it feels more like "automating the translation of a detailed human specification into code." The framework's success hinges on access to a "well-defined reward signal" , a limitation the authors admit in the conclusion. This raises significant questions about its utility in more realistic scenarios where the reward is sparse or the task goal is not so clearly defined.  
2. While simulation is a necessary first step, the paper makes strong claims about solving a key bottleneck for real-world robotics without providing any experiments or even a substantive discussion on the challenges of transferring these generated models to physical hardware. LLM-generated code might create brittle policies that overfit to simulation-specific dynamics or artifacts. An analysis of the sim-to-real gap would be essential for a paper with such practical claims.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LOAM (LLM-based design of Observation and Action Models), a framework that automates the design of observation and action representations in reinforcement learning (RL). Instead of relying on handcrafted feature and actuator mappings, LOAM uses large language models (LLMs) to generate executable Python functions that define observation and action models. It also introduces LOAM-Race a model selection mechanism that evaluates multiple LLM-generated candidates in parallel and adaptively allocates training resources to the most promising ones using upper-confidence-bound (UCB) estimates.  

Applied to **HumanoidBench**, a high-dimensional humanoid control benchmark, LOAM achieves up to **3× faster learning** than expert-designed models using the same RL algorithm (**FastTD3**). LOAM-Race further improves robustness by mitigating the variability of LLM-generated designs.

### Strengths
- **Novel contribution:** Automates a fundamental but underexplored RL design component, observation and action model specification using LLMs.  
- **Clear methodology:** The structured prompting framework for code generation (system, observation, and action prompts) is systematic and well-motivated.  
- **Strong empirical results:** Demonstrates consistent gains across locomotion and manipulation tasks on HumanoidBench, surpassing FastTD3 and LESR baselines.  
- **Comprehensive experiments:** Includes detailed ablations on observation-only vs. full design, candidate count, and racing behavior, alongside qualitative code analysis.  
- **Reproducibility:** Provides complete templates, prompts, and structured pipeline descriptions, making the approach easily replicable.

### Weaknesses
- **Limited theoretical grounding:** The paper is largely empirical and it lacks formal analysis of why LLM-generated designs improve representation quality or exploration.  
- **Dependence on LLM reliability:** Quality and efficiency depend on the correctness of generated code while the robustness under different model types (e.g., GPT-4 vs. GPT-5) is not explored.
- **Limited scope of environments:** Focuses solely on humanoid control in simulation. Additional results on other domains (e.g., vision-based or multi-agent tasks) would test generality.  
- **Novelty overlap:** Shares conceptual territory with recent LLM-for-environment design papers such as **LESR (Wang et al., 2024)**, **ExploRLLM (Ma et al., 2024)**, **Eureka (Ma et al., 2023)**, and **Text2Reward (Xie et al., 2023)**. The distinction lies mainly in targeting observation and action models rather than rewards.  
- **No real-world validation:** Physical robot experiments or noisy sensory settings would significantly strengthen claims of practical impact.


LOAM-Race is claimed to be the first use of LLM and checking different candidates. ExploRLLM (https://arxiv.org/abs/2403.09583) also does that. They also shape observation and action spaces.

### Questions
Along with weaknesses:
1. How does LOAM generalize to non-robotic domains (e.g., grid-worlds or visual navigation)?  
2. How sensitive is LOAM to LLM type and prompting format? Would smaller models (e.g., GPT-3.5) produce usable models?  
3. Could LOAM-Race be extended to also handle reward model design simultaneously?  
4. How does LOAM ensure the generated code’s physical plausibility (e.g., avoiding unfeasible joint mappings)?

### Soundness
3

### Presentation
4

### Contribution
3
