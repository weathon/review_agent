# PhysCodeBench: Benchmarking Physics-Aware Symbolic Simulation of 3D Scenes via Self-Corrective Multi-Agent Refinement

- Avg Score: 5.50
- Decision: Reject
- Scores: 10, 4, 4, 4

## Abstract
Physics-aware symbolic simulation of 3D scenes is critical for robotics, embodied AI, and scientific computing, requiring models to understand natural language descriptions of physical phenomena and translate them into executable simulation environments. While large language models (LLMs) excel at general code generation, they struggle with the semantic gap between physical descriptions and simulation implementation. We introduce PhysCodeBench, the first comprehensive benchmark for evaluating physics-aware symbolic simulation, comprising 700 manually-crafted diverse samples across mechanics, fluid dynamics, and soft-body physics with expert annotations. Our evaluation framework measures both code executability and physical accuracy through automated and visual assessment. Building on this, we propose a Self-Corrective Multi-Agent Refinement Framework (SMRF) with three specialized agents (simulation generator, error corrector, and simulation refiner) that collaborate iteratively with domain-specific validation to produce physically accurate simulations. SMRF achieves 67.7 points overall performance compared to 36.3 points for the best baseline among evaluated SOTA models, representing a 31.4-point improvement. Our analysis demonstrates that error correction is critical for accurate physics-aware symbolic simulation and that specialized multi-agent approaches significantly outperform single-agent methods across the tested physical domains.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This work proposes a benchmark for evaluating the quality of code-generation systems for generating accurate physics simulations. A training set is developed for this benchmark, which is then used to fine-tune local models in a new "self-corrective multi-agent refinement framework" that outperform SOTA proprietary models on this task.

### Strengths
- The topic of using code-gen agents to generate physics simulations is very relevant to the AI community, and the benchmark proposed in this paper is a solid contribution to the field.
- I appreciate the user study, which validated the use of ClipScore as an evaluation metric.
- The SMRF training pipeline follows standard practices (SFT and DPO), and the ablation study demonstrates the utility of each component.

### Weaknesses
- The pipeline for creating the dataset requires significant human oversight, including (1) creating initial seed prompts, (2) filtering AI-generated prompts, (3) validating simulation code, and (4) adding metadata and preference scores. I think this is excellent for validation/test sets, but this somewhat limits the scalability of the training set.
- Figure 2 is hard to parse without zooming in significantly; the inner text and images are too small.

### Questions
- At inference time, each model is provided with 100K tokens of documentation, but the paper also says that the maximimum context length is 32K tokens for the local models. What is actually done at inference time for these local models? Furthermore, it is commonly reported that LLM capabilities degrade at long contexts (along with being wasteful due to the quadratic attention of self-attention); have you tested reducing the length of the documentation or some tool-use system to provide a more manageable context length?
- Given the success of ClipScore in evaluation, have you tried using it (or other VLMs) as a filtering mechanism for outputs of SOTA models? In other words, generate k different outputs from a SOTA LLM, choose the output that gives valid code and has the best ClipScore. That seems like it could significantly boost scores for proprietary models.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel benchmark for simulation curation coding. The benchmark uses genesis simulator and test the ability to produce corresponding physical simulatin given the text prompt. The evaluation proposes a corresponding metric with a combination of code accuracy and visual accuracy. it also provides a dataset, curateing after semi-automatic process, which involves human expert labeling.
In addition to the benchmark, the method proposes a framework for finetuning LLM for the simulation coding task. It involves agenerator, a error corrector and involves DPO to improve based on human preference.
Experiment suggest that the benchmark is challenging and existing pipeline can solve simple cases like water drops.

### Strengths
1. The paper is well-written and easy to understand.
2. The problem setup is well-defined and the tools for benchmarking is well-provided.

### Weaknesses
1. The tasks are relatively simple. For benchmarking, when should include more challenging environment or stratify them to different difficulty level.
2. The metric for coding / visual is more qualitative. Looking similar does not emphasize accuracy of the code. For example, if i want to have a robot with 4 legs and it give me 5 legs. One should certainly punish such serious errors.
3. Limited to Genesis. A comprhensive benchmark should also consider other platforms like IsaacSim.

### Questions
1. How large is the dataset? How much effort is cost to build the dataset?

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
4

### Summary
This paper aims to solve the "semantic gap" that Large Language Models (LLMs) face when translating natural language descriptions into physically accurate 3D simulation code. The authors point out that code generated by existing LLMs often leads to simulation failures, bugs, or incorrect physical parameters.

### Strengths
- First-of-its-Kind Benchmark: PhysCodeBench is the first comprehensive benchmark in this domain. It provides not only a dataset but also detailed metadata (like difficulty and physical laws), laying a foundation for future research .
- Comprehensive Evaluation: The paper conducts not only quantitative analysis but also uses qualitative comparisons (Figure 5) to demonstrate SMRF's superiority in simulating fluid ripples and complex collapse dynamics . Furthermore, a user study with 10 participants was conducted, validating SMRF's lead in physical accuracy and code usefulness.

### Weaknesses
- Reliance on High-Level APIs: The entire benchmark and framework are heavily dependent on a single physics engine named "Genesis". The model primarily learns how to correctly call this specific library's API, rather than how to implement physics simulations from first principles (e.g., physical equations) .
- Superficial Evaluation Metrics: The PhysCodeEval evaluation framework (100 points) may fail to measure true physical accuracy.
  - Code Quality (50 points): Only evaluates whether the code can "successfully execute" and "generate files", not code efficiency or structural quality.
  - Simulation Fidelity (50 points): Relies on proxy metrics. S_clip measures "semantic similarity" between the video and text (e.g., if "ball" and "trampoline" are present), not physical correctness (e.g., if the ball's bounce follows Hooke's Law) . S_motion only assesses "motion smoothness" ; a simulation that is completely wrong physically (e.g., no acceleration) could still be "smooth".

### Questions
- Generalization: If the SMRF framework were applied to a completely new physics engine (e.g., MuJoCo or PyBullet), would it fail completely? How much of the knowledge learned by the framework is "physics logic" versus "Genesis API syntax"?
- Verification of Physical Laws: Given that PhysCodeEval relies on proxy metrics, to what extent do the simulations generated by SMRF truly adhere to core physical laws (e.g., conservation of energy, conservation of momentum)? Has any quantitative verification of this been performed?
- Robustness of Correction: How does the Error Corrector (EC) handle errors that are "physically unreasonable but syntactically correct" (e.g., setting gravity to an unrealistic value)? How strong is its diagnostic capability for this type of semantic error?

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
3

### Summary
The paper proposes PhysCodeBench, a benchmark for evaluating physics-aware symbolic simulation, a code gen problem where models must generate executable simulation code from natural language descriptions. The benchmark provides a dataset of 700 human-selected prompts and corresponding code for physical simulation scenarios across rigid and soft body simulation, fluid dynamics, and mechanics. The authors further propose the Self-Corrective Multi-Agent Refinement Framework (SMRF), which includes 3 modules: the Simulation Generator (generates initial code), Error Corrector (fix execution errors), and Simulation Refiner (improve code). The authors test SMRF on PhysCodeBench, where it achieves better performance compared to zero-shot and single-agent finetuned baselines.

### Strengths
- The Simulation Generator, Error Corrector, and Simulation Refiner framework is interesting, and their respective optimization procedures are presented clearly
- The data collection procedure is covered in detail in the appendix
- The authors evaluate against a number of relevant baselines, demonstrating improved performance with the multi-agent refinement framework

### Weaknesses
- Baselines include zero-shot and finetuned models, and ablations remove the Error Corrector or Simulation Refiner individually, however the baselines do not include a refinement framework which leverages a single model to fix and refine the generated code given error descriptions
- The human preference study has a fairly small sample size (10 participants)

### Questions
- How well does a single-agent iterative refinement framework perform compared to the SMRF framework?
- The authors write, "Robotics applications like SimGen (Zhou et al., 2024), VoxPoser (Huang et al., 2023), and Code as Policies (Arenas et al., 2024) generate simulation environments for robot task planning." However, to my knowledge, neither of these works generate simulation environments. Code as Policies writes policies with hierarchical code generation, and VoxPoser extracts affordance and constraint maps via LLM-generated code. Can the authors clarify what is meant by this statement?

### Soundness
2

### Presentation
3

### Contribution
3
