# PIPer: On-Device Environment Setup via Online Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Environment setup–the process of configuring the system to work with a specific software project–represents a persistent challenge in Software Engineering (SE). Automated environment setup methods could assist developers by providing fully configured environments for arbitrary repositories without manual effort. This also helps SE researchers to scale execution-based benchmarks. However, recent studies reveal that even state-of-the-art Large Language Models (LLMs) achieve limited success in automating this task. To address this limitation, we tune a specialized model for environment setup. We combine supervised fine-tuning for generating correct Bash scripts and Reinforcement Learning with Verifiable Rewards (RLVR) to adapt it to the task of environment setup. On EnvBench-Python, our method enables Qwen3-8B (a model runnable on consumer hardware) to perform on par with larger models–Qwen3-32B and GPT-4o. The training code and model checkpoints are available online: https://github.com/PIPer-iclr/PIPer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles repository environment setup by training a small open model with a two-stage recipe: supervised distillation from a stronger teacher to learn executable setup scripts, followed by online reinforcement learning with a lightweight, verifiable reward (LLM-as-a-judge predicting exit status and import issues). The goal is to close the gap between an 8B on-device model and larger closed/open foundation models under a single-turn “generate one script” protocol. The method is evaluated on EnvBench-Python, Repo2Run, and Terminal-Bench, reporting improved pass@N over same-size baselines and competitive scores relative to much larger models at substantially lower cost.

### Strengths
1) Practical problem with real engineering impact: automating per-repo environment setup is valuable for reproducibility, onboarding, and CI health.
2) Clear training recipe: SFT establishes executable script priors; online RL refines toward pass/failure-oriented behaviors with a cheap reward signal.
3) Sensible evaluation across multiple benchmarks (EnvBench-Python, Repo2Run), with pass@N reporting that reflects realistic multi-try usage.
4) Cost/efficiency perspective: shows that a tuned 8B model can approach larger models’ performance in the single-turn setting with better inference cost.

### Weaknesses
1) Only one base model (Qwen3-8B) is trained, which is not comprehensive enough.
2) The baselines are primarily foundation models; this does not sufficiently demonstrate the method’s effectiveness compared to task-specific systems.
3) Reporting under-emphasizes the common single-try metric: pass@1 is not tabulated across benchmarks. The paper includes a figure that visualizes an average pass@1 comparison but does not provide per-model pass@1 tables—and there is no explicit textual explanation for omitting tabulated pass@1—reducing comparability to standard single-attempt settings.

### Questions
1) Could you include results for a second open base model (e.g., another 7–9B family) to test robustness of the training recipe?
2) Can you add a task-specific baseline (e.g., a standardized agentic scaffold with controlled tools/budget) to better isolate method gains vs. model size?
3) Please provide per-repository runtime/cost breakdowns and scaling with repo size to support deployment decisions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes PIPER, a language model designed to configure terminal environments for successfully running projects and tasks. The authors fine-tune a smaller Qwen3 model using supervised fine-tuning (SFT) on training data distilled from a larger model of the same family. Given the high computational cost of executing generated bash scripts to verify correctness, the authors employ an LLM-as-a-Judge approach as a proxy for reward estimation during reinforcement learning (RL) training.

### Strengths
* The paper is clearly written and well-structured.
* The application of reinforcement learning to improve language models for environment configuration tasks is novel and practically relevant.
* **PIPER** achieves performance on par with larger models in benchmarks such as **Repo2Run** and **Terminal-Bench**, demonstrating promising efficiency gains.

### Weaknesses
### Overall

* There remains significant room for improvement: even the best-performing model (GPT-5) solves only **106/420** and **45/80** problems on **Repo2Run** and **Terminal-Bench**, respectively. Given that the larger **Qwen3-32B** model already performs poorly on these tasks (e.g., **29/329** on **EnvBench**), distilling from it is questionable. Typically, distillation is motivated by transferring the performance of a *highly capable* model to smaller ones, which does not seem justified in this setting.
* Using **LLM-as-a-Judge** as a proxy for script execution removes the verifiable reward component essential to reinforcement learning. It is unclear how accurate the judge model is at assessing correctness or identifying different error types. Although Figure 2 shows an increase in reward values during RL training, this does not translate to high absolute performance on **EnvBench**. This suggests that while Qwen3-8B is improving, its potential is constrained by the limited reliability of the feedback signal from the LLM-as-a-Judge. A systematic evaluation or profiling of the judge model’s accuracy is missing and should be included.
* Additionally, have the authors considered re-prompting the model with feedback from the environment when a setup attempt fails, and tasking it to iteratively refine the generated bash script? This strategy could be more effective than simply using pass@N, since both approaches involve multiple environment runs, but re-prompting may better guide the model toward the correct solution through targeted feedback rather than independent sampling.

---

### Data Collection

* It is unclear why the authors did not experiment with closed-source models such as **GPT-5** (or **Claude**) for collecting executable script data, even though these models are evaluated as baselines. Using more capable models for data generation could have produced higher-quality training data and potentially improved performance.

---

### Evaluation

* Given the high cost of executing model-generated shell scripts, **pass@1** would be a more suitable and interpretable metric for both in- and out-of-distribution benchmarks. What are the pass@1 numbers for all baselines and for PIPER on **Repo2Run** and **Terminal-Bench**? Such results would more clearly reveal each model’s ability to set up environments successfully on the first attempt.

---

### Generalization

* Although the authors defer exploration of other model families and scales to future work, an ablation or scaling analysis would be important to establish the general effectiveness of the proposed tuning method and to strengthen the paper’s contribution.
* The poor performance on **Terminal-Bench**, where PIPER underperforms even the base model, is insufficiently explained. One possible reason could be overfitting to the single-turn data distribution, resulting in the model only being effective on simpler, single-turn tasks from **Terminal-Bench**. The authors should discuss this and clarify the limitations of their approach.

### Questions
* See the Weaknesses Section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the task of automated software environment setup. The authors propose a two-stage training pipeline to fine-tune a small language model (Qwen3-8B) for this task. The method first uses supervised fine-tuning (SFT) on a dataset of executable scripts distilled from a larger "teacher" model (Qwen3-32B). This is followed by a reinforcement learning (RL) stage using a verifiable reward (RLVR), where the reward signal is provided by an "LLM-as-a-Judge" as a proxy for true script execution. The authors evaluate their final model on EnvBench-Python, Repo2Run, and Terminal-Bench, showing that their 8B model can achieve performance comparable to much larger models like GPT-4o on the primary benchmark.

### Strengths
1. The paper tackles a practical problem in software engineering. Automating environment setup is a well-known bottleneck, and a robust solution would have a significant real-world impact.
2. The core result—that a specialized 8B parameter model can match the performance of SOTA models like GPT-4o and a 32B model—is good.
3. The authors state their intention to release code, models, and generated scripts, which is commendable and valuable for the community.

### Weaknesses
1. The primary weakness of this paper is its limited contribution to methodology. The proposed "SFT + RLVR" pipeline is a well-established recipe. The SFT stage is standard knowledge distillation, and the RL stage applies known algorithms (REINFORCE++) with an LLM-as-a-Judge reward. This technique has been widely used. The paper presents a successful application of this pipeline to a new domain (environment setup) but does not introduce new learning principles or techniques that would be of broad interest to the ICLR community.
2. The paper's writing is clear but largely descriptive. It reads as a straightforward "we applied method X to problem Y and got good results". The narrative fails to build a strong case for why this specific application (environment setup) poses unique or interesting challenges that necessitate new ML insights, or what the broader ML community can learn from this specific application.
3. The paper's central claim of being "on par" with much larger models (GPT-4o, Qwen3-32B) is based on a flawed "apples-to-oranges" comparison. The authors compare their highly specialized, tuned 8B model against general-purpose, untuned (zero-shot) baselines. This comparison only proves the known fact that task-specific tuning is effective. A fair and more insightful comparison would require comparing against other tuned models. For instance, what is the performance of the Qwen3-32B "teacher" model after it also undergoes the SFT+RLVR training? The current setup compares a tuned small model to an untuned large model, which inflates the perceived relative performance of the 8B model.

### Questions
1. Given that the SFT+RLVR pipeline is standard, what do the authors believe is the core contribution? Did the environment setup task require specific, novel modifications to the RL algorithm or reward formulation that would be generalizable to other structured generation tasks?
2. The SFT data was filtered to include only successful executable scripts. Do the authors worry this introduces a bias? The model may not learn how to recover from or debug failed scripts, as it may not have seen such "failure-recovery" patterns in the SFT data. Did you experiment with including any "near-miss" or "productively-failed" scripts in the SFT dataset?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents PIPER, a specialized model for automating environment setup in software engineering projects. The authors address a persistent challenge in software development where configuring systems to work with specific repositories requires significant manual effort, which also limits the scalability of execution-based benchmarks for SE research. They develop their approach by combining supervised fine-tuning on scripts generated by larger models with RLVR, using a lightweight LLM-as-a-Judge reward function to avoid the computational overhead of executing scripts in isolated containers. The key contribution is demonstrating that their fine-tuned Qwen3-8B model—small enough to run on consumer hardware—achieves performance comparable to much larger models like Qwen3-32B and GPT-4o on the EnvBench-Python benchmark, showing more than 9× improvement over the base model. Through careful analysis of failure patterns in model-generated scripts and evaluation across multiple benchmarks including Repo2Run and Terminal-Bench, they show that their training approach develops genuine scripting capabilities that generalize beyond the training distribution, offering researchers and developers a more accessible and cost-effective solution for automated environment setup.

### Strengths
1. This paper tries to solve a very important problem, environment setup, which is useful for building coding environments for coding agents. This is a pain point in many questions. The work has strong practical significance for the SE research community. 
2. The key insight is to validate environment setup through LLM-as-a-Judge in reinforcement learning. In this way, it avoid execution overhead in containers. This can significanly reducing the training time and infrastructure requirements, while still preseve the debugging interpretability.
3. While the training pipeline itself is conventional, the analysis in this paper is very detailed. The paper evaluates across three diverse benchmarks (EnvBench, Repo2Run, Terminal-Bench) with different success criteria (static analysis, test collection, task completion), demonstrating the robustness of findings beyond a single evaluation protocol. The ablation studies clearly show both SFT and RL stages contribute meaningfully.

### Weaknesses
1. No validation of LLM-as-a-Judge reward wuality. The LLM-as-a-Judge is the core innovation, yet there's no empirical evidence it produces accurate predictions. 
2. The claim of 'on-device' is confusing. This paper is not about on-device training or on-device inference.

### Questions
1. Could you please provide correlation analysis between judge predictions and actual execution outcomes on a validation set? It would be interesting to see the false positive and false negatives. A detailed error analysis will be appriciated.
2. Why did GPT-4o and GPT-4o-mini fail as judges? Is there any detailed explanation?
3. How sensitive is performance to the judge prompt and criteria?

### Soundness
3

### Presentation
3

### Contribution
4
