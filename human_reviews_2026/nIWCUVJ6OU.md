# MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Modeling and solving optimization problems via large language models (LLMs) has attracted increasing attention recently. Although both prompt-based and learning-based methods have achieved progress, they remain limited by their reliance on large data volumes, high-quality annotations, expensive intermediate step verification, and huge computational overhead. From a data privacy perspective, a low-cost localized deployment of small-scale LLMs is of significant value. To train a small-scale LLM with excellent optimization generalization under limited resources, this paper proposes a reasoning to model and solve paradigm called MiniOpt based on reinforcement learning (RL) with verifiable reward. To reduce the demand for training data, MiniOpt adopts two-stage RL training. In the first stage the model quickly learns the model-and-solve paradigm and in the second stage it acquires strong optimization generalization ability. To reduce the cost of verifying the response of LLMs, OptReward in MiniOpt verifies the completeness of problem modeling and avoids the need for content validation. The above techniques enable the training of small-scale LLMs with strong optimization generalization ability under limited resources, thereby resulting in low inference cost for localized deployment and usage. Extensive experiments show that MiniOpt-3B exhibits strong optimization generalization across various optimization types and scenarios. For models with parameters fewer than 10B, MiniOpt-3B achieves the highest average solving accuracy (SA). For models with more than 10B parameters, MiniOpt-3B still shows competitive performance. Notably, MiniOpt-3B indicates superior SA on the hard OptMATH-Bench while only consuming 37.64% of the average output tokens required by DeepSeek-R1. The code is available at https://anonymous.4open.science/r/MiniOpt-6194.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **MiniOpt**, a framework for training small-scale large language models (LLMs) to model and solve optimization problems from natural language descriptions. It addresses key challenges—data scarcity, high verification costs, and computational overhead. By proposing a two-stage reinforcement learning (RL) pipeline with a verifiable reward function (**OptReward**) and an improved RL algorithm (**OptGRPO**). MiniOpt employs a structured “reasoning to model and solve” paradigm, where the model first formulates problems using a five-element representation and then generates executable Pyomo code. Evaluated across nine benchmarks, MiniOpt-3B outperforms many larger models, demonstrating strong optimization generalization with significantly lower resource requirements, making it suitable for local deployment in resource-constrained environments.

### Strengths
1.  **Strong Performance with High Efficiency:** MiniOpt achieves state-of-the-art or highly competitive solving accuracy (SA) on multiple benchmarks, even with its small 3B parameter size, outperforming many larger models like GPT-5 and LLMOPT-14B. Crucially, it does so while consuming significantly fewer computational tokens, making it highly efficient and cost-effective for local deployment.

2.  **Comprehensive and Rigorous Experiments:** The paper validates its claims through extensive experiments across nine diverse optimization benchmarks, spanning various problem types and real-world scenarios. This thorough assessment, complemented by detailed ablation studies, robustly demonstrates the model's generalization capability and the contribution of each proposed component.

3.  **A Novel, Resource-Conscious Training Framework:** The core contribution is a meticulously designed pipeline that effectively trains small-scale LLMs under limited resources. The integration of a two-stage RL process, a verifiable reward function (OptReward), and strategic data selection successfully overcomes the data-hungry and computationally intensive nature of traditional methods for this complex task.

### Weaknesses
### **Weaknesses 1**
A significant limitation is the framework's tight coupling with the **GurobiPy/Pyomo ecosystem** during both training and evaluation. The SFT data is derived from GurobiPy conversions, and the output is constrained to Pyomo code targeting a limited set of solvers. This design lacks **solver-agnostic flexibility**. A model might correctly understand a problem's structure but fail due to Pyomo-specific syntax or the selected solver's limitations, while an alternative solver or modeling language (e.g., CVXPY, PuLP, or PySCIOPT) could have succeeded. This restricts the generalizability and practical utility of the approach across the diverse landscape of optimization tools.


### **Weaknesses 2**
Another limitation is the **lack of diagnostic metrics** to disentangle the root causes of failure. The final reward and accuracy scores conflate two distinct types of errors:
1.  **Conceptual Modeling Errors:** The model fails to reason correctly and produces an **inaccurate mathematical formulation**.
2.  **Solver Programming Errors:** The model understands the problem but generates syntactically incorrect or semantically flawed Pyomo/Gurobi code.

Without a separate, fine-grained evaluation of the five-element formulation's correctness *before* code generation, it is impossible to diagnose whether failures stem from a lack of reasoning or merely implementation incompetence. This obscures the true bottleneck for model improvement.

### Questions
1.  **Error Diagnosis:** Can you break down where the model fails most? Is it primarily in the *mathematical reasoning* (producing a wrong formulation) or in the *code implementation* (writing bad Pyomo code)?

2.  **Generality:** Is your model's performance tied to Pyomo? Would it fail if asked to generate code for a different optimization tool like CVXPY, or has it learned a general skill for modeling optimization problems?

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
4

### Summary
This paper proposes a training and evaluation framework for learning to model natural language and solve optimization problems with small-scale LLMs under resource constraints. The main methods include (1) structuring the task into a paradigm with <think>...</think> (five-element model) and <answer>...</answer> (executable Pyomo code); (2) designing a verifiable reward function OptReward (format/five elements/numerical accuracy) to ensure stable output of the five elements through a two-stage RL training process; (3) selecting a suitable RL algorithm from the existing GRPO trick.

### Strengths
1.  A clear paradigm (<think> five elements + <answer> executable scripts) with verifiability, reducing the labor cost of annotation and verification.
2.  Detailed experimental analysis (demonstrating the contributions of modules such as SFT warm-up, OptReward, and OptGRPO).
3.  Advantages in practical evaluation metrics (SA, ER, and token cost).

### Weaknesses
1. The approach lacks innovation, as it merely enables the LLM to produce a five-element structure consistent with the verification format. Such verifiable schemes are fairly common in LLM optimization.
2. It does not offer significant improvements to the GRPO algorithm; rather, it incorporates a few well-established techniques from previous GRPO studies.
3. The paper failed to compare its approach with several closely related studies. Notable omissions include: [1] ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling. (2025). [2] Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation (2025).

### Questions
1. In the method description (Figure 1), the upper bound for gradient clipping in the RL algorithm is incorrectly specified. The formula should use a '+' symbol instead of a '–'."
2.  Could the authors provide ablation studies to demonstrate the impact of removing KL divergence and asymmetric clipping on training stability and performance?
3.  The paper would benefit from a performance comparison against other competitive learning-based LLM models for optimization modeling and solving.

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
5

### Summary
This paper proposes MiniOpt, a framework to train small-scale LLMs (e.g., 3B) to solve optimization problems under limited resources. The method uses a three-stage training pipeline (SFT Warm-up, RL Stage-1, and RL Stage-2). This pipeline is guided by a verifiable reward function (OptReward) and a RL algorithm (OptGRPO). Experiments show the MiniOpt-3B model achieves solving accuracy competitive with models larger than 10B parameters

### Strengths
1. **Strong Performance-to-Parameter Ratio:** The paper demonstrates a strong empirical performance-to-parameter trade-off. The 3B model, MiniOpt-3B, achieves an average solving accuracy of 56.94%. This result is shown to be competitive with several models in the >10B class. 

2. **Reported Inference Efficiency:** The method shows a clear advantage in inference cost, aligning with its primary goal of low-cost deployment. On the challenging OptMATH-Bench, MiniOpt-3B is reported to achieve a higher SA than DeepSeek-R1 while consuming fewer average output tokens.

3. **Systematic Training Pipeline and Reward Design:** The paper contributes a complete and structured training pipeline, beginning with an SFT warm-up to mitigate sparse rewards, followed by a two-stage RL process for paradigm acquisition and generalization .

### Weaknesses
1. **Stated Motivation Lacks Novelty:** The paper's primary motivation—addressing data privacy and deployability through small, open-source models—is not a new contribution to this specific research area. This motivation was already a core premise for foundational work like ORLM and has been a shared consideration in subsequent studies such as OptMATH , ReSocratic, and SIRL . Most of these works train 7B-8B models, and MiniOpt's 3B variant represents only a quantitative reduction in scale rather than a fundamentally different problem setting. Therefore, the problem framing does not offer a novel perspective or tackle a new challenge relative to the established literature.

2. **Contradictory Claims Regarding "Limited Resources":** The paper's premise is critically undermined by a direct contradiction. It claims to operate under "limited resources" and highlights the difficulty of acquiring high-quality data. However, the crucial SFT Warm-up stage relies on the OptMATH-Train dataset , which the paper states contains more than 100K samples. This is a massive dataset, far exceeding the ~30K samples used in related works like ORLM and ReSocratic. This methodology is not "resource-limited" and directly conflicts with the paper's central motivation.

3. **Incomplete Treatment of Related Work:** The paper omits critical comparisons with relevant work:
(1) Step-Opt[1]: This work also addresses stepwise optimization modeling with 7B-8B models but is not cited.
(2) Missing DAPO comparison: The paper states OptGRPO "builds upon" prior improvements like DAPO but provides no experimental comparison with DAPO itself.

[1] Wu Y, Zhang Y, Wu Y, et al. Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation[J]. arXiv preprint arXiv:2506.17637, 2025.

### Questions
1. Regarding the paper's motivation, the focus on deployability and privacy for small models is shared by prior work (ORLM, SIRL, etc.). Could you clarify what you see as the novel problem setting or challenge that MiniOpt addresses, beyond a quantitative reduction in model scale from the established 7B/8B range to 3B?

2. This paper is premised on operating under "limited resources," yet the SFT Warm-up stage relies on the 140K-sample OptMATH-Train dataset. This appears to be a much larger data requirement than related works. Could you clarify this apparent contradiction and explain how this massive SFT dataset aligns with the "limited resources" claim?

3. Please include a baseline comparison against DAPO to experimentally validate the algorithmic contribution.

### Soundness
3

### Presentation
3

### Contribution
2
