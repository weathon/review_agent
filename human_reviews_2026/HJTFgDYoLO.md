# GenCtrl -- A Formal Controllability Toolkit for Generative Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
As generative models become ubiquitous, there is a critical need for fine-grained control over the generation process. Yet, while controlled generation methods from prompting to fine-tuning proliferate, a fundamental question remains unanswered: are these models truly controllable in the first place? In this work, we provide a theoretical framework to formally answer this question. Framing human-model interaction as a control process, we propose a novel algorithm to estimate the controllable sets of models in a dialogue setting. Notably, we provide formal guarantees on the estimation error as a function of sample complexity: we derive probably-approximately correct bounds for controllable set estimates that are distribution-free, employ no assumptions except for output boundedness, and work for any black-box nonlinear control system (i.e., any generative model). We empirically demonstrate the theoretical framework on different tasks in controlling dialogue processes, for both language models and text-to-image generation. Our results show that model controllability is surprisingly fragile and highly dependent on the experimental setting. This highlights the need for rigorous controllability analysis, shifting the focus from simply attempting control to first understanding its fundamental limits.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper establishes a theoretical foundation for controllable generative modeling from a control-theoretic perspective, framing generation controllability in terms of the classic concepts of reachability and controllability. It provides a rigorous theoretical proof to derive the conditions for controllability. Experimental results ultimately demonstrate that current generative models are fragile in terms of their controllability.

### Strengths
1. The paper introduces a control-theoretic framework to formalize the controllability of generative models. The theoretical analysis yields specific conditions for achieving controllability, offering valuable foundational support for advancing controllability algorithms and control input design.
2. Through evaluations across different task configurations and various LLMs, the paper demonstrates that controllability performance varies significantly with both the task and the specific LLM employed.

### Weaknesses
1. According to the Kalman controllability rank condition, the controllability of a dynamical system is determined by its intrinsic properties. For an LLM, if its inherent capability in a certain attribute is weak, we can postulate that its controllability regarding that attribute will consequently be poor. For instance, a 3B parameter model inherently possesses limited mathematical and reasoning capabilities. Therefore, if evaluated on larger models with significantly enhanced overall abilities, does the authors' conclusion that "LLM controllability is fragile" still hold?
2. As noted in reference [1], if the attributes manifested in the LLM's input are opposite to the target attributes, a substantial and longer control input is required for correction, which also reflects weaker controllability. Thus, in the experimental setup, should the authors clarify the input conditions and endeavor to avoid inputs with such opposing attributes?

[1] Fundamental Limitations of Alignment in Large Language Models

### Questions
See the weakness.

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
2

### Summary
This paper provides an interesting stand - it formalizes user-model interaction as a control process, then gives PAC-style sample-complexity bounds for estimating reachable and (partially) controllable output sets of black-box generative models. A pair of MC algorithms (reachability and controllability) come with finite-sample guarantees. Experiments across a large variety of models suggest that, the controllability that we usually depend on is fragile and highly task/model dependent.

### Strengths
1) The formalization of dialogue-as-control is clear and rigorous.

2) The paper provides PAC-style, distribution-free bounds, avoiding intractable verification of global smoothness, thus suits black-box examinations well.

3) The proposed algorithms are simple and easy to follow.

4) The eventually alerting conclusion reveals the fragility of controllability in LLMs to the community, which could be of higher impacts.

### Weaknesses
1) Please correct me if I misunderstood this: how about when N -> \infty? In practice this is easy due to the dimensionality curse, and thus the total demanded sampling number by THM1 can explode to some extent.

2) The role of the readout h is a little bit less clearly demonstrated. In my understanding, isn't it the case that its calibration can be very hard in practice when considering the general case? I would appreciate if the authors can provide guidance for me to better understand how this part can work robustly.

### Questions
Please refer to Weakness).

### Soundness
4

### Presentation
4

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
This paper proposes a controllability framework to assess whether black-box generative models are controllable prior to designing control mechanisms. The authors model user-model interaction as a control process and introduce Monte Carlo algorithms to estimate the set of achievable outputs. These algorithms are supported by probably-approximately correct (PAC) guarantees. Applying this framework to several LLMs and T2IMs, the paper concludes that controllability is a fragile, task-dependent property, challenging the implicit assumption that these models are generally controllable.

### Strengths
1. The central idea of shifting focus from how to control models to if they can be controlled is a valuable and novel contribution. It brings a much-needed layer of formal scrutiny to a field dominated by empirical trial-and-error. 

2. The paper is well-written, and the authors do an admirable job of introducing concepts from control theory and adapting them to the context of generative models, particularly with their discussion of the discrete bottleneck.

### Weaknesses
1. The paper claims its framework can assess any control mechanism, including finetuning and representation engineering. Yet, the experiments are confined exclusively to prompting, arguably the weakest and least reliable form of control.

2. The framework is not scalable to complex, high-dimensional control problems. The sample complexity scales with the number of discretized output bins, N. For any realistic task involving the control of multiple attributes simultaneously (e.g., style, tone, and factual accuracy), the resulting number of bins would become combinatorially explosive, making the method computationally infeasible. 

3. Furthermore, the analysis is purely descriptive; it tells us that a model is uncontrollable for a given task, but provides no diagnostic insight into why it fails or how one might improve the system. This severely limits its practical value beyond a simple and potentially misleading assessment.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a novel and rigorous control-theoretic formalization of dialogue processes (DPs) in generative models, introducing definitions of reachability and controllability under a probabilistic and coarse-grained framework. The Monte Carlo–based estimation with PAC-style bounds is theoretically sound and well-motivated.
 However, several conceptual and practical issues remain: the definition of controllability in discrete, high-dimensional generative systems is somewhat ambiguous; key assumptions (boundedness, independence) are idealized . Overall, it is a theoretically interesting and well-written paper, but its practical implications and empirical validation need strengthening.

### Strengths
This paper proposes a control-theoretic framework to analyze and estimate the controllability and reachability of generative models (LLMs, T2IMs) viewed as black-box dynamical systems. The method provides PAC-style probabilistic guarantees for estimating controllable sets using Monte Carlo sampling, under minimal assumptions.
Strengths：
1.Theoretical originality – Introduces a formal and model-agnostic definition of controllability for generative systems, offering a new lens to study model behavior. 
2.Sound probabilistic formulation – Provides PAC-based guarantees that make the estimation process interpretable and principled.
3.Clear empirical illustration – Demonstrates differences in controllability across models and tasks, supported by open-source implementation.

### Weaknesses
Conceptual Level: Ambiguity of Controllability Definition
The core concept of the paper, controllability, lacks a clear physical and semantic interpretation in the context of generative models. In classical control theory, controllability implies the existence of a set of inputs capable of steering the system from any initial state to any desired final state. However, for LLMs or T2I models: The boundaries of the state space X and input space U are semantically unclear;  Prompts or noise vectors are not strict control signals—they are discrete, symbolic inputs mapped nonlinearly to a latent space。Suggestion: Include an analysis (e.g., in the appendix) of the mapping between control inputs and semantic behaviors, clarifying the real-world meaning of the chosen control variables.

Experimental Depth and Analysis of Controllability Failures：
While the empirical examples, such as formality control and object counting, are illustrative, their interpretation remains relatively shallow. The study does not compare against simpler or alternative baselines, such as linear regression, RLHF-based control, or direct output editing. Moreover, the analysis of controllability failures—like overshoot phenomena—is limited to surface-level descriptions, without a deeper causal diagnosis. 
Suggestion: Include qualitative case studies or detailed error analyses to clarify why certain models exhibit limited controllability and to provide more actionable insights.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
