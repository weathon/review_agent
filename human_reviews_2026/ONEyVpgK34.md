# Lang-PINN: From Language to Physics-Informed Neural Networks via a Multi-Agent Framework

- Decision: Reject
- Scores: 6, 4, 0, 4

## Abstract
Physics-informed neural networks (PINNs) provide a powerful approach for solving partial differential equations (PDEs), but constructing a usable PINN remains labor-intensive and error-prone. Scientists must interpret problems as PDE formulations, design architectures and loss functions, and implement stable training pipelines. Existing large language model (LLM) approaches address isolated steps such as code generation or architecture suggestion, but typically assume a formal PDE is already specified and therefore lack an end-to-end perspective. We present **Lang-PINN**, an LLM-driven multi-agent system that builds trainable PINNs directly from natural language task descriptions. **Lang-PINN** coordinates four complementary agents: a \emph{PDE Agent} that parses task descriptions into symbolic PDEs, a \emph{PINN Agent} that selects architectures, a \emph{Code Agent} that generates modular implementations, and a \emph{Feedback Agent} that executes and diagnoses errors for iterative refinement. This design transforms informal task statements into executable and verifiable PINN code. Experiments show that **Lang-PINN** achieves substantially lower errors and greater robustness than competitive baselines: mean squared error (MSE) is reduced by up to 3–5 orders of magnitude, end-to-end execution success improves by more than 50\%, and reduces time overhead by up to 74\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Lang-PINN, a novel multi-agent framework driven by large language models (LLMs) designed to automate the creation of trainable Physics-Informed Neural Networks (PINNs) directly from natural language descriptions of a problem. The traditional process of building PINNs is labor-intensive, requiring users to manually formulate Partial Differential Equations (PDEs), design neural network architectures, and implement training pipelines. Lang-PINN addresses this challenge by decomposing the workflow among four cooperating agents:
1.	PDE Agent: Parses the natural language description to formulate a symbolic PDE.
2.	PINN Agent: Selects a suitable neural network architecture for the formulated PDE.
3.	Code Agent: Generates a modular and verifiable implementation of the PINN.
4.	Feedback Agent: Executes the code, diagnoses errors, and provides feedback for iterative refinement.
The paper's main contributions are the proposal of the first end-to-end framework that transforms natural language task descriptions into complete and verifiable PINN solutions ; the construction of a new benchmark dataset, Task2PDE, for evaluating semantic-to-symbolic grounding ; and experimental results demonstrating that Lang-PINN substantially reduces mean squared error (MSE) by up to 3-5 orders of magnitude, improves code execution success rate by over 50%, and reduces time overhead by up to 74% compared to competitive baselines.

### Strengths
1.	Originality and Significance: The paper's primary strength is its novelty in proposing the first truly end-to-end framework for generating PINNs from natural language. This addresses a significant bottleneck in the practical application of PINNs, making a powerful scientific tool more accessible to domain experts who may not have specialized knowledge in programming or machine learning.
2.	Robust and Comprehensive Evaluation: The experimental validation is a major strength. The authors perform a thorough evaluation on a diverse set of 14 PDEs from a recognized benchmark. The comparison against multiple strong baselines and a human-expert-designed reference (PINNacle) clearly demonstrates the superiority of their method in terms of accuracy, executability, and efficiency.
3.	Well-Designed and Validated System Architecture: The decomposition of the complex task into a modular, multi-agent system is an elegant and effective design. Each agent has a clear responsibility, and the feedback loop ensures the system is robust and capable of self-correction. The value of each agent is individually confirmed through rigorous ablation studies, which adds a high degree of confidence in the proposed architecture.

### Weaknesses
1.	Lack of Wall-Clock Time Analysis: Efficiency is measured by the number of iterations required to generate an executable program. While this is a useful metric, it does not capture the full computational cost. The agentic workflow itself, involving multiple LLM calls, consensus voting, code execution, and feedback analysis, likely incurs significant latency. An analysis of the end-to-end wall-clock time, compared to baselines, would provide a more practical assessment of the system's efficiency.
2.	Insufficient Detail on the PINN Agent's Knowledge Base: The paper states that the PINN Agent uses a knowledge base K and a history cache H to guide architecture selection. However, the details of how this knowledge base is constructed, curated, and maintained are sparse.
3.	Critical Reliance on Initial PDE Formulation and Ambiguity in Error Attribution: The framework's overall validity is critically reliant on the fidelity of the initial PDE Agent. Any error in the language-to-symbol translation (e.g., misinterpreting operators or boundary conditions ) necessarily invalidates the entire downstream pipeline, as all subsequent architecture selection and code generation are predicated on a flawed physical assumption. Furthermore, a significant ambiguity exists in the feedback loop. The paper suggests the Feedback Agent can diagnose and escalate errors, but it does not present a clear mechanism for attributing the root cause of a failure. For instance, poor performance (e.g., high MSE ) could originate from a fundamental PDE formulation error, a suboptimal architecture choice (e.g., poor convergence), or a bug in the generated code. The framework lacks a demonstrated method to disambiguate these distinct failure modes, which may present similar symptoms but require entirely different corrective actions.

### Questions
1.	Regarding the PINN Agent's knowledge base (K): Could you please elaborate on the construction and contents of this knowledge base? Is it manually curated, and if so, how extensible is it? How does the agent perform when faced with a PDE whose characteristics do not have a clear mapping to an architecture within the existing knowledge base?
2.	Regarding end-to-end efficiency: In addition to the number of refinement iterations, could you provide a comparison of the total wall-clock time required by Lang-PINN to generate a final solution, versus the time taken by the baselines? This would be very helpful for understanding the practical overhead of the multi-agent framework.
3.	Regarding the PDE Agent's consensus voting: What is the system's behavior if the initial set of candidate PDEs generated via CoT does not result in a clear consensus? For example, in a case with several distinct and equally-plausible interpretations of the language, is there a fallback mechanism or a method to query the user for clarification?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Lang-PINN introduces a multi-agent LLM framework that automatically builds physics-informed neural networks from natural language descriptions, achieving 3-5 orders of magnitude MSE reduction and >50% execution success improvement over baselines.

### Strengths
1. The proposed method achieves substantial improvements over competitive baselines across multiple metrics—MSE, success rate, and convergence speed.
2. The Code Agent's modular generation with interface constraints and PDE loss verification ensures correctness and enables localized debugging, a clear improvement over monolithic approaches.

### Weaknesses
1. The framework primarily orchestrates existing techniques (CoT reasoning, consensus voting, modular code generation, feedback refinement) without introducing fundamentally new methods for PDE formulation, architecture search, or code synthesis.
2. In realistic scientific workflows, researchers typically know the governing PDE equations explicitly; the natural language-to-PDE translation step addresses a largely artificial problem rather than a genuine bottleneck in PINN deployment.
3. The Task2PDE dataset covers only 8 PDEs, and the linguistic complexity levels (especially Level 3-4) introduce irrelevant narrative details (e.g., "coffee machine" analogies) that test language understanding rather than scientific reasoning, undermining the validity of the PDE formulation evaluation.

### Questions
1. The caption of Figure 1 lacks vertical spacing from the main text, reducing readability—please add appropriate whitespace for consistency with standard formatting. 
2. The semantic consistency metric for PDE validation is mentioned but never formally defined—could the authors provide the exact formulation, embedding method, and similarity threshold used in Section 4.2?
3. Does the PINN Agent perform full hyperparameter optimization (e.g., layer depth, width, learning rate) via MCTS or similar search, or does it simply select from a fixed set of pre-configured architectures with predetermined hyperparameters?
4. Can Lang-PINN handle inverse problems, data assimilation, or parameter estimation tasks, or is it limited to forward PDE solving?
5. While Table 2 reports iteration counts, the paper lacks LLM inference cost, and memory consumption comparisons against baselines which is crucial since one of the core motivation is to accelerate the PDE solving.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper introduces Lang-PINN, an LLM-driven multi-agent system that automatically builds trainable Physics-Informed Neural Networks (PINNs) directly from natural language task descriptions. This framework uses four cooperating agents—a PDE Agent to formulate symbolic equations, a PINN Agent to select architectures, a Code Agent to generate modular code, and a Feedback Agent to execute and refine the solution. Experiments show this end-to-end approach significantly outperforms baselines.

### Strengths
The author leak the information that the paper is under review on ICLR 2026 in its preprint. The reviewer happens to have read this paper before, so my review might be biased. Please ignore my rating.

### Weaknesses
The author leak the information that the paper is under review on ICLR 2026 in its preprint. The reviewer happens to have read this paper before, so my review might be biased. Please ignore my rating.

### Questions
The author leak the information that the paper is under review on ICLR 2026 in its preprint. The reviewer happens to have read this paper before, so my review might be biased. Please ignore my rating.

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
4

### Summary
This paper proposes Lang-PINN, a novel multi-agent system that converts natural-language descriptions of scientific problems into executable Physics-Informed Neural Networks (PINNs). The framework orchestrates four specialized agents:
1.	PDE Agent – interprets natural language into mathematical PDE formulations using symbolic–semantic analysis and consensus voting.
2.	PINN Agent – selects appropriate neural architectures (e.g., MLP, CNN, GNN, Transformer) according to the PDE’s characteristics.
3.	Code Agent – performs modular code generation for the full PINN pipeline.
4.	Feedback Agent – executes, analyzes, and iteratively refines outputs based on runtime and convergence metrics.
Experiments on 14 standard PDEs (PINNacle benchmark) demonstrate dramatically improved execution success rate and reduced training time compared to previous LLM-based automation systems.

### Strengths
This work lies at a very interesting intersection between AI4Science and LLM-agent systems. Conceptually, the idea of translating language to executable PINNs is innovative and potentially impactful for scientific automation. The overall contribution is valuable but currently more proof-of-concept than fully convincing system.
•	Originality: First framework to bridge natural-language scientific description → PDE → trainable PINN code via explicit multi-agent collaboration.
•	Clarity: Modular design and workflow are carefully explained, with visual clarity and step-by-step examples.
•	Technical quality: Incorporates multiple robust reasoning strategies (symbolic equivalence, semantic matching, feedback metrics) that are well-engineered.

### Weaknesses
1.	Reproducibility and dependency on specific LLMs:
The paper does not clearly document the language model configuration—such as model type, version, temperature, or prompting strategy—used in each stage of Lang-PINN. As a result, reproducibility is limited, and the robustness of the method under different LLMs remains uncertain. The approach appears to rely heavily on the reasoning ability of a specific proprietary model, which might not generalize to smaller or open-source alternatives.
If the authors provide detailed model configurations and cross-model performance comparisons, my confidence in reproducibility and robustness would significantly improve.
2.	Unclear implementation of neural architectures (MLP, CNN, GNN, Transformer):
The paper briefly mentions that different network types are selected by the PINN Agent based on PDE characteristics, but it never explains how these architectures are actually implemented or adapted to the PINN setting. Traditional PINNs typically use simple MLPs, while CNNs, GNNs, or Transformers require specific spatial or graph structures to encode PDE constraints. Without details on network design, training loss formulation, or integration into the physics-informed loss, it is unclear how these variants differ from standard PINN implementations or how the framework ensures each model handles the PDE correctly.
3.	Lack of theoretical justification:
The symbolic–semantic similarity metrics and feedback indicators introduced in the paper are entirely heuristic. There is no theoretical analysis or formal reasoning explaining why these metrics should correlate with correct PDE reconstruction, stable network training, or improved convergence. The system demonstrates strong empirical results, but a deeper theoretical foundation would strengthen the overall credibility and generalizability of the proposed approach.

### Questions
1.	Reproducibility and LLM dependence:
Could the authors clearly specify the LLM configuration used at each stage of Lang-PINN (model name, version, temperature, prompting strategy, etc.)? Have you tested any smaller or open-source LLMs (such as Llama, Mistral, or DeepSeek) to evaluate whether the pipeline’s stability and reasoning ability generalize beyond a single proprietary model? 

2.	Implementation details of the neural architectures (MLP, CNN, GNN, Transformer):
The paper states that the PINN Agent selects different architectures based on PDE characteristics, but it remains unclear how these networks are specifically constructed and trained under the physics-informed setting. Could the authors elaborate on:
1）How the physics-informed loss is formulated for CNNs, GNNs, and Transformers (which require spatial/graph structures)?
2）Whether these architectures are pre-designed templates or automatically synthesized by the agent?
3）More concrete architectural or training details would help readers understand the novelty and scalability of this component.

3.	Theoretical understanding of metrics and feedback mechanisms:
I understand that current LLM-based agent frameworks and retrieval-augmented feedback loops are largely black-box systems, but some interpretability or empirical linkage would clarify the reliability of Lang-PINN’s internal metrics. Have you observed any quantitative correlation (e.g., higher semantic score → lower PDE reconstruction error or better final loss)? Providing such evidence or at least an empirical trend analysis could substantially strengthen the methodological validity of the system.
4.	Definition of success rate in Table 2:
In Appendix Table 2, Lang-PINN is compared against baselines (including PINNacle) using a “success rate (%)” metric averaged over 10 runs. Could the authors clarify precisely how success is defined?
Is a run deemed “successful” when the generated code executes without errors, or only if the resulting model achieves acceptable training convergence or physical consistency (e.g., loss threshold)?

### Soundness
3

### Presentation
3

### Contribution
3
