# ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Autoformalization, which translates natural language mathematics into machine-verifiable formal statements, is critical for using formal mathematical reasoning to solve math problems stated in natural language. While Large Language Models can generate syntactically correct formal statements, they often fail to preserve the original problem's semantic intent. This limitation arises from the LLM approaches' treating autoformalization as a simplistic translation task which lacks mechanisms for self-reflection and iterative refinement that human experts naturally employ. To address these issues, we propose ReForm, a Reflective Autoformalization method that tightly integrates semantic consistency evaluation into the autoformalization process. This enables the model to iteratively generate formal statements, assess its semantic fidelity, and self-correct identified errors through progressive refinement. To effectively train this reflective model, we introduce Prospective Bounded Sequence Optimization (PBSO), which employs different rewards at different sequence positions to ensure that the model develops both accurate autoformalization and correct semantic validations, preventing superficial critiques that would undermine the purpose of reflection. Extensive experiments across four autoformalization benchmarks demonstrate that ReForm achieves an average improvement of 22.6 percentage points over the strongest baselines. To further ensure evaluation reliability, we introduce ConsistencyCheck, a benchmark of 859 expert-annotated items that not only validates LLMs as judges but also reveals that autoformalization is inherently difficult: even human experts produce semantic errors in up to 38.5\% of cases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces ReForm, a novel Reflective Autoformalization framework designed to convert natural language mathematics into machine-verifiable formal statements more accurately. Unlike previous one-pass translation approaches that often lose semantic fidelity, ReForm integrates semantic self-validation directly into the autoformalization process.

### Strengths
- The paper provides a **ConsistencyCheck benchmark** that rigorously evaluates the reliability of **LLM-based judges** and quantifies the challenges of **autoformalization**.  
- Introduces **iterative reflection techniques from reasoning models** into the process of **autoformalization**.  
- The paper is **well-written**.

### Weaknesses
- LLM is not a perfect supervisory signal, and I would like to know to what extent the capability of the judge model affects the stability of training.  
- How did you handle **semantic alignment** in your method — was it **integrated into the reward design**?  
- Please provide an **ablation study comparing process-level supervision and outcome-level supervision**.

### Questions
Please check the **Weaknesses** section.

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
The paper presents ReForm, a novel reflective autoformalization framework that aims to improve the semantic fidelity of translating natural language mathematics into formal statements. Unlike traditional “one-pass” autoformalization models that generate a single formal output, ReForm introduces an iterative self-correction loop where the model alternates between formal statement generation and semantic self-validation. To train this reflective process effectively, the authors propose a new reinforcement learning algorithm, Prospective Bounded Sequence Optimization (PBSO), which integrates heterogeneous rewards for both the main task (final correctness) and auxiliary critiques (intermediate semantic validation). Extensive experiments across four challenging benchmarks (miniF2F, ProofNet, PutnamBench, AIME2025) demonstrate an average +17.2 percentage point improvement in semantic consistency over the strongest baselines.

### Strengths
- The target problem is well-motivated and addresses a clear bottleneck in formal reasoning: the semantic fidelity in autoformalization.
- The integration of reflective reasoning and reinforcement learning is novel and effective; the experiments demonstrate consistent and interpretable improvements across multiple benchmarks.
- The newly established ConsistencyCheck benchmark provides a valuable resource for quantitatively assessing the reliability of LLM-based metrics and for understanding the intrinsic challenges of mathematical autoformalization.

### Weaknesses
- The paper relies heavily on LLM-based semantic evaluation metrics, which, despite the ConsistencyCheck benchmark, may still introduce bias or circularity in measuring semantic consistency.

- The computational cost and efficiency trade-offs of the reflective multi-iteration process are not fully analyzed — it remains unclear how scalable the approach is for large-scale or more complex formal systems.

- Several important related works are missing; please refer to the recent surveys [1, 2] for a broader overview.

- (Minor) There are too many autoformalization papers recently, which may cause aesthetic fatigue in the community.

[1] A Survey on Deep Learning for Theorem Proving, COLM 2024. 


[2] Autoformalization in the Era of Large Language Models: A Survey, arXiv 2025.

### Questions
1. The paper both trains and evaluates with LLM-based judges (CriticLean for RL and Qwen3 for evaluation).  Could the authors verify cross-judge robustness, e.g., whether ReForm still shows similar gains when evaluated with a different LLM judge, such as Gemini or GPT-5?
2. In Sec. 4.4, the authors show that responses get longer during RL, but longer ≠ is better. Can we disentangle “more tokens” from “better critiques”? For example, is there an automatic or human rating showing that later reflections actually introduce new semantic constraints (quantifier scopes, hidden assumptions, edge cases), rather than just paraphrasing previous critiques? A precision/recall–style analysis on detected error types would clarify this.

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
This paper introduces **ReForm**, a reflective autoformalization paradigm that shifts from one-pass translation to an iterative process combining formal statement generation with semantic self-validation. To train this model effectively, the authors propose **Prospective Bounded Sequence Optimization (PBSO)**, a novel RL algorithm that uses heterogeneous rewards at different sequence positions to ensure both accurate autoformalization and faithful self-critiques. Extensive experiments on four benchmarks show ReForm achieves an average improvement of **17.2 percentage points** in semantic consistency over state-of-the-art baselines. The authors also introduce **ConsistencyCheck**, a benchmark of 859 expert-annotated items, which reveals that autoformalization is inherently difficult, even human experts make semantic errors in up to 38.5% of cases.

### Strengths
1.  **Novel Reflective Paradigm:** The core innovation is shifting autoformalization from a one-pass translation task to an iterative, self-correcting process. By mimicking the human expert's cycle of generation, validation, and refinement, ReForm directly addresses the critical challenge of semantic fidelity, moving beyond mere syntactic correctness.

2.  **Effective Training with PBSO:** The proposed Prospective Bounded Sequence Optimization (PBSO) algorithm is a clever solution to the multi-objective credit assignment problem. It effectively trains the model to produce high-quality final formalizations *and* accurate intermediate critiques, preventing the self-validation mechanism from degenerating into superficial or hallucinated feedback.

3.  **Rigorous and Comprehensive Evaluation:** The paper provides extensive empirical validation across four challenging benchmarks, demonstrating substantial and robust improvements. The creation of the expert-annotated **ConsistencyCheck** benchmark adds significant rigor, not only validating the use of LLMs as judges but also quantifying the inherent difficulty of the task itself.

### Weaknesses
1. **Heacy Dependence on LLM-based Evaluation**: The entire training and evaluation framework relies heavily on LLM judges (like Qwen3-235B and CriticLean-14B) to assess semantic consistency. While the authors rigorously validate these judges with their ConsistencyCheck benchmark, they still have a non-trivial error rate (about 17%). 
2. **Insufficient Ablation on Key Algorithmic Components**: While the paper demonstrates the overall effectiveness of PBSO, it lacks ablation studies on several critical design choices. The contribution of position-specific advantages is not isolated from the core bounded return mechanism, making it unclear if this complexity is necessary. Furthermore, the individual and interactive effects of the Task Reward &   Auxiliary Rewards are not thoroughly dissected.
3. **Lack of Details of Human Evaluation**: Number of the annotators in total? Number of the annotators per statement? Background of the annotators?  and so on...
4. The performance of 32B and 8B is very close, yet the author has not provided a reasonable explanation for this.

### Questions
See the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
