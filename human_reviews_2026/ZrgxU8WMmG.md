# StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large Language Models (LLMs) have shown promising capabilities for solving Operations Research (OR) problems. 
While reinforcement learning serves as a powerful paradigm for LLM training on OR problems, existing works generally face two key limitations. First, outcome reward suffers from the $\textit{credit assignment problem}$, where correct final answers can reinforce flawed reasoning.
Second, conventional discriminative process supervision is $\textit{myopic}$, failing to evaluate the interdependent steps of OR modeling holistically. 
To this end, we introduce $\textbf{\texttt{StepORLM}}$, a novel self-evolving framework with generative process supervision. 
At its core, $\texttt{StepORLM}$ features a co-evolutionary loop where a policy model and a generative process reward model (GenPRM) iteratively improve on each other. 
This loop is driven by a dual-feedback mechanism: definitive, outcome-based verification from an external solver, and nuanced, holistic process evaluation from the GenPRM. 
The combined signal is used to align the policy via Weighted Direct Preference Optimization (W-DPO) and simultaneously refine the GenPRM. 
Our resulting 8B-parameter $\texttt{StepORLM}$ establishes a new state-of-the-art across six benchmarks, significantly outperforming vastly larger generalist models, agentic methods, and specialized baselines. 
Moreover, the co-evolved GenPRM is able to act as a powerful and universally applicable process verifier, substantially boosting the inference scaling performance of both our own model and other existing LLMs. 
We release our models and code to facilitate future research (https://github.com/0xzhouchenyu/StepORLM).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes StepORLM, a self-evolving framework for modeling OR. It aims to address the credit assignment problem, where flawed reasoning might lead to a correct outcome , and the limitations of myopic process supervision that overlooks step interdependencies. The framework's core is a co-evolutionary loop between a policy model and a GenPRM. This loop uses dual feedback from an external solver for outcome verification and the GenPRM for holistic process evaluation. The paper reports that its resulting model achieves state-of-the-art results across six benchmarks and that the GenPRM can be used as a process verifier to improve the inference performance of other LLMs.

### Strengths
**Significant Adaptation of Process Supervision:** The paper’s primary strength is its effective adaptation of process-based supervision to the OR domain. By proposing a GenPRM to evaluate entire reasoning trajectories holistically , it aims to mitigate the known limitations of simple outcome-based rewards (the credit assignment problem) and myopic step-wise checks.

**Well-Motivated Co-evolutionary Design:** The paper proposes simultaneous refinement of policy and GenPRM, which is conceptually sound and avoids a static critic. Table 3 shows that freezing GenPRM reduces performance by 3.7% on average, validating the co-evolution mechanism.

**Strong Empirical Results: ** The framework's quality is supported by SOTA performance across six benchmarks. The 8B-parameter StepORLM is shown to outperform vastly larger generalist models and other specialized baselines.

### Weaknesses
**Unclear Attribution Due to Ambiguities in the Data Pipeline:** The paper's SOTA claims are complicated by ambiguities in its data strategy. The 50K-sample SFT dataset is significantly larger than baselines like ORLM (30K) , and Figure 4 shows this SFT-only model is already exceptionally strong . This makes it difficult to attribute the final SOTA performance to the co-evolutionary method (Stage 2) rather than just the high-quality initial data (Stage 1). This ambiguity extends to the DPO phase. Section 3.3.1 is unclear whether the DPO phase re-uses these static 50K instances or regenerates new ones , hindering reproducibility.


 **Omission of Relevant Concurrent Work:** The literature review overlooks highly relevant concurrent work that addresses the same problem of intermediate-step validation. Specifically, Step-Opt[1] is directly relevant as it proposes a "Stepwise Validation Mechanism" with explicit checkers for variables, constraints, and programs . This structured, explicit validation is a conceptually similar (though different in approach) solution to the same problem that StepORLM's learned GenPRM aims to solve. Additionally, Lima et al.[2] on verifiable synthetic data is also relevant and should be discussed .

**Backbone Choice Obscures Method Effects:** The paper's SOTA claims are ambiguous. StepORLM is built exclusively on Qwen3-8B , while principal baselines (like the 8B ORLM) rely on different 8B architectures (LLaMA-3-8B). Without cross-backbone controls (e.g., training StepORLM on the baseline's backbone), the comparison fails to isolate the method’s contribution from the backbone’s.

[1] Wu Y, Zhang Y, Wu Y, et al. Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation[J]. arXiv preprint arXiv:2506.17637, 2025.

[2] Lima V, Phan D T, Kalagnanam J, et al. Toward a trustworthy optimization modeling agent via verifiable synthetic data generation[J]. arXiv preprint arXiv:2508.03117, 2025.

### Questions
1. The Data Pipeline: The paper must clarify whether the DPO phase re-uses the static 50K SFT instances or dynamically regenerates new ones. The SFT-only model's results must also be provided to quantify the actual gain from the co-evolutionary method.

2. Omitted Work: The authors must discuss the highly relevant concurrent works by Wu et al. (2025) and Lima et al. (2025)  and situate their novelty accordingly.

3. Backbone Choice: The SOTA claim is confounded by the Qwen3-8B backbone. The authors need to provide experimental results on a common backbone (e.g., LLaMA-3-8B) to enable a fair comparison against baselines like ORLM.

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
The paper presents StepORLM, a novel self-evolving framework for LLM-based Operation Research (OR), in which a policy model and a generative process reward model (GenPRM) iteratively improve each other. The framework employs a dual feedback mechanism, combining outcome-based verification from an external solver with process-level evaluation from the GenPRM. The policy model is trained using this combined signal via weighted DPO, while the GenPRM is subsequently refined using new training data generated by the latest policy model. Experimental results demonstrate that the resulting 8B StepORLM model outperforms existing state-of-the-art models across six OR benchmarks, showing the effectiveness of the proposed approach.

### Strengths
1. The related work section is clear and reasonable.

2. The proposed self-evolving framework is reasonable and novel to me.

3. The experimental results are strong, demonstrating the effectiveness of the proposed method.

### Weaknesses
## Major Concerns

1. Unsupported Motivation (Myopic PRM). While the credit assignment problem is intuitive and makes sense, the myopia PRM problem appears weak to me since many existing PRMs generate process-level supervision signals based on the holistic model output. The authors need to provide stronger evidence or clearer citations to support their claim and better differentiate their work from existing PRMs. 

2. There is an apparent disconnect between the paper's claim of using "generative process supervision" and the described implementation. Specifically, according to Algorithm 1 and Equation (1), the detailed, step-by-step process-level correctness scores are seemingly collapsed into a single, global weight $w_{\tau_{w},\tau_{l}}$ for the weighted DPO. This suggests that the fine-grained, process-level feedback is lost and not directly utilized to supervise the model's optimization, which contradicts the paper's central claims.

3. Lack of Clarity on GenPRM Refinement: The methodology for refining the GenPRM is not described in sufficient detail. Key information about the training process is missing, particularly the exact format of the training data.

4. The experimental comparisons in Table 1 are incomplete, making it difficult to fully assess the performance of StepORLM. The missing models include the base model Qwen3-8B, SOTA commercial reasoning models (e.g., OpenAI o-series, Gemini 2.5) and SOTA open-source reasoning models (e.g., DeepSeek-R1, Kimi-K2).

5. Important implementation details are missing, including the number of self-evolving iteration, the prompts for process evaluation and inference scaling experiments, etc.

## Minor Issues

1. L98, “Extensive experiments on seven benchmarks” -> “Extensive experiments on six benchmarks”

2. L298, “ComplexORXiao et al. (2024)”

### Questions
While the proposed method demonstrates promising performance on OR tasks, the paper in its current form suffers from several major weaknesses concerning its core motivation, methodological clarity, and experimental rigor. The overall presentation is not up to the ICLR standard. In order to raise my rating, I would like the authors to address the different points in the major concerns.

### Soundness
2

### Presentation
1

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
StepORLM introduces a self-evolving 8B-parameter framework that couples a policy model with a generative process reward model (GenPRM) to train LLMs for operations-research tasks. By iteratively distilling dual feedback—solver-verified outcomes and holistic trajectory-level critiques—into weighted DPO updates, the system attains new state-of-the-art accuracy on six benchmarks, while the co-evolved GenPRM further boosts inference-time scaling for both its own policy and external OR models.

### Strengths
The proposed method is simple yet effective, achieving performance improvements.

### Weaknesses
GenPRM and Self-Improve are not novel concepts; they have been explored in prior research, eg. [1][2][3][4].

[1] R-PRM: Reasoning-Driven Process Reward Modeling

[2] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning

[3] rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

[4] Inference-Time Scaling for Generalist Reward Modeling

### Questions
Why is there such a large performance gap between LLMOPT (origin) and LLMOPT (reproduce)?

What model is the CoT on line 344 based on?

The StepORLM performance in Tables 2 and 3 is inconsistent with that in Table 1.

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
This paper introduces StepORLM, a novel framework for training large language models to solve operations research (OR) problems more reliably. The key innovation is generative process supervision through a Generative Process Reward Model (GenPRM) that holistically evaluates full reasoning trajectories, rather than step-by-step discriminative scoring. StepORLM optimizes a policy model through a dual-feedback loop: (1) outcome verification from an external solver ensures correctness of final results, and (2) process evaluation from GenPRM assesses reasoning quality. Training proceeds via Weighted Direct Preference Optimization (W-DPO) to align the policy with high-quality reasoning. Meanwhile, it also optimizes GenPRM during the process. Experiments on six benchmarks show StepORLM (8B parameters) outperforming larger models like GPT-4o, Qwen3-32B, and specialized ORLM baselines, achieving state-of-the-art Pass@1 accuracy. The paper also shows GenPRM’s transferability as a universal process verifier for inference-time scaling.

### Strengths
1. Introduces generative process supervision, a step beyond discriminative PRMs, enabling trajectory-level reasoning assessment and solving the credit assignment problem.
2. The co-evolution between policy and GenPRM provides a mutually reinforcing refinement process that avoids overfitting.
3. Empirical results are strong with demonstrated consistent SOTA performance across six diverse OR benchmarks, outperforming vastly larger generalist and agentic systems.

### Weaknesses
1. Most benchmarks are still synthetic or adapted from prior academic datasets; stronger demonstration on industrial-scale, more real-world tasks would enhance credibility.
2. The framework assumes deterministic, verifiable outcomes from OR solvers; applicability to open-ended or ill-posed reasoning tasks remains uncertain.
3. The dual-model co-evolution loop may be expensive to scale, and the paper provides limited discussion of training efficiency.
4. While numerical gains are clear, qualitative examples of how GenPRM critiques or refines reasoning are sparse. For example, any reward hacking examples from the PRM would be helpful.

### Questions
1. Can this method be used in other domains outside OR?
2. Do you observe any reward hacking using PRM? Is Generative PRM always reliable?
3. How good is the generalization of this algorithms across dataset and domain?

### Soundness
2

### Presentation
3

### Contribution
2
