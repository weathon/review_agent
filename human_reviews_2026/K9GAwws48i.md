# Teaching LLMs When to Stop Seeking and Start Acting

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Many tasks require machine learning models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper builds an approach to imbue LLMs with the ability to stop or terminate their ternal thinking process and/or environment interaction at the right point for maximal performance, without wasting computation or interaction. The high-level insight of the proposed framework is that reasoning itself can be used to learn accurate and generalizable termination behavior, as long as this reasoning is done comparatively. Then experimental results demonstrate the good performance of the proposed model.

### Strengths
1. The paper addresses a critical problem for deploying LLMs: deciding when to terminate. The idea itself is interesting. The proposed idea of combining counterfactuals with reasoning is novel and effective.
2. The related work section is comprehensive, systematically surveying the field from the perspectives of learning when to act, LLM-based information seeking, and optimal termination.
3. The experimental design is comprehensive and convincing. The evaluation across two distinct domains—interactive medical diagnosis and mathematical reasoning—effectively demonstrates both the generality and superiority of the proposed CaRT method.

### Weaknesses
1. The training and evaluation pipeline relies heavily on simulated environments and external reward models, which introduces a potential sim-to-real gap.
2. The method focuses solely on the termination policy ("when to stop") and does not address the exploration policy ("what information to seek"), which is a crucial limitation.
3. The training of CaRT itself appears computationally expensive, as it utilizes LLMs to generate counterfactual data and reasoning traces. The paper does not discuss this upfront cost in relation to the achieved inference-time efficiency gains, which is a practical concern for resource-constrained settings.

### Questions
1. Figure 1 is not cited in the main text.
2. A key limitation is the separation of termination ("when to stop") from exploration ("what to seek"); unifying them is a critical next step. Can the author discuss how to deal with this problem?
3. Using GPT-4o to generate reasoning traces potentially risks knowledge leakage. The performance gains of CaRT are confounded; it is unclear if they stem from the method itself or from the injected knowledge of a stronger teacher model. An ablation study is needed to isolate the true contribution of the reasoning framework.
4. The paper does not discuss the substantial computational overhead of training CaRT, which involves generating counterfactual data and reasoning traces with GPT-4o. Could the author provide an analysis of this upfront cost versus the achieved inference-time efficiency?
5. The definition of the "optimal termination point" (a ≥ 50% success rate increase) appears arbitrary. This threshold may not exist in cases of gradual information accumulation, causing the "Optimal Termination Rate" metric to be computed on a biased subset of examples. Is it possible to provide a threshold-agnostic metric that evaluates performance across all samples.
6.	Current baselines seem insufficient for a comprehensive evaluation. Adding an oracle baseline that selects actions to maximize future discounted reward at each step would better demonstrate the gap to the theoretical performance upper bound.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Authors propose Counterfacturals and Reasoning for Termination (CaRT), an augmentation plus fine-tuning approach to enable LLMs to balance the explore-exploit tradeoff from verbalized reasoning. CaRT is evaluated on multi-turn interactive medical diagnosis and math problem solving, and shows efficiency over the base model without fine-tuning and the base model fine-tuned on randomly samples training data.

### Strengths
The task is interesting and balancing information seeking and reward seeking (or avoid length penalty) is indeed a recurring theme in RL and in reality. The synthesized medical diagnosis domain reflect this tradeoff and is a great testbed.

### Weaknesses
1. Using GPT4o to generate termination reasoning as learning data poses a confounder of distillation to your argument, that is, the ability to terminate smartly is distilled from GPT4 and not solely from your "reasoning for termination" construction. Can you use Qwen for termination reasoning?
2. The ablations show that "reasoning for termination" is necessary, but it does not show that a separate termination model/perspective is necessary. I can think of two very simple baselines that can help justify/test the complexity of CaRT.
  * Aggressively greedy system prompt (e.g., "Only ask questions when absolutely necessary")
  * In-house termination reasoning prompt (e.g., each episode/turn, the model itself reasons about termination in context before outputting)
3. Related work such as RL with length penalty (as cited in paper) is excluded from head-to-head comparisons with CaRT without strong empirical evidence for their lack of adaptability.

### Questions
See weaknesses.

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
2

### Summary
This paper introduces CaRT (Counterfactuals and Reasoning for Termination), a method to teach large language models (LLMs) when to stop information gathering—either by terminating internal reasoning or ending interaction with an environment—in multi-step tasks. The approach combines supervised fine-tuning (SFT) with counterfactual examples and explicit reasoning traces to help the model learn an implicit value function for termination.

### Strengths
The paper formalizes and addresses the understudied problem of optimal termination in multi-step LLM reasoning and interaction settings. The combination of counterfactual data generation and explicit reasoning traces is a creative and well-motivated approach to teaching termination. Thorough ablations validate the importance of both counterfactuals and reasoning, and probe how reasoning improves representation generalizability.

### Weaknesses
Experiments are confined to two domains (medical diagnosis and math reasoning). Broader evaluation in more diverse or real-world interactive settings is needed.
CaRT only decides when to stop, not what to ask or reason about. This limits its overall impact on end-to-end task efficiency. Could the method be extended to jointly learn both what to ask and when to stop, and what challenges would that introduce?
The RL post-training step does not consistently improve performance and sometimes leads to longer reasoning traces, raising questions about its necessity.
The method relies heavily on accurate external reward labeling (e.g., using Llama-3.1-8B for medical diagnosis), which may not always be available or reliable. Were any experiments conducted with smaller or open-source models for generating reasoning traces, to reduce dependency on GPT-4o.

### Questions
See weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CaRT, a method for teaching LLMs to make strategic termination decisions during multi-step information-gathering tasks. The approach combines counterfactual trajectory pairs with explicit reasoning traces to help models balance task accuracy with resource efficiency. The authors evaluate CaRT on medical diagnosis and mathematical reasoning domains, demonstrating improvements in termination behavior compared to base models and standard fine-tuning baselines.

### Strengths
1. The paper addresses an interesting practical challenge: teaching LLMs when to stop gathering information and commit to a decision. This is highly relevant for resource-constrained deployments and agentic applications.
2. The use of counterfactual pairs to isolate critical information is creative and theoretically grounded. Contrasting trajectories where termination is appropriate vs. inappropriate provides a clear learning signal.

### Weaknesses
### 1. Model Evaluation
The experiments fine-tune only Qwen2.5-3B-Instruct (for the medical domain). Results from a single small model family are insufficient to support broad claims about the general effectiveness of CaRT. Testing across model sizes or families (e.g., Llama, Mistral) would strengthen the evidence.

### 2. Data 
The termination labels may not be stable across models or training stages. The paper uses Llama3.1-8B-Instruct to produce training/evaluation data, while fine-tuning is done on Qwen2.5-3B-Instruct. Models with stronger reasoning ability might succeed with fewer steps, while weaker models need more information. This raises concerns:

* The “golden label” (terminate vs. continue) may shift as base model capability changes. If the underlying reasoning model changes (fine-tuned or uses a different base model for questioning), the termination behavior may also shift, requiring new data generation and training each time.
* When GPT-4o generates rationales for why Llama3.1-8B cannot succeed with certain information, these rationales may not be meaningful: If GPT-4o itself could succeed with the same information, the rationale becomes mere justification rather than true reasoning about information sufficiency or the true uncertainty or capability of the target model.. This raises questions about whether the model learns genuine termination reasoning or simply mimics justification patterns

### 3. Metrics
Several key metrics lack clear explanation:
* Please explain more about FRQ SR Difference from Mean and fixed-budget heuristic baseline
* 'Optimal Termination Rate' methodology is unclear—what does the meaning of the results on data points "excluding conversations without steep increase" (line 301) mean precisely?

The medical domain results in particular seem less convincing—success differences are small and not clearly justified as meaningful.
* Qwen2.5-3B-Instruct + reason prompt achieves the highest success rate (3-4% better than CaRT), though at a higher termination index. While CaRT is more efficient, the natural saturation of benefits from additional information is expected behavior, not necessarily an achievement of the method. Figure 4 shows clear benefits for mathematical reasoning, but the medical diagnosis case lacks compelling evidence for CaRT's advantages beyond natural saturation effects

### 4. Method
* Why are the termination model and diagnosis/reasoning model kept separate? What prevents joint training?
* Were question orders randomized during training? This could significantly impact learning and evaluation.
* The paper compares CaRT with equal-sized SFT datasets. What about SFT with significantly more data (no counterfactuals)?
* How does explicit termination prediction compare with the model's own confidence scores on the target task?

### Minor Issues
* Figure 3 & 4: Mean success rate and Pareto frontier lines are difficult to distinguish visually

### Questions
1. How does the approach perform across different model sizes and families?
2. How do you obtain ground truth labels for termination during training, given the dynamic nature of model capabilities?
3. Have you considered joint training of termination and reasoning models?
4. Were question orders randomized during training?
5. How does performance compare when using larger SFT datasets without counterfactuals?

### Soundness
3

### Presentation
3

### Contribution
3
