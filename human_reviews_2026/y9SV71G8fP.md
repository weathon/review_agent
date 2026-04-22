# Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric **T**oken **E**ntropy **C**umulative **A**verage (**TECA**), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm---Explore Briefly, Then Decide---with an associated **C**umulative **E**ntropy **R**egulation (**CER**) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem-solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper observes that in overthinking models, the token entropy cumulative average (TECA) continuously increases during the thinking process. Based on this finding, the authors propose an entropy-based reinforcement learning algorithm that rewards the model based on the TECA during inference, encouraging the model—when producing correct answers—to use as few tokens as possible.

### Strengths
1. The paper experimentally demonstrates that TECA exhibits an upward trend during the thinking process of overthinking models.  
2. The proposed TECA-based reward mechanism reduces the average output length without significantly compromising model accuracy.

### Weaknesses
1. Prior work has shown that models may generate correct answers despite erroneous reasoning steps. Could the proposed method exacerbate this phenomenon?  
2. Would reducing TECA negatively impact the model’s exploratory behavior and creativity? Specifically, might the model under-explore and thus underperform on complex problems?  
3. The proposed algorithm appears to primarily balance performance and output length relative to the baseline. Could the authors plot results from different models on the same graph to demonstrate that their method indeed lies on the Pareto frontier between accuracy and output length?

### Questions
See weakness above.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework to address over-thinking in large language models by defining the new metric Token Entropy Cumulative Average (TECA) to quantify exploration during reasoning. Building on TECA, the authors propose the “Explore Briefly, Then Decide” paradigm alongside a Cumulative Entropy Regulation (CER) mechanism, which guides models to terminate reasoning when sufficient certainty is reached. Empirical results on multiple mathematical benchmarks show substantial reductions in response length while maintaining solving capability. The primary contributions are the TECA metric, the CER-based reasoning paradigm, and strong evidence of improved efficiency in reasoning without sacrificing accuracy.

### Strengths
1.Introduces a clear, quantitative metric for exploration behavior. TECA provides a novel and interpretable way to monitor when a model is overthinking by measuring uncertainty accumulation during reasoning.
2.Demonstrates large efficiency gains without major accuracy loss. On arithmetic and mathematics benchmarks the method reduced average reasoning length by up to 71 % while maintaining solving performance, showing strong practical value.

### Weaknesses
1.The paper focuses heavily on mathematical reasoning benchmarks, but the generalizability of the proposed TECA metric and CER mechanism to open-ended or conversational tasks is not demonstrated.
2.There is limited discussion on the potential trade-off between exploration suppression and creativity or robustness in reasoning. Suppressing exploration might hurt performance on novel or adversarial examples, but this risk is not fully evaluated.
3.In the CER mechanism, the segmentation of reward and the entropy threshold used to determine when to stop exploration or adjust weights appear to require manual tuning. The paper does not include an ablation study examining these hyper-parameters, nor does it describe the detailed workflow for adjusting them.
4.Although response length reduction is compelling, the accuracy drop or change in reasoning style and explanation detail is less thoroughly analyzed, making it harder to assess practical impacts on downstream use-cases.

### Questions
1.While CER has generally reduced output length while maintaining accuracy on multiple mathematical benchmarks, does it still suppress meaningful reasoning steps? Does the model exhibit a tendency to halt reasoning prematurely?
2.The current experiments are primarily based on mathematical reasoning benchmarks, but I wonder whether this approach can be generalized to more open-ended or non-mathematical reasoning tasks and undergo broader testing?
3.Based on the paper's description, the Cumulative Entropy Regulation framework appears to involve threshold setting and parameter tuning based on TECA. Could you present a sensitivity analysis demonstrating how different threshold settings impact the results?
4.The authors indicate that while the average response length was significantly reduced across multiple benchmarks, the accuracy rate only experienced a slight decline. However, can accuracy rates alone in mathematical reasoning benchmarks fully represent the completeness and effectiveness of the reasoning process? Does CER potentially lead to the truncation of certain important reasoning exploration steps?
5.The paper only provides theoretical analysis regarding the boundary tokens between exploration and decision-making. Could you present concrete examples of reasoning trajectories that illustrate the transition points between exploration and decision-making? Including such examples in the appendix could better support the theory proposed in the paper.
6.While reasoning efficiency has been improved, does CER training significantly increase training costs? How much additional training cost does it incur compared to the baseline?
7.We note that the experiments were conducted solely on Qwen3-4B and Qwen3-8B models. Can this methodology be extended to larger-scale models or different architectures? Does TECA's effectiveness remain consistent in larger-scale models?
8.I remain skeptical that CER can consistently reduce redundant reasoning while maintaining answer completeness for every test sample. Have there been instances where CER led to performance degradation or premature termination? What mechanisms have the authors implemented to mitigate such issues?
9.The "Explore Briefly, Then Decide" concept proposed in the paper is quite compelling. I wonder whether it could be integrated with other early-exit reasoning mechanisms to achieve better utility?
10.The experimental analysis in the paper appears overly simplistic, lacking ablation studies on components such as the segmented reward mechanism and TECA metric. Furthermore, details regarding the training environment and duration are not provided. Appropriate supplementation in the appendix could substantially improve the completeness of the paper.

### Soundness
2

### Presentation
2

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
This paper proposes TECA as a proxy for “exploration” during CoT inference, and introduces a GRPO-based reward shaping scheme (CER) that suppresses high TECA at the tail of the trajectory. Experiments show substantial reductions in average CoT length with minimal accuracy degradation.

### Strengths
The TECA metric is intuitively interpretable, simple to compute, and shows reasonable empirical correlation with response length.

The qualitative case study indeed shows fewer redundant “reflective” tokens in the CER-trained model.

### Weaknesses
**W1. Experimental scale is insufficient to support the claimed generality.**
Only Qwen3-4B and Qwen3-8B are used. GRPO RL training is notoriously sensitive to backbone initialization and emergent base-model bias. A two-model sample is not enough to justify the claim that the proposed mechanism reflects a general property of “overthinking” and adaptive reasoning. In fact, even for the 8B model, CER only provides a marginal advantage relative to CCoT in some metrics.

**W2. The work should also report Pass@k.**
Length reduction could come from eliminating alternative chains prematurely, not from eliminating irrelevant exploration. Pass@k is the standard instrument to distinguish “more decisive reasoning” from “shallower sampling.” Without this control, it is unclear whether CER suppresses useless branches or simply cuts off exploration globally.

**W3. Conceptual analysis is shallow.**
Preliminaries section repeats known GRPO and entropy definitions, while the core insight, *TECA is a principled proxy of exploration*, is not justified beyond curve visualizations. Two major omissions weaken the conceptual claim:
- The paper only establishes correlation between TECA trajectories and response length. There is no causal diagnostic: no intervention study (injecting entropy spikes), no ablation vs. simpler uncertainty surrogates (e.g. logit margin / variance). Thus, it is not shown that TECA is the right variable. It may simply correlate with the visible surface symptom (redundant tokens).
- The CER reward construction is not theoretically motivated. Rewarding TECA only on correct samples can encourage lucky short guesses, and the paper does not analyze this incentive misalignment. The design choices (equal weighting with accuracy reward, exponential form, applying TECA only at the end step) are hyperparameter-like knobs but are not justified mathematically.

### Questions
see weakness

### Soundness
2

### Presentation
2

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
The paper looks at the problem of overthinking in chain-of-thought reasoning efficiency by regulating token-level entropy. They propose a framework with two key components: TECA, which measures cumulative uncertainty, and CER, which penalizes excessive entropy to encourage concise reasoning. They explore (1) reasoning dynamics, where they analyze entropy changes during exploration and determination, and (2) RL integration, where they embed CER into GRPO to control reasoning depth. They also extend their work to mathematical reasoning benchmarks and show consistent reductions in reasoning length with minimal accuracy loss.

### Strengths
The idea of regulating token-level entropy to mitigate overthinking in LLM reasoning is both novel and well-motivated. Most existing work focuses on increasing reasoning accuracy through longer CoT generation or reinforcement learning rewards, but these approaches cannot adaptively control reasoning depth or prevent redundant exploration. This would limit their efficiency and interpretability. This paper introduces a method that improves reasoning efficiency without sacrificing accuracy. This makes the contribution practically impactful and valuable to the community.


The workflow is well-structured, as it integrates entropy-based reasoning control with GRPO to make the reasoning process adaptive and self-regulated. This method allows LLMs to decide when to stop exploring.

The experiments are extensive, with detailed analysis of the results. These experiments validate the effectiveness across different model sizes and reasoning complexities.

### Weaknesses
i. The experiments are now all on math tasks. In math tasks the reasoning process usually follows a clear logical progression: i. the model explores multiple computational paths ii. converges to a definitive conclusion. In this setting, token-level entropy does effectively reflect the model’s degree of cognitive divergence. The cumulative entropy can capture the transition from exploration to determination relatively easily. The assumption may not work well on open-ended and/or textual tasks. E.g., QA tasks lack a clear boundary between exploration and determination. The model simultaneously retrieves facts, interprets semantics, and formulates answers. This could cause the TECA to mix different cognitive phases and produce noisy signals. For open-ended or multi-answer tasks, the evaluation signals are ambiguous and entropy penalties may prematurely constrain generation. I would suggest that the entropy-based works dig deeper into diverse tasks.

ii. The contribution of each component is not fully isolated. Need an ablation study.

iii. A stability and sensitivity study is needed, as the entropy-based rewards may be sensitive to temperature, model calibration, and/or decoding policy.

iv. I would suggest adding a clock time report or computational cost, although the reasoning length is reduced.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
