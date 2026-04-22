# Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent shift associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a training-free framework for controlling the reasoning intensity of models at inference time. The framework involves (1) computing a latent steering vector based on multiple pairs of unlabelled data with and without CoT, (2) identifying the first principle component of these difference vectors, (3) add this vector to the original model's latent state scaled by a tunable factor $\alpha$. Given this tunable $\alpha$, an ensemble of responses with different $\alpha$ can then be produced and aggregated over to obtain better reasoning performance.

### Strengths
- The proposed concept of having a continuous tunable parameter to control reasoning intensity via latent space arithmetic is an interesting and innovative concept.
- The method is training-free and has been demonstrated to perform well compared to basic baselines, across various datasets and models.
-  There is analysis of the impact of $\alpha$ to model output, quantitatively through its correlation with generation length (Fig. 4) and some qualitative examples.
- The paper applied the approach to two different test-time strategies -- CoT and self-reflection strategies.

### Weaknesses
- The main motivation of the paper does not match the proposed intervention and experimental design. The authors proposed the ability to have tunable reasoning intensity as different questions may require different levels of reasoning intensity, but subsequently in the paper Sec.3 mentioned that it is inherently difficult to identify ideal prompt strength per question. The rest of the paper then discussed ensemble-based methods to improve reasoning with a uniform distribution of $\alpha$ to form the distribution, which is a different problem setting. It would be useful if the paper could be either repositioned, or include more experimental results specifically demonstrating how $\alpha$ is directly related to reasoning intensity as claimed.

- The paper compares against fairly basic LLM ensemble methods for improved reasoning (self-consistency with majority vote, best of N). The authors should consider discussing in related work and/or comparing with methods [1-3] that propose LLM prompt ensembles with improved reasoning performance since those are most directly comparable to the proposed Fractional Reasoning framework (it also effectively creates an ensemble with different prompts). It is unclear if the reported performance gains may stem mainly from more diverse 'prompting' of the LLM ensemble.

- The method requires first computing the latent steering vector based on the principle components of 'queries', though it is unclear from the paper how many or what type of queries would be required (held-out set of the same question distribution?). Providing greater clarity on this would help address concerns regarding generalizability of the method or the cost of applying such methods if the latent steering vectors are not general. 
   - Explicitly showing whether the latent steering vectors are generalizable across tasks and datasets would be useful. Related to this is providing a more thorough description (including what queries and how many are needed, from where) of the specific pipeline used for computing the steering vectors.

- The computational costs of the method seem large, especially compared to the basic baselines proposed to be used. The authors should consider reporting a more thorough analysis of the computational cost overheads compared to other methods, especially since the original motivation of the paper relates to efficiency. This includes the cost of generating the latent steering vectors along with vector addition and scaling at every token and layer.

- The change in methodology for Reflection seems ad-hoc, weakening the paper's unified framework. It is not clear why for Reflection (compared to CoT), the contrastive-pair approach should be changed (similarly why the single prompt latent approach should not be applied for the CoT setting). The authors should provide further elaboration and analysis on why there is this lack of consistency.

- The $\alpha$ hyperparameter seems relatively important. While there has been some ablation done, it would be helpful if the authors could further elaborate on whether this needs to be tuned for different model and task, and whether there is some intuition behind the selection of the range if it is important.



[1] Hu et al, Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning Tasks

[2] Chen et al, AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors.

[3] Du et al, Improving factuality and Reasoning in Language Models through Multiagent Debate

### Questions
Please see concerns and questions in the Weakness section.

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
This paper proposes Fractional Reasoning (FR), a training-free, model-agnostic framework that provides continuous, inference-time control over an LLM’s “reasoning intensity” by steering internal latent states. The core idea is to extract a steering vector that captures the latent shift induced by reasoning-promoting prompts (e.g., CoT or reflection) and then reapply this vector with a tunable scaling factor α during generation. This enables adaptive control of depth: from terse direct answers (low α) to richer multi-step reasoning and stronger reflection (higher α), without modifying input text or fine-tuning the model.

Methodologically, the paper formalizes prompts as latent state shifts in attention and constructs a steering vector via contrastive pairs (positive: “step-by-step reasoning” vs negative: “direct answering”), selecting the first principal direction of the difference. The steered latent states are norm-rescaled for stability. The framework supports two test-time scaling modes: breadth-based (e.g., Best-of-N, Majority Vote) by introducing structured diversity across α values, and depth-based (reflection) by controlling reflection strength. Experiments on GSM8K, MATH500, and GPQA with LLaMA-3-8B-Instruct and Qwen-2.5-7B-Instruct show consistent gains over standard prompting and over baseline test-time compute; results generalize to a reasoning-tuned model (DeepSeek-R1-Distill-Qwen-7B). Additional analyses show monotonic control over verbosity with α, favorable scaling with the number of generations, reflection improvements, robustness to the number of contrastive pairs, and saturation in α-range ablations. A sentence-level variant demonstrates dynamic adjustment of α based on feedback (e.g., PRM/consistency), correcting errors that instance-level control misses.

### Strengths
The paper introduces a simple, principled, and interpretable mechanism for inference-time control of reasoning via latent steering vectors, avoiding input rewrites or fine-tuning. It clearly connects instructional prompts to directional latent shifts and operationalizes this with a normalized additive intervention that is easy to implement and stable. The framework is broadly applicable: it enhances both breadth-based strategies (improving the quality/diversity of candidate generations for Majority Vote and Best-of-N) and depth-based strategies (controlling reflection strength, mitigating both under- and over-reflection). Empirically, results are consistent across multiple datasets and two strong open-source models, and even on a reasoning-specialized model. The analysis section is thoughtful: it demonstrates behavioral control (generation length vs α), scalable benefits with more generations, ablations on α ranges and number of contrastive pairs, and a compelling step-level reflection case. Overall, the contribution is timely and practically significant for the growing area of inference-time scaling.

### Weaknesses
While the contrastive construction of the steering vector is justified and practical, the paper would benefit from deeper analysis of where and how to intervene (per-layer/per-position granularity, keys/values/queries, attention vs MLP streams). The current choice (last-token representations concatenated across layers) is one of many possible designs; an ablation across layers or modules could strengthen claims of generality and inform best practices. The Best-of-N results hinge on an external PRM; more scrutiny of PRM sensitivity (e.g., swapping PRMs, PRM calibration vs α-induced diversity) would isolate the contribution of FR under different reward landscapes. Robustness and safety aspects are lightly treated: although inputs are public, the method changes internal activations; a brief discussion on failure modes (instability, hallucination shifts, adversarial α ranges) and safeguards would be useful. Finally, the dynamic (sentence-level) α control is presented as a case study; quantitative results comparing instance-level vs sentence-level control across benchmarks would substantiate its impact.

### Questions
How sensitive are the gains to the locus of intervention? For example: steering only specific layers (early/mid/late), only attention vs only MLP, or steering keys/values/queries separately. A systematic ablation could reveal more efficient and robust placements and perhaps reduce compute overhead. Can the authors quantify a compute-efficiency curve: accuracy vs total tokens/forward passes under fixed budgets, comparing FR-enhanced Majority Vote/Best-of-N against stronger baselines (e.g., self-consistency with calibrated temperatures, diversified prompts, or MC-SC)? This would clarify the “accuracy-per-sample” and “accuracy-per-token” advantages. In the reflection setting, can the sentence-level dynamic α be evaluated at scale with a simple feedback signal (e.g., PRM score thresholds) to provide a quantitative gain over instance-level reflection across GSM8K/MATH500/GPQA? Even a modest study (e.g., 1–2k instances) would help establish its general benefit. More broadly, can FR steer away from known pitfalls (e.g., spurious long CoT that sounds plausible but is wrong) by combining α control with lightweight detectors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a  training-free framework that effectively improves test-time compute by enabling fine-grained, continuous control over a model's reasoning intensity via latent steering vectors.

### Strengths
1. The paper is well-written,  and easy to follow . 
2. It presents a clear motivation and supports its claims with relatively thorough experimental validation.

### Weaknesses
1. The core technique is a direct application of existing work (e.g., Representation Engineering, Activation Addition). The paper applies a known method to a new axis ("reasoning vs. direct answer") and lacks a fundamental methodological contribution. It is more of an application case study than novel research. Moreover, several existing papers (e.g. [1]) present ideas that are highly similar to those in this work.
2. The paper wrongly assumes "reasoning" is a single, linear dimension controllable by a scalar α. This is a flawed model of a complex process. The evidence (e.g., Figure 4 showing correlation with length) strongly suggests the method is controlling verbosity, not true logical depth. Performance gains may just be an artifact of longer, more verbose outputs.
3. The fairness of the comparison between the FR method (using N different α  values) and the baseline (sampling N times from 1 fixed prompt) is worth further consideration. By design, the FR method introduces a form of "strategy ensemble" (mixing different reasoning intensities), whereas the baseline lacks this strategy-level diversity. As a result, it is difficult to disentangle from the current results how much of the performance lift comes from the merits of "fractional control" itself versus this built-in ensemble effect.
4. The paper's claims regarding "adaptivity" seem somewhat overstated. In the current implementation, the model itself does not appear to dynamically match its reasoning intensity to a specific problem's difficulty during generation. Rather, it seems to perform a non-adaptive search across N different reasoning styles, with the "intelligence" or "adaptivity" being deferred to the expensive post-processing selection step (e.g., Best-of-N or voting). 
5. The score for the DeepSeek model in Table 3 appear to contain a significant and counter-intuitive anomaly. The paper reports a score of 78.6% on GSM8K (relatively simple) but 92.4% on the much harder MATH500. This result is highly unusual, as the difficulty of MATH500 far exceeds that of GSM8K (a fact supported by the paper's other data and the wider literature). 

[1] Controlling thinking speed in reasoning models

### Questions
Please see the weaknesses.

### Soundness
2

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
3

### Summary
This paper proposes a framework called Fractional Reasoning, which enables adaptive control of a model’s reasoning depth and reflection strength during inference to improve the test-time computational efficiency of large language models (LLMs). By introducing a latent steering vector and a scaling factor, the method allows the model to dynamically adjust its reasoning behavior, striking a balance between concise direct answers and complex multi-step reasoning. Experimental results show that this approach significantly improves performance across multiple benchmarks, enhancing both the breadth and depth of reasoning while increasing sample efficiency. The framework is model-agnostic and can flexibly adapt reasoning strategies according to different task requirements and computational budgets.

### Strengths
1、The proposed method does not rely on model fine-tuning or modifying input texts, allowing it to be directly applied to existing LLMs. It supports a variety of LLMs, including both general instruction-tuned models and reasoning-specialized models, demonstrating strong generalizability.
2、The authors provide complete code and supplementary materials, ensuring high transparency and strong reproducibility, which enhances the credibility of the proposed framework.
3、The work is intuitively easy to understand and clearly explains how adjusting reasoning depth and reflection strength can control model behavior. By introducing latent steering vectors and tunable factors, the framework's design makes the reasoning process more controllable and easier to interpret.

### Weaknesses
1、Lack of explanation for the design of different latent steering approaches: The paper employs two different latent steering vectors for the reasoning (Chain-of-Thought) and reflection processes, but it does not clearly explain why these distinct designs are used. The absence of such an explanation weakens the interpretability of the framework and makes it difficult for readers to understand the specific impact of these design choices on model behavior.

### Questions
1、There are some minor errors in the writing process. In Section 5.2, the reference “As shown in Table ??” is incorrect; please fix this table citation.
2、Since two different latent steering vectors are used in the reasoning and reflection processes, it would be helpful to further explain in the paper why this design choice was made, so that readers can better understand it.

### Soundness
3

### Presentation
3

### Contribution
3
