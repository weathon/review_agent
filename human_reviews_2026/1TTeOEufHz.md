# Don't Throw Away Your Pretrained Model

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Alignment training has tradeoffs: it helps language models (LMs) gain in reasoning and instruction following but might lose out on skills such as creativity and calibration, where unaligned base models are better at. We aim to make the best of both worlds through model collaboration, where different models in the training pipeline collaborate and complement each other. Since LM responses feature interleaving skills that favor different models, we propose Switch Generation, where pretrained and aligned model versions take turns to ``speak'' in a response sequence. Specifically, we train a switcher LM by learning from outcomes of choosing different models to generate the next segment across diverse queries and contexts. At inference time, the switcher LM guides different model checkpoints to dynamically generate the next segment where their strengths are most needed. Extensive experiments with 8 model collaboration baselines and 18 datasets show that 1) model collaboration consistently outperforms individual models on 16 out of 18 tasks, and 2) Switch Generation further outperforms baselines by 12.9% on average. Further analysis reveals that Switch Generation discovers compositional skills to solve problems where individual models struggle and generalizes to unseen models and tasks, reusing and repurposing by-products in expensive model training pipelines that are otherwise discarded.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes to use a trained model selector to select from an aligned model and its pretrained version. The selection happens every few tokens, so the final response combines the capability of both models. The method improves over any single model, and is capable to answer some questions that both models cannot. Ablation study shows that the method can also be generalized to different models (instead of different versions of a single model).

### Strengths
1. The method is novel and provides valuable insights.
2. The paper provides thorough analysis of the results.

### Weaknesses
1. Some parts of the writing are unclear, e.g., in Table 3, we don't know the name of each model.
2. The size of the model selector seems to be the same as the pretrained and aligned models, which raises efficiency concerns. It would be nice to see either the result of a smaller selector model or an efficiency analysis.
3. Some experiment settings are unclear, and they might impact the fairness of the evaluation. See question 1.

### Questions
1. In the main experiment, has any of the baseline methods finetuned on these tasks (i.e., is the comparison with switch-g fair)? Has any of the baseline methods finetuned specifically for each of these tasks (i.e., is the comparison with switch-t fair)?
2. You mentioned that the switcher is a "small" model in section 2, but it's actually the same size of the pretrained and aligned models (8B), is there a reason for this?

### Soundness
2

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
This paper proposes SWITCH GENERATION, a collaborative inference framework that dynamically combines multiple model checkpoints (e.g., pretrained and aligned LMs) during response generation. Recognizing that alignment improves instruction following but harms creativity and calibration, the method uses a trained “switcher” LM to select, at each segment of the output, the model best suited for the current context. Evaluated on 18 tasks, it consistently outperforms individual models and existing collaboration baselines, demonstrating strong generalization, compositional reasoning, and efficient reuse of models typically discarded in training pipelines.

### Strengths
(1)Novel dynamic collaboration mechanism: Introduces fine-grained, trace-aware switching where models “take turns” generating response segments—more flexible than static ensembling or single-model inference.

(2)Effective and generalizable switcher design: The switcher LM is trained via supervised fine-tuning on rollout-based utility signals and generalizes well to unseen tasks and model combinations.

### Weaknesses
(1)The core idea of dynamically switching between models is not entirely new—it resembles prior work in other areas such as cloud-edge collaboration (e.g., ADASWITCH [1]) and LLM evaluation frameworks (e.g., Slide [2]). The paper does not sufficiently differentiate itself or provide deep mechanistic insights into why and when switching between base and aligned models works.

(2)The evaluation only uses same-sized (8B) checkpoints from the same model family (Tulu-v3). This fails to reflect practical scenarios where a large pretrained base model collaborates with a much smaller instruction-tuned model—a more common and resource-efficient deployment pattern.

(3)The paper does not compare against a simple but strong alternative: letting a capable base model refine or post-process the output of an aligned/SFT model. Such a baseline would better isolate the benefit of dynamic segment-level switching versus static post-hoc correction.

References:
[1]ADASWITCH: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning
[2]Slide: A framework integrating small and large language models for open-domain dialogues evaluation

### Questions
(1)Why not compare against a “refinement” baseline?
The paper should include a baseline where a strong pretrained (or larger) model processes or refines the output of an aligned/SFT model (e.g., by editing, re-ranking, or validating segments). This would help clarify whether the gains come from dynamic switching itself or simply from leveraging base-model capabilities in a post-hoc manner.


(2)Would the approach generalize to heterogeneous model scales or architectures?
The current evaluation assumes homogeneous model families and sizes. It remains unclear whether SWITCHGENERATION would work effectively when combining, say, a 70B base model with a 7B SFT model—or models from different families (e.g., Llama + Mistral). Have the authors tested or considered such settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the SWITCH GENERATION method to address the alignment tax problem in LLM alignment training. The core idea is that each model checkpoint has its advantages at different stages of the training process (pretrained/fine-tuned/aligned). Aligned models have strong inference capabilities, but pre-trained models are better at tasks such as creativity and calibration. The authors train a small switcher LM that dynamically selects which model to use to generate different text segments during generation (segment-level switching).

Experiments validated the method on 18 datasets, outperforming single models on 16/18 tasks, with an average improvement of 12.9% compared to 8 baselines. The method can generalize to unseen model combinations and tasks, and inference costs can be reduced through distillation (although only 57.5% of the gain is recovered).

Overall, this is an interesting task with a clear motivation, but the main problems are high computational costs and weak theoretical foundation.

### Strengths
1. **Important Problem, Strong Motivation:** Alignment tax is a real problem, supported by Table 1 and related literature. Pre-trained models do perform better on tasks such as WikiDYK and creativity. The idea of "not discarding pre-trained models" is quite meaningful given the high cost of training .

2. **Innovative Method Design:** Segment-level switching offers a new granularity option; using LM as a switcher leverages its language understanding capabilities to address the timing of switching, a reasonable design; model-labeled traces help the switcher learn the strengths of each model; and rollout evaluates the average performance of each choice.

3. **The experimental design is reasonable and comprehensive:** The 18 datasets are representative and cover different types (knowledge/reasoning/creativity/security); the 8 baselines cover different collaboration methods (API/text/logit/weight level); ablation experiments (patch size, training strategy) were conducted; the supplementary materials include statistical significance tests; and the generalization ability was also verified.

4. **Experimental results are good:** Significant improvements were observed on most tasks, outperforming single models on tasks 16 and 18, with an average improvement of 12.9% compared to the baseline.

5. **Strong generalization ability:** It can generalize to different model families (Qwen2.5), different numbers of models (2/4), specialized experts, and unseen tasks.

6. **Clear Writing:** Figure 1 effectively illustrates the motivation (the pretrained model does indeed perform better on some tasks), and Figure 2 fully demonstrates the methodology. Table 1 uses color to distinguish task types in the main results , facilitating understanding. The case studies (Tables 7-9) are compelling.

7. **High reproducibility:** We promise open-source code, data, and the switcher model. Appendix B contains detailed experimental information.

### Weaknesses
1. **High computational cost is the biggest problem.**
Inference requires n+1 models in memory (4 times the memory), and the switcher needs to be called every 50 tokens. Distillation reduces costs but only restores 57.5% of the gain. More importantly, the paper does not provide actual runtime and throughput comparisons, which are crucial for practical deployment. Hyperparameters are sensitive, and the patch size needs to be optimized for the specific task.

2. **Weak theoretical foundation**
Why is segment-level analysis superior to token-level or response-level analysis? There is a lack of theoretical explanation; the ablation in Table 2 is insufficient.
Why is tracing helpful? The R² in Figure 5 is 0.017, which shows almost no correlation.
- The definition of "new skills" (10.7%) in Table 5 is not rigorous enough and may be partly due to randomness.
How does the Switcher learn to recognize different skills? Figure 6 is only descriptive.
- The lack of in-depth analysis of these issues limits further improvement of the methodology.

3. **The difference between this method and routing methods is not significant enough.**
The essential difference lies mainly in the granularity and the use of traces; it is not a revolutionary breakthrough.

4. **Soundness-related issues**
Training the Switcher requires 10k samples × 32 rollouts (320k calls); sample efficiency was not discussed.
- Patch sizes vary greatly across different tasks (Table 2 shows 10-100), but there is no research on adaptive selection.
- The rule that the beginning and end must use aligned models is hard-coded and not elegant enough.

5. **Insufficient experimental analysis**
- Table 1 in the main text does not indicate statistical significance (although the supplementary materials do).
- Some datasets choose to be based on hypotheses rather than empirical evidence.
- Different evaluation indicators are not uniform
- On which tasks is switch generation inferior to single-model or baseline approaches? Why? These analyses are missing.

6. **Security Issues**
The Ethics Statement mentions that a misaligned base model may circumvent safety guardrails, but it lacks empirical assessment of the risks associated with harmful content generation. This is a serious concern.

7. **Generalization is still questionable.**
- Table 3 shows a significant decrease in generalization performance (setting 1: 5.8% → setting 4: 3.1%).
Can models with different architectures collaborate? For example, can GPT and LLaMA be combined ?
How did it perform with long text (>512 tokens)? Not evaluated.

8. **Minor Writing Issues**
The definitions of pretrained/finetuned/aligned in footnote 1 appeared relatively late. The R² of Figure 5 is 0.017, indicating extremely weak correlation; the statement of "positive correlation" in the text is overly optimistic.

### Questions
1. **What are the specific inference time, memory usage, and throughput?** This is crucial for evaluating usability. How does it perform with a single GPU/multiple GPUs?

2. Can the patch size be adjusted dynamically? Currently, we need to optimize for each task, which isn't very practical, is it?

3. Why use 8B as the switcher? Have you tried 1B? What was the bottleneck?

4. Can the training cost of 10k×32 rollouts be reduced? Have you tried importance sampling?

5. Where is the complete comparison of token-level switching? Table 2 only shows patch size, not token-level results.

6. Of the 10.7% "new skills" in Table 5, how much is a combination effect and how much is randomness? Can they be separated?

7. **How to ensure the security of the base model?** How to prevent harmful content? Can security constraints be added during switcher training?

8. Do long sequences (>512 tokens) still work?

9. Can models of different sizes (1B+7B+70B) collaborate? Wouldn't that be more practical?

10. In practical applications, how do you trade off performance and cost? Is a single model sufficient after distillation, or is multiple models necessary?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Switch Generation, an inference-time model-collaboration method that trains a small “switcher” LM 𝑓 to decide which model checkpoint (pretrained, finetuned, aligned) should produce the next patch of tokens in a response. During switcher training the authors generate traces by randomly switching among candidate models, roll out 𝑘 continuations for each candidate, score those continuations with task-appropriate metrics, and turn the argmax choice into supervised labels for 𝑓. At inference time 𝑓 is queried every patch and selection is sampled with top-p. Experiments use three-stage Tulu-v3 checkpoints across 18 diverse datasets (QA, reasoning, creativity, safety, etc.), compare to 11 collaboration baselines, ablate patch size and switcher training, show generalization to unseen tasks and model pools, and demonstrate that distilling SWITCH GENERATION outputs back into a single aligned model recovers a substantial fraction of gains.

### Strengths
The paper proposes a clear, novel operationalization of the who speaks when problem that is conceptually simple and practically appealing: reuse existing checkpoints rather than discarding them. The method bridges routing and multi-agent text-level collaboration in a way that is easy to implement (small switcher LM + rollouts) and is evaluated comprehensively: many real tasks, multiple baseline families, ablations (patch size, untuned/random switcher), and analyses. The empirical gains are substantial and well aligned with the paper’s motivation. The distillation result is particularly useful for deployment trade-offs.

### Weaknesses
1. Experiments only use Tulu-v3. The authors can validate SWITCH GENERATION using model families beyond Tulu-v3 to show the approach is not specific to one model.
2. Because SWITCH GENERATION is conceptually similar to ensembles/routing, include direct comparisons to established ensemble methods (e.g. stacking, weighted voting, MOE and simple ensembling of checkpoints) and clarify where switch generation outperforms or trades off against those baselines.
3. For tasks where some metrics are worse than the original single-model baselines, please analyze why SWITCH GENERATION degrades performance on those metrics and whether tuning patch size, switcher regularization, or constrained candidate sets can recover or explain the losses.

### Questions
See  Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
