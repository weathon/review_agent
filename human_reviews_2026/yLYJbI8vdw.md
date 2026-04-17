# Multimodal Function Vectors for Spatial Relations

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Large Multimodal Models (LMMs) demonstrate impressive in-context learning abilities from limited multimodal demonstrations, yet the internal mechanisms supporting such task learning remain opaque. Building on prior work of large language models, we show that a small subset of attention heads in two vision–language model, OpenFlamingo and Qwen3-VL, is responsible for transmitting representations of spatial relations. The activations of these attention heads, termed function vectors, can be extracted and manipulated to alter an LMM’s performance on relational tasks. First, using both synthetic and real image datasets, we apply causal mediation analysis to identify attention heads that strongly influence relational predictions, and extract multimodal function vectors that improve zero-shot accuracy at inference time. We further demonstrate that these multimodal function vectors can be fine-tuned with a modest amount of training data, while keeping LMM parameters frozen, to significantly outperform in-context learning baselines. Finally, we show that relation-specific function vectors can be linearly combined to solve analogy problems involving novel and untrained spatial relations, highlighting the strong generalization ability of this approach. Our results show that LMMs encode spatial relational knowledge within localized internal structures, which can be systematically extracted and optimized, thereby advancing our understanding of model modularity and enhancing control over relational reasoning in LMMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explore whether function vectors (FVs), that is activation of a sparse set of attention heads, can be extended from LLMs to LMMs for spatial relation reasoning. Using causal mediation analysis in OpenFlamingo-4B, the authors identify causally influential heads for each relation, sum their relation conditioned mean activations to form function vector. They further fine-tune only the function vector (freezing the LMM) to outperform few-shot ICL, and show that linear combinations of relation-specific function vector solve one-shot analogy tasks involving untrained composite relations. Experiments use both a purpose-built synthetic dataset and a carefully filtered GQA subset.

### Strengths
1. The work moves beyond heuristic head selection by using AIE/CIE to identify heads whose activations cause better relational predictions, then aggregates them into relation-specific FVs—bringing mechanistic interpretability tools to LMMs.
2. Merely injecting FVs improves zero-shot relation prediction; fine-tuning the FVs alone (freezing the model) significantly surpasses few-shot ICL, showing a compact, optimizable control handle for relational knowledge.
3. The synthetic set isolates relations; the GQA subset is stringently filtered to target relational reasoning, helping attribute gains to relation knowledge rather than object/category priors.

### Weaknesses
1. Results are limited to OpenFlamingo-4B; it remains unclear whether the same causal structure and FV utility hold for larger or different fusion architectures. The authors acknowledge this.
2. Only a few spatial relations are considered, and the GQA subset is small (201 images), limiting statistical power and real-world coverage.
3. Injecting or tuning FVs may perturb non-relational abilities (VQA, captioning, text understanding). No retained-capability analysis is reported.

### Questions
When injecting or fine-tuning the relation-specific FVs, what is the impact on other capabilities? Please report retained-capability results (e.g., standard VQA/captioning/text tasks) before vs. after FV injection and after FV fine-tuning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how multimodal function vectors can influence the spatial reasoning capabilities of large multimodal models (LMMs). Building on the idea of "function vectors" from LLM interpretability research, the authors identify a small subset of attention heads in OpenFlamingo-4B responsible for encoding spatial relations. By extracting and manipulating the activations of these heads, they can modify the model’s performance on relational reasoning tasks. 

The study uses causal mediation analysis, experiments on both synthetic and real-world datasets (e.g., GQA), and explores how these vectors can be fine-tuned or linearly combined to achieve compositional generalization. The results suggest that spatial relational knowledge in LMMs is modular, interpretable, and can be systematically controlled.

### Strengths
* The paper provides a novel and insightful exploration of how function vectors affect spatial perception in multimodal large models, extending interpretability research from LLMs to LMMs.

* Experimental results on both synthetic datasets and GQA show consistent improvement, demonstrating that function vectors can effectively manipulate spatial reasoning capabilities.

* The layer-wise and attention-head analyses are methodically performed, supporting the claimed relationship between specific attention structures and spatial reasoning.

* The composite function vector experiments exhibit a promising degree of generalization and compositionality, contributing new understanding to explainable and controllable AI.

### Weaknesses
1. The study is conducted only on OpenFlamingo-4B, which limits generalizability. Given the rapid progress in multimodal modeling, it would strengthen the paper to evaluate newer baselines such as Qwen-VL-2.5 (3B) or Gemma-3 (4B).

2. The selection of 32 object categories in the synthetic dataset is not well-justified. Some categories may appear ambiguous or prone to recognition errors by LMMs? Including more diverse or unseen object classes (in validation set) could better support the robustness of function vectors.

3. In the composite function vector experiments, the paper lacks zero-shot results or a discussion of correlation significance. This omission leaves unclear how strongly these vectors generalize without fine-tuning.

### Questions
1. Could you comment on how the identified function vectors might transfer across different LMM architectures?

2. What criteria guided the selection of object categories in the synthetic dataset?

3. Why are zero-shot results omitted in the composite vector experiments—were the correlations too weak or inconsistent?

4. Do you plan to test this method on stronger multimodal baselines to verify universality?

If these additional experiments or clarifications can be provided, I would be willing to raise my overall score, as the paper’s core idea is interesting and potentially impactful.

### Soundness
3

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
This paper investigates whether the concept of "function vectors" (FVs), previously explored in large language models, can be extended to large multimodal models (LMMs) for the specific task of spatial reasoning. The authors use OpenFlamingo-4B and employ causal mediation analysis to identify a small subset of attention heads responsible for encoding spatial relations.

### Strengths
The experimental design is logical and methodical, progressing from identifying causally relevant components to extraction, fine-tuning, and testing for generalization. 

The work extends a interesting line of research from LLMs to the multimodal domain, which is a potenial new direction for the community.

### Weaknesses
My primary concern is that the entire empirical investigation rests on a single, relatively dated and small-scale architecture and highly controlled, simplified datasets, as authors identified. Maybe the simplified spatial relation reasoning can be well-solved by latest VL model. 

The evidence for strong generalization is not convincing because the analogy task is too simple. The "untrained" relations like "above-left" are just trivial combinations of the trained ones ("above" and "left"). The success of this linear combination is predictable for simple geometric directions and doesn't prove the method works for truly complex spatial relations like "inside," "leaning against," or "occluding." The experiment only validates the approach on an overly simplistic, best-case scenario.

The evaluation and comparsion seems very weak.

### Questions
Can you provide a strong argument for why you believe these findings (e.g., the specific layers and sparsity of influential heads) would transfer to more powerful LMMs like Qwen 2.5-VL models? Did you perform any preliminary experiments to suggest this is the case?

### Soundness
2

### Presentation
2

### Contribution
2
