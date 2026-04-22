# Debiasing CLIP: Interpreting and Correcting Bias in Attention Heads

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Multimodal models like CLIP have gained significant attention due to their remarkable zero-shot performance across various tasks. However, studies have revealed that CLIP can inadvertently learn spurious associations between target variables and confounding factors. To address this, we introduce \textsc{Locate-Then-Correct} (LTC), a contrastive framework that identifies spurious attention heads in Vision Transformers via mechanistic insights and mitigates them through targeted ablation. Furthermore, LTC identifies salient, task-relevant attention heads, enabling the integration of discriminative features through orthogonal projection to improve classification performance. We evaluate LTC on benchmarks with inherent background and gender biases, achieving over a $>50\\%$ gain in worst-group accuracy compared to non-training post-hoc baselines. Additionally, we visualize the representation of selected heads and find that the presented interpretation corroborates our contrastive mechanism for identifying both spurious and salient attention heads.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper interprets and mitigates biases in CLIP from a mechanistic perspective by focusing on biased attention heads. The authors propose a novel framework, LTC, to measure the contribution of each attention head across layers towards the final prediction, and further isolate spurious and target contributions. With such insights, the authors then propose to debias by suppressing spurious attention states while enhancing target states. Experiments across various CLIP backbones and different bias types show the effectiveness and generalizability of LTC. The text interpretation and image interpretation results also further validate the interpretability of LTC.

### Strengths
1. Unlike existing methods which study the biases in the overall image or text embeddings, the proposed idea of using mechanistic interpretability to probe biases in ViT attention heads is novel and provides a new perspective for bias interpretation in CLIP  
2. In terms of debiasing, LTC outperforms existing methods (e.g., Ortho-Cali and Roboshot) across various benchmarks , further closing the gap between subgroups.  
3. The interpretation results, including text interpretation based on SHAP values and image interpretation based on pixels, provide coherent explanations of the SY, Y and S states. These findings further validate the interpretability of LTC.  
4. LTC supports both non-parameter-tuning and parameter-tuning setups, suitable for a wide range of debiasing needs  
5. Extensive debiasing, interpretability and ablation experiments across various CLIP backbones offer strong support of the effectiveness and interpretability of LTC on CLIP models

### Weaknesses
1. The proposed LTC framework is only for interpreting and debiasing attention head activations of the image representations. The biases in text representations, on the other hand, are not studied and addressed, thereby limiting the framework's ability to comprehensively address potential multi-modal biases in CLIP. Can the proposed LTC be further extended to text encoder of CLIP, which is also transformer-based and has attention heads?  
2. Since LTC assumes a transformer-based architecture for image encoders,  such as ViT, it cannot be applied on non-transformer-based CLIP variants, such as CLIP-ResNet, thus limiting the applications of LTC.  
3. The modeling of spurious and target state contributions in Sect. 4.2 is an interesting approach to decouple target and spurious states, but further clarification is required for the contrastive solution to V_S described in Eq. 10: specifically, how the inequalities in 220 lead to the monotone relations in line 223 after "normalizing each sample's contributions to unit mass" is not clear; additional explanations/derivations on how line 223 inequalities support the formulation of V_S in Eq. 10 will also be helpful for readers' understanding of Sect. 4.2.  
4. A minor suggestion is that more results and analyses on attention states interpretation can be included in the main paper, especially on the "spurious association" section: additional in-depth analysis on Z_SY, "the knowledge of associating S with Y", will provide more insights on the complex nature of biases in CLIP. Can Z_SY be also suppressed in the same way shown in Alg. 1 for additional bias mitigation?

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
2

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
This paper tackles the problem of bias in multimodal models like CLIP, where they learn spurious associations, such as linking a "waterbird" to "water". The authors propose Locate-Then-Correct (LTC), a framework that identifies the specific attention heads in the model responsible for the bias and those relevant to the task. By ablating the spurious heads and enhancing the task-relevant heads with orthogonal projection, LTC achieves over a 50% gain in worst-group accuracy on bias benchmarks compared to other post-hoc methods.

### Strengths
1. The paper is very well-written, well-presented, and easy to follow.
2. The methodology is technically solid, with clear and correct mathematical notations.
3. Illustrations and tables are clear and effectively support the paper's claims.

### Weaknesses
Despite the paper's strengths, the primary weakness lies in the discussion and comparison to related work. While the application of Causal Mediation Analysis is interesting, the core "locate-and-correct" idea is not unique. The authors, perhaps due to the timing of publication, seem to have missed several recent and highly relevant papers that explore similar ideas and show strong performance in debiasing.

The authors should discuss these papers and provide a clear rationale for why the proposed method is novel or advantageous in comparison. At a minimum, performance should be benchmarked against them.

Missing related works include:

- "Debiasing attention mechanism in transformer without demographics", Lu et. al., ICLR 2024
- "A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks", Jung et. al., NeurIPS 2024
- "NeuronTune: Towards Self-Guided Spurious Bias Mitigation", Zheng et. al., ICML 2025
- "EvA: Erasing Spurious Correlations with Activations", He et. al., ICLR 2025

On the other hand, the idea using direct effect for ViT for debiasing has been discussed in this new paper:
- "Model Editing for Vision Transformers", Huang et. al., NeurIPS 2025 (this paper might be accepted after ICLR submission though)


As the basic idea looks similar to me, authors should discuss these papers, and give audience to proper rationale why the proposed method is better than the papers above. If you don't find proper rationale, at least the performance should be better than them.



## Minor Point
The core of the proposed method is Direct Effect (from Causal Mediation Analysis). My concern here is twofold:
1. Technically, Direct Effect is not synonymous with an "importance score."
2. More fundamentally, I question the theoretical justification for applying Causal Mediation Analysis to a deep neural network. While I recognize that many empirical papers have used this approach, its theoretical validity is questionable. Neural networks are high-dimensional, non-linear systems that do not inherently represent a formal causal graph, which is a foundational assumption for CMA.

While a full theoretical defense is likely out of this paper's scope, the authors should be aware of this methodological limitation. As an out-of-scope discussion, the concern related to CMA doesn't affect my rating.

### Questions
I have one high-level question: If the authors chose to use causal effects as the basis for their method, why limit it to Transformer-based models? Causal effects can be (and have been) applied to CNNs as well (e.g., "Mediation CNN (Med-CNN) Model for High-Dimensional Mediation Data").

What was the specific technical justification for narrowing this approach to attention-based methods only, rather than applying it more broadly?

### Soundness
4

### Presentation
4

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
This paper introduces LOCATE-THEN-CORRECT (LTC), a novel, training-free framework designed to mitigate spurious correlations (such as background and gender biases) in Vision Transformer (ViT) based CLIP models. The authors evaluate LTC on benchmarks for background bias and gender bias. The results show that LTC outperforms other training-free baselines, achieving over a 50% gain in worst-group accuracy in some cases. The paper also provides extensive interpretability analyses (using SHAP and attention maps) to validate that the located heads indeed correspond to the hypothesized attributes.

### Strengths
1. Novel and Intuitive Location Method: The core contribution, the contrastive method for locating $P_S$ and $P_Y$ heads by comparing $G_{NC}$ and $G_{NW}$, is clever and well-motivated.
2. Targeted and Mechanistic Intervention: Unlike existing methods that apply a global correction to the final image or text representation, LTC performs intervention on specific attention heads. This mechanistic approach is more granular and effective.
3. Good Interpretability: The paper provides strong qualitative and quantitative evidence that the "located" heads are meaningful.

### Weaknesses
1. Clarity and Readability: The paper is difficult to read, with some parts of the exposition, particularly the dense methodology, being particularly difficult. Would benefit from clearer diagrams or worked-out toy examples.
2. Generalizability of Located Heads: The "Locate" step is inherently dataset-dependent, as it requires a set of samples (even if a validation set) to identify $P_S$ and $P_Y$. The paper shows one instance of generalization (reusing heads from GenderBias-VL for FairFace) and mentions reusing Waterbirds heads for CounterAnimal. However, the limits of this generalization are not fully explored. It remains unclear how robust these head locations are across different tasks, classes, or types of spurious correlation.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Locate-Then-Correct (LTC), a framework for debiasing CLIP by directly analyzing and modifying its attention heads. The key idea is to identify which attention heads are responsible for spurious correlations and correct them without retraining the model.

Specifically, LTC first identifies dominant attention heads that have a strong influence on model predictions by constructing a matrix V that measures each head’s direct effect across samples. Then, by contrasting correctly and incorrectly classified confounding samples (e.g., “waterbirds on land background”), the framework isolates which heads are spuriously correlated with confounding factors and which encode true task-relevant signals. Finally, LTC performs targeted correction: mean ablation, which replaces activations of spurious heads with their dataset mean to suppress bias, and knowledge injection, which reinforces task-relevant heads by projecting them onto text-derived feature directions from an LLM.

Experiments on Waterbirds, CounterAnimal, and GenderBias-VL show gains in bias reduction. Visual and textual analyses further confirm that LTC’s adjustments target semantically meaningful attention heads, offering an interpretable way to improve fairness in CLIP.

### Strengths
1. The method enhances interpretability by quantifying how much each attention head contributes to spurious predictions, revealing which internal components drive biased behavior. Moreover, by visualizing these heads, it allows clear inspection of which image regions lead to incorrect or biased decisions.
2. The proposed method introduces minimal modification to the original CLIP architecture, avoiding retraining and preserving the model’s original zero-shot capabilities.
3. The approach demonstrates consistent improvements across multiple datasets and bias types, showing strong generalizability and robustness of the framework.

### Weaknesses
1. My main concern is that the paper insufficiently discusses related work and does not provide a comparison in the table. A large body of recent literature has explored bias identification and mitigation in vision–language models, including both training-based and training-free approaches [1–9]. While the paper contrasts its approach with a few baselines, a more in-depth comparison is needed. For instance, B2T [1] mitigates bias by adding bias-related keywords to prompts, but such recent approaches are not clearly contrasted with LTC.
2. Presentation
    - The description of input supervision required for each method should be clarified. It is ambiguous whether baselines and LTC assume access to spurious attribute labels, target labels, or validation-only supervision.
    - Many important details are deferred to the appendix, which makes the paper harder to follow. For instance, Section 5.2 references about Table 6 and Figure 10 located in appendix.

[1] Kim et al., Discovering and Mitigating Visual Biases through Keyword Explanation, CVPR 2024.

[2] Gerych et al., BendVLM: Test-Time Debiasing of Vision-Language Embeddings, NeurIPS 2024.

[3] Dehdashtian et al., FairerCLIP: Debiasing CLIP's Zero-Shot Predictions using Functions in RKHSs, ICLR 2024.

[4] Jung et al., A Unified Debiasing Approach for Vision-Language Models across Modalities and Tasks, NeurIPS 2024.

[5] Phan et al., Controllable Prompt Tuning For Balancing Group Distributional Robustness, ICML 2024.

[6] Jang et al., Target Bias Is All You Need: Zero-Shot Debiasing of Vision-Language Models with Bias Corpus, ICCV 2025.

[7] Lu et al., Mitigating Spurious Correlations in Zero-Shot Multimodal Models, ICLR 2025.

[8] Yang et al., Debiasing Vison-Language Models with Text-Only Training, arXiv preprint.

[9] Zhu et al., Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness, CVPR 2025.

### Questions
1. Line 352. What is the precise difference between LTC and LTC*? Is the distinction based on which dataset is used (e.g., training vs. validation set), or does it involve a different optimization procedure?
2. Line 355. The description of LTC-JTT is somewhat vague. Is the only modification that CLIP is replaced with a JTT-trained CLIP model, or are there additional differences in how LTC is applied during or after training?
3. Line 351 & 358. It seems that all methods use zero-shot–inferred group labels for the training set and ground-truth group labels for the validation set. Please confirm if this interpretation is correct and specify any exceptions or differing configurations across baselines.

### Soundness
2

### Presentation
2

### Contribution
2
