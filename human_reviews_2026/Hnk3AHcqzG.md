# Error dynamics of symbolic context in small transformers

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Language models often recover from partial corruption in their inputs, yet the mechanism behind this spontaneous context restoration is unclear. We study controlled, label-preserving corruptions in symbolic arithmetic and find a consistent mid-to-late-layer elbow where later components integrate surviving cues to reconstruct the answer. We introduce two readouts, Repair Difference (RD), a logit-space contribution measure, and Token Agreement (TA), a layer-wise consistency score, and a linearity-scale test that predicts repairability. We find near-linear behavior on clean inputs and pronounced nonlinearity under corruption; the linearity residual predicts repair success. Across model families, accuracy degrades smoothly with corruption ($\rho \approx -1$) and yields compact robustness summaries ($\tau_{50} \approx 27$--$34\%$). RD/TA peak near the elbow, localizing where repair occurs. Brief fine-tuning at moderate corruption improves self-repair, whereas training on heavy corruption weakens it, giving a simple, data-efficient recipe. To test the linearity claim beyond arithmetic, we replicate the context perturbation correlation to the local non-linearity in the NLP corruption task. Together, RD, TA, $\tau_{50}$, and the linearity test form a concise toolkit for diagnosing and training for spontaneous context restoration on corrupted contexts and actionable guidance for when and how models repair corrupted context, offering practical levers for debugging, evaluation, and training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how small Transformer models recover from corrupted symbolic inputs—a phenomenon termed spontaneous context restoration. It contributes by: (1) Proposing three label-preserving corruption regimes (zero, in-range, out-of-range) to systematically study spontaneous context restoration under input corruption; (2) Introducing two new interpretable readouts—Repair Difference (RD) and Token Agreement (TA)—to quantify and locate internal repair dynamics; (3) Identifying a consistent mid-to-late layer “repair elbow” where self-repair occurs; (4) Proposing a linearity–scale test that measures local nonlinearity and predicts repair success; (5) Replicating the same repair and linearity behavior on natural language tasks; (6) Showing that moderate corruption fine-tuning enhances robustness, while heavy corruption weakens it.

### Strengths
Originality & Significance: 

(1) Unique and mechanism-driven perspective. The need for a mechanistic explanation of the “spontaneous context restoration” phenomenon is compelling. 

Quality & Clarity: 

(1) Comprehensive analytical framework. The paper introduces label-preserving tasks and custom metrics (RD, TA, and the linearity test), forming a complete, quantifiable, and reusable diagnostic toolkit.

(2) Theoretical insight with practical value. The paper provides fine-grained insights into the phenomenon of “spontaneous context restoration”, e.g., repair emerging in mid-to-late layers, and further proposes a practical robustness enhancement strategy through moderate corruption fine-tuning.

### Weaknesses
The central claim is that the model integrates surviving cues in the mid-to-late layers to repair corrupted inputs (spontaneous context restoration); however, the so-called “repair signal” observed in the experiments is merely a phenomenological interpretation of the RD/TA curves, lacking causal validation.

In other words, the main concern is that the analysis remains descriptive and does not provide causal or mechanistic evidence supporting the claimed repair process. The paper narrows the behavioral observation of “input corrupted → output still correct” from the model level to the layer level, quantified via token-level matching (TA), logit-level difference (RD), and linearity–scale tests. However, these analyses remain descriptive diagnostics rather than mechanistic explanations. The work does not verify that the model truly integrates surviving cues or reconstructs the answer, as it claims. Instead, it presents a functional hypothesis of “restoration” without providing causal or structural evidence.

### Questions
- Could the authors clarify what direct causal or representational evidence that such “integration and restoration” actually occurs?
- Presentation issue: Terms such as “repair signal,” “integration,” and “restoration” are used frequently but not rigorously defined.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper investigates how small transformers analyze and recover from partially corrupted inputs in symbolic arithmetic tasks. They introduce two metrics, Repair Differences (RD) and Token Agreement (TA), for evaluation. They connect local curvature to accuracy under corruption: when behavior is locally linear, models are predictable and repairable; when curvature grows, errors amplify. They replicated their findings to an NLP task and the pattern largely preserves.

### Strengths
1. The perspective is novel. The focus on "spontaneous context restoration" provides a fresh angle on robustness, which is a clearly novel idea compared to the standard notion of adversarial evaluation robustness

2. The metrics RD and TA are novel and appropriate. Per definition, they are intuitive and support the evaluation task

3. The experiments are comprehensive, including various levels of corruption and multiple types of hardware. Testing on NLP tasks in addition to symbolic tasks provides significant insights

### Weaknesses
1. The task diversity is limited. It largely focuses on symbolic tasks; although NLP tasks are included, the tasks are simple (e.g. SVA) and do not include more sophisticated tasks like reasoning or compositional tasks. The narrow scope of tasks downgrades the applicability of this paper

2. The practical implications of the findings are unclear. Other than noticing where the repairs occur, the work did not explain how they emerged, nor answer questions like "which neuron or attention head was responsible"

### Questions
Please see the Weakness section

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies error dynamics and spontaneous context correction in small Transformer models trained on symbolic arithmetic sequences. The authors introduce a quantitative framework—including Repair Difference, Token Agreement, and a local linearity test—to analyze how hidden representations evolve when the input context is partially corrupted. Through layer-wise probing, they find a consistent “elbow effect” in which early layers propagate errors while mid-layers partially repair corrupted representations, followed by late-layer stabilization or over-correction. The analysis links local linearity to repairability, showing that deviation from linearity increases roughly quadratically with perturbation strength, indicating that stronger curvature predicts reduced robustness. These findings generalize to pretrained language models on linguistic tasks such as Subject–Verb Agreement and factual cloze completion. Finally, the paper shows that targeted fine-tuning on moderately corrupted data  substantially improves robustness to input noise, while heavy corruption fine-tuning leads to over-specialization and collapse. Overall, the work provides a mechanistic link between local linearity, internal error correction, and model robustness.

### Strengths
1. The paper proposes a novel conceptual lens for studying robustness in small Transformers, viewing it as spontaneous context correction.
2. It proposes concise, quantitative metrics (etc. RD, TA) that make internal error dynamics measurable.
3. Experiments across custom and pretrained models consistently reveal mid-layer repair behavior and validate the theory.

### Weaknesses
1. The correlation between local linearity and repairability lacks causal validation or controlled intervention, leaving the direction of influence unclear.
2. The fine-tuning and evaluation results may be confounded by data distribution shifts and lack statistical significance reporting, making the robustness trends potentially noise-driven.
3. The generalization to larger models and natural language tasks is limited and qualitative, reducing the external validity of the findings.

### Questions
1. The paper’s claim that “local linearity predicts repairability” remains correlational rather than causal. It is unclear whether higher linearity enables repair, or whether successful repair merely preserves linearity. Without controlled interventions (e.g., curvature removal, normalization scaling), the direction of causality is not established, and potential confounding factors (layer width, LayerNorm placement, residual scaling) are not controlled. Can the authors clarify whether linearity is a cause or a byproduct of repairability, and rule out possible confounds?
2. The paper does not specify the number of runs, random seeds, or variability across samples. Can the authors provide confidence intervals, variance analysis, or seed-averaged results to ensure that the observed repair patterns are not artifacts of random initialization or noise?
3.  Can the authors clarify their operational definition of “small transformer” and examine whether repairability increases systematically with scale or pretraining diversity?
4. In §4.4, the authors extend their symbolic findings to NLP by testing only a single large model (GPT-Neo-1.3B). Can the authors provide more systematic evidence (e.g., multiple pretrained models or scaling baselines) to verify that the observed late-layer curvature and residual growth are consistent phenomena rather than idiosyncratic to GPT-Neo-1.3B?
5. Suggestion: The paper identifies the phenomenon of “error propagation → repair → over-correction,” but does not pinpoint which components (attention heads, MLPs, or residual pathways) drive the effect. Would the authors consider adding neuron- or head-level probing to reveal which submodules actually drive repair behavior?

### Soundness
3

### Presentation
2

### Contribution
2
