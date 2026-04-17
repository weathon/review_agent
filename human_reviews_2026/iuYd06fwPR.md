# SelfCAD: Protecting Your Efficient Reasoning Capabilities via Self Cautious Insertion

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Large reasoning models (LRMs) are increasingly deployed in modern AI systems due to their accuracy, efficiency, and transparency, as their reasoning traces enable users and auditors to interpret model outputs. 
However, publishing these traces introduces new risks. 
Adversaries may distill them to replicate efficient LRMs for their own purpose or build proxy models for malicious attacks, raising both copyright and security concerns that threaten the sustainability of the LLM ecosystem.
Existing defenses mainly detect distillation after violations occur or suppress transparency by masking or rewriting reasoning traces, which are impractical in real-world deployments. 
In this work, we propose a defense framework that preserves reasoning traces while preventing effective distillation. 
We begin with a systematic analysis of how different reasoning components affect model efficiency and accuracy. 
Our results reveal that the number of self-cautious sentences plays a crucial role: excessive self-cautious sentences lead to redundant outputs, while insufficient ones harm accuracy. 
Building on this insight, we propose $\textbf{SelfCAD (Self-Cautious Anti-Distillation)}$, a lightweight anti-distillation method that strategically manipulates self-cautious parts after models generate their reasoning traces. 
SelfCAD maintains the semantic clarity of reasoning traces for human users and LLM auditors, but significantly degrades the efficiency and accuracy of the downstream distilled models. 
Experiments on Llama and Qwen show that distilled models incur higher inference cost and lower accuracy, especially for Qwen-1.5B, whose token length is $4.8\times$ longer on GSM8K after distillation with our processed responses compared with distillation with vanilla responses. 
The results highlight a new efficiency-based perspective on safeguarding reasoning models from distillation while preserving interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SelfCAD (Self-Cautious Anti-Distillation), a method to protect large reasoning models (LRMs) from unauthorized distillation. The key insight is that self-cautious sentences in reasoning traces significantly affect both efficiency and accuracy of distilled models. The authors propose inserting additional self-cautious sentences after generation to make distilled models inefficient (producing 1.2-4.8× longer outputs) while preserving semantic clarity for legitimate users.

### Strengths
1. Novel perspective: Identifying self-cautious sentences as a key factor in reasoning efficiency is an interesting observation
2. Lightweight implementation: Inference-time processing without model modification is practical
3.Comprehensive experiments: Testing across multiple models and datasets shows consistency
4. Theoretical grounding: Provides mathematical analysis explaining the mechanism
5. Timely problem: Addresses important concerns about LLM intellectual property

### Weaknesses
1. No adaptive attack evaluation: Doesn't test against adversaries who might detect the pattern
2. Strong assumptions: Theoretical analysis assumes distributions remain stable (Eq. 4) without justification
3. Presentation issues: Poor writing quality, structural problems, missing implementation details

### Questions
1. Robustness to preprocessing: How does SelfCAD perform when adversaries use simple regex or pattern matching to remove the inserted sentences before distillation?

2. Template variations: Have you tested different self-cautious templates or randomized insertions to make detection harder?

3. Adaptive attacks: Can you evaluate against adversaries who train classifiers to detect artificially inserted vs. natural self-cautious sentences?

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
This paper proposes SelfCAD, a defense mechanism against unauthorized reasoning distillation. The key idea is to inject self-cautious sentences into chain-of-thought explanations—phrases expressing doubt or self-verification. These interventions increase the verbosity and reduce the efficiency of student models trained on the generated traces, while preserving the original model’s transparency and performance. Experiments on mathematical reasoning show that distillation from SelfCAD-protected data leads to significantly longer reasoning chains and modestly reduced accuracy.

### Strengths
1. Simple and low-cost deployment — does not require modifying the trainer or model architecture.

2. Maintains transparency — unlike encryption or CoT suppression, users still see the reasoning process.

3. Applies to real-world API scenarios — defense could plausibly be adopted by service providers immediately.

### Weaknesses
The most important thing is that I think defense seems trivial to bypass. While the idea is intuitive, the core mechanism—adding self-cautious phrases into reasoning traces—appears easily removable. A defender with basic text-processing capability (e.g., filtering, paraphrasing, style normalization) could simply strip or rewrite these caution sentences before distillation. This raises a fundamental question:

If the protection can be removed by a simple text post-processing step, is the defense truly effective?

Additionally, because the defense operates purely at the surface-form level, an attacker could bypass it entirely by:

• distilling from logits / token probabilities instead of CoT text

• accessing hidden states directly (a common research practice)

• supervised fine-tuning student confidence calibration back to normal

The paper would benefit from a more rigorous robustness evaluation, including adversarial distillation settings rather than only naïve student training pipelines. So far the method appears fragile under even basic threat models.

### Questions
See weakness.

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
3

### Summary
This paper proposes **SelfCAD (Self-Cautious Anti-Distillation)**, a lightweight defense mechanism designed to protect reasoning-capable LLMs from unauthorized distillation while preserving transparency.  
The authors first perform a fine-grained analysis of reasoning trajectories, decomposing them into *statement*, *reasoning*, *self-cautious*, and *conclusion* sentences. They discover that self-cautious sentences—phrases like “wait” or “let me double-check”—strongly influence both the *efficiency* (output length) and *accuracy* of reasoning models.  
Leveraging this insight, SelfCAD strategically inserts additional self-cautious sentences into reasoning traces. This manipulation keeps human readability intact but leads student models trained on these traces to produce **redundant and inefficient reasoning**, thereby degrading the effectiveness of model distillation.  
Experiments across Llama-3.2-1B/3B and Qwen2.5-1.5B/7B show that distillation with SelfCAD-processed traces increases output length by 1.3–4.8× and reduces accuracy by 2–8% while maintaining over 99% semantic equivalence for humans.  
The paper offers a practical and creative perspective on proactive anti-distillation for reasoning models.

### Strengths
- **Conceptual novelty:** Introduces a unique efficiency-oriented defense mechanism distinct from watermarking or audit-based approaches.  
- **Lightweight and practical:** Can be applied at inference time without model retraining or architecture modification.  
- **Empirical clarity:** Includes ablations and trajectory analyses that clearly illustrate the “self-cautious effect.”  
- **Transparency preserved:** Maintains semantic fidelity while effectively degrading distillation efficiency.  
- **Reproducibility:** Implementation details and evaluation settings are well documented.

### Weaknesses
- **Shallow methodological contribution:** The proposed method essentially inserts fixed *self-cautious* sentences (e.g., “*Wait, let me check again…*”) after each reasoning step. There is no adaptive component, learning mechanism, or optimization objective. As a result, the core technique is heuristic and lacks algorithmic or theoretical depth.  
- **Overstated novelty:** While the problem of protecting reasoning traces from distillation is timely and relevant, the proposed solution is minimal and does not introduce new principles beyond simple text augmentation. Most of the originality lies in the *problem framing* rather than the *technical approach*.  
- **Lack of rigorous baselines:** The paper does not compare SelfCAD against simpler or more intuitive baselines—such as random phrase insertion, noise injection, or partial reasoning truncation—that could yield similar degradation effects.  
- **Limited generalization:** All experiments focus solely on mathematical reasoning tasks. It remains unclear whether the proposed mechanism would transfer to other domains such as commonsense reasoning, logical inference, or code generation.  
- **Weak theoretical justification:** The included theorem is more descriptive than analytical—it essentially formalizes an intuitive observation that adding self-cautious sentences encourages longer reasoning. No provable guarantees or quantitative bounds are provided.  
- **Formatting and polish issues:** Several sections are overly bolded or inconsistently formatted, which negatively impacts readability. These should be fixed for the camera-ready version.

### Questions
1. Since the core mechanism is inserting templated *“wait”* phrases, how does SelfCAD differ from random or semantically neutral text insertion? Could a trivial baseline achieve similar anti-distillation effects?  
2. Have you evaluated whether paraphrasing or filtering the outputs before student training can remove these self-cautious tokens and thus bypass the defense?  
3. Does the anti-distillation effect persist when the student model is fine-tuned for more epochs or trained with RLHF-based objectives instead of simple supervised distillation?  
4. Beyond math reasoning, do the authors have evidence that SelfCAD generalizes to domains such as commonsense QA, coding, or multimodal reasoning?  
5. Could a future version of SelfCAD learn *where* and *when* to insert self-cautious sentences adaptively, rather than applying them uniformly across all reasoning steps?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SelfCAD, a lightweight, inference-time defense that preserves transparent reasoning traces while degrading the effectiveness of unauthorized distillation. The key insight is that the number of self-cautious sentences in reasoning trajectories critically impacts both efficiency and accuracy in distilled models. SelfCAD post-processes teacher outputs by inserting self-cautious sentences after each reasoning step, keeping the original reasoning intact for human and LLM auditors but causing student models trained on these traces to become less confident and over-verbose. Analyses show self-cautious sentences strongly reduce sequence termination probability and lengthen trajectories; training-time studies confirm that removing self-cautious sentences shortens outputs while adding them increases length and can reduce accuracy. A simple theoretical model explains how repeated self-cautiousness induces excessive reasoning.

### Strengths
(1) The paper identifies and isolates the role of self-cautious sentences in driving reasoning length and student confidence, providing both empirical and theoretical support for an efficiency-oriented protection mechanism.

(2) SelfCAD is practical and minimally invasive: it operates as post-processing at inference time, requires no teacher fine-tuning, preserves original reasoning content, and can be run on CPU in minutes.

(3) The methodology is clearly described, including sentence-type categorization, termination-probability analysis, a simple but principled theorem, and an explicit algorithm for insertion.

(4) Experiments span multiple student sizes and two distillation sources, with consistent increases in inference cost and modest accuracy reductions; the stealthiness evaluation suggests preserved semantic transparency for users and auditors.

### Weaknesses
(1) The insertion strategy is uniform and naive, applying the same self-cautious sentence after every step; more targeted placement tuned to correctness or step importance could strengthen effect or reduce accuracy loss, but is not explored.

(2) The defense primarily targets text-reasoning math datasets; generalization to other domains and modalities, longer-context tasks, or instruction-heavy settings is not demonstrated.

(3) Adversary adaptivity is not studied: a distiller could filter or downweight self-cautious spans, apply truncation, use RLHF to penalize over-caution, or use contrastive objectives to recover efficiency.

(4) The semantic-equivalence “stealth” check uses LLM judges on a binary yes/no criterion; human studies or more granular equivalence metrics would better validate transparency preservation.

(5) Theoretical assumptions (mixture model, stability to prior steps) simplify dynamics; empirical tests that vary $\lambda$ and distributional separability would strengthen the causal link to observed length inflation.

(6) Potential collateral effects on downstream evaluators or audit tools that rely on reasoning brevity or structure are not assessed.

### Questions
(1) How robust is SelfCAD to adaptive distillers that strip or downweight self-cautious sentences, truncate intermediate steps, or apply RLHF to penalize excessive caution; can you report results under such countermeasures?

(2) Can you design and evaluate a selective insertion policy that targets steps likely to be correct or pivotal, using lightweight heuristics or a small classifier, to maximize length inflation while minimizing accuracy degradation?

(3) How does SelfCAD perform beyond math, for example, in symbolic logic, code reasoning, tool-use chains, or long-context QA, and does the effect size persist with longer contexts and different tokenizers?

(4) Could you report human evaluation of transparency and usefulness, beyond LLM-judge equivalence, including perceived clarity, redundancy, and auditability of the modified traces?

(5) What are the impacts on student training stability and compute cost during distillation, such as gradient variance, convergence speed, and GPU hours; can you quantify the additional cost imposed on the distiller?

(6) Can you provide ablations on the content and style of self-cautious sentences, frequency of insertion, and position relative to sub-steps, to identify minimally invasive yet maximally effective variants?

### Soundness
2

### Presentation
1

### Contribution
2
