# Parameter-wise Weighted Model Editing for Efficient and Retentive LLM Unlearning

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
To unlearn certain entities in **large language models** (LLMs), model editing is performed by subtracting an entity-specific task vector (TV)--the parameter difference between the entity-tuned model and the original model--from the full LLM. Unlike training-based methods, it avoids costly iterative training. However, as the TV can overlap with LLM parameters essential for retaining knowledge, model editing may suffer from over-forgetting. Observing that each parameter may exhibit different importance for entities to be unlearned versus retained, in this paper, we propose a parameter-wise **weighted model editing** (WME) mechanism to rescale the TV, allowing flexible adjustment of the editing magnitude. These parameter-wise weights quantify the relative importance of each parameter for forgetting versus retention, estimated via ***grad**ients* (i.e., WME-grad) or the *diagonal **Fisher** information approximation* (i.e., WME-fisher). Furthermore, we extend WME to a more general form and provide a discussion of its effectiveness. Results on unlearning benchmarks show that WME outperforms the vanilla TV baseline, and even surpasses popular training-based unlearning methods in both forgetting quality and model utility. While preserving the efficiency of model editing-based approaches, WME maintains the retentive capacity for retaining knowledge, offering a new perspective for both LLM unlearning and flexible LLM editing. Our code is available at https://anonymous.4open.science/r/WME.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Weighted Model Editing (WME), a method for large language model unlearning. By applying parameter-wise weights to the task vector, WME selectively forgets target information while preserving general knowledge. It outperforms baseline methods and even rivals training-based approaches in effectiveness and utility.

### Strengths
1. The problem of LLMs is clear
2. The authors provide some experiments, validating the effectiveness of their method.

### Weaknesses
1. The weighting function (corresponding to Equations 4 and 5) currently appears to be more of a heuristic design. The authors do not sufficiently explain why the absolute value of the gradient or the squared ratio is used as the weighting proportion, nor do they provide corresponding theoretical analysis or approximate derivation to support this form.

 2. The authors choose to compute the gradients on the original model parameters theta_0 (rather than the full model theta_full) to estimate the weights W. This decision is somewhat counterintuitive, as the final editing operation (Equation 3) is actually performed on theta_full. The main text does not explain this, and the related discussion (Appendix C.3) is placed in the appendix, which may easily confuse readers.

3. Although TOFU and MUSE are commonly used evaluation benchmarks, they are primarily based on semi-synthetic or fictional data. If the paper could supplement with a small-scale, real-world privacy-related forgetting experiment, even as a simple validation, it would significantly enhance its persuasiveness.

4. As shown in Figure 7 and Algorithm 1, the parameters tau and alpha may have a considerable impact on the results. It is recommended that the authors add a brief explanation in the appendix or main text regarding the selection or tuning of these hyperparameters.

### Questions
1. Regarding the weighting function Wi: It is recommended to enhance the methodology section by incorporating an intuitive explanation or theoretical derivation for the design of the weighting function Wi. This would significantly improve the understandability and justification of the proposed approach.

2. Regarding the parameter choice (theta_0 vs. theta_full): The analysis and rationale behind the choice of computing gradients on the original parameters theta_0 instead of the full model theta_full, which is currently in Appendix C.3, should be concisely summarized and integrated into the main body of the paper. This clarification is crucial for reader comprehension, as the choice may appear counterintuitive at first glance.

3. Regarding experimental validation: To further strengthen the practical relevance of the method, consider adding a case study involving real-world or privacy-centric unlearning scenarios. Such an experiment, even on a small scale, would provide valuable validation of the method's applicability beyond synthetic benchmarks.

4. Regarding hyperparameters: The paper would benefit from a brief discussion or guidance on the selection and tuning of the hyperparameters tau and alpha, whose values appear to significantly influence the results as indicated in Figure 7 and Algorithm 1.

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
3

### Summary
This paper presents Parameter-wise Weighted Model Editing (WME), a lightweight approach to selective knowledge unlearning for large language models. Instead of applying a uniform scalar to the task vector, the authors introduce parameter-wise weights computed from the relative gradient or Fisher information between the forgetting and retention datasets. The method enables finer-grained control of parameter adjustments and achieves promising results on TOFU and MUSE benchmarks with Llama-3.2-1B and 3B models.

### Strengths
- The paper includes extensive quantitative and qualitative evaluations, covering multiple metrics (FQ, MU, ES) and ablation studies that demonstrate the benefits of both gradient- and Fisher-based weighting.
- The motivation, derivation, and algorithmic steps are logically presented. The figures and tables are well-designed, making the core idea easy to follow.
- The mathematical formulation is intuitive and consistent, with clear connections to prior task-vector and model-editing work. Overall, the paper reads smoothly and is technically sound.

### Weaknesses
- The paper frequently refers to the approach as “model editing.” However, the actual procedure—subtracting or adding task vectors—aligns more closely with model merging in the literature (e.g., task arithmetic, model merging). I suggest clarifying this terminology.
- All experiments are conducted on Llama-3.2-1B and 3B, which are relatively small. The scalability and behavior of WME on larger models remain unclear. An additional experiment on a mid-scale model (e.g., 8B or 14B) would substantially strengthen the paper’s empirical claims.
- The study focuses solely on the Llama-3.2 architecture. To demonstrate generality, it would be helpful to include results on other model families such as Qwen-3 or Gemma, which differ in pretraining corpus and architecture design. This would validate whether the parameter-wise weighting scheme generalizes across model types.

### Questions
- WME-fisher slightly outperforms WME-grad across benchmarks, but the paper provides no intuitive reasoning for why the squared-gradient form leads to more stable weighting. A brief analytical or empirical explanation would make this distinction more convincing.

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
5

### Summary
The paper introduces WME, a parameter-wise weighted variant of task-vector model editing for LLM unlearning. Instead of subtracting a single globally scaled task vector, WME applies per-parameter rescaling based on weights computed from either (i) gradient magnitude differences between forget and retain sets (WME-grad) or (ii) diagonal Fisher information approximations (WME-fisher). The approach is evaluated on the TOFU and MUSE benchmarks using metrics such as Forget Quality (FQ), Model Utility (MU), Extraction Strength (ES), and a gibberish detector.

### Strengths
The per-parameter reweighting strategy is intuitive, easy to implement, and integrates well with task-vector editing. WME demonstrates improvements over vanilla TV in both Forget Quality (FQ) and Model Utility (MU), and in some cases matches or exceeds gradient-based methods and NPO. The authors further justify their design through ablations on key hyperparameters (e.g., τ), supporting the choice of WME-grad and WME-fisher variants.

### Weaknesses
- The claim that WME outperforms training-based unlearning methods in both forgetting quality and model utility is not fully substantiated. Comparisons against strong baselines such as DPO [1] and LUNAR [2] are missing and should be included to support this claim.

- In Figure 9, it is unclear whether NPO timing reflects full-parameter or PEFT training. The authors should (1) include time-efficiency comparisons for PEFT-based GA, DPO/NPO, and LUNAR (with configuration details such as rank), (2) provide results for additional model sizes (e.g., 8B), and (3) provide analysis on FLOPs per token. Such analyses would substantiate the efficiency advantages claimed for task-vector-based methods.

- FQ relies on KS distance of “truth ratio” versus a retain-only “ground-truth” model. The connection of these surrogates to practical deletion risk remains indirect. The authors should clarify this connection and also report raw ROUGE scores for clearer interpretability.

- Robustness testing against known unlearning attacks (e.g., paraphrasing, quantization [3]) is missing. Given how TV are designed, they may be vulnerable to such attacks; including this analysis would help practitioners understand the robustness of the WME method.

- In Figure 10, the WME-unlearned sample appears to hallucinate plausible but incorrect information, which could mislead users. The authors should clarify why such behavior is considered an acceptable or ideal outcome and discuss possible mitigation strategies to ensure safer unlearning outputs.

[1] Direct preference optimization: Your language model is secretly a reward model
[2] LLM Unlearning via Neural Activation Redirection
[3] Does your llm truly unlearn? an embarrassingly simple approach to recover unlearned knowledge

### Questions
See above

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
This paper tackles LLM unlearning using task vector (TV) subtraction. The authors claimed that uniform scalar weighting of the negated TV causes over-forgetting due to parameter misalignment with retain gradients. To solve the problem, the authors proposed to replace scalar weight with element-wise weighting (WME), where per-parameter importance is estimated via gradient magnitudes or Fisher information. Experiments on TOFU and MUSE benchmarks show consistent performance gains over vanilla TV and even some training-based unlearning methods.

### Strengths
1. The problem studied in this paper is of great importance. 

2. The paper is generally well-written. The proposed method sounds reasonable to me. 

3. The experiment results look promising.

### Weaknesses
1. The first main concern I have lies in the technical novelty. The core contribution of this work is a fine-grained scaling of task vectors.
However, such element-/region-wise scaling of offsets is not new in multiple domains [1, 2, 3, 4]. In addition, the main way to determine such scaling factors is based on gradient magnitude (or its square), which are standard importance scores in model pruning. It seems that combining these two to scale a task vector poses very limited challenges. Also, the authors failed to discuss in detail how this work is distinct from existing works technically, except the setting. As a result, the technical contribution of this work is limited.

2. The second issue I see from the paper is about its practicability. According to Line 155, the dataset $D_{full}$ is new, where the knowledge that needs to be forgotten appears. This raises the question of why not directly fine-tune on $D_{retain}$, rather than fine-tuning a separate model with $D_{fgt}$ and applying TV. Admittedly, such a study can be helpful to understand the mechanism and theoretical properties, rather than serve as a realistic practice. But this contribution is undermined by the first weakness.

[1] TIES-MERGING: Resolving Interference When Merging Models, 2023.

[2] Knowledge Composition using Task Vectors with Learned Anisotropic Scaling. 2024.

[3] Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance. 2024. 

[4] Dynamic Fisher-weighted Model Merging via Bayesian Optimization, 2025

### Questions
Please see my comments above in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
1
