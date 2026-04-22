# Teaching Consensus Rules (and Exceptions) to LLMs for Trustworthy Medical Reasoning

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data  in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting  sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes teaching LLMs to follow medical consensus rules (deductive steps) and handle exceptions, using Sepsis-3 as a worked example. The authors fine-tune an 8B Llama-3 model on verbalized, step-by-step rule instantiations paired with patient data, add synthetic exceptions via ICD-10 codes, and couple the LLM with a time-series forecasting (TSF) model (pipeline or multimodal) to predict 24h-ahead clinical variables. They evaluate derivation correctness, forced derivation correctness, and value correctness, reporting near-perfect derivation correctness after fine-tuning, but much weaker value correctness for future variables/SOFA and modest Sepsis F1.

### Strengths
Fine-tuned 8B consistently beats one-shot (8B/70B) and a medical-pretrained baseline on derivation correctness, including forced-history tests. 
Thoughtful framing of the TSF bottleneck and exploration of pipeline vs multimodal coupling with a forecaster.

### Weaknesses
1.Clinical utility is limited by forecasting. Future SOFA value correctness is low; Sepsis F1 tops at ~0.31 with low sensitivity, which is problematic for early-warning use. Stronger TSF and equal-compute comparisons are needed. 
2.Synthetic exceptions only. The “exception” mechanism is shown on synthetic ICD scenarios; no real clinician-in-the-loop edits or real exception logs are used, so robustness and safety are uncertain. 
3.Parser-coupled evaluation. The metrics depend on output formatting and a numeric tolerance; this risks over-estimating competence if the model overfits the template. Ablations on phrasing/format would help. 
4.Potential leakage and narrow scope. “Suspected infection” is provided as input while also defining the label; even if aligned with prior labeling schemes, a leakage-control analysis is warranted. The study is single-center, single-definition (Sepsis-3) and doesn’t test other consensus families.

### Questions
1.Leakage control: Since “suspected infection” is both input and label component, can you report experiments excluding it from inputs, or using delayed/uncertain infection indicators? 
2.Evaluation robustness: How sensitive are derivation/value correctness scores to prompt paraphrases and output formatting? Please include format-randomized prompts and parser-free checks (e.g., structured outputs). 
3.Real exceptions: Can you provide a small real-world clinician-edited set (or retrospective EHR notes) to validate the exception mechanism beyond synthetic ICD insertions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The methodology transforms medical consensus guidelines (e.g., Sepsis-3/SOFA) into auto-generated "reasoning chains (scratchpads)" for LLM fine-tuning, enabling step-by-step rule-based calculations (e.g., extrema extraction, threshold mapping). Two metrics are introduced: derivation correctness (faithful rule-based inference) and value correctness (alignment with real-world data within error margins). An exception-handling mechanism uses synthetic ICD-10 histories to test "rules + exceptions" learnability, including OOD scenarios. The approach employs Llama-3 8B with LoRA fine-tuning and a Transformer-based TSF model, integrated via pipeline (prompt appending) or multimodal (embedding fusion). Results show near-perfect derivation correctness post fine-tuning, outperforming one-shot models (including 70B) and pre-trained medical LLMs. Exception learning achieves 100% correctness in ID/OOD, while value correctness for current variables is strong but declines for future predictions, revealing TSF as a bottleneck. Innovations include compositional rule learning, automated reasoning faithfulness evaluation, a trainable "rules + exceptions" framework, and TSF-LLM integration to enhance forecasting.

### Strengths
- This method shows potential as a foundational module for "guideline adherence + human-AI collaboration." The study highlights the bottleneck in TSF, pointing out clear directions for future improvements.
- Task definitions, rule diagrams (Figure 3), and examples (Figure 2) are intuitive; appendices provide clear SOFA tables, feature lists, hyperparameters, and hardware details.

### Weaknesses
- Inconsistent handling of missing TSF values: The main text (Sec.4) states that "missing values are only forward-filled from the previous day (carry forward)," while Appendix A.2 describes "hourly vectorization, where missing values are set to 0 (standardized mean) with an added missing mask."
- Forced derivation correctness interpretation: The paper claims "correct histories and predicted histories are very similar," which holds for fine-tuned models but not for one-shot models (Table 1 shows notable differences between forced and non-forced one-shot results, with SOFAdiff performing worse under forced conditions). A more precise explanation and analysis of the reasons are needed.
- Pipeline-related drop in current urine output: In Table 3, the pipeline approach shows a significant drop in current urine output performance (0.966 → 0.761), which remains unexplained.

### Questions
Which policy is actually used—LOCF (last observation carried forward) or mean imputation with a missingness mask? Are training and inference consistent? If both were tried, please provide an ablation and discuss the impact on results.

For training targets of future variables/future SOFA, do you use the ground-truth observations from the second 24h window, or fixed TSF predictions? Please clarify to rule out any information leakage and specify whether teacher forcing or scheduled sampling is used.

How exactly do you “force up to the last measurement token” (e.g., constrained decoding, special delimiters, alignment with a parser)? Why do one-shot models show lower SOFAdiff/SEPSIS performance under the forced setting than the non-forced setting? Is this due to alignment/parsing artifacts, truncation, or differences in node matching?

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
The paper investigates how machine learning for early prediction in medicine can be made more explainable and faithful. The authors aim to teach LLMs to follow consensus rules in their reasoning, these are rules that specify how practitioners proceed to diagnose certain diseases. To achieve this they finetune existing small models on this task and compare it against existing larger LLMs augmented with "one-shot" learning as well as models trained on medical text including consensus definitions. They limit their analysis and evaluation to the Sepsis-3 consensus definition. They find that small fine-tuned models perform almost perfect outperforming bigger models. To improve the forcasting using  sparsely and irregularly sampled clinical variables, they propose to use representations of a time series
forecasting model attached to the LLM using a trained conntector i.e. in a multimodal setting.

### Strengths
- The paper is clearly written and easy to follow
- The problem statement is clearly motivated
- The promised claims are supported

### Weaknesses
-  The paper is primarily an application of fine-tuning for reasoning and using a connector to embed a time-series forecasting (TSF) model. Hence, the contribution is limited. Although nobody has looked at this particular problem, it is rather unsurprising that this works well.

- The paper only focuses on one specific application, namely Sepsis-3.

- LLMs are known to be sensitive to the system prompt, and according to the experimental setup there was no prompt engineering performed. The “one-shot” prompt is shown in Appendix A.5, which suggests that the prompt lacks a sophisticated structure. This significantly weakens the claims, as performance could likely be improved by using a more sophisticated prompt template or more than one ICL example.

- The dataset is very commonly used; hence, it would not be surprising if there were leaks into the training set. Because the Llama 3 dataset is not fully public, focusing on models such as OLMo would have been better.

- The analysis misses a nuanced evaluation of failure modes—for example, whether the outputs of the untuned model are near-misses or whether the reasoning is completely off. This is very important to understand what fine-tuning is doing.

- Given that the goal is to investigate “reasoning,” it is not clear why you did not compare against open-source reasoning models such as Qwen.

- Moreover, it would be important to compare to methods such as DSPy or TextGrad, as these methods have been shown to be efficient while not requiring changes to model weights.

Minor:
- The tables are hard to parse. Please bold the best values and also round the results.

### Questions
None.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to train language models to follow medical consensus guidelines step-by-step when reasoning about diagnoses and predictions, focusing on transparent and faithful inference rather than only accuracy. It uses verbalized consensus rules (explicit textual versions of medical inference steps) linked to patient records to fine-tune large language models so that they learn both standard rules and their exceptions. The authors demonstrate their approach using the Sepsis-3 definition, which combines deductive rules (if-then mappings for organ dysfunction) with inductive ones (time-series forecasting of clinical variables). They generate textual training examples where each inference step (such as computing SOFA subscores or combining them into a sepsis label) is verbalized and paired with patient data. Experiments on MIMIC-III show that small fine-tuned models achieve nearly perfect derivation correctness and outperform much larger one-shot or pretrained medical LLMs under all evaluation metrics. They also introduce evaluation metrics distinguishing derivation correctness (logical faithfulness of reasoning) from value correctness (numerical accuracy of outputs). The paper finds that the key challenge for early prediction is not domain generalization but forecasting future, irregularly sampled measurements, which can be mitigated by combining LLMs with a time-series forecaster in a multimodal setup.

### Strengths
Below are the strengths of the paper in my opinion:

1. Clear and reproducible fine-tuning framework where LLMs are trained on verbalized consensus rules instantiated to patient data.
2. The distinction between derivation correctness and value correctness provides a structured and transparent way to evaluate the model’s inference faithfulness and numerical accuracy.
3. The integration of a time-series forecasting model with the LLM in a multimodal setup directly addresses the challenge of predicting future, irregularly sampled clinical variables that is identified as the core issue in the paper.
4. The experimental setup uses a publicly available dataset (MIMIC-III) and applies deterministic consensus-based labeling, ensuring internal consistency between training and evaluation and reproducibility.
5. The inclusion of rule exceptions (via ICD-10 preconditions) introduces a controlled mechanism to test model behavior under rule deviations.

### Weaknesses
I find the methodology looks very appropriate in the current realm of medical AI, however, there are some limitations that addressing them is missing in the current iteration of the work. The main weakness is about generalizability and can be addressed by looking beyond Sepsis-3. Below is a list of weaknesses about this work:

1. Although sepsis is very important but as a tool, the evaluation focuses on a single medical consensus guideline (Sepsis-3), limiting evidence that the proposed approach generalizes to other rule systems or disease contexts.
2. The paper lacks ablation studies to quantify the individual contributions of verbalization, fine-tuning, and multimodal integration to the final performance.
3. The generation of verbalized training data depends on manually designed templates and synthetic exception cases, introducing potential bias and limiting scalability.
4. The validation of derived reasoning chains relies on deterministic correctness checks rather than expert or clinical outcome validation, leaving open whether the reasoning aligns with real expert judgment.
5. While derivation and value correctness are well-defined, the methodology does not discuss uncertainty estimation or statistical significance of the reported near-perfect scores. (Bootstrapping can be used here for example).

### Questions
My main questions are directly related to the weaknesses I raised above.

1. How well does the proposed method generalize to other medical consensus definitions beyond Sepsis-3, particularly those that rely on qualitative or interview-based assessments rather than numerical thresholds?
2. Could you provide quantitative evidence or ablations to isolate the impact of fine-tuning on verbalized rules versus the multimodal integration with the time-series forecaster?
3. How were exceptions to the consensus rules (e.g., ICD-10 preconditions) validated to ensure that they represent realistic clinical edge cases rather than synthetic artifacts?
4. Could you clarify how sensitive the metrics are to small errors in the reasoning chain and whether they correlate with overall diagnostic accuracy?
5. How much human curation is required to construct these templates, and could the process be automated for other diseases?
6. Did the authors assess how robust the model is to noisy or incomplete clinical measurements, which are common in real EHR data?
7. And lastly for a minor comment: What are the main limitations that prevent the current system from being directly applicable to clinical decision support, and how do the authors envision addressing them?

### Soundness
3

### Presentation
3

### Contribution
3
