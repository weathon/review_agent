# Clinically Interpretable Rule–Guided Preference Optimization in Vision–Language Models for Radiology Report Generation

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
In modern healthcare, radiology plays a pivotal role in diagnosing and managing diseases. However, the complexity of medical image data combined with the variability of natural language generation often leads to inconsistencies, hallucinations, and a lack of clinical grounding, especially in automatically generated radiology reports. To address these challenges, we introduce a clinically interpretable rule-guided extension of direct preference optimization, tailored for radiology report generation. A typical radiology report comprises of findings and impression, findings capture the complex visual information from the medical image, for example a chest X-ray, and the impression is the implied conclusion. Our framework leverages on this phenomenon to design clinical rules from existing findings and impressions, that connect the finding and impression as a horn rule. The rules act as an additional, interpretable supervision signal, guiding the preference optimization of Vision–Language Models (VLM) toward outputs that are not only fluent but also clinically faithful. Unlike conventional preference optimization, which relies solely on lexical preferences, our approach enforces alignment with clinically meaningful predicates such as the presence, absence, or severity of key findings. A central feature of this framework is its ability to inject clinical rule guidance during optimization, ensuring that generated reports remain both linguistically coherent and clinically accurate. By integrating a neural verifier trained to evaluate rule satisfaction, our method provides an explicit mechanism for grounding preferences in interpretable clinical semantics via the clinical rules. Experimental results on benchmark datasets like MIMIC–CXR-JPG and IU–Xray, demonstrate that our approach substantially improves factual accuracy, and overall report quality compared to supervised fine-tuning and standard DPO baselines. We record a performance boost of 10% and 9% across lexical and semantic metrics. These results highlight the promise of clinically interpretable preference optimization as a pathway toward trustworthy and safe radiology report generation in medical AI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work is proposed to use keyphrases of clinical findings in the prompt to optimize the LLM-based report generation. 

The novelty is limited, where the idea is similar to [1] but no review or comparison is conducted. The experiments is limited, where no experimental results of comparison are reported to show the outperformance of proposed approach.
[1] Yasuhide Miura et al., Improving Factual Completeness and Consistency of Image-to-Text Radiology Report Generation. 

The contribution is also limited, where the results only show that finetuning LLM with keyphrases plus ground-truth reports are better than finetuning with ground-truth report.

### Strengths
Applying DPO in the LLM-based radiololgy report generation (RRG) research.

### Weaknesses
No enough comparison of the baselines are reported in the experiment. It is hard to see whether the proposed approach is state-of-the-art.
- This work use NER as the optimization objects for RRG. The related works of using entity accuracy as the optimization items should be compared, such as [1]. 
[1] Yasuhide Miura et al., Improving Factual Completeness and Consistency of Image-to-Text Radiology Report Generation. 
- No enough comparison with the related works of DPO+RRG. The results only shows the proposed approach is better than the conventional DPO. The related works of using DPO for RRG should be compared, such as [2], [3].
[2] Hong Liu et al., RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation.
[3] Oishi Banerjee et al., Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation

### Questions
For the example of ``cardiomegaly ∧ low lung volume ∧ no pneumothorax ∧ minimal right costrophrenic infiltrates→ cardiomegaly ∧ mild pleural effusion`` of illustrating the horn rule in the paper, why ``low lung volume`` can be used to diagnose ``cardiomegaly`` or ``mild pleural effusion``? It seems like not all the clinical findings in the _Findings_ can be led to every conclusion in the _Impression_. If there is not logical reasoning in the horn rule, then it is a list of keywords instead of ``rule'', or it is a wrong rule.

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
5

### Summary
The paper proposes a clinically interpretable, rule-guided extension to Direct Preference Optimization (DPO) for vision-language models (VLMs) focused on radiology report generation. The method introduces human-interpretable Horn rules derived from the findings and impression sections of reports, integrates these rules via a neural verifier, and enforces clinical alignment in the optimization procedure. The approach is validated on MIMIC-CXR-JPG and IU-Xray datasets, showing consistent improvements in both lexical and semantic diagnostic benchmarks compared to zero-shot and standard DPO baselines.

### Strengths
1. Novel Integration of Symbolic Rules: The paper presents a concrete framework for extracting and integrating structured clinical rules (Horn rules) into the VLM optimization procedure, moving beyond purely data-driven or black-box preference optimization.
2. Clinically-Grounded Objective: By combining a neural verifier (trained to detect rule-reported alignment) and DPO, joint optimization explicitly encourages both fluency and factual grounding—addressing a major need in medical AI for trustworthy, clinically faithful outputs.
3. Clear Algorithmic Pipeline: The overall methodology, from data preprocessing (entity/modifier extraction, rule formation) to preference data curation, neural verifier architecture, and constrained optimization, is systematically presented. Algorithm 1, supported by Figure 1 (img-0.jpeg), effectively illustrates this pipeline.
4. Empirical Results: Quantitative gains are robust across two prominent datasets and are supported by a comprehensive set of metrics (BLEU, ROUGE, ClinicalBERTScore, RG-F1, GREEN, LLM-based evaluation). Table 1 and Table 2 clearly show the stepwise improvements provided by DPO+Verifier over both zero-shot and plain DPO.
5. Interpretability & Clinical Faithfulness: Example outputs and qualitative analysis (Appendix, Section A.4) demonstrate improved preservation of critical clinical entities, with detailed examination of how grounded reporting is achieved by the proposed method.
6. Ethical Considerations: The paper demonstrates strong attention to clinical and ethical standards (see Ethics Statement), including data anonymization, bias mitigation, and responsible auxiliary LLM use.

### Weaknesses
1. Limited Baseline Coverage on Interpretability
2. Missing Related Work Citations and Contextualization
3. Missing or Unconvincing Theoretical Justification of Clinical Rule Formulation/Verifier Architecture
4. Incomplete Mathematical Exposition and Lagrangian Setup
5. Absence of Failure Analysis
6. Weakness in Generalization Claims: While Section 5.2 makes claims that the method generalizes well to non-medical VLMs or across tasks, Table 1 and Table 2 show only modest gains for non-medical baselines, and the paper downplays these results without systematic discussion. Are these models meaningfully improved in clinical practice, or is the argument more theoretical?
7. Figure 1  Underexplained: While this figure provides a high-level workflow diagram, there’s no reference in the main text explicitly interpreting what each module/conduit means in the context of (for example) tradeoff of reward signals, alignment failures, or possible control flow divergence between DPO and verifier decisions.

### Questions
1. Verifier Sensitivity: Can the authors provide ablation studies or calibration analyses regarding verifier threshold $\varepsilon$ and $\lambda$? Specifically, how does the choice of verifier threshold alter trade-offs between fluency (BLEU/ROUGE) and clinical consistency (RG-F1, GREEN)?
2. Failure Mode Profiling: Are there systematic failure cases—such as overconstrained generations, verifier misclassifications, or entity extraction ambiguities—leading to clinically invalid or nonsensical outputs? Can the authors provide frequency counts or examples beyond the case in Appendix A.4?
3. Generalization to Rare Findings and Out-of-Distribution Reports: How does the rule-guided framework handle rare or composite findings, or impressions absent in training data? Can the authors elaborate on coverage and error rates across different pathologies or anatomical regions?
4. Comparison to Knowledge Graph and Region-aware Baselines: Why are direct clinical knowledge or region-aware models (e.g., KiUT, RGRG) not included as baselines or discussed? What would be required to meaningfully compare with these systems?
5. Reference Implementation and Reproducibility: Will the authors release the code, annotation schema, entity/modifier lexicons, and verifier model checkpoints to enable independent reproducibility? If not, what is the anticipated effort required for replication?

### Soundness
2

### Presentation
2

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
This paper proposes a clinically interpretable rule-guided extension of DPO for RRG. The authors leverage inherent findings to impression structure in radiology reports to construct Horn-style predicate rules that capture clinical reasoning in natural language form. A neural verifier is trained to assess whether generated reports satisfy these rules, and its output is integrated as a constraint in a joint DPO objective to fine-tune VLMs. The method is evaluated on MIMIC-CXR-JPG and IU-Xray with BLEU, ROUGE, ClinicalBERTScore, RadGraph-F1, and GREEN metrics.

### Strengths
- integrates symbolic reasoning with modern preference optimization is pretty neat
- The rule verifier introduces an interpretable control signal, moving toward trustworthy medical text generation.
- Evaluation on both domain-specific and general VLMs

### Weaknesses
- Lacking a good central figure showing motivation and method - not enough visuals, all text and numbers is a bit hard to read
- Authors selection of metrics is pretty good, there may be a couple of others that you could consider? There are ones noted in the RadEval paper and ReXrank, try to see if there are ones missing from your paper.
- Limited human validation as there is no expert-radiologist evaluation or qualitative error analysis to assess clinical faithfulness beyond automated metrics
- Joint DPO + verifier training may be resource-intensive; runtime and scalability details are missing

### Questions
- How well does the neural verifier trained on MIMIC rules generalize to unseen institutions or imaging modalities?
- Have the authors considered using structured ontologies (e.g., RadGraph or SNOMED) to formalize rule predicates rather than purely text-based Horn rules?
- Did clinical experts review sample outputs? If not, how confident are the authors that improved metrics reflect genuine clinical correctness?
- What is the additional computational overhead introduced by the verifier constraint during training and inference?
- How does the system behave when findings contradict rules (e.g., conflicting entities in the same report)?
- Could the authors show results removing or varying the verifier threshold ϵ to demonstrate sensitivity of performance to constraint strength?

### Soundness
3

### Presentation
2

### Contribution
3
