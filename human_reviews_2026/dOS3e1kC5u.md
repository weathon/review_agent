# Stealing and Defending the Ends of LLMs

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Soft prompt tuning has emerged as a powerful and automated approach for adapting large language models (LLMs) to new tasks, eliminating the need for manual prompt engineering. The practical relevance of soft prompts is underscored by their support in major toolkits and APIs such as NVIDIA NeMo and IBM Watsonx AI. However, as soft prompts encode valuable, task-specific information, they have become attractive targets for adversarial extraction. In this work, we demonstrate that attackers can extract functionally equivalent soft prompts from prompt-tuned LLMs, effectively replicating their capabilities without access to the original training data or resources. By training a dedicated inversion model, we show that such extraction generalizes, enabling recovery of soft prompts for any downstream task on the given model. To counter this threat, we introduce CAP (**C**overage-**A**ware **P**erturbation), an active defense that substantially impairs extraction while maintaining task performance for legitimate use. Our framework highlights both new risks and practical solutions, paving the way for more trustworthy deployment of adapted LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the security of soft-prompt–tuned LLMs exposed via APIs that return next-token probabilities. It presents a two-stage attack: (i) distillation, which learns a functionally equivalent soft prompt by minimizing KL divergence between victim and surrogate outputs; and (ii) inversion, which trains a Transformer to map probability vectors directly to soft-prompt embeddings and claims cross-task generalization. To mitigate, the paper proposes CAP, an active defense that estimates query coverage in embedding space (via LSH bucket coverage, new-bucket rate, and spread) and injects Gaussian noise into the served prompt or logits in proportion to coverage. Experiments on T5 variants (plus some roberta-base) show strong Stage-1 performance; Stage-2 reports transfer across related tasks; CAP reduces extraction and increases last-layer RMSE. Additionally, a timing side-channel (SPLIT) for estimating prompt length appears in the appendix.

### Strengths
- S1: KL-based distillation on black-box probabilities seems simple, effective, and broadly applicable.
- S2: The authors target a realistic API settings (full/top-k probabilities), and frame risks/defenses at both the input (soft prompt) and output (last layer) ends.
- S3: Results show some transferability on T5 sizes and roberta-base; defense also touches Pythia/GPT-2 for last-layer extraction.
- S4: The separation into distillation and inversion clarifies assumptions and makes the attack path easy to reason about.

### Weaknesses
- W1 (Baselines): The Random prompt in Table 1 hovers around $50$% (even for MNLI where random chance should be $33$%?), which is hard to interpret. It would be great to include as a baseline T5-base with no soft prompt or a discrete prompt. This would clarify whether random soft prompts actively harm the model.
- W2 (Missing comparative baselines): Beyond ``random'', I believe this paper needs at least one additional baseline to contextualize the performance. For example this could be [Morris et al,. 2024: Language Model Inversion] to recover a discrete text prompt and compare its downstream performance to the presented results.
- W3 (Defense utility & adaptivity): I am skeptical about the assumption that benign queries are low-diversity (e.g. multi-task users). As this seems to be a fundamental assumption for the proposed defense method, it would be great to justify this with actual user data (e.g. WildChat dataset).
- W4 (CAP results interpretation): Table 2 shows strong perturbation on the attack queries; reporting per-run coverage metrics (C/N/S) would explain why these are deemed diverse. Additionally, in table 3, it would be important to add a row for LEGIT + CAP-OFF to show absolute utility drop. 
- W5 (Model choice): T5 seems a bit dated by now. Running some of the T5 experiments in stage 1 with e.g. Pythia would improve this.

In its current state I cannot recommend acceptance of the paper. However, the points I raised are adequately addressed, I am willing to increasing my score.

### Questions
- Q1: Could you clarify how exactly the LEGIT/ATTACK classification in Table 3 is defined?
- Q2: The adversary sometimes even outperforms the target (Table 2), and in summarization (Table 10) random $\approx$ target while reconstructed underperforms. Do you have an explanation for these?It might be good to add multi-seed means and standard deviations to make the findings more trustworthy.
- Q3: If trained on a single task, it is unclear to me why the inversion model should generalize? A small ablation varying the number/diversity of training tasks, and a naive cross-task baseline (i.e. reuse the $D_1$ recovered sot prompt on $D_2$) would clarify the incremental value.

### Soundness
2

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
4

### Summary
This paper introduces a technique to distill and invert prompt tuned models through soft prompts.
From the system provider side, the paper also introduces a defense against the proposed inversion attack through monitoring the adversaries' query diversity.
Experiment are carried out on small scale models and datasets showing promising results.

### Strengths
The main strengths of this work are:

1) The proposed approach is generic. Distilling the inverting through soft prompts does not assume any specific architecture nor dataset.

2) The related work section in this paper is thorough.

3) The proposed attack is easy to implement and provides good results under small scale experiments.

### Weaknesses
Despite the aforementioned strengths, this paper has major weaknesses that need to be addressed before getting it published.

1) Experiments are conducted under small scale models and datasets. It is hard to measure actual usefulness. To demonstrate the effectiveness of the proposed attack, experiments should be carried out on realistic size models (such as Llama3-7B), and on realistic tasks/datasets.

2) The assumption that the architecture is shared seems to be restrictive. It is often the case that the API does not provide any information about the architecture. Experiments should also include the case where there is a mismatch between the architecture of the target and the victim models. 

3) The baseline (LLM with random SP initialization) is not convincing. At least, the performance of the base LLM without any SP should be included as a weak baseline. Further, the impact of the learnt soft prompts on different tasks (e.g. reasoning/coding/solving math problems) should be measured.

4) Experiments with comparison with other PEFT methods are missing (e.g. LoRA). While I understand the advantages of soft-prompts compared to LoRA, the argument in this paper would have been significantly strengthen if a comprehensive comparison against other PEFT methods is included.

5) The writing of this paper can be vastly improved. Many parts (such as Figure 1, Section 5.2) are very hard to parse and overcomplicates the simple message of this paper. For example, Equation (2) can be simplified to $TotalCost= \lambda + w_c (\frac{\alpha C}{\lambda \beta} -1 ) + \alpha w_n N + \alpha w_s \min (\frac{S}{S_{max}}, 1)$. Further, the tables can be located in the page they were mentioned in to ease the reading of the manuscript.

6) The proposed defense CAP is both naive and makes unrealistic assumptions. For instance, such defense can be surpassed with multiple attackers setting (each attach does not diversify the queried topics, but different attacks query different buckets). 

7) The paper mentions that the inversion accelerates the distillation (e.g. in line 361), without providing a comparison nor quantification of the time/cost saved. A discussion along these lines is necessary with its corresponding experiments.

Overall, while I extremely appreciate the practicality of the proposed method, I believe that this work has to address significant concerns, provide important extra experiments and ablations before getting published.

### Questions
Please refer to the weaknesses section

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a systematic framework for auditing LLMs' susceptibility to generating malicious code, demonstrating significant real-world impact through the discovery of active scam sites. However, the analysis of model-specific vulnerabilities lacks depth, and the scope is limited to URL-based threats, potentially overlooking other malicious code vectors. The guardrail evaluation is limited to a single system without exploring alternative defenses (Sec. 6.1, Sec. 6.2).

### Strengths
* Novel Attack Formulation
  - Introduces a two-stage attack combining distillation (KL divergence minimization, Eq. 1) and inversion, enabling cross-task prompt extraction (Sec. 3).
  - Demonstrates generalization to unseen tasks (e.g., 87.2% vs. 88.8% on YELP when trained on AMAZON, Table 2), reducing computational costs vs. tuning from scratch (Table 5).
  - Validated on diverse architectures (T5/Roberta) and tasks (classification/NLI), showing robustness to OOD queries (Table 1).

* Practical Defense Design
  - CAP leverages embedding-space coverage metrics (bucket coverage, spread) to distinguish adversarial vs. benign users (Sec. 5.1).
  - Perturbs prompts/outputs based on coverage (Eq. 2), reducing stolen utility to random levels (Table 3) while preserving benign user performance (e.g., 86.7% vs. 91.08% on AMAZON).
  - Extends to defend against last-layer extraction (RMSE increase from 1.96e-5 to 18.21 for T5-base, Table 4).

* Comprehensive Experimental Validation
  - Tests 4 LLMs in 2024 with varied Prompt/Codegen LLM combinations (Table 1), demonstrating robustness across model pairs and architectures.
  - Applies benchmark to 7 state-of-the-art 2025 models (Table 2), showing consistent vulnerability across diverse providers and model sizes.

### Weaknesses
* Limited Threat Model Realism
  - Assumes full access to next-token probabilities, but real-world APIs (e.g., OpenAI) often expose only top-k tokens (Sec. 4.1). Experiments with top-5 access (Table 8) show minor degradation, but broader constraints (e.g., rate limits) are unexplored.
  - Ignores ethical implications: No discussion of misuse risks (e.g., IP theft) or mitigation beyond CAP (Sec. 7).

* Inadequate Analysis of Guardrail Effectiveness
  - Only tests one guardrail (NeMo Guardrails), with no comparison to alternative safety mechanisms or discussion of why it failed.  
  - No investigation into specific policy gaps (e.g., S24: "Use of scam API/website" in Fig. 9) that might improve detection.  
  - Fails to address why guardrails missed all malicious outputs despite clear scam API references, leaving mitigation strategies unexplored.  

* Insufficient Discussion of Training Data Poisoning Mechanisms
  - Claims data poisoning but provides no evidence of how poisoning occurred (e.g., specific training data sources or crawl processes).  
  - Hypothesizes OpenAI models’ higher malicious rates due to "more extensive data containing scam-related content" (Sec. 6.1) without supporting evidence or analysis.  
  - No investigation into why certain models (e.g., gpt-4o-mini) consistently produce higher malicious rates across combinations (Table 1), limiting understanding of root causes.

### Questions
1. In Section 1, the authors state that "the actual rate of malicious code generation likely exceeds this figure when considering attack vectors beyond URLs." Could the authors provide a preliminary analysis of other malicious code vectors (e.g., backdoors or worms) or discuss how their framework could be extended to include them? (Sec. 1)

2. In Table 1, the authors report varying malicious rates based on Prompt and Codegen LLM combinations. Could they provide more details on how specific Prompt LLM characteristics (e.g., keyword diversity or prompt specificity) correlate with higher malicious rates? (Sec. 6.1)

3. The paper claims "malicious content contamination is an industry-wide problem persisting despite advances in safety alignment" (Sec. 8). Could the authors discuss potential training data sanitization strategies that could mitigate this issue, and whether any such strategies were tested in their framework? (Sec. 8)

### Soundness
3

### Presentation
2

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
The paper studies the vulnerability of soft-prompt-tuned LLM deployments and proposes: (1) Attack: A two-stage black-box attack, including distillation stage (learn a functionally equivalent soft prompt by minimizing KL divergence to the victim’s next-token probabilities) and inversion stage (train a small transformer to map next-token probability vectors to soft-prompt embeddings that transfer across tasks); (2) Defense: Coverage-Aware Perturbation (CAP), track query diversity via LSH-based bucket coverage, new-bucket rate, and spread; map coverage to a perturbation budget and inject Gaussian noise either into the soft prompt or outputs to frustrate extraction, claiming minimal harm to benign users.

### Strengths
1. The paper solved a timely and important problem, and the authors provide a clear and realistic threat model.
2. Proposed attack is simple and effective at stage 1 and the learnt inversion model yields usable prompts on unseen tasks at stage 2.
3. The paper provides a concise end-to-end runtime analysis showing that the proposed two-stage attack is far faster than prompt tuning. CAP adds only moderate overhead suitable for deployment.

### Weaknesses
1. Limited evaluation beyond mid-scale classification: experiments focus on T5/RoBERTa/GPT-2/Pythia and classification tasks. Evidence for generation tasks (e.g., summarization) is weak or inconsistent and should be expanded with generation metrics (ROUGE/BERTScore) and CAP-on vs CAP-off comparisons.
2. Table 2 shows when trained on YELP and evaluated on MOVREV, the Adversary (CAP Off) accuracy is 89.33% (+7.2 points to target). This is intriguing, could we provide some insight for the underlying reason? Or a variance analysis with multiple seeds may help.

### Questions
1. Will identical inputs from the same benign user may produce slightly different behavior over time? Can we guarantee stability and user-visible variance?
2. Can CAP be fooled by coherent multi‑task users (e.g., a legitimate pipeline with naturally diverse inputs)?

### Soundness
2

### Presentation
3

### Contribution
3
