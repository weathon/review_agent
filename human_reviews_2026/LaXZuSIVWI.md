# AutoClinician: Structured Clinical Guideline Integration for Trustworthy Diagnostic Reasoning in Healthcare

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Recent advances in large language models (LLMs) have enabled clinical conversational systems with impressive diagnostic capabilities. However, existing approaches often lack alignment with real-world clinical workflows and fail to provide interpretable, evidence-grounded reasoning. In this work, we propose AutoClinician, a unified and training-free framework that integrates clinical guidelines to support stepwise and explainable diagnosis on real-world electronic health records (EHRs). AutoClinician first extracts and summarizes narrative guidelines into Clinical Evidence Graphs (CEGs). These graphs are further automatically verified and refined using a consistency-based strategy. To support trustworthy and patient-specific diagnosis, we utilize CEGs by conducting context-aware, evidence-grounded clinical reasoning on EHRs with Deterministic Finite Automaton (DFA). Our framework outperforms both general-purpose and clinically specialized LLMs, and exhibits stronger interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose AutoClinician as a method for aligning LLMs with practical clinical workflows that can integrate existing medical guidelines. The paper demonstrates that AutoClinician can outperform alternative baseline methods by first turning narrative medical guidelines into structured ``Clinical Evidence Graphs'' (CEGs). It then tests and refines these graphs for accuracy using a self-consistency protocol without needing constant expert supervision. When diagnosing a patient the system uses these graphs to guide its reasoning tracking the conversation with a deterministic finite automaton to ensure logic stays on track. This means it is better at recommending the right lab tests and making an accurate final diagnosis compared to models that are just given raw guideline text or no guidance at all

### Strengths
* The methodology is an interesting and novel solution to a generally well-motivated problem
* The writing in the paper is written to a generally good standard
* Evaluation procedures are convincing in that every step of the methodology are essential particularly through the ablation studies

### Weaknesses
* **Consideration for data privacy in healthcare**: If I have understood correctly, the framework's graph generation relies on GPT-4o, a proprietary model accessible only through a third party API. Healthcare institutions handle very highly sensitive patient data and operate under strict privacy regulations. In my opinion, the need to send clinical guideline data (and potentially patient data for future applications) to an external service creates a significant barrier to real world adoption. This point is not discussed within the paper. Analyzing the performance of generating the graphs with open-source (and practical) LLMs is not explore. 
* **Practicallity of real-world deployment**: The paper explores using open source models such as Qwen3-32B, but the overall diagnostic accuracy reported is very low. This level of accuracy is obviously insufficient for a clinical diagnostic tool where errors could have serious consequences for patient safety. The paper does not justify if this performance level is acceptable for practical implementation, which I believe is essential for the paper
* **Technical novelty**: the methodology, whilst sound, does not present much technical sophistication or novelty. I'm not sure it passes the high bar of ICLR

Minor:
* authors do not provide any analysis of the statistical significance of their method. Results in the main tables appear to be from a single run with no indication of variability, for example standard deviations from multiple runs with different seeds. This makes it difficult to assess the robustness and reliability of the findings
* The text in Figure 1 is too small

### Questions
* If a hospital cannot use a third-party API (e.g, due to data privacy concerns), can your pipeline generate CEGs with an on-prem open-source LLM? Did you try this in practice? If not explored, why not? What performance delta would you expect vs GPT-4o, and what evidence supports that expectation?
* The diagnostic accuracy with open source backbones looks very low. What minimum clinical threshold are you targeting, and how far are you from it?

### Soundness
2

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
3

### Summary
AutoClinician proposes a training-free framework to enhance the trustworthiness and interpretability of Large Language Models in EHR-driven diagnostic conversations by rigorously integrating clinical guidelines. The framework first compresses complex narrative guidelines into structured Clinical Evidence Graphs (CEGs), which are automatically verified and refined using consistency-based adversarial pseudo-patient examples to ensure accuracy and fidelity to clinical logic. Diagnostic reasoning is then modeled as a deterministic, multi-step process guided by a Deterministic Finite Automaton (DFA), which acts as a state tracker, retrieving relevant CEGs and interpreting EHR data (including numerical lab values) to guide the LLM's conversational output. Experiments on EHRSHOT and TriNetX demonstrate that AutoClinician significantly outperforms standard zero-shot, few-shot, and iterative RAG baselines across diagnosis accuracy and lab test recommendation metrics, validating the benefit of structured guideline integration and DFA-guided reasoning.

### Strengths
- The structured integration of clinical guidelines via Clinical Evidence Graphs (CEGs) and the novel use of a Deterministic Finite Automaton (DFA) to manage multi-step, evidence-grounded conversational state are highly original for aligning LLMs with real-world clinical workflows.
- The proposed consistency-based refinement strategy using four types of adversarial pseudo-patient examples is a strong methodological contribution for unsupervised validation and correction of LLM-generated knowledge graphs.
- Modeling the diagnostic process as a DFA provides a highly interpretable and explicit reasoning scaffold that directly addresses the need for transparency and evidence-based medicine (EBM) principles in clinical AI systems.
- The comprehensive experimental validation across two challenging EHR benchmarks (EHRSHOT and TriNetX) and multiple ablation studies convincingly demonstrates the empirical superiority of the framework over competitive LLM baselines.

### Weaknesses
- The reliance on GPT-4o for the critical steps of CEG extraction and refinement introduces a potential limitation regarding reproducibility and scalability, as these steps are not validated with open-source models.
- While the DFA provides a strong structure, the definition of the five input categories ($\Sigma$) seems somewhat manual, and the robustness of the LLM's abstraction layer to accurately categorize complex EHR inputs into these abstract classes is not fully demonstrated.
- The paper claims the CEGs are automatically refined, yet the process relies on an LLM-as-a-judge scoring and then in-context refinement using failed examples, which is a powerful but potentially resource-intensive LLM operation whose exact cost/speed tradeoff is not discussed.
- The DFA design currently limits reasoning paths based only on the five retrieved CEGs, which might restrict the model's ability to handle highly rare or complex cases requiring broader medical context or non-standard guidelines.
- Although the CEG structure handles logical dependencies, the current 5-tuple format appears limited in capturing complex multi-factorial interactions or parallel diagnostic pathways common in highly heterogeneous diseases.
- The evaluation metrics, while standard (Acc, MRR, Lab Acc), do not fully capture the quality of the conversational output beyond the final outcome, particularly how the DFA handles the *ConflictResults* state and subsequent re-reasoning in practice.
- The human evaluation provided in the appendix is based on only one diabetes-related case example, which is insufficient to generalize the claims regarding guideline faithfulness and clinical utility across the full range of diseases covered in the benchmark.

### Questions
- Can the authors quantify the complexity (e.g., average number of nodes, depth, branching factor) of the generated CEGs for the different disease subsets (Metabolic, Respiratory, Circulatory), and how does CEG complexity correlate with refinement success?
- Given the reliance on GPT-4o for CEG extraction, have the authors explored validating the CEG generation and refinement process using alternative strong open-source models (e.g., Qwen3-32B or Yi-34B) to ensure the scalability and accessibility of the crucial knowledge preparation step?
- Could the authors provide a more detailed breakdown of the *Failure* input category ($\Sigma$) usage during DFA navigation in the experiments, specifically quantifying how often the model transitions to *Unresolved* and relies on the LLM itself, and how this rate compares between the disease subsets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AutoClinician, a structured, training-free clinical reasoning framework that integrates medical guidelines into a graph-based representation called the Clinical Evidence Graph (CEG). By converting guideline text into structured conditional logic and using a deterministic finite automaton (DFA) to track patient states, the system aims to achieve interpretable, evidence-aligned diagnostic reasoning.

### Strengths
1. The paper is well-written and conceptually clear, with figures that effectively illustrate how CEG and DFA mirror real clinical workflows.

2. The approach is novel in aligning LLM reasoning directly with medical guidelines, producing a structured and interpretable reasoning path rather than unconstrained text generation.

3. The multi-step data construction (guideline parsing, graph generation, consistency validation, and DFA inference) reflects solid engineering design and an appreciation for clinical process logic.

### Weaknesses
1. Despite being “training-free,” the method still heavily depends on large language models for guideline extraction, pseudo-patient synthesis, and consistency scoring, which may introduce bias and circular evaluation.

2. The evaluation relies largely on an LLM-as-judge setup over limited samples, lacking confidence intervals or inter-rater validation, making statistical robustness unclear.

3. The comparison set omits stronger structured baselines such as knowledge-graph or rule-based RAG methods, so it remains uncertain whether the improvement stems from the CEG structure or general engineering advantages.

### Questions
1. How consistent are the results across different judging models or threshold settings for pseudo-patient validation?

2. Could the authors clarify the scope and versioning of the guideline corpus—how broadly does it cover specialties, and how are updates or conflicts handled?

3. When all retrieved CEGs fail to match a case and the system falls back to free LLM reasoning, how is guideline alignment maintained, and how often does this fallback occur?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
AutoClinician is a training-free framework that integrates official clinical guidelines into EHR-driven diagnostic conversations. It compresses guidelines into Clinical Evidence Graphs (CEGs), refines them via pseudo–patient consistency checks, and conducts stepwise, evidence-grounded reasoning with a Deterministic Finite Automaton (DFA). Across EHRSHOT and TriNetX, it outperforms zero-/few-shot baselines and RAG-with-planning on diagnosis, MRR, and lab-test recommendation, with improved interpretability and guideline adherence.

### Strengths
- Aligns LLM reasoning with real clinical workflows and EBM principles.
- Structured, guideline-derived CEGs capture thresholds and ordered test logic beyond plain text retrieval.
- Scalable, training-free refinement using pseudo–patient consistency checks.
- DFA-based state tracking yields transparent, auditable reasoning steps (missingness, conflict, confirmation).
- Consistent performance gains on two EHR benchmarks; ablations show CEGs and DFA both matter.

### Weaknesses
- Heavy reliance on LLMs for CEG extraction/refinement and judging; potential bias, limited human gold standards.
- Pseudo–patient checks may miss rare or subtle guideline nuances; coverage breadth unclear.
- Evaluation uses embedding-based matching; limited details on human IRR and exact-match metrics.
- DFA design may oversimplify complex workflows; limited sensitivity analyses (e.g., CEG count, thresholds).
- Generalization to multimorbidity, unit normalization, and broader diseases needs more evidence.

### Questions
- What guideline sources/versions and disease coverage are included? Any clinician audits or agreement metrics on CEG correctness?
- How sensitive are results to judge thresholds and pseudo–patient volume/types? Can you quantify CEG error reduction over refinement iterations?
- How do results vary with alternative DFA designs and retrieval counts beyond top-5?
- Can you add exact-match metrics and blinded human ratings with inter-rater reliability for dialogue faithfulness?
- Do RAG baselines access the same guideline corpus, and how are context/latency constraints handled for deployment?

### Soundness
2

### Presentation
2

### Contribution
2
