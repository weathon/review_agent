# Critic–Adviser–Reviser Cyclic Refinement: Towards High-Quality EMR Corpus Generation with LLMs

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Electronic medical records (EMRs) are vital for healthcare research, but their use is limited by privacy concerns. Synthetic EMR generation offers a promising alternative, yet most existing methods merely imitate real records without adhering to rigorous clinical quality principles. To address this, we introduce LLM-CARe, a stage-wise cyclic refinement framework that progressively improves EMR quality through three stages, each targeting a specific granularity: corpus, section and document. At each stage, a Critic, an Adviser, and a Reviser collaborate iteratively to evaluate, provide feedback, and refine the drafts. This structured, multi-stage process produces records that better satisfy clinical quality standards. Experiments show that LLM-CARe significantly enhances EMR quality across all levels compared to strong baselines and yields improved performance on real-world clinical tasks such as diagnosis prediction. Unlike prior work, our method requires no real EMR text for training or prompting, demonstrating the effectiveness of stage-wise, cyclic refinement for generating high-quality, privacy-preserving EMR datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes LLM-CARe (Critic–Adviser–Reviser Cyclic Refinement), a multi-agent, stage-wise framework for generating high-quality synthetic electronic medical records (EMRs) using large language models without access to real data. The framework operates at three granular levels—corpus, section, and document—each involving cyclic collaboration among three agents: a Critic that evaluates, an Adviser that provides feedback, and a Reviser that updates drafts. The stages sequentially enforce corpus-level distributional realism, section-level content completeness, and document-level logical and clinical consistency. Evaluations on a de-identified dataset of 192k EMRs across 302 disease categories demonstrate that LLM-CARe produces synthetic data of significantly higher intrinsic quality and downstream utility than GAN-based, autoregressive, and direct LLM baselines. Notably, synthetic corpora produced by LLM-CARe enable superior model performance on diagnosis, examination, and treatment prediction tasks, all without using any real EMRs for prompting or fine-tuning

### Strengths
- The paper articulates a pressing problem and approaches it with a structured, multi-agent refinement paradigm inspired by human editorial cycles. This Critic–Adviser–Reviser decomposition is conceptually elegant and operationally grounded.
- The division into corpus-, section-, and document-level quality control provides a principled way to address EMR quality at increasing semantic granularity. This explicit separation reflects a solid understanding of how clinical documentation structure affects utility and consistency.
- Extensive experiments compare LLM-CARe to autoregressive (LSTM), GAN-based (mtGAN), and LLM-based (MedSyn, LLM Direct) baselines. LLM-CARe consistently achieves top scores across all five quality dimensions (completeness, correctness, consistency, demographic typicality, and knowledge coverage) and outperforms on downstream tasks
- The framework improves EMR quality across several backbone models (LLaMA 3.1, Meditron, R1-Distill), showing that the refinement process generalizes beyond a single model or domain setting.
- The ablation study isolates each agent’s contribution, empirically confirming that cyclic interaction—not one-shot prompting—is essential to achieve quality improvements.

### Weaknesses
- While baselines like MedSyn and mtGAN rely on real EMRs for conditioning or training, LLM-CARe is evaluated without them, creating a methodological imbalance. A fair comparison should include a variant of LLM-CARe that also conditions on limited real data to isolate the contribution of cyclic refinement rather than data access differences.
- Please include a recent citation found related to LLM based synthetic EMR generation at a recent ML for health conference (https://proceedings.mlr.press/v287/lin25a.html)
- The paper asserts privacy preservation without providing empirical checks for memorization or data leakage—critical for synthetic data generation claims.
- Because the evaluation tasks (diagnosis, examination, treatment) are derived from the same structure as the generation prompts, there may be an inductive bias favoring models trained under similar textual formats. Try following a clinical task derived from MEDS or something unrelated (https://openreview.net/forum?id=IsHy2ebjIG)
- The multi-agent cyclic process is likely computationally heavy, but the paper does not report inference or generation latency, making it difficult to assess the feasibility of applying LLM-CARe to million-record corpora.
-

### Questions
- How do you control for potential overfitting between the quality evaluator (LLM judge) and the generator, given that both are LLMs possibly from the same family?
- Can you quantify the computational cost of one full cyclic refinement iteration (critic–adviser–reviser triplet) and estimate scalability for large corpora?
- How stable is the framework under different prompting styles or initial draft qualities? Does the cyclic loop always converge, or can it oscillate or degrade?
- Given that human evaluation was performed on 100 samples, can you report inter-annotator disagreement cases or examples where LLM and clinicians diverged significantly? I think this would make for some nice analysis in the appendix in general to showcase divergences in thinking between specialists and AI

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LLM-CARe, a novel framework for generating high-quality synthetic electronic medical records using large language models. The approach employs a three-stage cyclic refinement process with specialized agents (Critic, Adviser, Reviser) that progressively improve EMR quality at corpus, section, and document levels without requiring access to real patient data during generation.

### Strengths
1.  LLM-CARe aims to generate realistic EMRs without relying on real patient records, addressing a critical privacy barrier in healthcare AI. This approach could enable large-scale data creation for medical model training without exposing sensitive information.
2. The paper conducts detailed ablation studies isolating the Critic, Adviser, and Reviser components, as well as corpus-, section-, and document-level refinements. These experiments clearly demonstrate that each module contributes meaningfully to the overall EMR quality improvements, validating the necessity of the multi-agent architecture.
3. Evaluation is multi-layered—combining automatic LLM-based judging, human clinician assessment, and downstream task performance
4. The framework is tested across several LLM backbones (Qwen 2, LLaMA 3.1, Meditron, R1) and maintains performance advantages on all of them, indicating robustness to model architecture and training data differences.
5. The paper includes detailed prompts, backbone configurations, and ablation setups, signaling commitment to transparent and reproducible research

### Weaknesses
1. The paper’s structure is fragmented: key definitions, evaluation criteria, and implementation details are scattered across main text and appendices. This hampers readability and makes it difficult to follow the end-to-end methodology. 
2. The five quality dimensions (content completeness, medical correctness, context consistency, demographic typicality, knowledge coverage) are not well defined in the main text. Critical terms — e.g., “major” vs “minor” symptoms — are undefined.Additionally, the paper provides insufficient detail about baseline methods (particularly MedSyn, described in only one sentence )
3. The introduction claims to address “insufficient coverage of less typical clinical cases,” yet the dataset construction retains only common diagnoses (≥500 records) and explicitly excludes rare conditions. This directly contradicts the motivation presented by the authors and undermines the paper’s central claim.
4. The clinician validation is small (100 records, 4 clinicians) and the paper reports aggregate clinician–LLM agreement without per-criterion reliability (e.g., per-criterion Cohen’s κ, confusion matrices). Given the paper relies on an LLM judge, stronger human validation and per-criterion calibration are needed.
5. The corpus-level alignment step matches synthetic data to a reference distribution. The paper never specifies whether 𝑇\_𝑑  is computed from public statistics, synthetic approximations, or the authors’ real EMR dataset. If it is derived from real EMRs, it contradicts their claim that the generation “requires no real EMRs”. The source and exact construction of this distribution must be stated explicitly.
6. Aligning synthetic data to a reference distribution derived from real data risks reproducing existing demographic or clinical biases. The paper lacks subgroup/fairness analyses to evaluate whether corpus-level alignment amplifies or mitigates such biases.
7. The experiments predominantly use a single model family (Qwen) for generation, judging, and downstream evaluation, which risks correlated errors and vendor-specific bias. This open the possibility that performance gains may be partially due to vendor/model-specific bias rather than general methodological benefit.
8. Missing critical details such as exact sample sizes for each downstream task.
9. The authors provide no statistics on EMR length, number of sections, or content depth, nor do they compare these properties between synthetic and real EMRs. Without such analysis, claims of realism and practical utility are unsupported.
10. Medical correctness is assessed via surface-level alignment checks rather than deeper clinical plausibility. The evaluation does not test whether records reflect realistic multi-disease interactions or plausible event chronology.

### Questions
1. Please clarify whether the corpus-level reference distributions used for alignment are derived from real EMRs, public statistics, or synthetic approximations. If real EMRs were used, specify exactly which aggregate statistics (e.g., age, gender, disease frequency) were extracted.
2. Report per-criterion Cohen’s Kappa values between clinician and LLM judgments, as well as inter-clinician Kappas.
Include confusion matrices for each quality dimension on the clinician-labeled samples.
3. Why was exact string matching chosen instead of concept-level or semantic matching (e.g., UMLS normalization or fuzzy embedding similarity)?
4. Provide comparative results using a semantic approach or at least an analysis quantifying the under-count due to exact matching.
5. Supply (a) the distributions of age, gender, and disease before and after each filtering stage, and (b) an analysis of how the ≥ 500-record disease cutoff affects downstream generalization to less-common conditions.
6. Provide paired statistical significance tests comparing LLM-CARe with each baseline on primary metrics. Include per-disease performance tables or aggregates grouped by disease frequency.
7. Consider evaluating with an independent LLM judge (from a different vendor), expanding clinician evaluation beyond 100 EMRs, and including edge or rare cases.
8. Provide subgroup analyses (e.g., by gender, age bracket, or ethnicity if available) for both EMR quality metrics and downstream task outcomes.

### Soundness
3

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
The paper presents LLM-CARe, a large-language-model framework for generating synthetic electronic medical records (EMRs) without using real patient data. It employs three agents—a Critic, Adviser, and Reviser—that work in a cyclic refinement loop to iteratively improve data quality. The process operates in three stages: corpus-level alignment with real-world demographic and diagnostic distributions, section-level completion of clinical fields, and document-level consistency of medical logic. Experiments on a large de-identified EMR dataset show that LLM-CARe surpasses traditional GAN- and LLM-based baselines such as MedSyn in both intrinsic quality metrics and downstream predictive tasks, including diagnosis, examination, and treatment recommendation

### Strengths
S1. The authors build an appealing three-agent loop (Critic – Adviser – Reviser) and wrap it in a staged pipeline.

S2. The authors perform extensive evaluations, including both intrinsic “LLM-as-a-judge” metrics and downstream predictive tasks, and even conduct a clinician study to validate alignment between automatic and human assessments. 

S3. The experiments are thorough and cover several baselines across different LLM backbones, providing a thorough empirical picture

### Weaknesses
W1. The claimed “clinical quality principles” are human-defined checklists, not learned constraints; therefore the method is rule-driven text refinement, not genuine reasoning or data synthesis.

W2. The multi-agent design seems overcomplicated for its marginal gains, with no comparison to a single self-refining LLM.

W3. The approach relies on static, non-temporal EMR data, which limits realism and generalizability to real clinical settings.

W4. The definition of “content completeness” assumes that every section should be fully filled, which may contradict real-world medical documentation patterns where incompleteness reflects diagnostic uncertainty or missing data. Have the authors tested whether enforcing strict completeness might produce clinically implausible or overly templated records?

### Questions
Q1. Why three separate agents? Could a single unified agent with self-critique and iterative prompting achieve similar results with lower computational cost?

Q2. Why does the proposed EMR quality not align with downstream task performance. For instance, why does MedSyn, which scores poorly on section-level quality metrics, still achieve competitive results in downstream prediction tasks?

Q3. The document-stage refinement shows limited incremental benefit in Figure 6, raising concerns about its necessity and efficiency. Why is the document-level stage retained when its contribution appears minimal?

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
3

### Summary
This paper proposes LLM-CARe, a  Critic–Adviser–Reviser cyclic refinement framework for synthetic EMR generation. The method refines drafts progressively at corpus, section, and document levels. Each level targets specific quality principles such as distributional alignment, completeness, and consistency. Experiments show that LLM-CARe outperforms baseline GAN- and LLM-based EMR generators on both intrinsic quality metrics (using LLM-as-judge) and downstream clinical tasks (diagnosis/exam/treatment prediction). The paper emphasizes that its method requires no real EMR text, thereby ensuring privacy.

### Strengths
* The motivation is clear and well grounded in real-world needs. The paper is clearly written and polished throughout.

* The Critic–Adviser–Reviser framework is intuitive and well explained. The three-stage design (corpus, section, document) aligns nicely with EMR structure and gives the approach a coherent logic.

* Experiments are extensive and consistently favorable. The method improves both intrinsic quality metrics and downstream task accuracy, with convincing ablations and backbone comparisons.

* The inclusion of a clinician study strengthens credibility.

### Weaknesses
* The novelty is limited. The multi-agent cyclic refinement closely follows prior self-reflection or debate-style frameworks (e.g. Self-Refine [1]). The contribution is mostly in domain adaptation rather than algorithmic innovation.

* The major intrinsic metrics rely on LLM-based judgments (Qwen-32B), which risks circularity—since the same model family is used for generation and evaluation. Although clinician validation is reported, the sample size (n = 100) is small and does not fully calibrate the quantitative scores in Table 1.

* No comparison to recent diffusion-based EMR generators (e.g., EHRDiff [2])

* No discussion of computational cost or typical failure patterns is provided, leaving scalability and robustness uncertain.

-------
References

[1] Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Clark, P. (2023). Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 46534-46594.

[2] Yuan, H., Zhou, S., & Yu, S. EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models. Transactions on Machine Learning Research.

### Questions
* How many iterations per stage were run, and what is the stopping criterion?

* Does the system risk “mode collapse,” where cyclic feedback converges to repetitive templates?

### Soundness
3

### Presentation
3

### Contribution
2
