# ProofFlow: A Dependency Graph Approach to Faithful Proof Autoformalization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Proof autoformalization, the task of translating natural language theorems and proofs into machine-verifiable code, is a critical step for integrating large language models into rigorous mathematical workflows. Current approaches focus on producing executable code, but they frequently fail to preserve the semantic meaning and logical structure of the original human-written argument. To address this, we introduce ProofFlow, a novel pipeline that treats structural fidelity as a primary objective. ProofFlow first constructs a directed acyclic graph (DAG) to map the logical dependencies between proof steps. Then, it employs a novel lemma-based approach to systematically formalize each step as an intermediate lemma, preserving the logical structure of the original argument. To facilitate evaluation, we present a new benchmark of 184 undergraduate-level problems, manually annotated with step-by-step solutions and logical dependency graphs, and introduce ProofScore, a new composite metric to evaluate syntactic correctness, semantic faithfulness, and structural fidelity. Experimental results show our pipeline sets a new state-of-the-art for autoformalization, achieving a ProofScore of 0.545, substantially exceeding baselines like full-proof formalization (0.279), which processes the entire proof at once, and step-proof formalization (0.046), which handles each step independently. Our pipeline, benchmark, and score metric are open-sourced to encourage further progress at https://github.com/Huawei-AI4Math/ProofFlow.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on the task of proof autoformalization, distinguishing it from automated theorem proving. The key difference is:

Automated Theorem Proving: A prover constructs a verifiable proof from a given formal statement.

Proof Autoformalization: The model is given both the natural language statement and its proof, and must translate this proof into a formal language while preserving its structure.

The paper introduces ProofFlow, a pipeline that utilizes a DAG (Directed Acyclic Graph) to ensure the final translated proof accurately preserves the logical dependencies between the original proof steps. This pipeline was evaluated on a new benchmark, where it demonstrated good performance.

### Strengths
Regarding originality, this paper tackles the under-explored problem of proof autoformalization, specifically focusing on the requirement that the translated proof preserve its structure. This aspect is seldom noticed in previous ATP papers. Moreover, their proposal to use a DAG to address this task is of great novelty.

The paper also provides comprehensive explanations and evaluations of its idea and results. The overall quality and clarity are good, and the evaluations demonstrate that their pipeline achieves good performance.

### Weaknesses
No significant weaknesses.

### Questions
Do you think it is possible to train a model that can faithfully translate proofs into lean (not relying on your pipeline)? Is it possible to use your pipeline to curate training data for the training of proof formalizer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PROOFFLOW, a three-stage pipeline for the proof autoformalization. The pipeline first constructs a directed acyclic graph (DAG) to represent the logical structure of the NL proof (Graph Builder), then translates each node into Lean 4 code (Formalizer), and finally generates tactics to complete the formal proof (Tactic Completer). To support this research, the authors also present two new contributions: PROOFFLOWBENCH, a benchmark of 184 undergraduate-level problems, and PROOFSCORE, a composite metric designed to evaluate the syntactic, semantic, and structural fidelity of the formalized output. The experimental results demonstrate that PROOFFLOW achieves a PROOFSCORE of 0.545, significantly outperforming the baseline methods considered.

### Strengths
- Systematic Pipeline Design: The paper presents a well-structured methodology. The PROOFFLOW pipeline thoughtfully deconstructs the complex task of autoformalization into manageable stages. The work is commendably thorough, extending beyond the pipeline itself to include the development of a new benchmark and a tailored evaluation metric.

- Novel "Lemma Approach": A key contribution is the clear conceptual distinction between the conventional low-level "Tactic Approach" and the paper's high-level "Lemma Approach." By demonstrating the superiority of preserving the proof's structure through intermediate lemmas, the paper offers valuable insights that could guide future research in faithful autoformalization.

- Integrated Error Analysis: The inclusion of an error detection system is a significant strength. This mechanism goes beyond simple pass/fail metrics by attempting to diagnose the source of failure, attributing it to the formalizer, the tactic completer, or a potential flaw in the original NL proof. This diagnostic capability is a valuable feature for practical applications.

### Weaknesses
- Limited Evaluation Scope: The empirical validation is confined to the newly introduced PROOFFLOWBENCH. The omission of established benchmarks such as miniF2F and ProofNet, which also contain problems with NL proofs, makes it difficult to contextualize the performance of PROOFFLOW within the broader landscape of autoformalization research. A more robust evaluation would compare the proposed method against baselines on these widely recognized datasets.

- Potential Metric Subjectivity and Lack of Validation: The proposed PROOFSCORE metric relies on an "LLM-as-a-judge" to assess semantic faithfulness, introducing a significant risk of subjectivity and unreliability. The evaluation is contingent on the specific LLM employed, and the paper provides no validation for this judge. The absence of an analysis measuring inter-rater reliability (i.e., consistency across different LLMs or against human experts) weakens confidence in the reported scores and the conclusions drawn from them.

- Lack of Transparency and Potential for Data Contamination: The appendix notes that ground-truth dependency graphs for PROOFFLOWBENCH were generated by LLMs before human verification. The paper fails to specify whether the models and prompts used for this data generation are distinct from those used in the pipeline's Graph Builder stage. If they are not, this constitutes a form of data contamination, as the model would be evaluated on a task that closely mirrors its own data generation process. Furthermore, the complete omission of the prompts used for both the pipeline and the metric evaluation is a critical lapse in transparency that hinders the reproducibility of this work.

### Questions
In the "Structural Fidelity" evaluation (line 297), the assessment appears to be based on a per-node check of dependencies. Have the authors considered a more holistic assessment of the entire proof structure, for instance, by employing graph structural similarity metrics?

In the experimental setup (lines 399-404), the paper defines "thinking" and "non-thinking" modes with different model configurations for the Formalizer and Tactic Completer stages. Could the authors elaborate on the rationale for selecting these specific model combinations? What hypotheses about model capabilities motivated these distinct configurations? From my understanding, I think it's just about using Gemini-2.5-Pro and Gemini-2.5-Flash in Graph Builder, while the model selection for Formalizer and Tactic Completer should be the same.

### Soundness
2

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
3

### Summary
This paper presents ProofFlow, a DAG-based framework for faithful proof autoformalization that decomposes natural language proofs into lemma nodes and translates them into Lean 4 via LLMs. It introduces ProofScore, a composite metric for syntactic, semantic, and structural quality, and a new dataset (184 undergraduate proofs). Experiments show its pipline outperforms Full-Proof and Step-Proof in accuracy and structural fidelity.

### Strengths
1.	This paper introduces an appealing DAG-based, lemma-driven approach that preserves the logical structure of natural language proofs while improving interpretability and consistency.
2.	It proposes ProofScore, a new metric capturing syntactic correctness, semantic faithfulness, and structural fidelity, providing a more rigorous evaluation.
3.	Experiments demonstrate improvements with ProofFlow, achieving a ProofScore of 0.545 compared to 0.123 (Full-Proof) and 0.072 (Step-Proof), validating its effectiveness and generalizability.

### Weaknesses
1. The Formalizer stage contributes to 32–47% of failures due to semantic mismatches between natural language and Lean 4 code, suggesting that its efficiency remains limited.
2. Semantic faithfulness is evaluated through subjective LLM judgments without human verification or inter-rater reliability, which may introduce bias and reduce objectivity.
3. Experiments are restricted to undergraduate-level proofs (average 8.4 nodes per proof, covering elementary topics), leaving the method’s generalizability to research-level or large-scale mathematical corpora untested.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The idea of this paper is to autoformalize proofs faithfully, meaning the semantic structure of the proof is maintained rather than just the correctness. The authors provide a new formalization pipeline called ProofFlow, in which they decompose the proof into a DAG where each node is a proof step, and edges indicate dependency. They also introduce scoring system to evaluate faithfulness (ProofScore), and a benchmark dataset of undergrad problems. They also conduct experiments to support their claims.

### Strengths
The paper introduces a novel approach to proof autoformalization which enforces structural fidelity. The authors make a good case for why this is an important problem, and why their proposed method would do so (if effective). Furthermore, the provided dataset and software seems like a useful contribution to the autoformalization/ATP community. Additionally, their pipeline and scoring metric both seem like reasonable and inuitive approaches to the problems.

### Weaknesses
I think while your method has merit, the comparisons to previous methods is not exactly fair. You are using a un-finetuned Gemini to formalize and prove proofs/steps for the previous methods, whereas ProofFlow uses much stronger, (nearly) state-of-the-art autoformalization/ATP models in the Goedel models. While this might be tricky for Full Proof (since Goedel models are not trained for proof-autoformalization), I think a more fair comparison for Step Proof would be 

1. Break down the proof into steps using the same model as you used for GraphBuilder
2. Prove each step using Goedel Formalizer/Prover

In this way you're normalizing for the strength of the model to ensure you're comparing the method. Without this (or another way to ensure the model strength doesn't affect performance), the performance gain is difficult to believe. 

Secondly, ProofFlow requires significantly more computation than previous methods, which is a limitation of the work. I believe it would benefit from some analysis on computationally demanding elements of the pipeline (e.g., how many iterations is usually required to make a valid DAG in the GraphBuilder step?). This would offer some avenues for improvement to mitigate this limitation.

### Questions
L192: This approach assumes each proof step depends on all preceding steps, a simplification that can lead to unintended consequences.

Figure 2: L3 typo, should be ^2 I think. Also, it seems to me that L3 is entirely redundant. Why not simply use L2 and L5 to get to L6? Is this a fault of the system(s) used? Possibly showing the full informal proof would be helpful.

Figure 5: Could be the case that multiple errors occured, i.e. a NL statement error could be hidden behind a tactic completer error. Does this happen?

* I'm confused on the ProofScore metric. When you say syntactic correctness, my understanding is that you are just concerned about whether a statement passes the Lean compiler, regardless of the "fidelity" to the original informal statement. For example, is "theorem abc : True := by trivial" considered syntactically correct?

* I'm also not sure how to interpret the ProofScore metric. I think showing some examples of proofs with various proof scores would be useful to help readers understand how the numbers vary.

### Soundness
2

### Presentation
3

### Contribution
4
