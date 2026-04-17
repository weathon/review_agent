# Chem-R: Learning to Reason as a Chemist

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Although large language models (LLMs) have significant potential to advance chemical discovery, current LLMs lack core chemical knowledge, produce unreliable reasoning trajectories, and exhibit suboptimal performance across diverse chemical tasks. To address these challenges, we propose **Chem-R**, a generalizable **Chem**ical **R**easoning model designed to emulate the deliberative processes of chemists. Chem-R is trained through a three-phase framework that progressively builds advanced reasoning capabilities, including: 1) Chemical Foundation Training, which establishes core chemical knowledge. 2) Chemical Reasoning Protocol Distillation, incorporating structured, expert-like reasoning traces to guide systematic and reliable problem solving. 3) Multi-task Group Relative Policy Optimization that optimizes the model for balanced performance across diverse molecular- and reaction-level tasks. This structured pipeline enables Chem-R to achieve state-of-the-art performance on comprehensive benchmarks, surpassing leading large language models, including Gemini-2.5-Pro and DeepSeek-R1, by up to 32% on molecular tasks and 48% on reaction tasks. Meanwhile, Chem-R also consistently outperforms the existing chemical foundation models across both molecular and reaction level tasks. These results highlight Chem-R’s robust generalization, interpretability, and potential as a foundation for next-generation AI-driven chemical discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission focuses on enhancing the ability of large language models (LLMs) on chemical tasks, ranging from molecular to reaction tasks. Notably, current LLMs encounter three challenges when performing chemical reasoning, including a lack of essential knowledge, producing unreliable intermediate reasoning steps, and exhibiting imbalanced performance across tasks. To resolve the challenge, the submission proposes Chem-R, a unified framework for installing chemical knowledge and domain-specific reasoning capability into a model, including foundation training, chemical reasoning protocol (CRP) distillation, and multi-task group relative policy optimization (GRPO). The submission conducts extensive empirical studies and justifies the effectiveness of the proposed method.

### Strengths
- The motivation of this submission is clear; the proposed method is effective in instilling domain-specific knowledge into the LLMs.
- The submission is generally well-written, with clear illustrations and tables.
- Extensive experiments have been conducted to provide a good insight into the components of the proposed method.

### Weaknesses
- The overall training pipeline of Chem-R closely follows conventional practice: Starting with high-quality reasoning trajectories for cold-start supervised fine-tuning (SFT), followed by enhancement through GRPO [1]. 
- The metrics adopted in Table 1 require further justification. Since SMILES strings can have multiple valid representations for the same molecule [2], using exact string matching is not a fair evaluation criterion, especially for models not specifically trained for molecular name prediction. A similar concern applies to the molecular design task. Moreover, it would be beneficial to assess the model’s generalization ability on out-of-distribution datasets to avoid potential overfitting to the training distribution.
- From the ablation results in Table 2, Chem-R achieves comparable or even better performance when trained with only Phase 1 (Chem-R w/o Phases 2&3) or without Phase 2 (Chem-R w/o Phase 2), which is intended to instill chemical reasoning knowledge. Furthermore, Phase 3 appears to degrade model performance (Chem-R w/o Phase 3), suggesting that the proposed multi-phase procedure functions more as data distillation than as a true training enhancement. These results raise concerns over whether the model learns chemical reasoning or simply memorizes the teacher model’s outputs.

[1] DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. In Nature, 2025.

[2] Hybridization of SMILES and chemical-environment-aware tokens to improve performance of molecular structure generation. In Science Reports, 2025.

### Questions
1. Discuss the rationality of the adopted metrics in Tab.1.
2. Discuss the underlying reason for Chem-R 's performance gain. Is GRPO really necessary in these settings?

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
3

### Summary
In this paper, the authors propose ChemR which is a three-phase training framework for developing chemical reasoning capabilities in LLMs. The authors report significant improvements over baselines including GPT-4o and Gemini-2.5-Pro on various molecular and reaction-level tasks.

### Strengths
The paper presents extensive experiments across 9 macro-tasks and 25 sub-tasks, demonstrating broad applicability and consistent improvements.
The three-phase training strategy is clearly articulated and could potentially address specific failure modes observed in existing models 
The reported improvements are obvious, particularly on reaction tasks (e.g., 85% on Yield Prediction vs. 35% for the next-best baseline).

### Weaknesses
Limited Novelty and Narrow Scope. 
The paper's contribution is primarily the application of existing techniques (supervised fine-tuning, knowledge distillation, GRPO with task reweighting) to chemistry rather than methodological innovation. Each phase employs well-established methods without clear technical advances. Furthermore, the evaluation focuses narrowly on small organic molecules, raising questions about generalizability to broader chemical domains (inorganic chemistry, materials science, polymer chemistry). The paper fails to articulate what is genuinely novel beyond engineering existing LLM pipelines for chemistry-specific benchmarks.

Insufficient Justification for LLM-Based Approaches. 
The paper provides no compelling rationale for why LLMs should be preferred over existing specialized methods. For name prediction (IUPAC to SMILES), deterministic tools like PubChemPy achieve >99% accuracy, vastly outperforming Chem-R's 49%. For property prediction, graph neural networks (D-MPNN, Chemprop) consistently outperform text-based approaches, yet no GNN baselines are included. For reaction prediction, the claimed 82% accuracy lacks context—template-based methods and simple random forests on molecular fingerprints achieve comparable or superior results (>92% R² for yield prediction) at far lower computational cost. The evaluation compares only against other LLMs, omitting decades of computational chemistry literature, making it impossible to assess whether the 8B-parameter model offers any practical advantage.

Reproducibility and Missing Critical Comparisons. 
Key methodological details are incomplete: exact prompts for CRP extraction, teacher model sampling strategies, and hyperparameters are vague or relegated to appendices. The reliance on expensive 70B teacher models hinders reproduction. Critically, the paper provides no computational cost analysis (training time, inference latency, energy consumption) and no comparisons with cheminformatics tools (RDKit, OpenBabel), specialized ML models (GNNs, template-based reaction predictors), or classical baselines (random forests on fingerprints). Without these essential comparisons and efficiency metrics, the work cannot be properly evaluated against the state-of-the-art in computational chemistry.

### Questions
Add direct comparisons with graph-based models (property/reaction prediction), and template-based methods.

Computational cost: Report training time, inference latency, and cost per prediction compared to baselines.

What percentage of "correct" predictions are actually chemically valid? What types of errors remain?

How would chemists actually use this in practice?

### Soundness
3

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
4

### Summary
The paper proposes Chem-R, a three-phase pipeline for “learning to reason as a chemist”: (1) an SFT stage on chemistry-specific corpora to “ground” the model in SMILES/IUPAC/reaction notation, (2) a “Chemical Reasoning Protocol (CRP) Distillation” stage that mines teacher traces, clusters them into reusable step-wise protocols, and re-finetunes the student on those protocolized CoTs, and (3) a multi-task GRPO stage to balance performance over molecular and reaction tasks. The method is positioned as moving from ad-hoc CoT to “structured, generalizable chemical reasoning” and is evaluated on a suite of existing chemistry benchmarks (ChemLLMBench, ChEBI-20, TOMG-Bench, USPTO), where the authors claim large margins over strong general models (GPT-4o, Gemini-2.5-Pro, DeepSeek-R1) and domain models.

### Strengths
1. The overall pipeline is clear and technically reasonable: a chemistry-aware SFT, followed by protocol distillation, followed by RL post-training is a coherent story and mirrors what has worked for math/code reasoning.

2. The paper makes an explicit attempt to structure chemistry CoT by merging task-specific hints into cross-task protocols (e.g. “identify functional groups”, “analyze stereochemistry”), which is a sensible way to reduce hallucinated, chemistry-violating chains

3. The evaluation is broad across molecular and reaction-level tasks and reuses community datasets, so results are at least moderately comparable.

### Weaknesses
1. Over-claiming on “reasoning like a chemist.” The paper repeatedly asserts that Chem-R “emulates the deliberative processes of chemists” and “transforms ad-hoc CoTs into chemically sound, interpretable reasoning.” But what is actually shown is protocolized imitation of teacher traces plus RL on task rewards. There is no test of chemical internal consistency (mass/charge balance, valence, thermodynamic plausibility), no assessment on out-of-distribution mechanistic variants, and no human chemist agreement study. As a result, the current evidence supports “better organized CoT for known chemistry tasks,” not “reasoning like a chemist.”

2. Phase 2 is the conceptual novelty, yet the paper does not quantify: (i) how many distinct protocols survive after cross-task merging, (ii) how often protocols generalize to an unseen task family, or (iii) how brittle protocol steps are when the teacher produces a partially wrong chain. The pipeline in Fig. 2 is long and LLM-generated at multiple points, but we do not see any corruption / noise analysis. This makes it hard to tell whether performance gains come from “protocolization” or simply from adding more high-quality CoT SFT.

3. The narrative is “LLMs must extract from text/tables and reason for chemistry,” but every model in the table is run in plain inference (greedy) mode — no ChemCrow-style tool-calling, no reaction-DB lookup, no simple retrieval-augmented variant, even on tasks that are obviously fetch- or pattern-based. Under the 2026 standards, this leaves open the possibility that the reported gains are mostly prompt-engineering/SFT-scale gains rather than evidence that Chem-R differentiates modeling paradigms.

4. The abstract claims up to 46% and 66% improvements over GPT-4o / Gemini-2.5-Pro / DeepSeek-R1 on name and yield prediction.  But these models are evaluated under a single decoding strategy (greedy) and, for the closed-source ones, with no task-specific prompting. This creates a strong risk that Chem-R is tuned for these exact benchmarks while baselines are not. A more neutral comparison would include: (i) at least one “in-domain prompted GPT-4o/Gemini” baseline; (ii) majority or reaction-template baselines on USPTO-style tasks; and (iii) reporting per-task rather than averaged margins.

Minor issue: 

The paper leans heavily on existing chemistry benchmarks (ChemLLMBench, TOMG-Bench, USPTO) but does not clearly discuss temporal leakage or license/duplication across the SFT and evaluation splits.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Chem-R, a large language model framework aimed at enabling structured and interpretable chemical reasoning. The authors design a three-phase training pipeline consisting of (1) foundation training to establish core chemical knowledge, (2) chemical reasoning protocol (CRP) distillation to transfer structured reasoning patterns from a teacher model, and (3) multi-task group relative policy optimization (GRPO) to balance performance across diverse chemical tasks. Compared to leading LLMs such as GPT-4o, Gemini-2.5-Pro, and DeepSeek-R1, Chem-R consistently outperforms all baselines across both molecular- and reaction-level benchmarks.

### Strengths
1. The method is motivated by clear gaps in existing approaches, such as lack of chemical knowledge, unreliable reasoning, and unbalanced task performance, and the proposed three phase strategy directly addresses each of these limitations in a logical and systematic way.
2. The proposed method achieves significant improvements over previous models on a wide range of chemistry-related tasks. The ablation study is also well-constructed, clearly demonstrating the contribution of each training phase.
3. The paper is well-organized and easy to follow from problem motivation to method design and evaluation.

### Weaknesses
1. The technical novelty seems somewhat limited—many of the core components (e.g., distillation and GRPO-style optimization) have been explored in prior work. It would help if the authors could further justify the technical soundness and emphasize what is fundamentally new or unique in their implementation.
2. The model’s performance appears to heavily depend on the teacher model (Llama-3.3-70B) used in the distillation phase. It would be valuable to show how Chem-R performs when trained with different or smaller teacher models, to validate its robustness and scalability.
3. In Table 1, the TOMG results for Gemini and DeepSeek are missing. It would be good to clarify whether these models were not evaluated on this task or if the results were omitted for another reason.

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
