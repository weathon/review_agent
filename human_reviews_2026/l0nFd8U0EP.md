# Coder as Editor: Code‑driven Interpretable Molecular Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Molecular optimization is a central task in drug discovery that requires precise structural reasoning and domain knowledge. While large language models (LLMs) have shown promise in generating high-level editing intentions in natural language, they often struggle to faithfully execute these modifications—particularly when operating on non-intuitive representations like SMILES. We introduce MECo, a framework that bridges reasoning and execution by translating editing actions into executable code. MECo reformulates molecular optimization for LLMs as a cascaded framework: generating human-interpretable editing intentions from a molecule and property goal, followed by translating those intentions into executable structural edits via code generation. Our approach achieves over 98\% accuracy in reproducing held-out realistic edits derived from chemical reactions and target-specific compound pairs. On downstream optimization benchmarks spanning physicochemical properties and target activities, MECo substantially improves consistency by 38-86 percentage points to 90\%+ and achieves higher success rates over SMILES-based baselines while preserving structural similarity. By aligning intention with execution, MECo enables consistent, controllable and interpretable molecular design, laying the foundation for high-fidelity feedback loops and collaborative human–AI workflows in drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MECo (Coder as Editor), a framework that reformulates molecular optimization as a code generation task. Instead of directly producing SMILES strings, it uses a reasoning LLM to propose interpretable editing intentions, and a coder LLM to translate them into executable RDKit scripts. Experiments on both synthetic and real benchmarks show that MECo significantly improves success and consistency rates across six property and activity optimization tasks. The approach demonstrates a promising paradigm for making LLM-based molecular design verifiable and reproducible rather than purely generative.

### Strengths
1. The code-based interface makes every molecular edit explicit, traceable, and executable, improving transparency and human understanding.

2. The framework consistently outperforms SMILES-based baselines across six property and activity optimization tasks.

### Weaknesses
## Motivation needs to be clarified with more evidence

1. In Abstract: The statement "LLMs have shown promise in generating high-level editing intentions in natural language" lacks supporting evidence. Is there any data in the paper or related work demonstrating this claim? Are there quantitative results (e.g., percentage, accuracy, or error rate) showing that the generated intentions (such as functional groups) are useful for a downstream task?

2. In Line 73, the authors skip a more straightforward approach, i.e., supervised fine-tuning (SFT). Why not simply construct molecular pairs before and after editing to evaluate the LLM’s performance in optimization? Another straightforward idea is to combine deterministic algorithms (such as MOLLEO, which the authors also discuss) or to apply multiple sampling with LLMs, which would be computationally inexpensive since each SMILES string is relatively short. If such simple methods were not used, an analysis explaining why and a comparison with them as baselines are needed.

## Details on the method are missing

1. The coder LLM edits molecules based on a set of predefined SMARTS patterns. However, the coverage and diversity of these patterns are unclear. A comprehensive analysis is needed to support the claimed generalization ability of the method.

2. In the coding process, one molecule may contain multiple substructures that match a given SMARTS pattern. How does the method decide which one to modify, or are all matching substructures replaced simultaneously? If the coder LLM writes the script and makes decisions, how do we know that the coder LLM strictly follows the instructions provided by the reasoning LLM?

## Related work and baseline

The work overlooks many previous efforts on inverse design prior to the use of LLMs and includes no non-LLM baselines. It is unclear why these earlier models are omitted, especially since there is no evidence showing that LLM-based molecular design outperforms non-LLM methods.

### Questions
1. All baselines are SMILES-based LLMs. Molecular generation constrained by structure has also been studied using graph-based models [1,2,3,4]. The work needs a more comprehensive comparison with these baselines. 

2. The actions in the “reasoning” component are limited to predefined edit rules based on SMARTS. What if no suitable pattern from the predefined SMARTS rules exists for a given task?

3. What does "Qwen2.5-7B-Instruct-FT" mean in Table 1?

4. Why are there many "-" in Table 2?

5. Is there any complete case for the generation result of reasoning/coder LLMs?

Reference:
[1] Hierarchical Generation of Molecular Graphs using Structural Motifs. ICML 2020.
[2] Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018.
[3] Multi-Objective Molecule Generation using Interpretable Substructures. ICML 2020.
[4] DiGress: Discrete Denoising diffusion for graph generation. ICLR 2023.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MECo, a framework that reformulates molecular optimization as a code-generation problem. Instead of directly generating SMILES strings, a reasoning LLM first produces high-level editing intentions, and a separate coder LLM translates these intentions into executable scripts that perform the structural edits.

### Strengths
1. Reformulating molecular editing as a code generation problem is an original and practical idea.
2. Each generated edit is represented as an explicit Python/RDKit script, which allows human chemists to read, check, and reproduce the exact transformation.
3. The model demonstrates that LLMs can indeed produce executable and syntactically correct chemical code.

### Weaknesses
1. The framework is only compared with general-purpose LLMs, not with specialized molecular optimization models, which makes its competitive value for drug discovery unclear.
2. The system appears to follow a linear pipeline: the reasoning LLM generates an edit, and then the executor applies it. However, this design underutilizes the LLM’s reflection ability. In the future, the authors could consider allowing the reasoning LLM to reflect on the executor’s results and perform multi-round optimizations.

### Questions
see weakness

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
5

### Summary
The paper proposes MECo (Molecular Editing via Code generation), a cascaded framework for molecular optimization: a reasoning LLM generates human-interpretable editing actions, and a coder LLM translates these actions into executable RDKit code that edits the input molecule. A synthetic training set of moiety/linker substitutions is used to fine-tune a code model; evaluation uses three “realistic” edit sets (reaction-derived, target-specific terminal, target-specific core) and ChemCoTBench optimization tasks. MECo reports ~98% edit execution accuracy on realistic edits and higher success/consistency on downstream optimization versus direct SMILES generation by LLMs.

### Strengths
- Clear problem decomposition. Separating “why edit” (reasoning) from “how to edit” (code execution) addresses a known brittleness of SMILES-level generation.
- Strong execution fidelity. The code LLM fine-tuned on synthetic edits achieves >=98% accuracy on all three realistic edit categories, a large margin over direct SMILES generation and zero-shot code prompting baselines. 
- Meaningful end-task gains. On ChemCoTBench, MECo improves success rate and similarity while reaching ~90–98% structure–action consistency, which is important for traceable design. 
- Reproducible ingredients. RDKit-based edits, explicit prompt wrapper, and synthetic data construction details improve reusability.
- Case studies with mechanistic narrative. The examples show plausible edits that retain cores and avoid score hacking behaviors seen in direct generation.

### Weaknesses
- Reliance on synthetic-to-real transfer without ablations. The model is trained only on synthetic replacements yet attains ~98% on realistic edits; stronger leakage checks and ablations (e.g., reducing pattern overlap, altering SMARTS syntax, training set size scaling) are not reported. Filtering based on Tanimoto >= 0.6 to training moieties may be insufficient to exclude near-variants of patterns. 
- Manual consistency scoring introduces subjectivity. Structure–action consistency depends on a manual protocol that accepts “discernible” but syntactically imperfect actions. No inter-rater agreement, adjudication process, or blinded evaluation is reported.
- Baseline breadth and parity. Comparisons focus on direct LLM generation and simple code prompting. Strong graph-editing or reaction-transform baselines (e.g., JT-VAE/graph AF variants configured for localized edits, template-guided transformations) are not evaluated.
- Metric choices can favor conservative edits. Emphasis on success rate and similarity, with mean property improvement deferred to the appendix, may bias toward small edits. A multi-objective view (e.g., Pareto fronts of improvement vs. similarity) is not shown.

### Questions
- How robust is MECo to stereochemistry and formal charge preservation? Please report chirality retention rates and valence error checks after edits.
- Does ChemicalReaction editing ever produce multiple products or atom-mapping ambiguities? How are products filtered and how often?
- What is the impact of training set size and pattern diversity on execution accuracy? Please add scaling-law style ablations.

### Soundness
3

### Presentation
4

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
This paper introduces MECo, a framework for interpretable molecular optimization that leverages large language models (LLMs) to generate high-level molecular editing intentions in natural language, which are then translated into executable code for structural modification of molecules. MECo decouples chemical reasoning (intent formation) from execution (precise code-driven edits), utilizing a code LLM (Qwen2.5-Coder) to translate rationales into robust RDKit-based editing scripts. The authors present a systematic synthetic and realistic data construction pipeline for training and evaluation. MECo demonstrates strong accuracy on edit execution benchmarks and achieves higher success, similarity, and intention-consistency rates than SMILES-based generation and previous LLM approaches on molecular optimization tasks.

### Strengths
1. The MECo framework proposes a well-motivated separation between intent (reasoning LLM) and edit execution (code LLM), resulting in interpretable and robust molecular modifications. This mirrors expert workflows and addresses a long-standing bottleneck in LLM-driven molecule design (Section 1, Figure 1).

2. The evaluation is comprehensive, involving both synthetic and realistic molecular edits, with clear and challenging test splits (see Section 3.3, Figure 2B). Filtering out test samples with high similarity to training moieties ensures a genuine assessment of model generalization.

3. MECo achieves impressive execution accuracy ($98%+$) on realistic edits (Table 1) and consistently higher performance (success rate, similarity, and edit-consistency) across multiple property and bioactivity optimization tasks (Table 2, Figure 3).

4. The code-based approach produces auditable and modifiable editing scripts, greatly aiding transparency and reproducibility. Figure 4B demonstrates how intention, code, and edit outcomes remain aligned in complex cases.
Careful Metrics & Human Evaluation: The authors avoid misleading optimization metrics (see discussion in Section 3.5, Appendix C.4) by focusing on success rates and consistency, and they clearly describe robust manual criteria for edit intention realization (Appendix C.2, Figure 5).

### Weaknesses
1. Several highly relevant recent works on code-driven molecular optimization and code LLM reasoning, such as Yu et al. (2025) (Collaborative Expert LLMs Guided Multi-Objective Molecular Optimization), are not cited or compared. These works also address LLM-guided optimization with structured intermediates and could expose theoretical or empirical overlaps.
Influential studies on LLM code analysis (Fang et al., 2024), and on splitting code generation between LLMs and formal program synthesis (Murphy et al., 2024), are omitted from the related work and could further contextualize the challenges and limits of code-driven approaches in molecular tasks (see Section 2, missing from the discussion).

2. While Table 2 covers a range of LLM-based approaches, the comparison almost exclusively focuses on direct SMILES generation or "reasoning LLM + direct execution" without including strong recent graph-based or programmatic editing methods as empirical baselines (e.g., advanced molecular graph editing, or hybrid neuro-symbolic models).
No dedicated ablation of the prompt format choices or an investigation into how prompting and code granularity impacts robustness and execution success, especially in edge or ambiguous editing scenarios (prompting decisions are described in Section B.1, but not experimentally dissected).

3. The reliance on DeepSeek-R1 (and to some extent Gemini-2.5-Pro) as the intention LLM leaves unclear how the overall system behaves under less capable or non-scientific LLMs, and whether the overall gains are primarily due to the code LLM, reasoning LLM, or their synergy (Section 4.3, Table 2). There is some analysis of improvements across reasoning LLMs (Figure 3), but a more detailed breakdown would strengthen claims of generality.

4. For multi-site (core) replacements and edits involving complex attachment point mapping, the formal definition of permutation handling (Appendix A.4) is described, but the main paper does not fully explore the algorithmic consequences (e.g., how ambiguities in fragment symmetry or atom mapping are resolved in executable code). This could affect robustness in less-structured or fuzzy design scenarios.

5. Human-in-the-loop evaluations (see Section 4.3, Appendix C.2 & Figure 5) are clear and motivated, but reliance on manual consistency checks for intention realization raises concerns of scalability, subjectivity, and reproducibility on very large or diverse benchmarks.

6. While the formalism for code generation is largely clear (Section 3.3, Algorithm 1), the mechanism for fragment identification, attachment point matching, and the translation of ambiguous instructions into unambiguous, executable RDKit code is not fully formalized. For example, in iterative moiety replacement (Algorithm 1), how does the model (or supporting code) resolve cases where multiple matching substructures are present, or when multiple attachment permutations are valid? These details are nontrivial and could be potential failure modes.

7. Furthermore, in Table 1, some discrepancies between reaction-derived and target-specific replacements are discussed (Appendix C.3), but specific SMILES syntax challenges (such as dummy atom placement) and code-generation edge-cases are not quantitatively analyzed in detail.

8. Although the authors claim strong generalization from synthetic to realistic edits, the training data’s moiety/replacement set is well-defined (Table 3 and 4), and it remains an open question whether broader, less formulaic edits—such as those arising from less constrained drug design—can be captured without an explosion of training diversity.

### Questions
1. In cases where the same edit action (natural language) could map to multiple code realizations (e.g., due to substructure symmetry or ambiguous atom mapping), how does MECo determine which to prioritize, and what are the observed failure rates as a function of edit ambiguity? Explicit discussion of error distributions and representative code translation mistakes would be welcome.
2. Can the manual edit-intention consistency checks be reliably scaled to order-of-magnitude larger or more diverse test sets, or is there a feasible path toward automating this evaluation in a way that preserves interpretability and trust?
3. What is the impact on MECo’s downstream success and consistency if a less performant or non-scientifically trained reasoning LLM is used? Have the authors observed any bottlenecks or cascading failures rooted in the upstream intent-generation stage, and how sensitive is MECo to noisy or partially incorrect edit actions?
4. Is the code-based editing approach extensible to larger classes of modifications, such as multi-step or scaffold-hopping edits, or to more open-ended design objectives (e.g., multi-objective, combinatorial libraries)? What (if any) bottlenecks emerge in code complexity, execution speed, or model training for such cases?

### Soundness
3

### Presentation
3

### Contribution
2
