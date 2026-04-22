# DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal statements lack direct mappings to mathematical theorems and lemmata, nor do those theorems translate trivially into the formal primitives of languages like Lean.
To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable "sub-components". This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. 
We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 42.25% and 37.14% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an end-to-end autoformalization pipeline that converts informal mathematical statements into Lean 4 declarations. The system couples a retriever (trained by the authors) with a generator (commercial or open LLMs) and evaluates outputs using automated Lean checks and a symbolic-equivalence metric. The main claims are (i) training a task-specific retriever improves grounding for generation, and (ii) the pipeline can robustly produce syntactically valid, type-checked Lean statements across a curated benchmark. The paper reports head-to-head comparisons against several general-purpose LLMs and ablations around retrieval.

### Strengths
- Clear end-to-end systemization of a practical autoformalization pipeline for Lean 4, with sensible stages (retrieve → generate → check). The paper’s “illustrate” section is genuinely helpful: the qualitative visual walkthroughs of retrieved contexts, intermediate rewrites, and final Lean artifacts improve interpretability and reproducibility.
- Ablations on retrieval are valuable; retrieval quality is a major bottleneck in Lean autoformalization, so any careful analysis there is welcome.
- Timely topic & potential impact. There is fast-moving SOTA on Lean autoformalization and proof generation (Goedel/ Kimina/ process-supervised approaches); a robust, reproducible pipeline can help the community measure progress on statements specifically.

### Weaknesses
W1. Retriever novelty & baselines are weak.

The only trained component is a DPR-style dual encoder, which is basically the same as one in RAutouformalizer. Moreover, Lean-specific retrieval methods already exist (LeanSearch’s semantic search[3]; LeanExplore’s hybrid multi-signal retrieval[4]), and the paper neither compares against nor leverages them as baselines or components. This makes it hard to justify training a new DPR when stronger plug-ins are available. 

W2. Missing SOTA autoformalizer baselines.

 The paper compares primarily to generic LLMs (GPT-4.1, DeepSeek-V3.1) but omits direct comparisons to autoformalizer-specialized systems, notably Goedel-Formalizer[1] and Kimina-Autoformalizer[2], which are expressly designed to translate informal math to Lean 4 statements and are publicly available. Given their focus and reported quality, they are the most relevant baselines for this task. Including them (or explaining why they cannot be included) is essential for positioning. 

W3. Frontier model coverage is incomplete/out-of-date.

For the generator, the paper focuses on GPT-4.1 and DeepSeek V3.1. The current frontier for mathematical/logic tasks prominently features DeepSeek R1 (0528), OpenAI’s o-series (o3), Gemini 2.5 Pro, and Claude 4.1; these models publicly advertise stronger reasoning/coding capabilities and should be part of the comparison, at least in a retrieval-on vs retrieval-off ablation to substantiate the retriever’s benefit. Having only a single frontier model (Claude 4) is not persuasive in 2025. 

W4. Stale toolchain / dataset snapshot raises representativeness concerns.

 Experiments are run on Lean 4.7.0 / an older mathlib snapshot. Lean and mathlib have evolved significantly (Lean 4.25.0-rc2 exists; mathlib’s scale has expanded beyond 200k theorems), and style/namespace changes accumulate. Results confined to an older snapshot may under- or over-estimate real-world robustness. Authors should either (i) re-run on a contemporary toolchain and a recent mathlib commit or (ii) justify the choice and discuss compatibility gaps. 

W5. Evaluation metric (BEq+) may under-state performance without human adjudication.

 BEq+ is a reasonable automated proxy, but even its authors note a relatively high false-negative rate; strict symbol-level equivalence can mark semantically correct paraphrases as wrong. The paper reports low success rates (sub-25% in places); without a human-adjudicated subset or complementary metrics, it is hard to interpret practical significance. A small-scale human study or relaxed-equivalence cross-check (e.g., type-equivalence under definitional unfolding) would strengthen claims. 

[1] Yong Lin, et al. "Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction" arXiv preprint arXiv:2508.03613 (2025)

[2] Wang, Haiming, et al. "Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning" arXiv preprint arXiv:2504.11354 (2025).

[3] Gao, Guoxiong, et al. "A semantic search engine for Mathlib4." arXiv preprint arXiv:2403.13310 (2024).

[4] Asher, Justin. "LeanExplore: A search engine for Lean 4 declarations." arXiv preprint arXiv:2506.11085 (2025).

### Questions
Toolchain & dataset. What constraints led you to Lean v4.7.0/that mathlib commit? Please discuss how brittle your pipeline is to syntax/tactic drift across versions. Additionally, could you explain in detail how you conduct data extraction from mathlib and prepare them for embedding training in detail?

Ground truth. How do you construct ground truth (oracles) for decomposition and retrieval tasks? Detail the pipeline used to obtain ground truth (oracles) from Lean.

Interpreting BEq+. Given BEq+’s known false negatives, do you have a human-adjudicated subset to calibrate precision/recall? How often do your “failures” reflect symbol-level mismatches vs true semantic errors? Consider reporting: (a) case study on type-checks but BEq+-fails; (b) human-judged correctness. 

Ablations on retrieval → generation sensitivity. Please report end-to-end success vs top-k retrieval quality (e.g., R@k buckets) to quantify how much the generator depends on retrieval depth and filtering.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the DRIFT framework , which aims to enhance the process of mathematical statement autoformalization. DRIFT decomposes complex informal mathematical statements into sub-queries, then accurately retrieves the corresponding formal dependencies. Based on these retrieved dependencies, it provides illustrative contextual examples to guide the model in applying them correctly. Through this structured process, DRIFT achieves improved performance in automatic formalization of mathematical statements.

### Strengths
1.DRIFT introduces an innovative approach by decomposing complex informal statements into sub-queries, which allows for precise retrieval of the required formal dependencies. In addition to retrieving relevant premises, the framework also provides illustrative examples of their usage, effectively guiding the model to apply the dependencies correctly during formalization.

2.The experiments demonstrate both the generality and effectiveness of the proposed framework. Moreover, the ablation studies clearly reveal the individual contributions and roles of different modules within DRIFT, strengthening the empirical support for the proposed design.

### Weaknesses
1.The ablation study shows that the retrieval module plays a crucial role in the overall performance of DRIFT. However, the paper does not compare this module with existing Lean premise retrieval methods, such as Lean Search or other established retrieval baselines. Including such comparisons would provide a clearer understanding of the advantages and limitations of the proposed retrieval component.

2.The experiments primarily focus on general-purpose reasoning models such as GPT and DeepSeek. However, there are now several large models that have been specifically trained or fine-tuned on Lean. It remains unclear whether DRIFT has been tested on these Lean-specialized models, and such evaluation could further demonstrate the framework’s adaptability and robustness.

### Questions
Please refer to the Weakness section.

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
The paper proposes DRIFT: a four-stage framework for autoformalizing informal math statements into Lean. DRIFT (i) Decomposes an informal statement into atomic, concept-focused sub-queries (with predicted formal “anchors”), (ii) Retrieves dependent premises from a formal library using a finetuned dense retriever, (iii) Illustrates usage via a small set of demonstrative theorems chosen by a greedy coverage algorithm, and (iv) Formalizes the statement conditioned on the retrieved context. Across ProofNet (in-distribution), MiniF2F-test (largely self-contained), and ConNF (out-of-distribution), DRIFT improves dependency retrieval F1 and downstream formalization, with especially large gains on ConNF where it even surpasses an oracle* retrieval setting.

### Strengths
1. The framework adds an Illustrate step that selects a minimal set of theorems to demonstrate how retrieved premises are used, addressing the gap between definition and usage, an underexplored angle in prior work.

2. Clear end-to-end design validated on three complementary benchmarks; the method substantially boosts BEq+ and type-check rates over strong retrieval baselines and zero-shot, with striking OOD gains on ConNF.

3. Ablations show the Illustrate step is crucial (removing it sharply reduces BEq+ on ProofNet/ConNF), and quantify contributions of Decompose vs. Retrieval.

4. The pipeline is easy to follow, and can inform future RAG design for formal methods.

### Weaknesses
1. The decomposer appends predicted formal representations; the paper argues this helps anchoring, but does not quantify robustness when anchors are wrong/noisy.

2. The ablation discussion suggests decomposed retrieval may introduce diverse noise that requires illustrative scaffolding; more error taxonomy and qualitative failure analyses (on both ProofNet and MiniF2F) would bolster the explanation and guide adaptive retrieval strategies.

3. There is a chance that I missed this somewhere, but the paper does not seem to report compute/latency costs for decomposition + retrieval + illustration vs. baselines? Given practical adoption, cost-quality tradeoffs matter.

4. The paper observes retrieval can distract in low-dependency regimes (MiniF2F). It would help to show an adaptive gate (e.g., predicting when to skip retrieval or to down-weight illustration) and quantify decision quality.

### Questions
1. How does performance vary with the number of sub-queries, top-k per sub-query, and the illustration budget m? Any evidence of diminishing returns or overfitting with larger m?

2. You surpass Oracle* on ConNF; can you provide diagnostics (e.g., premise coverage of selected theorems, overlap with ground truth, qualitative examples) explaining where illustrative theorems help beyond oracle dependencies?

3. Some slight inconsistencies in terminologies: e.g., "DeepSeek-3.1" and "DeepSeek-V3.1" both appeared.

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
4

### Summary
The paper proposes DRIFT, a new technique for autoformalization. Their pipeline first decomposes a statement into sub-problems, retrieves relevant premises and their usage in sample theorems, and finally uses all the included retrievals to perform formalization.

### Strengths
- The pipeline for autoformalization is novel and has not been explored before in the literature. The results are also strong, and the authors also ablate removing several modules (Sec 5.3), providing interesting insights into the value add of each module.
- The techniques used for the modules are interesting, and the method outperforms zero-shot and retrieval-augmented baselines.

### Weaknesses
- As discussed in the paper, I suspect that the benchmarks could be contaminated, especially miniF2F and ProofNet. There are many instances of these two popular benchmarks on GitHub, and it would be surprising if models had not seen them before, even if the results are low. It is possible that retrieval could remind or steer the model to a certain distribution that can elicit its recall ability of seeing these benchmarks.
- The pipeline consists of a set of modules, but none of them seem to be particularly optimized for performance. For example, for the decompose module, only one decomposition prompt seems to have been tested. 
- The previous issue could lead to misleading interpretations of the ablation study: the authors noted that in one of the experiments, removing the "decompose" module does not degrade performance, but I wonder if this would be different if the retrieval and formalizer models were replaced with stronger models.
- The paper does not compare with other autoformalization techniques in the literature, making it hard to assess its significance and effectiveness

### Questions
- Can the authors demonstrate gains on proof autoformalization using this method as well?
- Why was m=3 selected for the illustration stage? Is it possible that scaling this up to much more examples will improve more (e.g. in https://arxiv.org/abs/2404.11018)
- How does DRIFT compare to other autoformalization techniques in performance?

### Soundness
3

### Presentation
3

### Contribution
3
