# Find tailored step example for next step: A Targeted Step-wise Retrieval Framework for Guiding LLM Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large language models (LLMs) have shown strong performance in mathematical reasoning, supported by approaches such as In-Context Learning (ICL) and Retrieval-Augmented Generation (RAG). However, existing methods often provide entire problems as examples, which is too coarse-grained for multi-step reasoning. Many steps in a retrieved problem may not align with the reasoning trajectory of the target problem, and some may even mislead the inference process. To address this limitation, we propose TSS (Tailored Step Search), a framework that enhances LLM reasoning through targeted step-level retrieval. TSS enables a model to dynamically decide when to retrieve a single logically consistent next step, using the current problem and its intermediate state as the query. The framework consists of two main components. First, we design structured training data and develop a Step Retriever, trained with a contrastive learning strategy to capture the logical flow between consecutive steps. Second, we train a Generator with a two-phase curriculum: it first learns to predict whether retrieval is necessary, and then learns to generate step-by-step reasoning in a structured format. Experiments on four mathematical reasoning datasets, across four backbone LLMs and multiple few-shot settings, demonstrate that TSS substantially improves reasoning accuracy and reliability. 
Our code is available at https://anonymous.4open.science/r/TSS-D930/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on retrieving the examples for mathematical reasoning via Large Language Models (LLMs). Existing methods often provide entire problems as examples, which may not align with the reasoning trajectory of the target problem, and some may even mislead the inference process. To this end, the authors propose Tailored Step Search (TSS), a framework that dynamically decides when to retrieve a single logically consistent next step (still, I do not think that the authors have given a clear definition to the ``logically consistent''). Experiments on four mathematical reasoning datasets demonstrate that they can improve reasoning accuracy and reliability compared to multiple baseline methods.

### Strengths
1. The proposed method outperforms multiple baseline methods in the experiments.

### Weaknesses
1. The writing could be improved. From lines 258 to 280, the line break is confusing. Element substitution negatives for replacing the problem stem and the targeted step can be presented in a single paragraph for better understanding. In Figure 1, the meaning of the red and blue font colors is not explained in the caption.

2. Some important information is missing from the paper. From 370 to 372, for the baseline methods, the authors should give a brief introduction to SPELL and IDS methods, at least their key differences in working mechanism compared to the proposed method.

3. The motivation is not clear enough to me. Based on Figure 1, I still do not understand what it looks like for a perfectly retrieved example. Regarding the given retrieved example in Figure 1, what do you mean by "correspond one-to-one with the corresponding steps in the golden steps"?

### Questions
1. As retrieving good examples for ICL is common in different applications of LLMs, how can the method be generalized to other tasks like question answering?

2. What is the computational cost for the extra procedure for judging and retrieving the examples?

### Soundness
3

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
This paper proposes TSS (Tailored Step Search), a framework for improving LLM mathematical reasoning through targeted step-level retrieval. Unlike traditional RAG methods that retrieve entire problems, TSS dynamically retrieves single next-step examples during reasoning. The system consists of: (1) a Step Retriever trained with contrastive learning to capture logical flow between steps, and (2) a Generator trained in two phases to predict retrieval necessity and generate structured reasoning steps in [CONDITION, PROCESS, CONCLUSION] format. The paper evaluates on AIME, GSM8K, MATH500, and a custom STEP dataset  across four LLM backbones (Llama-2-7B, Llama-3-8B, Qwen2.5-3B, Qwen2.5-7B).  They introduce new metrics (DSM, TCN, RCR) for evaluating intermediate 
reasoning quality.

### Strengths
1. The paper identifies a real limitation of problem-level retrieval in mathematical reasoning - many retrieved steps are misaligned.
2. The [CONDITION, PROCESS, CONCLUSION] decomposition provides an interpretable structure for step-wise reasoning, which could benefit explainability and error analysis.
3. The introduction of DSM (Dynamic Step Matching), TCN (Trajectory Correction Number), and RCR (Retrieval-Corrected Rate) offers fine-grained evaluation of intermediate reasoning quality beyond final answer accuracy.
4. The method shows improvements across multiple model backbones and datasets, suggesting some degree of generalizability.

### Weaknesses
1. The paper lacks comparison with directly  relevant prior work: 1) Self-RAG: Also performs adaptive retrieval with similar decide-then-retrieve mechanism. This is a glaring omission; 2) Problem-level retrieval using the SAME knowledge base: Without this comparison, the claimed superiority of step-level over problem-level retrieval is not convincingly demonstrated; 3) Simple retrieval methods (e.g., BM25) as lower bounds.
2, On AIME (122 problems), improvements of 0.82→2.46 represent only ~1-2 additional correct answers, which may lack statistical significance.
3. Missing comparisons on existing benchmarks like MMLU-Math, TheoremQA.
4. No analysis of retrieval quality (precision/recall of relevant steps)

### Questions
1. Why is Self-RAG (2024) not included as a baseline? It directly addresses adaptive retrieval in LLM reasoning and should be the primary comparison point. Can you provide results comparing TSS with Self-RAG?
2. Can you provide a controlled comparison where both problem-level and step-level retrieval use the EXACT SAME knowledge base?
3. What is the accuracy of the retrieval decision mechanism during inference? (i.e., when the model predicts <retrieval> vs <no-retrieval>, how often is this decision correct?)
4. How does the automatic annotation quality (using GPT-4o-mini) affect results?
5. How sensitive is the method to the number of negative examples in contrastive learning?

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
This paper proposes **TSS**, a novel framework designed to improve LLM performance on multi-step reasoning tasks like mathematics. It addresses the "granularity mismatch" of traditional RAG, which retrieves *entire problems* based only on the initial query, often resulting in irrelevant or misleading examples. TSS instead uses the **current problem-solving state** to dynamically retrieve more fine-grained, **step-level examples**. The framework features a **Step Retriever** trained with contrastive learning to capture logical flow, and a **Generator** trained in a two-phase curriculum to first predict *when* to retrieve and then to *utilize* the retrieved step for the next reasoning action. Experiments across four LLMs and multiple math datasets demonstrate that this targeted, step-wise retrieval significantly improves reasoning accuracy compared to standard RAG baselines.

### Strengths
1.  **Strong Motivation and Intuitive Solution:** The paper identifies a clear and significant limitation in traditional RAG for multi-step reasoning: retrieving examples based only on the initial problem (problem-level) often fails to provide relevant guidance for intermediate steps. The proposed TSS framework, which incorporates the *current problem-solving state* to retrieve fine-grained, *step-level* examples, is a highly logical and necessary solution. This approach correctly shifts the retrieval focus from "what the problem looks like" to "what the next logical step is," finding examples that are similar in *method* rather than just *form*.

2.  **Robust, Process-Oriented Evaluation:** The authors rightly point out that final accuracy is an insufficient metric for reasoning tasks, where correct answers can be "unfaithful" (i.e., derived from a flawed process). The introduction of new process-supervision metrics—**Dynamic Step Matching (DSM)**, **Trajectory Correction Number (TCN)**, and **Retrieval-Corrected Rate (RCR)**—is a major strength. These metrics provide a far more precise instrument to measure the *actual* impact of the retrieval, assessing its effect on the reasoning *trajectory* itself, not just the final result.

3.  **Clever and Effective Training Methodology:** The paper's technical execution is sophisticated.
    * For the **Step Retriever**, the construction of four distinct types of "hard negatives" (e.g., Order Negatives, Backtracking Negatives) provides a strong supervisory signal, training the model to learn logical flow, not just semantic similarity.
    * For the **Generator**, the two-phase training curriculum is elegant. It first trains the model to explicitly decide *when* to retrieve (predicting `<retrieval>` or `<no_retrieval>`) and then trains it to *use* that retrieved information. This "on-demand" retrieval mechanism is a practical and efficient design choice.

### Weaknesses
1.  **High Implementation Cost and Limited Applicability:** The proposed framework's reliance on a highly specific `[Condition, Process, Conclusion]` data format, along with the need to train two separate, specialized models (the Step Retriever and the two-stage Generator), creates a significant barrier to entry. This complex setup limits the method's practical applicability compared to other ICL baselines, as it:
    * Cannot be easily extended to new domains (e.g., domains outside of mathematics) without a new, costly data-structuring pipeline.
    * Is incompatible with non-finetunable, closed-source models (e.g., proprietary embedding or generation APIs).
    * This contrasts sharply with baseline methods like IDS, which are more "plug-and-play" and can operate with arbitrary frozen retrievers and generators.

2.  **Flaws in the Dynamic Retrieval Mechanism:** The core "on-demand" retrieval mechanism has two theoretical flaws:
    * **Intrinsic Difficulty and Speculative Nature of the Retrieval Trigger:** The framework's design forces the Generator to perform an intrinsically difficult task: it must predict *if* retrieval is needed for the *next* step ($S_i$) using only the context of the problem $X$ and the *previous* step $S_{i-1}$. To accurately judge if the retrieval is needed, the model must essentially *predict its own future difficulty*. It needs to implicitly anticipate what $S_i$ is supposed to be, assess the complexity of generating that step, and *then* decide whether it needs help, all based only on past information. Training the model to decide whether to retrieve without any clues about the next step is relatively harder than the data generation stage, which provides GPT-4o-mini with both context and the subsequent step.
    * **Lack of a Fallback Mechanism:** The framework commits to using whatever the retriever provides. If the retrieved example is low-quality or misleading, it can derail the entire reasoning chain, potentially performing worse than `no-retrieval`. The framework lacks a crucial "grading" step at inference time to *validate* the retrieved example and fall back to its own generation if the hint is poor. Furthermore, the paper provides no analysis or ablation study on how the quality of the retrieved example (e.g., "Perfect Analogy" vs. "Poor," as defined in their own appendix ) impacts the generator's final performance.

3.  **Weaknesses in Evaluation and Metric Definitions:**
    * **Statistical Power:** The results on the AIME dataset are of questionable statistical significance. The dataset is small (~122 questions), and the models' base performance is very low. The reported 1-3% gaps represent a difference of only a few questions, which could be due to random chance.
    * **Vague Process-Oriented Metrics:** While the new metrics (SRR, LUR, TCN, DSM) are well-motivated, their definitions and implementations are unclear.
        * **TCN:** The definition of an "erroneous step" seems to assume any deviation from the single Ground Truth (GT) solution is an error. This incorrectly penalizes valid alternative solution paths, different step orders, or different (but correct) levels of granularity.
        * **DSM:** The "one-to-one matching" is brittle. The appendix does not explain how the metric handles cases where the model's output merges or splits steps differently than the GT solution (e.g., model Step 1 = GT Step 1+2), which would cause all subsequent steps to be misaligned and marked as incorrect.
        * **SRR/LUR:** These metrics are only vaguely described in the appendix, and their calculation relies on an LLM-as-judge with prompts that are not clearly specified, adding subjectivity to the evaluation.

4.  **Minor Presentation and Clarity Issues:**
    * In Section 4.2.1, the text describing the final training data formats appears to have a copy-paste error, showing the *same* pattern (`<retrieval> S_i <p> E_j </p>`) for both retrieval and no-retrieval instances.
    * In Figure 4 (a) and (b), the Y-axis range (e.g., 0-100) is too wide, making the performance differences between methods visually indistinguishable. The plots should be re-scaled to a more appropriate range to clearly show the relative improvements.

### Questions
1.  **Applicability to Frozen/API-Based Models:** The current TSS framework requires significant custom data construction and fine-tuning for both the Step Retriever and the two-stage Generator. This limits its applicability to open-source models. Have the authors considered or experimented with adapting this framework to a "frozen model" or API-based setting, which is a common and practical use case, similar to the baseline methods? For instance:
    * Could a similar *step-level* retrieval be achieved by using a strong, pre-trained (but not fine-tuned) retriever (e.g., a generic embedding API) where the query is a simple concatenation of the problem and the current reasoning history?
    * Could the two-stage generator be simulated by prompting a large, closed-source model (like GPT-5, Gemini, or Claude) to generate step by step and only keep the next step by hand, thereby mimicking the step-wise generation?

2.  **Clarity and Fairness of Process Metrics:** The introduction of process-oriented metrics (DSM, TCN, RCR) is a strong point, but their definitions and application are ambiguous.
    * Could the authors provide more precise, algorithmic definitions for **TCN** (e.g., how is an 'erroneous step' defined beyond simple mismatch with the GT, especially concerning valid alternative solution paths?) and **DSM** (e.g., how does the one-to-one matching handle granularity differences like merged or split steps)?
    * How were these process metrics (like SRR, LUR, and DSM) applied to the *baseline* methods? Baselines retrieve *full problem solutions*, not steps. Were these full solutions also decomposed into the steps? The TSS generator is trained on data derived from PRM800K, which, like the GT solutions in the STEP dataset, was processed and rewritten by GPT-4o-mini into the specific `[C, P, C]` format. What if the baseline methods retrieve a full solution or generate their own full solution in a format, style, or granularity that is *very different* from this standardized GT format?
    
    In such a scenario, wouldn't the TSS generator automatically achieve a better score on these metrics (especially DSM) simply because its output *structure* matches the GT's structure, while the baseline's (potentially correct) reasoning is unfairly penalized for being stylistically different? To clarify this, could you provide a qualitative case study that compares all five components side-by-side:
    1.  The GT step-by-step solution
    2.  TSS's retrieved example steps
    3.  TSS's generated steps
    4.  The baseline's retrieved full solution (and how it was 'split' for metric calculation)
    5.  The baseline's generated full solution (and how it was 'split')

3.  **Analysis of Retrieval Quality Impact:** The framework currently lacks a mechanism to "grade" the quality of a retrieved step at inference time, forcing the generator to use it even if it's low-quality. Have the authors considered an ablation study to measure this impact?
    * For example, by using an oracle (like GPT-4o-mini, similar to the data annotation process) to rate the *quality* of each retrieved step during inference (e.g., 1-5 stars, as in Appendix F.3.2). This could show: (a) the downstream accuracy when using only 5-star examples, (b) the accuracy when using 1-star examples (to measure negative impact), and (c) the accuracy compared to `no-retrieval`.
    * If this grading proves to be highly correlated with success, could this "grading" step be incorporated into the framework itself, perhaps as a learned 'retrieval-quality' model to decide whether to ultimately *use* the retrieved example or fall back to a `no-retrieval` generation?

### Soundness
3

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
4

### Summary
This paper proposes a step-wise retrieval framework for large language model (LLM) reasoning, named Tailored Step Search (TSS). The method refines the traditional RAG paradigm from semantic-level retrieval to logic-step-level retrieval, and explicitly models each reasoning step’s condition, process, and conclusion through the structured representation $S_i = (C_i, P_i, Q_i)$. The retriever is trained via contrastive learning to capture *logical adjacency* rather than mere *semantic similarity*.

In addition, the authors design a two-stage generator consisting of (1) a retrieval-decision module (<retrieval> vs <no-retrieval>) and (2) a step-wise generation stage. The proposed framework achieves performance improvements on multiple mathematical reasoning datasets, including AIME, MATH500, and GSM8K. The paper further introduces new metrics—DSM, RCR, and TCN—to evaluate the quality of intermediate reasoning steps and the effectiveness of retrieval-based correction.

### Strengths
- High originality: This work is the first to introduce the concept of *logical adjacency* into a retrieval-based reasoning framework.
- Structured interpretability: The design of $S_i = (C_i, P_i, Q_i)$ makes each intermediate reasoning step more analyzable and interpretable.
- Broad experimental coverage: The method is evaluated across multiple reasoning datasets and model scales, with particularly strong results observed on the Qwen3-7B model.

### Weaknesses
1. Simplified modeling assumption:
    Reducing multi-step dependencies to a first-order Markov assumption may cause the model to lose contextual consistency over long reasoning chains.
2. Inconsistent training logic:
    Positive samples are constructed in a predictive form ($\rightarrow S_i$), while the retrieval stage uses a key–value form ($key \rightarrow value$). The paper should analyze the potential gap between these two paradigms.
3. Lack of complexity analysis:
    The paper does not report the average token length of $S_i$ or the token consumption during the generation phase, making it difficult to assess computational efficiency.
4. Limited statistical significance:
    The reported improvements are not particularly strong. Multiple-run statistics should be provided. Currently, the performance gains appear significant only on Qwen7B, while on other models the advantage over the second-best result is minimal and could fall within experimental variance.
5. Narrow baseline selection:
    The paper does not include comparisons with structurally similar systems such as AirRAG, Self-RAG, or ProcessRAG, which would help contextualize its contribution.

### Questions
- Does the construction of $S_i = (C_i, P_i, Q_i)$ include all previous steps?
 Please clarify whether each step explicitly contains the entire reasoning history or only the immediate predecessor $S_{i-1}$. This distinction directly affects token length, computational cost, and the validity of the first-order Markov assumption.

- Can the retriever embedding capture such logic-level relationships?
 The retriever is trained using a predictive form ($(X, S_{i-1}) \rightarrow S_i$); why was this approach chosen instead of a key–value alignment objective? If the retriever were trained directly on $(key \leftrightarrow query)$ matching, would the performance differ?

- How is *one-shot* defined in this context? Does it mean that the model retrieves examples only once during the entire reasoning process, or that it can perform multiple retrievals but uses a single example (one-shot) each time?

- Could the authors report the mean and variance over multiple runs, as well as statistical significance tests? This would clarify whether the observed improvements are consistent or within noise margins.

- Have the authors compared their method against structurally related logical retrieval approaches such as AirRAG, Self-RAG, or ProcessRAG? Such comparisons would better position TSS in the current research landscape.

- How does the proposed retrieval mechanism perform when combined with *Chain-of-Thought (CoT)* prompting? Does TSS still provide measurable improvement when CoT reasoning is already applied?

### Soundness
3

### Presentation
3

### Contribution
3
