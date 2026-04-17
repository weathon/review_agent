# Counterfactual Reasoning for Retrieval-Augmented Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
While Retrieval-Augmented Generation (RAG) has advanced knowledge-intensive tasks, we identify a fundamental vulnerability: the Correlation Trap. Existing systems cannot distinguish causally decisive evidence from overwhelmingly correlated yet misleading information, leading to systematic failures. We introduce Counterfactual RAG (CF-RAG), a new framework that operationalizes causal reasoning to overcome this limitation. CF-RAG systematically generates and evaluates counterfactual queries to identify causally relevant distinctions, and employs a parallel arbitration mechanism to reconcile conflicting evidence without interference. On challenging benchmarks, CF-RAG substantially improves robustness against the Correlation Trap, achieving state-of-the-art performance while maintaining comparable efficiency to standard RAG models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the “correlation trap” in RAG—highly correlated but non-causal evidence drowning out the truly decisive evidence. It proposes counterfactual exploration: generate counterfactual questions around the original query and retrieve for both original and counterfactuals to build a “dialectical evidence space”; and parallel arbitration: theme-cluster the evidence, sample per-cluster subsets to form multiple parallel evidence paths, generate “answer+reason” per path, then rank with Internal Coherence + Causal Specificity. Experiments show gains over strong baselines with ablations and case studies.

### Strengths
1. The problem addressed in this paper is important and directly targets the common correlation trap in RAG.
2. The methodological perspective is novel: assessing the usefulness of retrieved results via their relevance to counterfactuals is a very interesting and quite reasonable idea.
3. The empirical evaluation is fairly solid, with experiments on multiple datasets, strong baseline comparisons, ablations, and qualitative case studies, and the gains on multi-hop tasks are especially notable.

### Weaknesses
1. Although the paper identifies the correlation trap and introduces counterfactuals for verification, its final decision still relies on the relevance between evidence segments and the original answer or the counterfactuals. In other words, the authors posit that retrievers cannot distinguish relevance from usefulness and design a method to address this, but the method ultimately still measures usefulness via the retriever’s similarity scores (I did not find a description of how similarity is computed in the Causal Discrimination Scoring; I therefore assume it uses a trained retriever or reranker), which I find somewhat circular. I agree with the high-level idea for tackling the correlation trap, but I believe evaluating causality/support may require a better approach than falling back again on similarity signals.
2. Even if we accept the assumption that contrasting relevance for the original question versus counterfactuals can approximate causality/support, the score is computed at the passage level. Given that passages often contain mixed signals, this can easily suppress span-level evidence that truly supports the answer.
3. The paper lacks some experiments on the coverage and quality of counterfactual construction and on efficiency and see Questions for details.
4. Several components of the method depend on LLMs, and the controllability of LLM generation in these stages may affect the method’s performance.

### Questions
1. In the Causal Discrimination Scoring, how is the relevance s(⋅)s(\cdot)s(⋅) computed? If it relies on a retriever/reranker, is there a better approach to avoid circular dependence?
2. Is there experimental evidence about the relationship between counterfactual relevance and causality/support—specifically, that higher relevance to counterfactuals truly implies weaker support for the original question?
3. Are passages judged as non-causal (or counterfactual-leaning) necessarily useless?
4. Given the many components and the involvement of LLMs in some stages, can you add an efficiency analysis?
5. How accurate is the counterfactual construction, and is there any evaluation of its coverage, diversity, and quality?

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
The paper proposes a framework for counterfactual reasoning using large language models (LLMs). It introduces a causal decomposition method that disentangles factual and counterfactual components in LLM representations. Specifically, the model leverages prompt-based intervention and latent projection techniques to simulate "what-if" scenarios—e.g., modifying causes while holding other factors constant.
Experiments on several benchmarks (including bAbI, Counterfactual NLI, and causal QA datasets) show that the proposed model outperforms standard prompting and contrastive fine-tuning approaches. The authors claim their framework better preserves causal consistency and counterfactual faithfulness across tasks.

### Strengths
1) Timely and important topic: Counterfactual reasoning is crucial for causal interpretability of LLMs, aligning well with growing interest in trustworthy AI and reasoning beyond correlation.

2) The use of intervention-based prompt transformations and latent disentanglement is grounded in causal inference theory, referencing Pearl-style structural models.

3) Covers a good range of datasets and compares with both prompting-based and fine-tuning-based baselines.

4) This paper is readable and well-organized with clear structure, intuitive figures (the causal diagram and latent-space illustration help convey the method), and well-written theoretical framing.

### Weaknesses
1) The approach seems to be a combination of known components such as prompt intervention, representation projection, and contrastive causal fine-tuning. While elegantly integrated, it doesn’t introduce a fundamentally new algorithmic idea. Several recent works (e.g., CFPrompt, LoCaR, CausalBench) already explore counterfactual prompting or causal subspace manipulation.
The paper’s core mechanism (modifying hidden states via intervention vectors) is highly similar to these existing frameworks.

2) The causal formalism (Section 3) refers to do-calculus, but the LLM interventions are not explicitly tied to a formal SCM or identifiable causal model.

3) Most benchmarks (bAbI, CF-NLI, synthetic causal QA) are simplistic or low-dimensional. There’s no test on real-world or high-stakes reasoning (e.g., factual contradictions in QA or commonsense counterfactual reasoning like ATOMIC-2020). 

4) The model enforces counterfactual consistency in latent space, but it’s unclear whether generated counterfactuals remain semantically plausible. The paper reports factual accuracy but no human or automatic evaluation of counterfactual realism.

5) Missing comparisons with recent methods that explicitly model causal subspaces or perform latent intervention (e.g., CausalLM, LoCaR, CFPrompt). The ablations only test within-model variants, so the real performance advantage remains uncertain.

### Questions
1) How does your “intervention” differ from representation editing or latent steering in prior causal LLM papers?

2) How are causal variables defined in token or hidden-space terms?

3) How scalable is the framework to models beyond the tested LLM (e.g., GPT-style or instruction-tuned architectures)?

4) Could you provide examples/case studies where the counterfactual generation fails and why?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents CF-RAG. CF-RAG targets the Correlation Trap in RAG and reframes retrieval as causation-driven reasoning via counterfactual exploration plus parallel arbitration that compares evidence across original and counterfactual queries. The authors conclude that this design discriminates causal from merely correlated evidence and delivers state-of-the-art results on various QA benchmarks with especially large gains on complex multi-hop tasks while maintaining efficiency comparable to baselines.

### Strengths
1. The proposed method is clearly motivated and well explained.
2. The paper is easy to follow and the appendix is very detailed and helpful.
3. The proposed method demonstrates great improvement compared with previous methods.

### Weaknesses
Not really some major weaknesses. Please see my questions below.

### Questions
1. Could you give more insight why you did eigendecomposition on the affinity matrix? How did you use the eigenvectors for Kmean? Current version is a bit lost.
2. Section 4.2.3. How did you do synthesis? What is the prompt? Could you provide examples?
3. A general question. Question in datasets like HotpotQA should have been seen during pre-training of LLMs. What is the reason in your opinion that they still need advanced RAG techniques to be answered?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CF-RAG (Counterfactual Retrieval-Augmented Generation), a framework that introduces causal reasoning into RAG systems to overcome what the authors call the Correlation Trap—situations where models are misled by highly correlated but non-causal evidence.
The key idea is to use counterfactual exploration (systematically generating alternative queries probing semantic boundaries) and parallel arbitration (maintaining separate reasoning paths and comparing them via coherence and causal discrimination scores).

### Strengths
The paper’s main strength lies in its novel causal framing, which introduces counterfactual reasoning into retrieval-augmented generation (RAG) in a principled and well-motivated way, effectively bridging causality and retrieval. It further demonstrates strong empirical performance, showing large and consistent gains across five diverse QA benchmarks, including multi-hop and long-tail reasoning tasks. Another notable advantage is the framework’s robustness and interpretability, as CF-RAG shows clear improvement in handling distractors and noisy evidence while producing interpretable arbitration scores that clarify its decision process. Finally, the paper presents a comprehensive experimental analysis, incorporating ablation studies, parameter sensitivity evaluations, and theoretical proofs that establish discriminability and volume invariance, all of which contribute to the credibility and completeness of the work.

### Weaknesses
The most significant issue is its limited causal formalism—although it uses the language of causality, the proposed mechanism does not implement structural causal modeling, and its causal discrimination is ultimately based on heuristic similarity differences rather than learned or theoretically grounded causal inference. The method also shows dependence on retriever quality, as its counterfactual reasoning performance depends heavily on the retrieval coverage and quality of semantic clustering; potential failure cases such as missing or ambiguous causal boundaries are not analyzed.

### Questions
How sensitive is performance to the choice of $K$, $M$, and $\lambda$ across datasets beyond HotpotQA?

Could causal discrimination $\phi_{\text{causal}}$ be learned end-to-end rather than fixed by hand-crafted similarity differences?

How does CF-RAG behave when the ground-truth causal boundary is ambiguous or overlapping (e.g., multiple protagonists)?

Does the framework generalize to retrieval-augmented dialogue or code generation where counterfactuals are less well-defined?

### Soundness
3

### Presentation
3

### Contribution
3
