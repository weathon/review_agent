# LlavaCode: Compressed Code Representations for Retrieval-Augmented Code Generation

- Avg Score: 3.20
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2, 6

## Abstract
Retrieval-augmented generation has emerged as one of the most effective approaches for code completion, especially when context from the surrounding repository is important. However, adding this context substantially increases sequence length, which slows inference—an important limitation for interactive settings such as IDEs.
In this work, we introduce LlavaCode, a framework that compresses context into compact, semantically rich representations that remain interpretable to code LLMs. This improves generation quality while reducing prompt augmentation to only a few compressed single-token vectors.
Our approach requires training only a small projector module and introduces negligible additional latency, yet it significantly improves the prediction quality of code LLMs.
Our experiments show that LlavaCode enables a 20–38\% reduction in Time-to-First-Token (TTFT) on line-completion tasks compared with uncompressed RAG.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes LlavaCode, a retrieval-augmented code generation framework that compresses retrieved code into dense embeddings aligned with the LLM’s input space. A lightweight projector is trained with a mix of cross-entropy, REINFORCE, and cosine-alignment losses to preserve semantic fidelity while reducing input length. Experiments on The Stack (Python) show improved EM and Edit Similarity with notably lower latency, suggesting LlavaCode’s potential for efficient IDE code completion.

### Strengths
The paper is easy to follow and well-structured. The motivation—to reduce latency and input length in retrieval-augmented code generation—is clearly articulated.

### Weaknesses
This paper has several serious weaknesses.  

First, the technical contribution is limited. The idea of embedding or compressing retrieved contexts has been well explored in recent RAG literature, and this work does not offer meaningful methodological innovation beyond applying it to repository-level code generation.  

Second, the experimental setup is unsound and fails to validate the proposed method’s effectiveness. The method is designed to compress retrieved context, yet in Section 4.1 the evaluation is performed under a single-file setting without any cross-file context, making the comparison with baselines largely meaningless. The paper provides no evidence that the proposed compression preserves model performance (e.g., EM, ES) when cross-file context is included. Under such a flawed setup, the reported efficiency gains have little discussion value.  

Finally, the experimental scope is very limited. The authors do not evaluate on any standard RAG or code completion benchmarks, nor do they compare with strong existing baselines. All results are based on a small self-collected dataset and a single model, which is insufficient to support the claimed contributions.

### Questions
In Table 4.1, it seems that the main experiments are conducted under the in-file setting, without including cross-file context. Could you please clarify whether any of your evaluations are performed with cross-file context (CFC)? If not, how can we be confident that the proposed compression method does not compromise model performance (e.g., EM, ES) when cross-file context is provided?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents LlavaCode, a method to compress the retrieved context into tokens by a separate frozen encoder and a learned projector, in order to reduce the TFFT latency. To train the projector, a composite loss combines token-level cross-entropy, a REINFORCE-style sequence reward on EM/ES, and a cosine-similarity alignment term. Results on a Python dataset show that the method can improve EM & ES over a non-RAG baseline (though quality remains below uncompressed RAG); and lower TFFT.

### Strengths
1. Clear and important problem framing
2. Its focus on TFFT latency reduction is relevant for interactive IDE settings
3. Empirical TTFT measurements under two serving regimes and across multiple model sizes provide a useful operational perspective.

### Weaknesses
1. Novelty claims can be overstated. Despite the difference in methodology, like the cosine-alignment loss and RL on new metrics, the evaluation lacks a direct comparison to xRAG as the baseline, which already demonstrated “one-token” RAG with frozen retriever and LLM. A discussion of prior code-focused prompt/context compression works, such as RepoFUSE and CodePromptZip, is also missing. 
2. Section 3.1 states top-10 retrieved chunks per completion and 512-token truncation per chunk, which would imply thousands of extra tokens, yet the “w/ CFC” baseline adds only 512 tokens (2000→2512). To clarify, does the CFC baseline only include the top-1 retrieved chunk, or does it truncate the concatenated top-10 chunks to 512?
3.  The paper states the gradient is approximated “using a single Monte-Carlo sample from pθ,” but then computes rewards on greedy rollouts and omits a variance-reducing baseline (contrary to SCST). This produces a biased, high-variance estimator with unclear benefits, and there is no analysis of reward scaling or stability.
4. Experiments only use Qwen2.5-Coder-1.5B for quality evaluation; larger or different code LMs are not tested for accuracy (only TTFT). Also, the best EM/ES numbers still come from uncompressed cross-file context, indicating the compression method is not lossless -- an analysis on the quality–compression trade-off would be useful. 
5. EM and ES are surface-form metrics that reward lexical overlap and exact phrasing; they can penalize semantically correct alternatives (e.g., different variable names or algorithmic implementations). Why not use execution-based metrics, or code-aware similarity metrics like CodeBLEU?
6. In section 5, when measuring the latency, does it take the encoder and projector inference latency into account? If not, the reported TTFT gains only reflect prefill compute savings under synthetic prompt shortening, not the actual compressed-RAG system.
7. Writing issues: 
- typos like “freezed,” “weight dacay”
- wrong in-text citation format -- use \citep

### Questions
1. Consider adding stronger baselines like code-specific compression (RepoFUSE, CodePromptZip) and xRAG adapted to code.
2. Clarify why does “w/ CFC” add only 512 tokens?
3. Across different sizes of Qwen2.5-Coder, does projector weights transfer across backbones or require re-training?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on compressed representations of retrieved texts to handle the long context problem in RAG. Inspired by Llava, the authors use an embedding model and a projection layer to condense the retrieved text representations into one token representation that is fed to the LLM for generation. To address several limitations of the proposed framework, the authors further incorporate reinforcement learning and cosine alignment loss. Evaluation results demonstrate the effectiveness of the LlavaCode.

### Strengths
1. The studied problem is valuable. Indeed, code generation tasks suffer a lot from the long context problem in RAG, much more severe than general generation tasks.
2. The proposed LlavaCode demonstrates better performance on ES and EM.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The proposed LlavaCode seems highly similar to the related work xRAG mentioned in the paper, which limits the novelty of this paper. The authors should explicitly discuss the distinction between xRAG and LlavaCode. I don't agree that xRAG was not evaluated under the code generation task is a solid reason. Besides, it seems that the authors do not compare the performance between xRAG and LlavaCode in the paper. If I ignore the comparison, please correct me.
2. Exact Match and Edit Similarity are not good metrics for evaluating code generation performance. Correct codes can have highly different linguistic forms due to variable naming preferences, different running logics, etc. This was largely discussed by previous works, which were referred to as style difference [1, 2]. Thus, recent code generation works often evaluate functional correctness through the pass rates of test cases, not the matching percentage of tokens. 
3. The authors only evaluate the LlavaCode on Qwen-series LLMs and only on one dataset. More Experiments should be conducted to demonstrate the generalization ability of LlavaCode.

[1] Wang, Yanlin, et al. "Beyond functional correctness: Investigating coding style inconsistencies in large language models." Proceedings of the ACM on Software Engineering 2.FSE (2025): 690-712.

[2] Li, Haochen, Xin Zhou, and Zhiqi Shen. "Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

### Questions
1. It would be interesting to investigate why the compressed representations become highly similar without cosine alignment loss. Does it mean that not all the retrieved texts are meaningful in RAG? I would like to hear some explanations from the authors.
2. The discussion of the limitations of CE loss confused me. The authors said that Exact Match, which measures the percentage of predictions character by character, is a sequence-level metric. I would like to hear some explanations from the authors.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission studies code representation compression for retrieval-augmented code generation (RAG). To support RAG for code completion, a contextual representation must be generated, which raises efficiency concerns. Prior work has explored context compression, among which embedding projection (e.g., LLaVA) is a typical approach. The current paper adapts LLaVA to code completion, resulting in LLaVACode. The authors evaluate their method using two metrics—Exact Match and Edit Similarity—and employ reinforcement learning (REINFORCE) to optimize these objectives.

### Strengths
- The idea of exploring compression techniques for code generation/completion is interesting and potentially useful for improving both efficiency and quality.

- The originality of the paper is low (cf. weakness). 

- The paper is readable, partially because of the simplicity of the approach.

### Weaknesses
- Incremental contribution. The method is a direct adaptation of LLaVA. As shown in Figure 1(b), the encoder architecture is nearly identical to LLaVA. The adaptation process appears straightforward and does not deeply consider the unique characteristics of programming languages or the specific challenges of code generation compared with natural language. Although the paper incorporates AST information, this component is relatively weak and does not seem to yield notable performance gains.

- Inadequate evaluation metrics. The chosen metrics, Exact Match and Edit Similarity, do not faithfully capture the quality of code completion. While they may be suitable for natural language text, they are insufficient for code, where even minor syntactic differences can lead to uncompilable outputs. More meaningful metrics—such as functional correctness or execution-based accuracy—should be included to better assess code quality.

- Limited evaluation scope. The experiments are conducted exclusively on Qwen2.5-coder and limited to a Python dataset, leaving it unclear how the proposed approach generalizes to other models, languages, or programming paradigms.

### Questions
1. Why are Exact Match and Edit Similarity the only metrics considered?
2. Why choose REINFORCE? Why not choose state-of-the-art RL methods?
3. What is the essential novelty of LLaVACode compared to LLaVA?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes LlavaCode, a framework that compresses retrieved code context into compact embedding tokens for retrieval-augmented code completion. Inspired by LLaVA and xRAG, it uses a lightweight projector to align encoder embeddings (e.g., from Qwen3-Embedding or UniXCoder) with a frozen code LLM. The model is trained with a composite loss combining Cross-Entropy, REINFORCE (directly optimizing EM and ES), and Cosine Alignment to prevent projection collapse. Experiments on a self-constructed Python dataset show 20–38% latency reduction and small accuracy gains over a no-context baseline.

### Strengths
- Novel idea in a new domain: The first attempt to apply LLaVA-style projection and compression to the code RAG problem. This is conceptually creative and well-motivated by latency constraints in IDE-style code completion.

- Thoughtful loss design: The use of a composite CE + RL + cosine alignment objective is well-argued and clearly improves stability and alignment.

- Thorough ablation: The authors carefully isolate the effects of encoder choice, loss terms, and projection architecture, which adds credibility to the analysis.

- Clear motivation and solid writing: The methodology section is coherent and easy to follow, with intuitive visualizations (Figure 1–3) and detailed latency analyses.

### Weaknesses
- Limited effectiveness: Despite the clever design, LlavaCode’s generation quality is underwhelming. EM only rises from 45.97 → 47.66 (vs no-context), while full-context QwenCoder achieves 50.87. This small margin makes it hard to claim the approach “works” for code RAG in a meaningful way.

- Dataset limitation: All evaluations are conducted on a self-created Python dataset, not on established benchmarks such as CrossCodeEval or RepoEval. This leaves both generalization and fairness in doubt.

- Questionable compression validity: Code logic and dependency resolution are highly structured; compressing an entire retrieved snippet into a single embedding may lose critical syntactic and semantic cues. The current results seem to confirm this weakness.

- Scope of results: The experiments focus narrowly on EM/ES and TTFT. There is no human or task-level evaluation of completion usefulness or correctness, which matters for latency–quality trade-offs in IDEs.

- No comparison to advanced baselines: The paper compares only to “with” and “without context” Qwen baselines, missing broader comparisons to more sophisticated retrieval or context-pruning methods.

### Questions
- Why was evaluation restricted to self-constructed data? How would LlavaCode perform on standard repo-level benchmarks like CrossCodeEval or RepoEval?

- Could the authors show qualitative examples where compressed context helped or failed, to better illustrate interpretability?

### Soundness
2

### Presentation
3

### Contribution
3
