# SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training, or imposing significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), a self-supervised alignment approach for enhancing faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with context versus without context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burdens. 
We analyze this faithfulness alignment process and provide empirical evidence that it leverages a benign form of likelihood displacement, shifting probability mass from parametric-based tokens to context-aligned tokens.
Based on this insight, we adapt the DPO loss using a weighting scheme that encourages likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art results in faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness while preserving general instruction-following capabilities. We release our code at: https://anonymous.4open.science/r/SSFO

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on improving LLMs’ ability to follow context knowledge rather than relying on parametric knowledge. Based on DPO, the authors propose a self-supervised alignment approach, and further provide an analysis using likelihood displacement to explain why this method improves performance. They then introduce Shift-DPO to further enhance the results.

### Strengths
1. The proposed method outperforms the baselines without requiring any external ground-truth data.


2. The experiments cover multiple model families, demonstrating the generalization ability of the method.


3. The attribution of the performance gains to the benign application of likelihood displacement is insightful, and the corresponding analysis is reasonable.

### Weaknesses
1. Self-DPO is not a new technique, and similar approaches have already been explored. For example, [1] applies self-DPO to improve truthfulness.


2. The further proposed Shift-DPO is also introduced in prior work. Additionally, the paper claims that Shift-DPO “leads to a more pronounced suppression of the likelihood of the parametric response during optimization.” However, this seems inconsistent with the stated motivation of enabling better likelihood displacement, and the connection between the two is unclear. This point is therefore confusing.

3. Including results on additional model sizes would provide stronger evidence for the generalizability of the proposed method.


[1] Chen, Weixin, Dawn Song, and Bo Li. "GRATH: Gradual Self-Truthifying for Large Language Models." Forty-first International Conference on Machine Learning.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Self-Supervised Faithfulness Optimization (SSFO), a RAG alignment method that creates a preference dataset by contrasting a model's own context-based generation (preferred) against its parametric-only generation (dispreferred). The model is then aligned using DPO. The authors posit this induces a "benign likelihood displacement," shifting probability from parametric tokens to context-aligned tokens and introduce the hyper-parameter to encourage this process. The paper claims state-of-the-art faithfulness, strong robustness to conflicting knowledge (e.g., NQ-Swap, MemoTrap), good generalization, and high data efficiency from only hundreds of training examples.

### Strengths
- The method's primary strength is its simplicity. It avoids costly human or teacher annotation, and achieving significant gains from <1000 self-generated examples is a notable practical contribution.
- The "benign likelihood displacement" concept provides a comfirmation of the method prioritizes contextual information over parametric memory.
- The method demonstrates empirical gains on benchmarks specifically designed to test robustness against conflicting internal knowledge (NQ-Swap, MemoTrap). It also shows that SSFO generalizes well cross-lingually and preserves general instruction-following capabilities.

### Weaknesses
- The entire SSFO framework is built on the critical, unstated assumption that the retrieved context is always the source of truth. The method explicitly trains the model to suppress its (potentially correct) internal knowledge in favor of any provided context. This is a major blind spot. In realistic RAG scenarios involving noisy, irrelevant, or factually incorrect context, SSFO would likely amplify this critical failure mode, forcing the model to "faithfully" repeat misinformation. The paper fails to evaluate this obvious scenario or acknowledge this foundational weakness. There is an entire line of works on dynamic contextual faithfulness ([Huang et al., ICLR 2025](https://openreview.net/forum?id=K2jOacHUlO), [Wang et al., ACL 2025](https://aclanthology.org/2025.acl-long.1476/)), which is ignored in the paper.
- The paper omits the exact prompt template used for the "Instruct Model" baseline. This is a crucial omission, given that SSFO's goal is to force strict adherence to the context, it is plausible that its entire training effect could be replicated by a well-engineered prompt (e.g., "Using only the provided context, answer the following..."). The authors do not provide evidence that SSFO achieves an optimization benefit beyond what simple prompting could accomplish, which would significantly weaken the paper's contribution.
- The empirical evaluation against other baselines is also undermined by several issues. Several baseline results are abnormally low (e.g., SCOPE on Mistral-7B, CAD on Llama-3-8B) without explanation, casting doubt on the implementation's correctness and, consequently, the validity of SSFO's superiority. Although the self-training protocol is a core proposal, the fact that the baselines are trained on different data makes the comparison even harder to interpret.

### Questions
- There are widespread citation format errors (e.g., inline parenthetical `(2020)` instead of `\citep{...}`).
- Typo in Table 3: "Crose-language".
- Formatting in Table 1: The "Human/Superior AI Supervision" column has excessive horizontal space, making the other content too small to read.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Self-Supervised Faithfulness Optimization which is a post-training method for RAG that builds preference pairs from the same model by contrasting answers generated with retrieved context against answers generated without context. The pairs are optimized with DPO, and analyzed empirically supported mechanism is "benign likelihood displacement", shifting probability mass from parametric tokens to context aligned tokens. 
The authors also propose a simple variant of SSFO-Lamda which unweights the pressure to suppress the parametric response in the DPO objective. Across different models (llama, qwen and mistral), both the proposed methods show improvement in Span-EM and long form metrics, with no extra inference time costs and using only hundreds of self generated pairs.

### Strengths
1) The overall idea is simple, yet gives a strong empirical payoff, when doing self supervised the preference pairs are easy to generate and avoid costly human labels, this directly aligns to a context adherence principle. This reduces supervision and avoids extra inference burden.  
2) The paper explains why encouraging displacement can be beneficial when preferred examples are silver, then introduces SSFO-lambda that rescores the DPO objective so the gradient puts stronger negative weight, which is a strong motivation with has a broad application in the current field. 
3)  ​Results in Table 1 shows how SSFO-lamda outperforms strong decoding strategies and post training baselines on both robustness and quality response for diverse models (llama and qwen)
4) The authors also show the data efficiency of the proposed method, and how only 400-500 pairs can achieve upto 85% of the total gains over the instruct baselines.

### Weaknesses
1) Results depend on retrieval quality, and the paper states a “standard RAG prompt” and datasets, but does not detail the retriever configuration or ablations to retrieval quality. 
2) The authors report LFS which uses GPT-4 with a provided prompt, while standard, it introduces judge bias and lacks calibration against human labels or alternative factuality metrics, and there is no reliability analysis

### Questions
1) The paper compares against key faithfulness methods but omits other displacement-aware alignment baselines like DPO-shift and AlphaPO in experiments even though they are mentioned, adding these results in the table could strengthen their claims. 
2) Consider adding RAGTruth which is cited in the paper, to show faithfulness under human-curated hallucination settings.
3) The paper misses an explicit evaluation under retrieval noise (distractor setting), missing or contradictory contexts, long-context settings, such experiments could strengthen the claims of the paper.

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
The paper introduces SSFO, a post‑training alignment method for RAG that creates preference pairs without human or stronger‑LLM supervision. For cheaper training, SSFO constructs preference data pairs by contrasting the model’s outputs generated with context versus without context. The model is then optimized with a DPO‑style objective that widens the margin between these two responses. They show that their methodology improves the performance on several short‑form QA datasets (NQ‑Swap, MemoTrap, NQ‑Open, SQuAD) and long‑form metrics on ELI5/WikiPassageQA, preserves instruction following (FollowBench), and shows cross‑lingual generalization (XQuAD‑ES, DuReader‑ZH), often with only hundreds of self‑generated training pairs.

### Strengths
- They propose a self-supervised and efficient training method. Preference pairs are self‑generated, and only hundreds of examples suffice.
- They conduct a systemic evaluation with several benchmarks from various aspects, including Robustness, Response Quality, Cross-language Response Quality, and Instruction Following Ability. 
- They prove that their method is effective across LLMs.
- They not only propose the effective training method but also try to explain why their method works through benign likelihood displacement.

### Weaknesses
Actually, I don't see many weaknesses in this paper. However, one question is about how the authors ensure that the no-context generations are indeed dispreferred responses. In some cases, no-context generations might still be factually correct and meaningfully not different from the context-grounded response. The paper could be clearer about how it guarantees that these no-context responses are dispreferred and meaningfully distinct from context-grounded responses.

Moreover, they use GPT-4 as LLM-as-Judge to calculate LFS. This might introduce some LLM bias during the evaluation. Did you check some responses manually to verify the validity and consistency of GPT-4? 

It's minor, but for readability, please use "\citep" for citations.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
