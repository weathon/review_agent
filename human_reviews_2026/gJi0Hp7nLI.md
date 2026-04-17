# Information Flow Reveals When to Trust Language Models

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large language models (LLMs) have emerged as powerful tools for real-world applications, but their utility is often undermined by a fundamental flaw: a tendency toward overconfidence and guessing that leads to unreliable responses. This issue is particularly critical in retrieval-augmented generation (RAG), which is explicitly designed to provide factually grounded answers with retrieved context. Current approaches to quantifying LLM uncertainty are often inadequate, as they rely on surface signals from either the input embeddings or the output space, such as token probabilities or semantic consistency across multiple generations. This work opens the black box of transformers and assesses response reliability by analyzing the information flow within language models. Specifically, we uncover the contributions of context tokens to the generated output, providing an interpretable basis for evaluating reliability. From this analysis, we introduce two measures. The first, simulatability, assesses the alignment between the context token contributions and their relevance, and the second, concentration, quantifies the extent to which a response's support stems from a narrow subset of tokens. Our experiments demonstrate that these information-flow signals offer a more effective and interpretable basis for assessing reliability than existing methods, outperforming baselines across multiple metrics and advancing the development of more trustworthy LLM deployments. Meanwhile, we also discuss computational considerations and our method’s application scope.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an uncertainty-aware confidence estimator for RAG that looks inside a transformer to quantify how much each context token contributes to a predicted answer. The authors compose per-layer contribution matrices to attribute the answer to specific context tokens, yielding two signals—simulatability (alignment with a reranker’s token-level relevance) and concentration (sharpness of support). A light calibrator combines these with the reranker score to predict confidence. On LLaMA-3.2-3B-Instruct model, it outperforms standard UQ baselines and shows better calibration.

### Strengths
* The core idea of estimating confidence by tracing the model’s internal information flow rather than just looking at outputs or inputs is interesting, and the token-level explanations are intuitive to read and discuss.
* The two signals work well together: alignment with an external relevance view plus how narrowly the model concentrates its support, and this combo translates into clear gains in discrimination and calibration on a standard QA benchmark.

### Weaknesses
* The core signal (“simulatability”) is defined by how closely the model’s internal attributions match a reranker’s token relevance, so both the metric and the final predictor depend on the reranker—creating a dependency loop that inherits the reranker’s biases. Some potential tests:
    - In addition to §5.3, swap in a very different reranker or two and see how much the confidence quality moves.
    - Build a small human-labeled “relevance layout” set and compare the model’s simulatability to humans vs. to the reranker.
    - Vary the rank-agreement hyperparameters (e.g., how top-heavy the agreement metric is) and see if conclusions flip.
* A large share of the reported gains may come from the reranker’s single scalar score rather than the new information-flow features; the paper does not show a clean ablation of “reranker-only” vs. “proposed features without the reranker,” which is needed to prove incremental value.
* The evaluation is narrowly scoped.
    - The number and variety of models tested are very limited.
    - The authors only use very short answers on single-passage SQuAD) and never tests a true end-to-end RAG setup with multi-document retrieval and distractors, so it’s unclear whether the benefits hold in realistic settings. 
     - Baselines are not on equal footing: the proposed method uses a tuned post-hoc calibrator while several baselines appear “as is.” Applying the same calibrator to all methods (and reporting both raw and calibrated results) would make the comparison fair.
* Faithfulness is assumed rather than demonstrated; there are no causal tests (e.g., ablating top-ranked vs. low-ranked tokens) to show that the extracted token flows actually drive correctness rather than merely correlate with it.

### Questions
* What are the runtime/memory costs for building/composing contribution matrices at 2k–8k context lengths? How does the method affect inference speed/latency?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this submission, the authors introduce a series of novel criteria that offer a fresh perspective on uncertainty quantification for LLMs. By leveraging established concepts such as information flow and contribution matrices, the authors present a framework for assessing the relative importance of input tokens with respect to model outputs. The proposed measurement captures both the emergence order and contribution layout of input tokens, thereby providing deeper insight into the relationship between inputs and outputs and enabling an estimation of response confidence.  
The experimental results include comparisons between the proposed method and several existing UQ baselines on a specific dataset. The findings suggest the proposed method exhibits certain advantages over baselines.

### Strengths
- The authors introduce a novel perspective by leveraging information flow to reveal the importance of input tokens.
- The workflow for calculating contribution matrices and deriving the subsequent uncertainty quantification criteria is clearly described and easy to follow.
- The experimental results demonstrate the improvements of the proposed method compared to baselines.

### Weaknesses
- The motivation for applying information flow in uncertainty quantification is not clear. The authors are encouraged to provide further discussion on this point. For example, what specific advantages are brought by introducing information flow, and what makes the proposed method outperform existing studies in uncertainty quantification tasks? Does applying information flow improve the generalizability and robustness of uncertainty quantification, or make it applicable to a wider range of scenarios?
- While the proposed method is demonstrated in the context of RAG, the experimental setup is limited to a single base model and one dataset. As a result, the experiments in this paper are somewhat limited and not fully convincing.
- The writing of this submission should be further polished. For example, the inconsistent usage of $\mathbf{y_i}$ and $\mathbf{y}_i$ in Equation 5.

### Questions
Please refer to the weaknesses.

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
2

### Summary
The paper proposes an uncertainty quantification framework for RAG that traces the information flow through attention and residual connections. Two main constructs are introduced, emergence order which is a greedy backward extraction of a principal flow that ranks input tokens by the order they enter dominant computation, and contribution layout which is a path composed summary that aggregates token to token contributions across layer. 
These are compared against a a token level relevance layout to yield simulatability, and concentration. A learned calibrator combines these with a context-level relevance score to predict answer reliability. They show performance across different benchmarks and show outperforming several output space and input aware UQ baselines.

### Strengths
1) The authors present the paper which is methodologically clear and grounded, the paper recasts multi-head attention to define per-layer contribution matrices via an attribution vector and a normalized distance, forming lower-triangular structures consistent with causal masking. This provides a principled basis to trace token interactions across layers
2) A clear, constructive definition of Emergence Order via a greedy backward search over a selection pool yields a ranked, rather than thresholded, importance ordering which improves resolution over prior binary criteria
3) The approach improves AUROC and shows better calibration relative to Perplexity, P(True), (Semantic) Entropy, and two input-aware baselines

### Weaknesses
1) All core results are on SQuAD2.0 with a single base model (LLaMA-3.2-3B-Instruct) and short answers. This setting is comparatively “clean,” with relatively local evidence and it lacks representing the multi hop reasoning, long contexts, noisy retrievers and OOD drift, which are all common in RAG applications. 
2) The simulatability requires a token-level ground truth relevance layout created by shapley decomposing a rerankers single relevance score. This can create a potential circularity and bias, if the rerankers inductive bias differ from the QA models, simulatability may reward mimicry of the reranker rather than grounding.

### Questions
1) Can you also report the computation per-layer contribution matrices and their product, which could be expensive in memory.
2) Results reported are point estimates without confidence intervals, variance, or even seeds. Given a modest AUROC detas, formal uncertainty estimation is essential.
3) Minor typo - Line 224: "embeddning"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper uses mechanistic interpretability sorts of ideas to propose what I believe is a method for white-box confidence estimation, as it relies on having access to the model weights. The work proposes what it terms “simulatability”, which assesses the alignment between context token contributions and their relevance, and “concentration”, which is measure of how many tokens the model relies on for its generation. From my understanding, these can be seen as features for calibrating confidence. The work is positioned as UQ for RAG problems, but I felt that it could apply more broadly. Maybe I did not fully appreciate the RAG aspect in the paper, which is possible. Limited experiments are performed on Squad 2.0 to show how the confidences perform on standard prediction metrics such as AUROC and AUPRC as well as others.

### Strengths
While I am not an expert on mechanistic interpretability, I found it interesting to see the process by which features were derived from the underlying attention mechanism of transformers.

The issue of UQ for RAG seems useful but I could not follow the specific challenges in this direction. Perhaps a revised version of the paper can clearly describe the scope of problems that the work is particularly suitable for.

### Weaknesses
I found experimentation to be lacking. I understand that it is tricky to do this for many models, particularly large ones, but one model and one dataset is not enough to gauge the value of the method, in my view. I also generally did not fully appreciate the exact way that ground truth was gauged, forming the basis for evaluating confidence of responses. How is an answer deemed correct? 

I’m concerned about the choice of baselines and find them somewhat lacking. Did the authors consider using features from logits (like G-NLL from this work: https://arxiv.org/abs/2412.15176) or similarity features (like from this work: https://www.arxiv.org/abs/2510.13836) to calibrate? In other words, what about using other scores/features to train a model with training data? Also, did they consider other white-box methods that use model weights?

I found the method to be quite involved, without enough justification for what was going on with all the transformations, normalizations, distributional distance computations, etc. It was hard to follow. So much is going on and it’s unclear that it’s that valuable in that end – at least based on the evidence in the submission.

Having access to model weights makes the method quite limiting in my view. Much more needs to be mentioned about the severe limitations of the work. There is a reason why there is so much attention in the literature on methods that are either black-box or gray-box (using token logits).

Please note that my limited understanding is reflected in my low confidence. I’m open to changing my opinion as I read other reviews and participate in the discussion.

### Questions
Here are some additional comments and questions:

As noted previously, I had a hard time following evaluation. A brief and clear explanation of how correctness of responses is deemed would be appreciated.

I recommend mentioning the limitations of the work in sufficient detail, including the abstract. Issues are raised about competing methods but not about the proposed method, which I think is misleading.

The citation of Vaswani et al. for LLMs is strange. I suggest modifying this line to make the citation about transformers and not LLMs.

The authors say that all the cited work is black-box, but that is not true – some are gray-box methods using logits. Not enough white-box methods, including classic ones on UQ, are cited in the paper.

I had a hard time following the import of the P^{total} computation on page 5, and also much of the content on page 6.

Is the relevance score “r” used for evaluation? On page 7, it seems it is also used as a feature?

### Soundness
2

### Presentation
2

### Contribution
2
