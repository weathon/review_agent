# AdaptiveResidual: Inference-Time Trust Calibration for Contextual Knowledge Injection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
In modern large language models (LLMs), injecting external knowledge via the context to guide models' outputs toward desired outcomes (e.g., through RAG) is a standard practice. 
However, recent research reveals that once conflicts arise between the contextual information and the internal parametric knowledge, LLMs tend to underutilize the external evidence, leading to unreliable or even contradictory outputs. 
This raises a fundamental question: *how can we dynamically reconcile these knowledge conflicts to ensure faithful integration of contextual information* ? 
Inspired by mechanism interpretability findings that identify the `Attention` module as the primary aggregator of external context and the `FFN` module as the locus of internal knowledge lookup, we pinpoint the vanilla residual pathway as the crucial junction where these two information streams are integrated. 
Based on this insight, we introduce AdaRes (*Ada*ptive *Res*idual), a lightweight, parameter-free trust calibration mechanism that operates at test-time. 
Specifically, AdaRes recalibrates the standard residual connection to dynamically balance the influence of external knowledge (from `Attention`) and internal knowledge (from `FFN`). 
This balancing is guided by two instance-specific "trust scores", which are calculated on-the-fly by probing how much the input query relies on contextual versus parametric knowledge sources. 
By adaptively reweighting these contributions without altering any model parameters, AdaRes effectively mitigates knowledge conflicts. 
Experiments on different benchmarks verify the effectiveness of AdaRes in regulating contextual and parametric knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes AdaRes (Adaptive Residual), a lightweight, training-free inference method for dynamically reconciling knowledge conflicts in large language models (LLMs). The method reparameterizes the residual connections in selected layers to adaptively balance the influence of contextual knowledge (from attention) and parametric knowledge (from the feed-forward network) at some heuristically chosen layers. Experiments on knowledge editing benchmarks showcase the effectiveness of the proposed approach.

### Strengths
- The paper was well-written and easy to follow.
- The paper provides very detailed explanations of its experiments, fostering reproducibility.

### Weaknesses
1. *Methodological justification is weak.*
The core assumption that the entire attention module represents contextual knowledge while the entire MLP (FFN) module represents parametric memory is insufficiently supported. The paper provides no empirical or theoretical evidence for this decomposition. Prior work has shown that attention layers can themselves act as associative memory mechanisms [1, 2], directly challenging this simplification. Moreover, Equation (4) implicitly assumes that all contextual information is trustworthy, which rarely holds in realistic settings. Although the authors categorize four types of knowledge conflict in Figure 3, they only address the “context-preferred” case (Scenario #4), leaving the other scenarios unhandled by the proposed formulation. This narrow scope leads to potential overclaiming of generality. In addition, the use of dynamically computed trust values $\alpha$ and $\beta$ lacks clear motivation or theoretical/empirical grounding. The paper does not explain why these specific scaling forms are appropriate or how they relate to the underlying model dynamics, making the mechanism appear ad hoc.
2. *Limited novelty and contribution.*
The paper’s conceptual framing overlaps substantially with existing literature on scaling and analyzing attention heads and FFN modules (e.g., works in [3]). While the implementation is lightweight, it does not introduce new insights or mechanisms that significantly advance the mechanistic interpretability’s understanding of contextual–parametric interactions.
3. *Experimental design and evaluation issues.*
The experimental setup raises several concerns. Although the work claims to address knowledge conflict, it evaluates primarily on knowledge editing benchmarks, which are conceptually distinct. (1) Datasets: For the contextual-conflict case (Scenario #4), standard benchmarks such as NQ-Swap [4] and Memo-Trap [5] should be included. If the paper intends to cover other conflict types (e.g., Scenarios #1 and #2), corresponding datasets should also be used; otherwise, these discussions should be removed for focus and clarity. (2) Baselines: In Table 1, the comparison set omits a number of decoding-based methods explicitly designed to mitigate contextual hallucination and knowledge conflict, such as [6-10] and there are more missing ones. Including these would provide a fairer and more meaningful evaluation.
4. *Key related works are missing*. [11] also discusses the role of MLP and attention in much more detail, and [12] shows that intervening the entire attention module could lead to superposition. Both works can reconcile knowledge conflicts in both Scenario 1 and 4 with only intervening in the attention module. These omissions weaken the contextualization of the proposed approach and raise questions about its incremental contribution.


[1] Memorization capacity of multi-head attention in transformers. ICLR'23

[2] Understanding factual recall in transformers via associative memories. ICLR'25

[3] Attention Heads of Large Language Models: A Survey. ArXiv'24

[4] Entity-Based Knowledge Conflicts in Question Answering. EMNLP'21

[5] https://huggingface.co/datasets/Albertmade/memo-trap

[6] Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. NAACL'24

[7] Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation. EMNLP'25

[8] Sled: Self logits evolution decoding for improving factuality in large language models. NeurIPS'24

[9]  Dola: Decoding by contrasting layers improves factuality in large language models. ICLR'24

[10] AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge. NACCL'25

[11] Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in Language Models. ACL'24

[12] Taming Knowledge Conflict in Language Models. ICML'25

### Questions
Aforementioned in the Weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces an inference method for adapting the reliance of an LLM's output more on the context provided, as opposed to its internal knowledge. The key idea is that the residual connection which adds the attention outputs to the FFN outputs can serve as a modulator between the two, with the attention output serving as a proxy for the reliance on the context. Empirical results on Qwen and Llama models in the 7-8B range demonstrate that this can indeed improve the contextual dependence of the model responses.

### Strengths
- The observation that residual connections can serve as a modulator of context vs internal knowledge dependence is, to my knowledge, new and quite interesting.
- The paper provides quite extensive empirical results in terms of both the models and datasets, as well as the hyperparameter choices and other variables involved in the methods.
- The paper is well written and quite easy to follow, even if it is unnecessarily math-y when describing the methods.

### Weaknesses
- Section 3.2.1 for trust estimation for \alpha seems to be described incorrectly. In the formulation presented, \alpha is computed as an average of the per-row softmax outputs of query -> context attention. But softmax outputs sum to exactly 1, so it is not clear why these would sum up to anything other than 1/M. This is clearly not the case based on the results presented later, so I suspect the issue is in the description of the method.
- The intro and motivation seem to position the method as a general "dynamic" scheme for selecting between context and internal knowledge. But in practice, it only applies to the setting where the context is correct and the internal knowledge is incorrect (Figure 3). The claims would be supported more strongly if there were also experiments studying the reverse direction -- internal knowledge is correct and the context is wrong.
- On a similar note, the paper is lacking important baselines: (i) simply prompting the model to trust the context instead of its internal knowledge (after all, the trust method already assumes that the context is correct); and (ii) baselines from a very relevant paper published at ICLR 2024. (This paper is actually completely missed from the related work discussion).
- The layers used for AdaRes seem to be quite sensitive to the choice of model and dataset. There is no discussion if the layers selected for one setup will generalize to other setups, limiting the practical applicability of the method.
- Contrary to what the text claims, there seems to be quite a significant impact on the latency of the model (in some cases 50-100% slowdown in Figure 6).

### Questions
- Why are the main results in the paper on base models instead of instruction tuned ones? The latter seem more relevant for QA tasks.
- What is the vanilla res method in Table 2?

### Soundness
2

### Presentation
3

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
AdaRes (AdaptiveResidual) is a training-free, inference-time trust calibrator to resolve knowledge conflict in LLM parametric knowledge and contextual knowledge. . It probes each layer on-the-fly to compute two “trust scores” (query-to-context attention and FFN memory affinity), then asymmetrically rescales the residual contributions to prioritize the more trustworthy source; the only hyperparameter is which layers to apply it to (chosen by a simple greedy search).  Across conflict-centric evaluations (ZsRE, CounterFact, ConflictQA variants), AdaRes strongly improves adherence to the supplied context and preserves locality, often outperforming editing and parameter-editing baselines

### Strengths
1. Resolving knowledge conflicts is a timely and important problem. 
2. The paper’s methodology is presented and written clearly, with a well-structured description that makes the approach easy to follow.

### Weaknesses
1. **Problem Scope**. While the paper is framed as resolving knowledge conflicts, in practice it mainly addresses how to make LLMs more faithful to external contexts, that is, how to prioritize retrieved evidence over internal memory. This effectively reduces the problem to *enforcing context faithfulness rather than truly deciding between conflicting knowledge sources*. The more interesting challenge is how to determine which side deserves trust; if we already assume the external context is more reliable, the task becomes much simpler. In that case, one might wonder why not simply optimize the prompt or training objective to explicitly instruct the model to follow the context. The long-context scenario might make this harder, but benchmarks used in the paper (e.g, ConflictQA) involve short passages that are far below the model’s context limit.

2. **Missing benchmarks**. There exist several datasets [1, 2, 3] that explicitly evaluate knowledge conflict resolution, but the paper only reports results on ConflictQA, while the rest are knowledge editing benchmarks, which only partially capture the intended problem.

3. **Missing baselines.** Numerous prior works, both prompting-based and training-based, directly tackle knowledge conflict resolution, yet none are included as baselines [3, 4, 5, 6]. The authors should either compare with or at least discuss why these methods were omitted.

[1] “ClashEval: Quantifying the tug-of-war between an LLM’s internal prior and external evidence”, NeurIPS 2025  \
[2] “FaithEval: Can Your Language Model Stay Faithful to Context, Even If 'The Moon is Made of Marshmallows'”, ICLR 2025 \
[3] “To Trust or Not to Trust? Enhancing Large Language Models' Situated Faithfulness to External Contexts”, ICLR 2025 \
[4] “KnowPO: Knowledge-aware Preference Optimization for Controllable Knowledge Selection in Retrieval-Augmented Language Models”, AAAI 2025 \
[5] “FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation”, ACL 2025 \
[6] “Trusting Your Evidence: Hallucinate Less with Context-aware Decoding”, NAACL 2024

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes AdaRes, a parameter-free, inference-time mechanism to address knowledge conflicts between an LLM's internal (parametric) knowledge and external (contextual) information. The method is inspired by interpretability findings that identify the Attention module as the context aggregator and the FFN as the knowledge store. AdaRes calculates instance-specific "trust scores" for each source ($\alpha^{(l)}$ for context, $\beta^{(l)}$ for parametric) and uses them to reweight the respective contributions within the residual pathway. The method reports empirical results on knowledge editing and conflict-aware QA benchmarks.

### Strengths
The paper introduces a novel, training-free mechanism to address the critical problem of knowledge conflicts in LLMs, which is highly relevant for improving the reliability of RAG. The core idea is well-motivated by mechanistic interpretability findings, specifically the distinct roles of the Attention (context) and FFN (parametric) modules.

### Weaknesses
- The method's design is heavily "context-first" (as seen in the focus on Scenario #4) and seems to require a priori knowledge that the context should be trusted. This is a significant limitation that is not clearly acknowledged. A simple but crucial baseline is missing: prompt engineering. The results in Table 3 show low performance for the "Original" baseline even on instruction-tuned (it) models, suggesting that a simple, well-crafted prompt to "follow the context" was not explored as a point of comparison. The "IKE" results hint on a positive role of the instruction for Qwen 2.5. The actual prompts used are not disclosed, which harms reproducibility.
- The paper's description of the methodology is lacking. Important details about the FFN probing mechanism (for $\beta^{(l)}$ estimation) are relegated to the appendix. Furthermore, a "Top-n" selection is depicted in Figure 2 but never explained in the text, and it appears to be a hyperparameter. This contradicts the claim that the set of target layers $\mathcal{H}$ is the "sole hyperparameter" (line 229). The description of the context trust estimation ($\alpha^{(l)}$) is also unnecessarily convoluted.
- The submission is missing highly relevant citations in its related work (Section 2.2) regarding dual-response or fusion strategies. For example, [Huang et al. (ICLR 2025)](https://openreview.net/forum?id=K2jOacHUlO) addresses the identical problem of "dynamically calibrating trust" to "resolve knowledge conflicts" and should be discussed.
- The claim of "negligible runtime cost" is questionable. Algorithm 1 and the three-stream design imply that at least one additional, full forward pass is required for the probes. The significant increase in inference time in Figure 6 demonstrates this cost, while the paper downplays it.

### Questions
- How is AdaRes intended to operate when it is not known a priori whether the context or the parametric knowledge is correct? What happens if it is applied in Scenario #2 (correct model, wrong context)?
- Can you please disclose the prompts used for the "IKE" baseline? A comparison against a strong, instruction-based prompt (e.g., "Follow the context provided exactly") seems like a critical and missing baseline.
- Please clarify the "Top-n" selection from Figure 2. Is this a hyperparameter, and how was it set? This contradicts the claim that the layer set $\mathcal{H}$ is the sole hyperparameter.

### Minor Issues

- Table results (e.g., Table 1) would be more readable as percentages.
- Tables 2 and 3 are difficult to parse; adding a separate header row for model labels (e.g., Phi3, Gemma3) would improve clarity.
- There are minor typos (e.g., "an" -> "a" in line 248).

### Soundness
2

### Presentation
3

### Contribution
3
