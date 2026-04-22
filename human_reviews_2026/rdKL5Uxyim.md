# Expert Heads: Robust Evidence Identification for Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Large language models (LLMs) exhibit strong abilities in multi-document reasoning, yet their evidence identification is highly sensitive to input order. We trace this limitation to attention mechanisms, where many heads overemphasize sequence boundaries and neglect central content.
We systematically analyze attention distributions under document permutations and discover a small subset of heads that consistently prioritize task-relevant documents regardless of position. We formalize these as Expert Heads, identified via activation frequency and stability across permutations.
Experiments on LLaMA, Mistral, and Qwen reveal architecture-specific patterns: mid-layer heads in LLaMA and Mistral dominate semantic integration, while deeper-layer heads in Qwen specialize in evidence selection. Moreover, Expert Heads exhibit concentrated focus during understanding and more distributed engagement during generation. Their activation strongly correlates with answer correctness, providing diagnostic signals for hallucination detection.
Leveraging Expert Heads for document voting significantly improves retrieval and ranking on HotpotQA, 2WikiMultiHopQA, and MuSiQue, outperforming dense retrievers and LLM-based ranking with minimal overhead. Ablations confirm that even a small subset achieves robust gains.
Our findings establish Expert Heads as a stable and interpretable mechanism for evidence integration, offering new directions for context pruning, hallucination mitigation, and head-guided training of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how attention mechanisms in Large Language Models handle evidence identification across multiple documents, with particular focus on their sensitivity to input order. The authors identify a small subset of attention heads termed "Expert Heads" that consistently focus on task-relevant documents regardless of their position in the context. They demonstrate that these Expert Heads show different layer-wise distributions and can be leveraged to improve document retrieval and ranking tasks.

### Strengths
The identification of architecture-specific patterns (mid-layer specialization in LLaMA/Mistral vs. deeper-layer specialization in Qwen) provides valuable insights into model behavior.

### Weaknesses
1. Threshold Selection Appears Ad-Hoc: While Table 2 shows different thresholds for different models/sources, the paper doesn't provide principled guidance for selecting these thresholds in new settings. The choice of τ_f = 0.6 and τ_p = 0.9 seems empirically driven without theoretical justification.
2. Only using the HotpotQA training set may not be sufficient to establish robust patterns, and using multiple randomly sampled datasets may be better.

### Questions
How do Expert Heads identified on HotpotQA transfer to other datasets? Can Expert Heads be identified once and reused, or must they be recomputed for each new task/dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the order sensitivity of evidence identification in multi-document reasoning with LLMs. The authors observe that a small subset of attention heads consistently focuses on gold documents across input permutations and define these as **Expert Heads**. They report architecture-specific layer patterns (e.g., LLaMA/Mistral peaking in mid layers, Qwen in deeper layers) and claim strong P@2/NDCG/MAP on HotpotQA/2Wiki/MuSiQue by using Expert Heads to identify and rank gold documents among 10 candidates.

### Strengths
- Simple method validated across multiple architectures, with clear analysis and visualization.
- Consistent improvements are shown across multiple families (e.g., LLaMA, Mistral, Qwen).  
- Consistent improvements on HotpotQA/2Wiki/MuSiQue under the proposed evaluation.
- Implementation details (code and prompts) aid reproducibility.

### Weaknesses
- The selection of Expert Heads relies heavily on supervision from a HotpotQA training subset. This raises a concern that the selected heads may overfit to HotpotQA-specific structure, style, and difficulty, potentially overstating cross-domain generalization. Although you report results on 2Wiki/MuSiQue, the paper lacks evidence that the head-selection procedure itself can be reproduced stably outside HotpotQA.
- The authors fix the condition of having two adjacent gold documents, which is a strong assumption. Does the method still work when gold positions are more flexible (e.g., non-adjacent)?
- The number of candidate documents is fixed at 10; ablations that vary this number are missing.
- The hyperparameters are effectively retrofitted to match the target outcome (the number of selected heads), which weakens claims about generalization.
- Please state the data sizes and the precise train/validation/test splits explicitly.
- While the proposed method substantially outperforms the baselines, I worry that the baselines may be too weak.
- Ranking based on Response→Document attention is only available after full-context generation, making it ill-suited for pre-filtering (cost reduction). Despite this, Expert(R) often yields the best results in the main tables; this is useful for offline analysis but offers limited benefit in practical deployments.
- Several notation/formatting issues remain; please see the notes below and incorporate any that are helpful.

### Questions
- Do the heads selected as Expert Heads change depending on factors such as document length or topic differences within the dataset?
- How does accuracy vary with the size of the training data used for selecting Expert Heads?
- If the candidate documents vary widely in length, does the proposed algorithm remain robust?
- You evaluate 7B and 8B models; what differences do you observe for larger models?
- Your experiments target retrieval rather than QA; if applying the proposal to QA, would it be used in a two-stage pipeline—retrieving supporting documents -> generating the answer with the LLM?
- How do you explain that Expert Heads (R) generally outperform Expert Heads (Q)?
- At line 361 the HotpotQA test size is reported as 2,269, whereas Figure 6 says “HotpotQA test (2,323 samples).” Which number is correct?
- If the training data changes, do the selected Expert Heads change substantially?

(Other Comment)
- Line 108: Consider clarifying whether $D$ denotes ``document tokens''; the current notation is ambiguous.  
- Line 108: The symbols for “document $D$” and “distractor document $D$” collide; please disambiguate.  
- Line 111: Define $l$ (layer) and $h$ (head) explicitly.  
- Line 138: Improve clarity by illustrating gold–distractor relatedness in HotpotQA (e.g., with a concrete example or similarity scores).  
- Line 245: Add a definition for $P_{\tau_p}(\cdot)$.  
- Figure 6 (right): The x-axis is discrete; a line plot may mislead—consider bars or points.  
- Numerals: Add thousands separators to 4+ digit numbers (e.g., 2,269).  
- References: Prefer final proceedings over arXiv where available (e.g., Ren et al., 2024 to ACL Findings). Unify citation formats for Mistral and Qwen.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an empirical investigation into the functional specialization of "expert heads" within a deep learning architecture (presumably a multi-headed attention mechanism). The authors identify emergent specialization patterns and analyze how these specialized components contribute to the model's performance. The core contribution lies in the detailed empirical observations demonstrating this specialization and the subsequent high performance achieved on specific benchmarks, primarily those focused on precision.

### Strengths
- The paper provides interesting and potentially valuable empirical data regarding how specialization emerges and functions. These observations offer a foundation for future theoretical work or architectural improvements.
- The analysis clearly demonstrates the effectiveness of the expert heads in scenarios where minimizing false positives is the priority.

### Weaknesses
- The most significant weakness of this work is the narrow scope of the evaluation, which relies almost exclusively on precision. This provides an incomplete picture of the model's efficacy. In many critical applications (_e.g._, anomaly detection), recall (sensitivity) is equally or more important than precision. By neglecting recall or balanced metrics (e.g., F1-score, PR-AUC), the paper fails to demonstrate the overall utility of the expert heads.

### Questions
- Could the authors provide a holistic evaluation by including Recall and F1-scores for all experiments? Furthermore, presenting the Precision-Recall curve (PR-AUC) would significantly strengthen the evaluation.
- I would also recommend the authors include experiments on tasks specifically designed to test high-recall performance. How do the expert heads perform when the loss function or evaluation criteria are optimized for recall rather than precision?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates attention distributions in LLMs under document permutations in multi-hop question answering setting. They conduct an analysis of attention patterns with LLama3, Mistral and Qwen2.5 models, and then identify a small set of attention heads, named expert heads, that consistently attend to task-relevant documents across all permutations based on activation frequency and average attention score. Their experiments suggest that the layer-wise distribution of expert heads varies across different model architectures, and the activation of expert heads is correlated with answer correctness. Additionally, they leverage expert heads to rank candidate documents on HotpotQA, 2WikiMultiHopQA and MuSiQue and show that expert heads are effective for document retrieval and ranking.

### Strengths
- They conduct extensive analysis of attention patterns under document permutations with three model families, providing interesting insights into the cause of position bias and attention behavior across architectures.
- Their method to identify expert heads is clearly defined and reasonable.
- The paper is clearly written and easy to follow. Figures are well presented and intuitive to understand.

### Weaknesses
Overall the paper is well-structured. It would be clearer to add some clarification about the experiment setup and evaluation metrics used in the document ranking experiments.

### Questions
- Could you explain more details about the evaluation metrics used in the ranking experiments? ie. what do you measure by Precision@2, NDCG@2 and MAP?
- In the left figure in Figure 6, what do the two dashed lines indicate? Why does the performance remain the same across different layers?

### Soundness
3

### Presentation
4

### Contribution
3
