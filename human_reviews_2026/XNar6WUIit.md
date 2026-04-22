# From Bias to Benefit: Place Good Documents in Good Positions

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) exhibit a U-shaped positional bias in processing input information, characterized by heightened attention to tokens at the beginning and end of the prompt while ignoring information in the middle, also known as the Lost-in-the-Middle phenomenon. In this paper, we investigate the internal mechanisms underlying this phenomenon by analyzing how positional bias influences attention weights across both horizontal (input-level) and vertical (layer-level) dimensions of the model. Based on these findings, we propose U-shaped Placement, a strategy that leverages inherent positional bias of the model by assigning documents to positions that align with its attention pattern. By combining this placement strategy with the importance estimations of documents, effectively placing good documents in good positions, we enhance the model’s ability to utilize documents within two iterations. Experimental results demonstrate that our method consistently outperforms existing baselines across multiple models and datasets, indicating that leveraging positional bias can bring improved document utilization capability. Our codes are submitted with the paper and will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a 2-stage method that puts more important documents on the beginning or end of the input. The motivation is to utilize the U-shaped position bias of LLMs to make the more important documents get more attention. 
In stage 1, to rank the importance of each document, it uses averaged attention weights from answer tokens. Moreover, it separates a "position bias score" to decide whether a document should be placed at the beginning or end of the input. In stage 2, LLM uses the reordered input to generate the answer.
Experiments on 3

### Strengths
1. Clear structure and easy to understand
2. Experiments on many baseline methods and datasets

### Weaknesses
1. This method is similar to baseline methods such as Attention Sorting, so the novelty is limited. The only difference is to put the document either on the beginning or the end based on a "position bias score".
2. The findings and insights are trivial for me. Most are already found by previous works about lost-in-the-middle, but these works are not cited in Related works. For example:

[1] Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization

[2] Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models

[3] Eliminating Position Bias of Language Models: A Mechanistic Approach

[4] Mitigate Position Bias in Large Language Models via Scaling a Single Dimension

3. This method relies on attention weights. However, calculating attention weights means more computing, and the inability to use FlashAttention2 in generation. This will significantly slow down the inference speed, especially when the input is very long. So it is not so practical.
4. The improvement compared to baselines is not so great.

### Questions
none

### Soundness
3

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
This paper investigates the U-shaped position bias in LLM-based RAG systems. The authors propose a new method to isolate position bias in LLMs and design a U-shaped placement strategy to organize input documents. Extensive experiments are conducted to validate the effectiveness of the proposed approach.

### Strengths
1.The paper provides a thorough and detailed study of the U-shaped position bias phenomenon.

2.The related work is well-summarized, and the connection to prior research is clearly articulated.

3.The proposed solution is simple and efficient.

### Weaknesses
1.In Step 1, the authors use attention weights to estimate document importance. However, since attention weights themselves may be influenced by position bias, it is unclear whether same documents placed in different positions would yield consistent importance scores. This potential confounding effect needs to be clarified and ideally supported with empirical evidence.

2.In Lines 193–195, when discussing the relationship between position bias and the number of documents, the authors should provide the exact document counts for each dataset. Moreover, this relationship should also be tested within the same dataset by varying the number of documents to ensure a fair comparison.

3.Regarding the finding in Lines 257–259—“positional distinctions are more pronounced in the lower layers”—this may not hold for ordered inputs. As shown in Appendix D.2, the curves for lower and higher layers appear nearly identical under ordered settings, suggesting that the distinction is much less evident.

4.In addition to Figure 3, the accuracy and stability of the calculated position biases should be further justified. For instance, when document orders are randomized multiple times, do the computed biases remain consistent? Providing more quantitative evidence and analysis of variance across random orders would strengthen this point.

5.The experiments are limited to small-scale LLMs. It would strengthen the paper to evaluate the method’s scalability on larger models (e.g., 10B or 30B parameters) to confirm its scalability.

### Questions
1.Several figures (e.g., Figures 2 and 3) are of low resolution. They should be redrawn using vector formats such as .eps or .pdf. Additionally, increasing line weight and font size would improve readability.

2.There are some typos in the paper, e.g., line 152 Dlanguage and line 758 lowercase "we".

### Soundness
3

### Presentation
2

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
This paper investigates the "Lost in the Middle" phenomenon—where models tend to assign higher attention to the beginning and end of input prompts while neglecting the middle—from the perspective of the model's attention mechanism. It constructs a position score by aggregating attention weights before and after answer tokens. Based on document importance scores and position scores, the paper proposes a strategy called U-shaped Placement, which rearranges documents to align with the model’s inherent positional bias, ensuring that highly relevant content is placed in positions that receive greater attention.

### Strengths
1. The investigation of the "Lost in the Middle" phenomenon from the perspective of the attention mechanism may better reveal its underlying causes.

2. Based on the insights gained from this investigation, the proposed U-shaped Placement strategy is supported by comprehensive experiments.

### Weaknesses
1. There is a lack of ablation studies on the token-level scores. A more rigorous validation of the rationale behind using token-level (as opposed to document-level) scores would involve replacing only the scoring granularity while keeping all other factors constant.

2. The experiments are limited to smaller-scale models, leaving it unclear whether the U-shaped Placement strategy generalizes to larger LLMs with more parameters.

3. The paper's exposition could be clearer. The distinction between related prior work and the novel contributions of this work is not sufficiently emphasized.

### Questions
1. Why do the results for "U-shaped Original" in Table 3 differ from those in Table 2? Are the results in Table 3 obtained by replacing the token-level score with a document-level score? If so, what distinguishes this document-level version from existing methods?

2. What is the specific difference between the token-level method proposed in this paper and the token-level method(s) mentioned in Table 4?

​Comment:​​

If the authors can clarify the distinctions between their designed method and existing approaches, thereby providing a clearer understanding of its advantages, I would consider raising my score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the U-shaped positional bias of large language models (LLMs) — their tendency to attend more strongly to tokens at the beginning and end of a prompt while neglecting the middle (“lost in the middle” phenomenon). The authors analyze this behavior from both horizontal (input-level) and vertical (layer-level) perspectives using attention weight analysis.

Building on these insights, they propose a U-shaped Placement strategy that leverages positional bias by placing documents according to both their relevance and the attention distribution of the model. The first LLM pass estimates document relevance via attention weights and extracts positional bias from attention patterns. In the second pass, documents are rearranged to align important ones with high-attention positions (front and back of the prompt).

Experiments on multi-document QA benchmarks (HotpotQA, Musique, 2WikiMHQA) and multiple LLMs (Vicuna-7B, Llama-3.1-8B, Qwen2.5-7B) show consistent improvements over baselines such as RankGPT, ICR, and SELFELICIT without any additional training.

### Strengths
+ Turning positional bias into a benefit by explicitly optimizing for it is creative and conceptually appealing.

+ The paper systematically confirms the U-shaped bias across models and layers using both horizontal and vertical attention analysis (Fig. 2), adding interpretability to a known but underexplored phenomenon

+ The proposed algorithm requires no retraining or model modification, making it simple to deploy across different open-source LLMs and datasets.

+ The method achieves consistent gains across datasets and models (up to +5–10% EM improvement) over comparable two-pass baselines

.

### Weaknesses
- While the empirical motivation is solid, the paper lacks a formal justification or deeper theoretical model for why the U-shaped bias emerges and persists across architectures.

- The method assumes that attention weights correlate well with relevance, which has been debated. Without cross-validation (e.g., gradient-based attribution or causal probes), this assumption may not fully hold.

- The approach requires two LLM inference passes. Although cheaper than fine-tuning, it doubles latency and inference cost, which may matter for real-time systems.

- All datasets are QA benchmarks. It remains unclear whether the approach generalizes to non-QA tasks (e.g., summarization, reasoning, dialogue) that also suffer from “lost in the middle.”

- Gains vary across models. More analysis on model scaling or prompt length sensitivity would strengthen generality claims.

### Questions
1) How robust is attention as a measure of document importance? Could you provide any validation—e.g., correlation with relevance scores from gradient-based attribution, SHAP, or token perturbation tests—to support the assumption that attention weights reflect semantic importance rather than just syntactic salience?

2) If attention occasionally misaligns with relevance (e.g., focusing on stopwords or question tokens), how does that affect the placement strategy? Is there any mechanism to filter or normalize attention signals before using them for ranking?

3) Could you clarify how horizontal and vertical attention distributions are aggregated across layers and heads?
Are all attention heads equally weighted, or are only a subset of “dominant” heads used?

### Soundness
2

### Presentation
2

### Contribution
2
