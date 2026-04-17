# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess‑and‑verify strategy, but existing training-free variants face trade‑offs: retrieval‑based drafts break when no exact match exists, while logits‑based drafts lack structural guidance. We propose **RACER** (**R**etrieval‑**A**ugmented **C**ont**e**xtual **R**apid Speculative Decoding), a lightweight and training‑free framework that integrates retrieved exact patterns with logit‑driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM demonstrate that RACER consistently accelerates inference, achieving a speedup of $2.2{\sim}2.8\times$ compared to autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding. Our source code is available at [this anonymous repository](https://anonymous.4open.science/r/racer_anonymous-9464).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes RACER (Retrieval-Augmented Contextual Rapid Speculative Decoding), a lightweight
and training-free framework that integrates retrieved exact patterns with logit-driven future cues. Experiments on Spec-Bench, HumanEval, and MGSM demonstrate that RACER consistently accelerates inference and can achieve 2.2~2.8x speedup.

### Strengths
This paper is technically sound and easy to understand.

The experimental results show the effectiveness of the proposed method.

### Weaknesses
The paper focuses on generating tree structure and improve the overall MAT to speedup the large language model.

However, they only conduct experiments with HuggingFace Transformers framework. Here comes a problem that the method may not have such speedup on the popular inference framework such as vLLM. In fact, HuggingFace Transformers framework does not optimize the speed of LLMs very well, which makes the ratio of the latency of tree generation process smaller. When using vLLM framework where the operations in LLMs are optimized very well, the tree generation process will take more time and reduce the speedup.

The author should verify their method on such inference frameworks to show that their method is actually useful in reality.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes RACER, a training-free decoding acceleration framework that combines two types of speculative signals: (1) a logits tree constructed using a copy-logits strategy to extrapolate high-probability tokens, and (2) a retrieval tree maintained by an AC automaton to leverage repeated patterns in the context. These two sources are integrated into a unified speculative draft, which is then verified using standard speculative decoding. The paper reports empirical results on several benchmarks, showing that RACER achieves a 2.2-2.8x speedup over standard autoregressive decoding and outperforms other training-free baselines such as TokenRecycling and LogitSpec.

### Strengths
1. The paper addresses an important and practical problem: accelerating LLM decoding without additional training. This is highly relevant for both research and deployment.

2. The proposed method integrates retrieval-based and logits-based drafting in a unified framework, which is conceptually simple and compatible with existing speculative decoding systems.

3. The method achieves consistent speedups across tasks and shows robustness with respect to hyperparameters such as draft size, retrieval tree capacity, and n-gram depth.

### Weaknesses
1. **The paper is poorly written.** For example, the main contributions are the use of copy-logit for building the logits tree and the use of an AC automaton for maintaining the retrieval tree. However, the entire article uses only one sentence to explain “copy-logit”, and that sentence is not understandable (see Weakness 2). In addition, the acronym “LRU” is explained only the third time it appears, which is not reader-friendly. Moreover, **there are many very short paragraphs with only 2 or 3 lines** (e.g., lines 162–167 and 216–222), which makes the paper look unprofessional. There is also a missing period at the end of line 185.

2. The definitions of the last-logit and copy-logit strategies are unclear. In particular, the meaning of “the same token” in line 125 is ambiguous; the paper should specify what the token is “the same as”. It is strongly recommended to explain these concepts more clearly. In Figure 1, the differences between white and green circles, as well as between solid and dashed arrows, should be explicitly explained in the figure caption.

3. The node index definition in lines 165–167 is non-standard. Using proper mathematical symbols would make it clearer and more precise.

4. The choice of the MGSM-ZH dataset is somewhat questionable. From Table 1, it seems this dataset was chosen mainly because EAGLE-3 performs poorly on it, allowing RACER to surpass EAGLE-3 on average. However, the reproduced EAGLE-3 model was not trained on Chinese data, making this comparison problematic. If my guess is true, I don't think there's any need to be so secretive. It's normal for retrieval-based methods to not outperform EAGLE-3, and it would be strange if they could.

5. The novelty is somewhat weak. The authors claim three contributions. The copy-logit strategy is not clearly explained. The retrieval tree with AC automaton is acceptable, but the integration is rather trivial.

### Questions
1. How do copy-logit and last-logit work? In Figure 1, what is the difference between the white circles and the green ones? What is the difference between a solid arrow and a dashed arrow?

2. What is b in Equation 3? Is it the same as k in a k-ary tree?

3. Why is the experiment done on MGSM-ZH, which is not popular in reasoning tasks? It is advised to do the experiments on more popular datasets, such as the original dataset of MGSM-ZH, GSM8K, or some other popular datasets such as AIME and AMC.

### Soundness
2

### Presentation
1

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
This paper introduces RACER, a training‑free framework that integrates retrieval-based speculative decoding (SD) with logit‑based drafts. Experiments on Spec-Bench, HumanEval, and MGSM demonstrate that RACER consistently accelerates inference, achieving a speedup of compared to autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding.

### Strengths
1. This work presents a valuable exploration to integrate retrieval-based SD methods with logit-based drafting. It provides an insightful idea that SD should not only leverage retrieval contents, which provide seen information through exact pattern matches, but also logit-based contents, which supply unseen information for future tokens.
2. The proposed method is assessed on a wide range of text generation benchmarks, including Spec-Bench, HumanEval, and MGSM. The evaluated models cover the Vicuna series and Qwen3 series.
3. RACER consistently accelerates inference, achieving a speedup of 2.2x-2.8x compared to autoregressive decoding, and outperforms prior training-free methods, such as token recycling, logitspec, PLD, and REST.

### Weaknesses
1. **Comparison with LogitSpec**: A similar idea of combining retrieval-based SD with logit-based drafts has been proposed in LogitSpec [1], which appears to be the most closely related work to RACER. A more thorough comparison between LogitSpec and RACER is needed to better highlight this paper’s contributions.
2. **Writing Refinement**: The manuscript would benefit from significant improvements in writing clarity. For example, Section 3.1 fails to clearly explain the “copy-logit strategy.” Although I carefully reviewed Lines 124–126 and Figure 1, the explanation lacks sufficient detail. Given that the copy-logit strategy is a central contribution of the logit-based drafts, the lack of a clear and detailed description creates confusion for the reader.
3. **Insufficient Methodological Details**: (1) In Figure 1, the relationship between frequency (y-axis) and the Mean Accepted Tokens (MAT) is unclear. (2) The term “failure link” in Lines 213–214 is not well defined. (3) Section 3.2 omits several critical details. For instance, what specific retrieval method is employed? Additionally, the update rule described in Lines 216–219 is presented too briefly, without adequate explanation.
4. **Lack of Experimental Clarity**: It is unclear what the maximum number of draft tokens is for each SD step. Among these draft tokens, what is the proportion of retrieval tokens versus logit tokens? It is also confusing that Line 425 states the draft size ranges from 16 to 64, while the maximum breadth of the logit tree is 8, and the retrieval tree contains up to 10,000 nodes with an n-gram length of 10. The relationships among these parameters should be clearly clarified.
5. **Missing Baseline Comparison**: Why is RACER not compared against EAGLE-3 using the Vicuna series? 
6. Table 2 illustrates that the logit-based drafting contributes more to the efficiency of RACER, while the improvement from the retrieval part seems to be much smaller. The contributions and necessity of each designed module should be further analyzed.

### Questions
1. The font size in Figures 1 and 2 is too small and difficult to read.
2. The phrase “P50 and P85 quantiles” in Lines 148–149 could be revised to “50th and 85th percentiles” for clarity and consistency.
3. A grammar issue: the paragraph spanning Lines 183–186 ends without a period.


[1] LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation. Liu et al. 2025.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper unifies logit-based and retrieval-based training-free speculative decoding within a single framework. Experimental results demonstrate superior performance compared to either approach used independently.

### Strengths
1.	The paper unifies two subcategories of training-free speculative decoding, achieving a better trade-off between acceptance rate and inference efficiency.
2.	The methodology and presentation are clear and easy to follow, making the approach straightforward to reproduce.

### Weaknesses
1.	The experiments primarily focus on a batch size of 1, which limits the generality of the throughput evaluation and may not reflect performance in higher-throughput settings.
2.	The main contribution appears to be the integration of two existing methods, offering limited novelty and conceptual insight beyond their combination.

### Questions
Throughput scaling: Could you report results for batch sizes > 1 to assess how throughput and latency scale beyond the single-batch setting?

Model-size ablation: The current results suggest the method may benefit more on larger models. Since most evaluations are on small–to–medium scales, could you include an ablation across model sizes to quantify how effectiveness changes with model capacity?

### Soundness
3

### Presentation
3

### Contribution
2
