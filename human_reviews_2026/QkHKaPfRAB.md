# ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Large Language Models (LLMs) excel in various domains, but their safe deployment faces the challenge of balancing safety and utility. Existing alignment strategies often strengthen refusal mechanisms to reduce harmful outputs, but harmless instructions with superficial risky words are mistakenly rejected, which is known as over-refusal. 
This work first reveals that over-refusal stems from a cognitive bias in the model's internal representation space: LLMs naturally encode safety attributes in hidden states, and pseudo-harmful instructions overlap with harmful features, causing over-harmful encoding. 
To address this, we propose ProSafePrune, a subspace-projected low-rank parameter pruning framework for mitigating LLM over-refusal. By projecting pseudo-harmful features into subspaces and removing low-rank directions corresponding to harmful components in the most discriminative layers, we significantly reduce over-refusal while preserving the model’s ability to reject genuinely harmful requests, improving performance on general tasks. In experiments, across different models, our method significantly lowers the average false rejection rate while slightly improving general task performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ProSafePrune, a low-rank parameter pruning method to mitigate over-refusal in large language models. By identifying and pruning harmful components within pseudo-harmful subspaces, the approach reduces false refusals while preserving genuine safety. Experiments on multiple LLMs show improved compliance and balanced safety performance.

### Strengths
1. The proposed method addresses an important and timely issue in LLM alignment with clear motivation and practical relevance.

2. This paper proposes a lightweight, training-free pruning framework that is theoretically grounded and computationally efficient.

### Weaknesses
1. The claimed “cognitive bias in internal representations” remains largely conceptual — the paper lacks deeper interpretability analysis or causal evidence supporting this mechanism.

2. The proposed pruning relies on white-box parameter access, limiting applicability to real-world LLMs.

3. The novelty is incremental, as the idea of subspace-based low-rank pruning has been explored ; this paper mainly repurposes it for the over-refusal context.

4. While benchmarks show strong quantitative results, the evaluation misses human evaluation.

### Questions
1. The paper attributes over-refusal to “over-harmful encoding” in internal representations. Could the authors elaborate on how this relates to recent work on safe unlearning? Conceptually, is ProSafePrune pruning a similar subspace that unlearning methods aim to erase, or does it operate on a different mechanism?

2. How were the pruning hyperparameters (e.g., λ and rank r) selected, and how sensitive are the results to these choices?

3. Would combining ProSafePrune with post-training alignment or inference-time steering further improve the safety–utility trade-off?

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
This paper focuses on mitigating over-refusal in LLMs. The authors argue that this behavior comes from harmful encodings overlapping with pseudo-harmful ones in the model’s latent space. To address this, they propose a training-free and subspace-based low-rank pruning method. The key idea is to use (truncated) SVD to identify and remove directions in parameter space that push pseudo-harmful instructions toward harmful representations while not compromising the model’s refusal and general abilities. Experiments across model scales in Llama and Qwen families show lower false rejection rates and stable safety performance, while maintaining or improving general-task performance.

### Strengths
1. **Simple and elegant method.**
- Low-rank pruning based on subspace overlap is lightweight and does not require retraining or inference-time overhead.  

2. **Strong empirical results.**
- Overall, the proposed method improves compliance on multiple pseudo-harmful benchmarks while maintaining safety on harmful ones (Table 1, Figure 4).  

3. **Good theoretical evidence support.**
- The energy-removal guarantee in Theorem 3.2 supports the claim that pruning minimally disrupts the model’s general capacity.  

4. **Interesting insights connected to alignment tax.**
- The additional experiments in Section 5.2 add conceptual depth to understanding how overly strong safety alignment can constrain model capabilities.

### Weaknesses
1. **Suspiciously large gap on OR-Bench despite training on it.**
- In Table 1, the proposed method shows promising performance on average, but for LLaMA‑3‑8B on OR-Bench the score (71.0) is far lower than Self-CD (86.0). Since the pseudo-harmful subspace is built using OR-Bench data, this is a bit suspicious and raises the question of whether the subspace construction is stable.

2. **The results would be more robust with additional evidence.**
- For example, confidence intervals or std errors for model performances for Table 1 and Figure 4, how pruning rank and λ are selected for Table 3, and potential evaluation on do_sample=True to give a better sense of the method’s stability under more realistic decoding settings.

3. **Small evaluation samples.**
- The model performances are reported on a few hundred examples per dataset (according to Section B.2), which has the concern that small fluctuations could produce the reported gains.

### Questions
- Qualitative analysis on cases where the proposed method helps or hurts.
- Submodule-level insight. The pruning is done on Q/K/V/O/MLP modules with no analysis on which parts of the network contribute most to improvements.

### Soundness
3

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
4

### Summary
This paper studies the over-refusal issue in LLMs, where harmless instructions are incorrectly rejected due to overlapping safety and harmful feature representations. The authors introduce ProSafePrune, a subspace-projected low-rank pruning framework that removes harmful components from key layers, effectively mitigating over-refusal while preserving genuine safety behavior and slightly improving overall task performance across various models.

### Strengths
1. Addressing the over-refusal issue in LLMs through parameter subspace-projected low-rank pruning is a novel approach, and the extensive experiments convincingly demonstrate its effectiveness.

2. The paper is well organized and clearly written, making it easy to follow.

### Weaknesses
1. The advantage of the proposed parameter subspace low-rank pruning approach over representation editing in mitigating over-refusal in LLMs is not convincingly demonstrated. Incorporating additional theoretical analysis or interpretability-oriented empirical studies could strengthen the comparative argument and better substantiate the contribution of this work.

2. The finding that the proposed framework does not degrade, and even slightly improves, the general capabilities of LLMs is intriguing. However, the underlying causes of this improvement remain unclear. A more in-depth discussion would help elucidate the underlying mechanisms of this phenomenon.

3. The evaluation is currently limited to LLMs within the 7B–14B parameter range. Assessing the scalability and effectiveness of the proposed framework on larger models (e.g., Qwen3-32B, Llama-3.1-70B) would further validate its robustness and practical applicability.

### Questions
1. The finding that the proposed framework does not degrade, and even slightly improves, the general capabilities of LLMs is intriguing. What do the authors consider to be the underlying causes of this improvement?

### Soundness
3

### Presentation
3

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
This paper introduces ProSafePrune, a training-free framework designed to mitigate the "over-refusal" phenomenon in LLMs. The authors identify the root cause as a cognitive bias within the model's internal representation space, where pseudo-harmful instructions are "over-harmfully encoded". ProSafePrune uses subspace projection via truncated SVD to identify and prune low-rank components corresponding to harmful amplification directions in the most discriminative layers.

### Strengths
1. This paper shifts the focus from activation-level interventions to directly modifying the model parameters to mitigate the over-refusal phenomenon, which is a new direction.
2. ProSafePrune consistently demonstrates superior performance across diverse LLMs (LLaMA-2/3) and a range of over-refusal benchmarks (OR-Bench, PHTest, XSTest, OKTest).

### Weaknesses
1. Sensitive to prune layers and hyperparameter lambda: From table 3 and figure 8, there is a close relationship between model performance and careful tuning of those hyperparameters. It is better to explain how ProSafePrune selects the prune layers since authors only claim high-scoring middle layers as candidates for pruning but without presenting the score threshold for selection.
2. Unclear scalability: The experiments focus on models up to 14B parameters. It’s not clear how well the approach would scale to larger models that may have different internal representations.
3. Lack of pruning time report: Although ProSafePrune employs once static pruning, it is better to report the time cost compared to those training-free baselines.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
3
