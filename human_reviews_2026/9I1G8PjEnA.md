# Representation Drift Compensation: A Zero-Cost Enhancement for LLM Decomposition

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
While low-rank decomposition offers potential for LLM size reduction, its application is limited by considerable performance degradation. In this work, we identify and formalize a key overlooked issue in LLM decomposition: \textit{representation drift}. We show that approximation errors introduced by decomposition propagate and amplify non-linearly through the deep layers of the transformer architecture, progressively distorting internal representations and degrading downstream performance. To mitigate this, we introduce a conceptually simple but principled compensation mechanism, named ``\our'', that operates by suppressing error at its source. By learning to align the output distribution of decomposed transformer blocks with their original counterparts, our method effectively counteracts representation drift, achieving notable performance recovery with zero inference overhead. Extensive experiments across OPT, LLaMA-2, LLaMA-3, and QWen exhibit remarkable improvements in language modeling, common-sense reasoning, knowledge-based reasoning, and vision-language tasks. For instance, on LLaMA-3-8B and OPT-13B at 40\% compression, perplexity is reduced by more than 70\% while reasoning task accuracy improves by over 10\%. Our code is available at this anonymous URL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Decomper to align decomposed block outputs with original ones. It tests on OPT, LLaMA-2/3, QWen, showing over 70% perplexity reduction and 10% accuracy gain at 40% compression.

### Strengths
1.The paper is well organized and well written.

2.The theoretical foundation is sound and presented with clarity, providing solid support for the proposed approach.

3.The experimental setup is rigorous and comprehensive, effectively validating the theoretical claims.

### Weaknesses
1. **In Table 1, the color blocks used in the “ratio” section are visually confusing.**
   The inconsistent coloring makes it difficult for readers to interpret the results clearly, and the authors are encouraged to unify or clarify the color scheme for better readability.

2. **From Table 1, it can be observed that while the proposed method performs well on simpler tasks such as CommonsenseQA, it suffers larger performance drops on more challenging benchmarks like MMLU.**
   Moreover, the paper lacks evaluation on difficult text generation or reasoning tasks (e.g., GSM8K, HumanEval), which are crucial for demonstrating the generality and robustness of the method. Without such experiments, it is hard to justify that the proposed approach has broad applicability.

3. **The practical value of the proposed method remains unclear.**
   As shown in Table 1, *Decomper* exhibits substantial performance degradation compared to the original model, especially on more complex tasks such as MMLU. This raises the question of whether the method offers any real advantage over simpler quantization techniques. Although the “Compatibility with Quantization” section claims that *Decomper* can be combined with quantization, the paper does not provide comparisons with standalone quantization methods. More detailed experimental analyses should be included to validate the method’s practical value, including but not limited to comparisons in effectiveness, inference speed, and computational cost.

### Questions
As shown in the weaknesses.

### Soundness
4

### Presentation
4

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
This paper identifies “representation drift” — the propagation and amplification of approximation errors in low-rank decomposition of LLMs — as a key cause of performance degradation. The authors propose Decomper, a zero-overhead compensation method that learns bias corrections to align the outputs of decomposed layers with the original model. Experiments across multiple LLMs and benchmarks show the method effectively mitigates performance drops while adding no inference cost.

### Strengths
S1: The paper is well-structured with logical flow, facilitating reader comprehension. Section 3 presents a step-by-step, tightly connected process from error quantification to propagation analysis, enhancing readability and reinforcing technical rigor.
S2: The diagnosis of “representation drift” combines theoretical analysis with empirical validation. The proposed Decomper mechanism features zero deployment overhead and demonstrates strong generalizability across SVD/PCA-based decomposition, quantization, and vision-language models.
S3: Experiments cover diverse models and benchmarks, with in-depth analysis validating robustness and efficiency compared to fine-tuning/matrix updates, ensuring thorough method evaluation.

### Weaknesses
W1: Novelty: Decomposition-based compensation has been extensively explored in this field, as evidenced by prior works such as FLAP and AFM cited in the paper. This study primarily tests similar approaches on low-rank decompositions without substantial conceptual advancement.
W2: Although the paper conducts numerous comparative experiments, it fails to adequately contrast its approach with several outstanding related studies it cites.
W3: Section 3 presents critical formulas but omits intermediate derivation steps, and the Appendix does not supplement them. The paper directly states the linear layer reconstruction loss expectation without showing how to expand the squared norm expectation into mean and covariance terms.
W4: Equation 3 contains critical issues: the left-hand side is defined as the drift of the L-th block, but the right-hand side sums drift from l=1 to L, creating a logical contradiction. The formula also fails to retain the expectation term or connect to the squared L2 norm, breaking the link between theoretical setup and propagation analysis.
W5: The paper compares Decomper with recovery fine-tuning and least-squares matrix updates but omits critical experimental details. It remains unclear which dataset (WikiText-2 or C4) was used for fine-tuning/calibration, which benchmark the “Avg. Acc.” corresponds to, and how Decomper performs in domain-specific scenarios where fine-tuning typically excels.

### Questions
Please refer to Weaknesses.

### Soundness
3

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
4

### Summary
This paper studies the low-rank decomposition of LLMs. The authors discuss the layerwise decomposition and show that the error from layerwise decomposition can accumulate over different layers. They propose to add a bias term to compensate for the additional error. The resulting optimization is solved using gradient descent. The authors also provide numerical experiments over a wide range of baselines and benchmark datasets.

### Strengths
- The paper studies an important problem
- The numerical experiments are extensive

### Weaknesses
I think the writing can be improved. Some examples:

- Proposition 1 is quite handwavy. What is random? What is variance? With respect to what distribution? The results should be written properly.

- Why should one care about (3)? We only care about the final error, not the average layerwise error.

- It is not immediately clear how problem (6) is solved. One has to look into the appendix to find the algorithm, which I don't think it is referred to in the main text.

- Table 4 is missing dense baselines. I'm not sure how accurate the comparisons in the section named "Contribution of Compensation Strategy" are.

### Questions
Some of the models used in the experiments are rather old (OPT, Llama 2). Can the authors please present more benchmarks for newer models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies "representation drift" as a key problem in low-rank decomposition of LLMs, where approximation errors accumulate and amplify non-linearly through deep transformer layers, progressively distorting internal representations. The authors propose "Decomper," a compensation mechanism that learns bias vectors for each decomposed linear layer to align decomposed block outputs with original counterparts. The method is optimized on a small calibration set and adds zero inference overhead by fusing learned compensation into existing bias terms. Extensive experiments on OPT, LLaMA-2, LLaMA-3, and QWen2.5-VL etc, are shown.

### Strengths
- The core insight about error propagation through Equation 3 is well-motivated and provides theoretical grounding for the empirical phenomenon
- Proposition 1 offers a clear bound on the expected local error after compensation
- Experimental validation is comprehensive across multiple model families, scales, and benchmarks and the ablation studies convincingly isolate the contribution of the compensation mechanism

### Weaknesses
- The core idea of learning bias corrections is incremental—FLAP (An et al., 2024) already uses bias correction for pruning, and this work essentially adapts it for decomposition
- No comparison with other alignment strategies (e.g., feature matching, KL divergence minimization between distributions)
- The theoretical analysis in Section 3.1 uses first-order Taylor expansions without discussing when this approximation is valid or quantifying higher-order terms
- Proposition 1's proof assumes convergence to c* = Eμ but doesn't address the non-convex optimization landscape mentioned in Section 3.2
- The claim that Equation 6 "consistently converges to a strong local optimum" lacks empirical evidence (e.g., convergence curves, sensitivity to initialization)
- The theoretical closed-form solution c* = Eμ is dismissed as insufficient, but no rigorous analysis explains why the learned bias systematically outperforms this baseline
- Missing analysis of how the compensation mechanism interacts with different compression ratios per layer (mentioned but not studied)


Minor:
- Figure 2's caption could better explain what "23rd Transformer block" represents (out of how many?)
- The connection between the local error term (Equation 1) and the block-level propagation (Equation 3) could be made more explicit
- Section 3.2 introduces multiple ideas (theoretical c*, averaging trick, learned bias) that feel somewhat disconnected
- The paper claims "zero-cost deployment" but doesn't discuss memory overhead of storing compensation vectors during training or calibration time costs
- Some experimental details are unclear: what is "data-whitening SVD" as distinct from vanilla SVD?

### Questions
- Neat but not very well explained:why does the learned bias outperform the closed-form solution? 
- Which layers contribute most to drift? Are all layers equally important to compensate? How does compensation allocation across layers affect results?
- What is the wall-clock time and memory cost of the compensation optimization phase? How does this scale with model size?
- Figure 2 shows drift recovery, but can you quantify the alignment more rigorously (e.g., KL divergence, Wasserstein distance between original and compensated distributions)?
- Can you provide error bars or confidence intervals for the main results (Table 1-3) to assess statistical significance?

### Soundness
3

### Presentation
3

### Contribution
3
