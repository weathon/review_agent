# Learning is Forgetting; LLM Training As Lossy Compression

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 0

## Abstract
Despite the increasing prevalence of large language models (LLMs), we still have a limited understanding of how their representational spaces are structured. This limits our ability to interpret how and what they learn or relate them to learning in humans. We argue LLMs are best seen as an instance of lossy compression, where over training they learn by retaining only information in their training data relevant to their objective(s). We show pre-training results in models that are optimally compressed for next-sequence prediction, approaching the Information Bottleneck bound on compression. Across an array of open weights models, each compresses differently, likely due to differences in the data and training recipes used. However even across different families of LLMs the optimality of a model's compression, and the information present in it, can predict downstream performance on across a wide array of benchmarks, letting us directly link representational structure to actionable insights about model performance. In the general case the work presented here offers a unified Information-Theoretic framing for how these models learn that is deployable at scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work applies the idea of information bottleneck to study LLM training. The authors propose an entropy estimation method and use it to quantify the mutual information between the representations and inputs/targets. By studying these quantities, they conclude that LLM training consists of two phases: a fitting phase and a compression phase, so training can be viewed as lossy compression. They further show that their framework reveals a correlation between model performance and the optimality of the compression.

### Strengths
1. The work applies the interesting idea of information bottleneck to study models/datasets at much larger scale than prior works.
2. A tractable method for estimating entropy at LLM scale is used.

### Weaknesses
1. The existence of two phases is not clear from the figures. How do we know that the change in I(Y; Z) and I(X; Z) is significant?
2. In Figure 1, the two phases on the left and right don’t quite look similar.
3. If the changes are indeed significant, then the compression phase seems to consist of a significant decrease in expressivity (mutual information with the label). This casts doubts on a) whether the entropy is estimated accurately enough, and b) whether the “compression phase” is really about compressing irrelevant information between X and Z.
4. In Figure 1, the x and y axes have numbers between 0.2-0.22. Doesn’t that imply that there’s barely any difference in the mutual information throughout training? In other words, how do we know that the differences are significant?
5. Commonly used language datasets are highly imbalanced. While I understand the authors’ desire to avoid weighing tokens like “the” and “a” too heavily, I question whether weighing everything equally skews the estimate too much.

### Questions
1. In the context of the information bottleneck, there were several papers that questioned the relation between compression and generalization. Could the authors add a literature survey on that topic and elucidate what those papers got wrong?
2. Is the method used for estimating mutual information similar to the Nystrom method in kernel methods?
3. Is there any prior work that also measures compression of token embeddings and looks into whether such properties correlate with generalization?
4. Do we have a reason to expect that the method would actually work once the number of points w_i is small (which is required for estimating mutual information based on the resulting softmax probabilities)?
5. Did you conduct experiments testing the robustness of observations to the number of points w_i in eq. 2?
6. Does expressivity alone predict performance? What is the correlation between them?
7. How do you expect the trajectories to change as the context length used to estimate entropy increases?
8. Does data quality impact the presence of the two phases?
9. In the original IB paper, the "knee" in the curve was due to the interpolation of the training data (transition from getting zero error on ERM to compression). Could you show that the same thing occurs in your experiments?

### Soundness
3

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
3

### Summary
The paper proposes a method to frame pre-training as lossy compression, by leveraging the representational space, i.e. the learned representations with information-theory methods. They use entropy and mutual information between input, hidden representations and output, using a scalable soft-entropy estimator, to compute complexity (compression) and expressivity (task information). The authors  demonstrate that during training LLMs, there are two phases under the Information Bottleneck theory perspective: an initial expansion of expressivity followed by a compression phase. They propose that models approaching the optimal compression bound tend to perform better on downstream tasks. The approach is novel and well-motivated, and the findings regarding compression during training are a valuable contribution.

### Strengths
- **S1** While the soft-entropy estimator is not a contribution of this paper itself, it seems to be the first to explain and implement the approach from Conklin (2025) to compute entropy and mutual information within LLMs to distinguish between complexity (compression) and expressivity (task-relevant information).
- **S2** The use of such metrics to track the behaviour of LLMs during pretraining is, to the best of my knowledge, novel. While the results for smaller models are inconclusive, the results for olmo2 7B and 32B during training align with the Information Bottleneck theory and paired with the results at the end of training for a vast array of models constitutes a welcomed new perspective on LLM training.
- **S3** The paper is clear in its motivation for why information-theoretic analysis is needed for understanding LLMs, and it provides a nice overview on representation learning in the context of compression.
- **S4** The proposed methods to assess expressivity and compression during training are likely to spark future work exploring these further.

### Weaknesses
- **W1. Attribution and Contextualization of Contributions**
   - One of the main contributions of the paper is the implementation of the theoretical framework from Conklin (2025) to estimate soft-entropy. You should better contextualize this, not to diminish your own contributions, but to clarify that much of Section 3.1 is a reinterpretation of prior work.
   - Then, emphasize that this paper is the first to leverage the soft-entropy estimator in the context of LLMs.

- **W2. Inconsistent Level of Detail**
  - The level of detail varies across the main body, which may be challenging for readers who are not experts in compression.
  - For example, Section 3.2 glosses over the n-gram backoff mechanisms and provides little explanation of I(Z;Y) in the main text.
  - Readers would not know that I(Z;Y) is better explained in Appendix E.2 from reading the main body
  - In that vein, the paper seems to expect readers to assume that the mechanisms for I(Z;X) and I(Z;Y) are identical (see last sentences of E.2), but this is only made explicit in the appendix.
  - Meanwhile, the Introduction and Background sections are lengthy (up to page 5), and could be condensed to make room for more technical details that are currently omitted or relegated to the appendix.

- **W3. Section 4.2 Results and Interpretation**
  - The results in Section 4.2 are not fully convincing.
  - While it is good to see other use cases for the mutual information values, this section feels lackluster in achieving its goal.
  - It seems almost granted that I(Y;Z) will correlate with downstream performance.
  - The paper argues that the ratio with compression (expressivity/compression) is important, suggesting that two models with the same expressivity may not generalize equally and that compression may be the key difference.
  - However, the paper does not fully explore whether it is solely I(Y;Z) that is driving the correlation, and the claim of "optimal compression" may be more semantic than substantive without further analysis.

- **W4. Practical implications**
  - The paper focuses on empirical results, reporting results for a vast array of models, and checkpoints for the case of Olmo/Pythia.
  - However, not that much emphasis is given to practical implications (beyond what is discussed in W3).
  - This seems a missed opportunity. For instance, could these findings be used to predict early stopping? 
  - What about when to expect the change between phase1 and phase2? Are there any implications related to that in terms of performance? (i.e. does the point in which a model start compressing, predict performance?) It does seem that larger models take proportionally more steps in phase 1. 
  - Going back to W2, some content could have been condensed to leave place for more practical implications.

### Questions
- Have you analyzed whether the correlation with downstream performance is primarily driven by I(Y;Z) alone, or does the expressivity/compression ratio provide additional predictive power? Could you provide results or discussion to clarify this point?
- Are there other practical implications of your findings for LLM training? You could place here the questions from W4.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates LLM training through the lens of Information Bottleneck (IB) theory, framing learning as a process of lossy compression of input information. It introduces an information-plane analysis that tracks the mutual information between model representations, inputs, and outputs during training, revealing a two-phase trajectory: first, expansion, then compression of representations. The authors develop a soft-entropy-based estimator to compute mutual information efficiently at LLM scale, enabling comparisons across model families and token contexts. Their empirical results show that models approaching optimal compression achieve higher accuracy on downstream benchmarks such as MMLU-Pro, and that post-training fine-tuning primarily increases human preference information rather than token-level complexity.

### Strengths
1. The paper presents a clear, interpretable information-theoretic framework for understanding representational dynamics of LLMs.

2. This paper evaluates multiple open-weight models, scaling classical IB analyses to billions of parameters.

3. This paper provides a readable exposition with intuitive information-plane plots.

### Weaknesses
1. The core argument that LLM training is a process of lossy compression approximating the Information Bottleneck largely restates conclusions from prior work, such as Language Modeling is Compression [1] and Norm of Word Embedding Encodes Information Gain [2]. The paper extends scale but does not introduce a new theoretical insight or training principle.

2. No comparison with established compression baselines. The study fails to relate its “compression optimality” metric to practical model compression methods like pruning [5, 6], quantization [7, 8], or low-rank adaptation [9, 10]. Without such baselines, it is unclear whether the proposed metric offers predictive or practical value for compression performance.

3. The observed correlation between compression optimality and MMLU-Pro accuracy does not imply causality. Unlike controlled studies of feature emergence or SGD-noise-driven IB dynamics [3, 4], the paper lacks interventions demonstrating that manipulating compression affects downstream accuracy.

4. The entropy estimation approach is based on a soft-entropy approximation but is not compared against common mutual-information estimators such as MINE or CLUB. Without quantitative validation, the robustness of the reported information-plane trajectories remains uncertain.

References

[1] Delétang et al., Language Modeling Is Compression, NeurIPS, 2023.

[2] Oyama et al., Norm of Word Embedding Encodes Information Gain, EMNLP, 2023.

[3] Tishby and Zaslavsky, Deep Learning and the Information Bottleneck Principle, ITW, 2015.

[4] Saxe et al., On the Information Bottleneck Theory of Deep Learning, J. Stat. Mech., 2019.

[5] Frantar and Alistarh, SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, arXiv, 2023.

[6] Ma et al., LLM-Pruner: On-the-Fly Structured Pruning for Large Language Models, NeurIPS, 2023.

[7] Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers, arXiv, 2022.

[8] Lin et al., AWQ: Activation-Aware Weight Quantization for LLMs, MLSys, 2024 (preprint 2023).

[9] Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR, 2022.

[10] Zhang et al., AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, ICLR, 2023.

### Questions
Under matched compute/bit-budget/latency, how does the proposed “compression optimality” metric predict retained accuracy compared with established compression methods?

How robust are the information-plane trajectories to the choice of MI/entropy estimator? Can you benchmark your soft-entropy approach against common alternatives (e.g., MINE, CLUB)?

What concrete theoretical insight or training principle, beyond prior claims that “language modeling is compression” and IB-style dynamic, does this paper introduce?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This work frames large language models as lossy compressors that retain only information essential to their training goals. The paper tries to show that that pre-training drives models toward Information Bottleneck optimality, and differences in compression across model families reflect data and training choices. Very limited and insufficient empirical studies are provided to support the claim.

### Strengths
None noted.  See explanation in the weaknesses.

### Weaknesses
1. The authors ignore that the understanding LLMs from the perspective of information bottleneck (IB) has been explored in the literature, such as [1-3]. These existing works understanding the compression effect of variants of LLMs by IB, and none of these works are cited and discussed in this paper. The main claimed contribution of this paper, the compression effect of LLMs from an IB persecuted, is not new and already discussed in these existing works.

2. While this paper was submitted as a "learning theory" paper, no formal theoretical results are presented and proved, far below the standard of major machine learning venues including ICLR.

3. Again, this paper lacks a key formal definitions to support its main claim, While "optimal compression" of LLMs are mentioned across this paper for many times, there no formal result about when "optimal compression" can happen (that is, minimum of Eq. (1)), and there is no result about why training LLMs can achieve such "optimal compression".

4. The formatting and presentations are chaotic, far below the bar of the a major ML venue. A lot of sentences are ungrammatical, and there are a lot of formatting issues (large space between equations or paragraphs).

5. This paper also lacks sufficient empirical justification: experiments on large-scale LLMs on standard benchmark should be used to support the claims, such as LLaMA-4 and Qwen-3.


[1] Protecting Your LLMs with Information Bottleneck. NeurIPS 2024.

[2] An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation. ACL 2024.

[3] QUITO-X: A New Perspective on Context Compression from the Information Bottleneck Theory. EMNLP 2025.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
