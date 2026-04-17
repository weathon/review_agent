# Semantic differentiation for tackling challenges in watermarking low-entropy constrained generation outputs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
We posit and demonstrate that while the current approaches for language model (LM) watermarking are effective for open-ended generation, they are inadequate at watermarking LM outputs for constrained generation tasks like machine translation and abstractive summarization due to the lower entropy of the output space. We investigate the reasons for such shortcomings in a variety of prominent watermarking approaches, and propose an effective solution based on sequence-level watermarking with semantic differentiation to watermark LLM outputs for constrained generation tasks that balances the output quality, watermark detectability, and imperceptibility of the watermark. Specifically, we show that token-level watermarking algorithms that modify logits over vocabulary during autoregressive generation, fail because they under-utilize the sequence entropy of the LM available for watermarking constrained generation outputs. While sequence-level semantic watermarking algorithms are promising alternatives for exploiting the higher sequence entropy compared to the low-levels of token-wise entropy, we identify a different fundamental drawback termed region collapse in the operationalization of current approaches that causes poor watermarking performance. Current approaches pseudorandomly partition the sequence-level representation space into valid and invalid regions for watermarking, but their operationalization encourages most high-quality output embeddings to all collapse into a single region causing a trade-off in output quality and watermarking effectiveness. To mitigate this, we devise a scheme SeqMark to differentiate the high quality output subspace and partition it into valid and invalid regions for watermarking, ensuring the even spread of high quality outputs among all the regions for effective watermarking without compromising the output quality. SeqMark substantially improves watermark detection accuracy (up to 28\% increase in \fscore) while maintaining high generation quality in constrained generation settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues that existing LM watermarking methods are ill-suited for constrained generation because token-wise entropy is low even when sequence-level entropy remains usable. It diagnoses a core failure mode of prior sequence-level methods: region collapse, where semantically similar high-quality outputs cluster into the same red/green partitions, forcing a quality–detectability trade-off. The authors introduce SeqMark, which (1) samples likely high-quality candidates to estimate the “good output” subspace, (2) mean-centers this set to increase pairwise distances, and (3) applies LSH-based partitioning so good outputs distribute more evenly across regions; generation then uses rejection sampling on valid (“green”) regions. Empirically, SeqMark improves detection F1 by up to ~28% while keeping translation/summarization quality roughly unchanged; a lighter Fast-SeqMark predicts the mean embedding to avoid LM access at detection, trading some accuracy.

### Strengths
1. The “region collapse” concept is well-motivated and contrasted with LSH/k-means behavior in SemStamp/k-SemStamp, including a helpful schematic and an explanation of why constrained tasks exacerbate it.

2. The mean-centering transform is elegant, cheap, and shown to spread high-quality candidates across partitions, increasing “region entropy” and reducing cosine similarity among candidates

3. Evaluations cover sentence translation, paragraph translation, summarization, and open-ended generation. Open-ended results show SeqMark matches prior work without quality loss, suggesting no regression in high-entropy regimes.

### Weaknesses
1. Results are mainly point metrics (P/R/F1). ROC/PR curves, confidence intervals, and explicit operating thresholds across tasks would give a clearer picture of false-positive trade-offs, especially for human vs. non-watermarked-LM negatives.

2. This paper focuses on the entropy and the semantic watermark. However, the discussion of related work is limited. More related works should be discussed.

3. How sensitive is SeqMark to the encoder family (domain/language mismatch), and to alternative embedding spaces? A cross-encoder robustness study is missing.

4. SeqMark’s detector samples to estimate the candidate mean, implying access to the (watermarked) LM at verification time; Fast-SeqMark mitigates this but loses F1. A fuller analysis of deployment models (closed-weight LMs, third-party auditors) and security implications (e.g., keying, secret seeds, and adversary knowledge) would strengthen the case.

### Questions
Please see above.

### Soundness
3

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
This paper tries to address the challenge of applying language model watermarks to low-entropy generation tasks.
The authors posit that previously token-level methods (e.g., SWEET) are inadequate because they only consider token-level entropy and fail to utilize the much richer *sequence-level entropy* . While other sequence-level semantic watermarks (e.g., SemStamp, k-SemStamp) are better suited, the paper also identifies "region collapse" in these methods: in constrained tasks, all viable, high-quality outputs are semantically similar. SemStamp utilize semantic partitioning methods like LSH or k-means are designed to group similar items together, which causes all high-quality outputs to "collapse" into a single valid (green) or invalid (red) region, forcing a severe trade-off between output quality and watermark detectability. To solve this, the paper proposes a variant of SemStamp, which transforms the embedding spacek, allowing LSH to partition the "high-quality output subspace".

### Strengths
- The identification of "region collapse" is a novel and insightful critique of existing semantic watermarking methods.The empirical evidence in Table 1 (showing low region entropy for baselines) strongly supports this claim.

- The presentation is good and clear.

### Weaknesses
1.  The core premise of SeqMark is to "isolate the high-quality output subspace". However, the paper implements this by "sampling *n* high likelihood responses... under low temperature". The authors incorrectly and fatally equate **high probability** with **high quality**. This assumption is flawed. The method only ensures that *high-probability* samples are spread across partitions, not necessarily *high-quality* ones. This fails to solve the paper's stated problem: ensuring *high-quality* options are available in the "green" region. Also, the method was only tested on "easy" tasks (translation and summarization) where, for modern LLMs, the high-probability set is largely the high-quality set. This assumption would almost certainly fail in more complex, low-entropy domains like code generation or mathematical reasoning. The paper's claims of generalizability are unsupported without experiments in these challenging tasks.

2.  The paper argues that semantic aggregation methods (like SemStamp) are flawed due to "region collapse". However, it ignores that the entire purpose of semantic aggregation is to achieve **robustness** to paraphrasing attacks. SeqMark is built on the opposite principle of "semantic differentiation", which, as the authors admit in Appendix A.2, is more likely to be vulnerable to paraphrasing. The paper presents no experiments evaluating watermark strength under any attack (e.g., paraphrasing, editing, etc.). This is a major weakness.

3. The paper evaluates detection using Precision, Recall, and F1. These metrics are highly dependent on the classification threshold. The standard metrics for evaluating watermark strength are AUC-ROC or TPR@FPR (e.g., TPR@FPR=1%), which should be used for all detection results.

### Questions
1.  The definition of "constrained generation" is vague. Is this simply a synonym for "low-entropy tasks"? A clearer definition would be helpful.
2. How is "region entropy" (Table 1) calculated? The paper provides no formula or explanation.
3. Section 3 argues that token-entropy-based methods (like SWEET) fail because they under-utilize sequence entropy. However, the key experiment in Figure 1 uses KGW (a non-entropy-aware method) as the token-level baseline. To be consistent, the comparison should have been against SWEET

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a semantic watermarking method. In the partition of the embedding space, the proposed method aims to make high-quality generations fall in different red/green regions, which avoids the region collapse phenomenon that is observed for previous semantic watermarking methods. By doing the transformation of the embedding of the high-quality responses, the method decreases their cosine similarities and increases the region entropy. In machine translation and summarization tasks, the proposed method demonstrated better detection power while preserving similar generation quality.

### Strengths
The paper addresses the hard and important task of watermarking for LLMs in tasks with low-entropy generations. The paper provides intuitive explanations of the failure mode of existing semantic watermarking methods. Empirical results demonstrate the effectiveness of the proposed method

### Weaknesses
1. The paper would benefit from providing an algorithm with detailed descriptions of the full procedure. The proposed method uses locality sensitive hashing (LSH) to partition the embedding space, which is the same as some previous work. But it will still be beneficial to introduce how the partition works. The reject sampling generation and detection for red/green list type methods are relatively standard, but it would still be good to provide the detailed procedure, especially considering the general audience for the conference.

2. Some more explanation of why transforming $c_i$ by subtracting the mean can prevent region collapse would be beneficial.

3. No source code provided

There are also some minor typos, e.g., in line 293, "we simply subtract the sample mean is subtracted from"

### Questions
Please refer to weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SeqMark, a sequence-level watermarking method that leverages sequence entropy and subspace semantic differentiation to avoid region collapse and achieve highly imperceptible watermarking in low-entropy constrained generation tasks.

### Strengths
1. The paper effectively identifies and tackles the poor watermark performance under low-entropy constrained generation by analyzing sequence entropy in the continuous embedding space. It also introduces the novel region collapse phenomenon and proposes a principled design to mitigate it.
2. The use of LSH partitioning on a transformed high-quality subspace is conceptually elegant and ensures the watermark remains semantically hidden while maintaining output naturalness.
3. Written perspective is very clear and formulation is great.

### Weaknesses
1. Mean-centering does not fundamentally alter angular relationships in the embedding space; it merely shifts the cluster mean. Thus, the claim that LSH over the transformed subspace eliminates collapse is not theoretically substantiated. The increase in region entropy could be stochastic, not a structural guarantee.
2. The mean-centering transformation is purely heuristic. The paper provides no theoretical justification or citation for why subtracting the sample mean should effectively decorrelate embeddings or prevent region collapse. 
3.  SeqMark’s detection procedure depends on re-sampling from the LM to re-estimate the mean embedding, introducing stochasticity and potential non-reproducibility. Even small differences in temperature, sampling seed, or model checkpoint may shift the embedding manifold, leading to inconsistent detection. The paper does not quantify this sensitivity or propose a robust verification protocol.

### Questions
Will the reproduction of results be stable so that maintaining an effective watermark checking progress?

### Soundness
4

### Presentation
4

### Contribution
3
