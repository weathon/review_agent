# Efficiently Identifying Watermarked Segments in Mixed-Source Texts

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
Text watermarks in large language models (LLMs) are increasingly used to detect synthetic text, mitigating misuse cases like fake news and academic dishonesty. While existing watermarking detection techniques primarily focus on classifying entire documents as watermarked or not, they often neglect the common scenario of identifying individual watermark segments within longer, mixed-source documents. Drawing inspiration from plagiarism detection systems, we propose two novel methods for partial watermark detection. First, we develop a geometry cover detection framework aimed at determining whether there is a watermark segment in long text. Second, we introduce an adaptive online learning algorithm to pinpoint the precise location of watermark segments within the text. Evaluated on three popular watermarking techniques (KGW-Watermark, Unigram-Watermark, and Gumbel-Watermark), our approach achieves high accuracy, significantly outperforming baseline methods. Moreover, our framework is adaptable to other watermarking techniques, offering new insights for precise watermark detection.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes an efficient watermark localization when the text under scrutiny is a mix of generated (and watermarked) sentences and human (and non-watermarked) passages.

### Strengths
**S1. This paper tackles an important issue in LLM watermarking**

**S2. The connection with TV regularized estimator is nice**

### Weaknesses
**W1. The Document False Positive Rate is not under controlled**

There are multiple reasons:
- For a given chunk in the GCD, the detection translates the score into a p-value with the correct formula except for Unigram where this is done via a Z-statistic. Since Unigram and KGW are very similar, I wonder why Unigram deserves special treatment. The Z-statistic approximation may not be accurate for a length of 32, especially when targeting low segment-FPR $\tau$. Moreover, Fernandez et al. advice to remove repeating tokens windows. This is done for Unigram, but not for Aaronson and KGW.
- The algorithm first sets the segment-FPR (also called FPR for intervals) and then extrapolates the Document-FPR (also called Family-Wise Error Rate). It should be the other way around.
- This extrapolation is done empirically (line 369)
- The only approximation is an upper bound $n\tau$ (line 255) which is vacuous for the experiments: $\tau=10^{-4}$ and $n$ in the order of $\approx 10^4$ (Table 3). Note that the Document-FPR is not given in Table 3.

**W2. Dilution of the watermark**

The authors argue that classical watermark detection makes a global decision, which is less reliable when a small fraction of the text is watermarked. I do agree, but it also holds for their proposal. For a given Document-FPR (which should be requirement number 1, independent of the text length), there are more intervals in GCD. Therefore, the threshold at the interval level is a lower p-value, increasing the probability of missing the detection. This difficulty is inherent to the problem. 

**W3. Strong limitations**
    
In the same way, "*Additionally, positive samples created by inserting the generated watermark paragraph into natural text may not be detectable with our approach.*" , "*we assume that our method needs only to detect reasonably high quality watermarked text segments*".
Limitations are hidden in the appendices. I disagree that this is due solely to the watermarking technique. This is partly due to the new detection scheme (see W2). Appendix A.3 reveals the experiments are biased by a filtering technique sieving the good examples.

**W4. Completeness**

A recap of the ALIGATOR algorithm as applied on watermarking detection in the appendix would be a good idea as the papers from Baby are not easy to read.

**W5. Advantage**

The main advantage is the complexity in $O(n\log n)$. I am not so sure it makes a big advantage. One first computes the score per token in $O(n)$. This is the slowest computation. Then analyzing these $n$ real values one way or another should not make a big difference (compared to the first step).

**W6. Experiments**
The experiments consider only one configuration: normal text + watermarked text + normal text. What happens if the watermarked text is spread all over the document?

### Questions
**Q1. Some parameters are missing**

The value of $\zeta$ (line 228)? The value of the window size $h$ for Aaronson and KGW?

**Q2. Baselines**

The baseline using RoBERTa is very surprising. If $h=0$ (like for Unigram), why not! But for $h>1$, I just don't understand how it could work.
Why the baselines mentioned in Section 2.3 are not considered in the benchmark?
In the end, the problem seems to be similar to a segmentation-based object categorization. This opens the door to many baseline solutions like Graph min-cut using spectral techniques.

**Q3. Typos**

- line 161. $\mathcal{M}(x)$ to be replaced by  $\mathcal{M}(y)$?
- Eq.(3) is not 100% `correct'. $\mathcal{I}^{(k)}$ is not one interval, but a subset of intervals. $\mathcal{I}$ is not a union of intervals but a set of intervals.
- Eq. line 298. I have some doubt. First, I have difficulty finding this result back in the paper of Baby. Second, $\frac{1}{j-1}$ to be replaced by $\frac{1}{j-i}$? $\theta_t$ is compared to a unique value not depending on $t$?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes methods to identify and pinpoint watermarked segments in long and mixed-source texts.

### Strengths
The paper proposes two methods, not only detecting the watermark segments in long texts, but also identifying the position of the watermarked segments.

### Weaknesses
The technical contribution of the paper appears limited. It does not propose a novel watermarking scheme or detection method. Instead, it leverages the Geometric Cover technique introduced by Daniel et al. for designing the collection of intervals used in watermark detection, and it utilizes the Aligator algorithm proposed by Baby et al. for watermark localization. The paper seems to be a straightforward combination of these existing ideas. There is little connection between these two methods, and the paper does not address whether the complexity could be further optimized by applying both methods simultaneously.

To enhance its value, the paper could benefit from a more comprehensive evaluation. One of its key strengths is efficiency, and it would be beneficial to include experiments that assess efficiency in terms of both time and computing resource utilization.

### Questions
How did you choose the thresholds in Algorithm 2?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims to identify individual watermark segments within longer, mixed-source documents. 
The authors utilized geometry cover to partition the document into segments, then compute a detection score for each segments. If one segment is detected as watermarked, then the whole document is watermarked. Then they  introduced an adaptive online learning algorithm Alligator to pinpoint the precise location of watermark segments. Evaluations show that the approach achieves high accuracy, significantly outperforming baseline methods.

### Strengths
The topic is interesting. The idea to exploit online learning prediction to localize watermark segments is attractive.

### Weaknesses
1 The proposed method lacks a clear description of its underlying intuition.
2 The reason for choosing the compared baseline method is missing.
3 Some parameter settings require further explanation.
4 Some variables lack clear explanations, such as "set N "on line 213, which set? and FPR-1, FPR-2, et al.

### Questions
1 What is the difference of partial watermarked text localization for traditional and LLM scenarios? 
2 The authors described some related works on identifying watermarked portions in long text in Section 2.3, and claimed that the AOL improves the efficiency of detecting watermarked portions, why not choose them as baseline? Why not compared the time cost?
3 According to GCD, some segmentation boundary is fixed across different I(k), how about the situation that the fixed boundary is in the middle of some watermarked content?
4 In section 3.2, why start from 32? 
5 In the experimental settings, the watermarking percentage is set as 10% of the mix-sourced text. Why? I suggest evaluating the method  under different watermarking percentages.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a new method for detecting watermarked short text in long texts. First, multi-scale division is performed based on GC algorithm to roughly identify whether there is a watermark (paragraph-level identification). Next, AOL is used for adaptive and precise positioning to determine the specific watermark location (token-level identification). The proposed method can effectively reduce the time calculation complexity and greatly improve the recognition of the original watermark algorithm.

### Strengths
It is proposed to first use GC to quickly and roughly determine whether a long text contains a watermark. Then AOL is used for "denoising" to accurately identify the watermark position.

Based on three watermark algorithms on the C4 and Arxiv datasets, watermark detection experiments in long texts were carried out, demonstrating the superiority of the proposed method over some baseline detection methods.

### Weaknesses
The threat model is not discussed enough. The design of this method assumes that the watermark detection algorithms can achieve perfect detection (or encounter minimal false positives and false negatives), which is overly idealistic. Furthermore, common adversarial watermarking techniques, such as manual modifications or rewrites of watermark texts and the adversarial manipulation of certain tokens, have not been formalized in this context.

The original contribution is not solid enough: For the recognition of target short texts in long texts, it is a natural idea to use GC or similar divide-and-conquer algorithms, binary search, etc. for multi-scale fast detection. The proposed method does not make enough innovations for watermark recognition scenarios. In addition, the AOL used seems to be a complete copy of existing work, and it also lacks original improvements.

Experimental settings are too narrow:
First, the watermark text is fixed at a 10% proportion, which lacks a reasonable justification. What trends in detection results would emerge if the watermark text proportion were lower (1%) or higher (99%)?

Second, tokens modified by LLMs may inherently be more detectable. In other words, the detection of watermark texts might not stem from the presence of watermarks but rather from their generation by an LLM. Therefore, it is recommended to add additional comparative experiments to evaluate whether text generated by LLM but without watermarks will not be detected by the proposed method.

Finally, there is a lack of experiments to evaluate time complexity. This paper mentioned several times in the introduction that it can significantly improve the time complexity (theoretically), but did not show any quantitative comparative experiments.

### Questions
Why does Table 1 show a 20-30% difference in TPR between the baseline and the proposed method? Is this due to the baseline processing the entire long document as input for detection? If so, would the baseline achieve detection performance similar to that of the proposed method if it were to input single tokens for detection, albeit requiring a longer runtime?

### Soundness
2

### Presentation
3

### Contribution
2
