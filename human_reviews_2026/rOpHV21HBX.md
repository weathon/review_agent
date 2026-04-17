# Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
In deep time series forecasting, the Fourier Transform (FT) is extensively employed for frequency representation learning. However, it often struggles in capturing multi-scale, time-sensitive patterns. Although the Wavelet Transform (WT) can capture these patterns through frequency decomposition, its coefficients are insensitive to abrupt changes in the time series, leading to suboptimal modeling. To mitigate these limitations, we introduce the multi-order Wavelet Derivative Transform (WDT) grounded in the WT, enabling the extraction of time-aware patterns  spanning both the overall trend and subtle fluctuations. Compared with the standard FT and WT, which model the raw series, WDT operates on the derivative of the series, selectively magnifying rate-of-change cues and exposing abrupt regime shifts that are particularly informative for time series modeling. Practically, we embed the WDT into a multi-branch framework named **WaveTS**, which decomposes the input series into multi-scale time-frequency coefficients, refines them via linear layers, and reconstructs them into the time domain via the inverse WDT. Extensive experiments on multiple benchmark datasets demonstrate that WaveTS achieves state-of-the-art forecasting accuracy while retaining high computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes the Wavelet Derivative Transform (WDT) based on the Wavelet Transform. WDT operates on the derivative of the time series, selectively amplifying rate-of-change cues and revealing abrupt regime shifts that are especially informative for time series modeling. The paper provides extensive theoretical analysis of WDT, including proofs related to the inverse transform, differentiation, and other properties.

### Strengths
- The paper provides numerous proofs, which appear meaningful.

- Developing WDT and IWDT seems non-trivial, although the paper does not clearly articulate the specific challenges involved.

- The experimental results in the paper validate the effectiveness of the proposed model.

### Weaknesses
The introduction is poorly written. It fails to convey why the paper is important, even though the proofs presented later seem significant.

1. The paper does not effectively connect its theoretical proofs with the broader discussion. The introduction completely omits these key elements—rendering it, in a sense, largely vacuous. It fails to clarify the difficulty of the problem being addressed, the significance of the contribution, or the approach taken to solve it. If the proofs are indeed as important as I understand them to be, the authors should have summarized their challenges and implications right in the introduction. Otherwise, readers less familiar with the field will struggle to grasp the significance of these results, even if they read the conclusions of the proofs.

2. When X is a discrete sequence, is directly computing its derivative inherently difficult? Do WDT and IWDT (Inverse Wavelet Derivative Transform) address this challenge?

3. Is it difficult to construct a basis that possesses the inverse transform property? What is the practical or theoretical significance of having an inverse transform?

4. What is the meaning (or required properties) of wavelet basis functions? Does WDT satisfy these properties? Does WDT offer additional desirable properties beyond those of conventional wavelet transforms?

5. The paper employs $\psi^{(n)} $ (the n-th derivative of the mother wavelet $\psi$ ), yet for many mother wavelets $\psi$ , computing such derivatives is nontrivial. How are these derivatives obtained in practice? This constitutes a significant technical challenge that the paper should explicitly address—either through explanation or formal justification.

### Questions
Please address Weaknesses 1–5.

If Weakness 5 is adequately answered, I will raise my rating to 6.
If the responses to the other weaknesses are also satisfactory, I will consider raising my rating to 8.

### Soundness
3

### Presentation
1

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
This paper introduces the **Wavelet Derivative Transform (WDT)** for deep time series forecasting, addressing the limitations of Fourier-based and standard wavelet transforms. WDT applies wavelet analysis to the derivative of a time series, improving sensitivity to abrupt changes while preserving trends. The authors propose **WaveTS**, a multi-branch architecture that processes different derivative orders, refines coefficients with Frequency Refinement Units, and reconstructs sequences using the inverse WDT. On several time series benchmarks, WaveTS outperforms state-of-the-art models.

### Strengths
1. The paper is well-structured, clearly written, and easy to follow.

2. The core innovations are well-expressed and make sense.

3. The experiments are thorough, and the results surpass state-of-the-art methods.

### Weaknesses
1. The layout of Figure 2 appears cluttered, with too much text and dense labeling, making it hard to interpret. A revision is recommended.

2. Line 302 mentions the benefits of adding backcast loss, but there is no ablation study to support this. It should be added.

3. The citation format in Table 3 is incorrect; `\cite` should be replaced with `\citep`.

### Questions
See **Weakness.**

### Soundness
3

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
3

### Summary
This paper focuses on the task of time series forecasting. The existing methods still struggle in capturing multi-scale,  time-sensitive patterns. To solve these challenges, the proposed method uses multi-order Wavelet Derivative Transform (WDT), supported by theoretical analysis.

### Strengths
S1: The method is supported by theoretical analysis.

S2: The experiment is extensive.

S3: The idea of using multi-order Wavelet Derivative Transform is novel.

### Weaknesses
W1: The explanation of Figure 1, which is central to the paper's motivation, requires further clarification. The text states that ''(e)(f)the Fourier-derivative makes the spectrum stationary yet discards macro-trend information''. However, it is not immediately clear from the visual evidence provided in the figure how this ''discarding'' of the macro-trend is demonstrated. The similar problem exists for the sentence of  ''(e)(g)(h) The Wavelet Derivative Transform (WDT) retains those trends while offering complementary detail.'' It's hard to figure out the subfig (e)(f)(g)(h), which makes the motivation confused.

W2: Lack of clarity on the method's domain of superiority and failure modes. The paper provides a comprehensive evaluation of WaveTS on standard benchmarks but lacks a clear delineation of the specific time series characteristics for which the proposed method is most and least suited. Given the non-trivial complexity introduced by the multi-branch WDT architecture, it is crucial to understand its performance boundaries. 

W3: The figure 2, i.e., the framework of WaveTS, is too complex to understand.

W4: Some typos. According to LIne 322, Electricity is used. Is this called ECL in Table 1?

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

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
This paper proposes a new frequency-domain forecasting architecture grounded in multi-order wavelet derivatives, achieving solid empirical gains and offering an interpretable multi-scale design. The work effectively bridges traditional wavelet analysis with modern deep learning frameworks.

### Strengths
1. The paper introduces a new idea, i.e., using multi-order wavelet derivatives, that extends traditional wavelet transforms to a learnable, multi-resolution representation explicitly tied to derivative order, bridging signal processing and deep learning perspectives.

2. The model achieves consistent performance gains across diverse benchmarks, demonstrating robustness to both long-term and short-term forecasting scenarios.

### Weaknesses
1. My main concern is that the paper appears conceptually similar to DeRiTS, with the primary difference being the replacement of the Fourier-based derivative operator by a wavelet-based one. The overall framework and motivation seem closely aligned, which raises questions about the degree of novelty.

2. The ablation section does not sufficiently isolate the contributions of wavelet decomposition, derivative order, and fusion strategy. It is unclear which component contributes most to the gains.

### Questions
1. Are the wavelet parameters (e.g., basis, scale) fixed or learnable during training?

2. Could the proposed wavelet derivative operator be integrated into other architectures, such as CNNs or MLP-Mixers, beyond the current framework?

### Soundness
3

### Presentation
3

### Contribution
3
