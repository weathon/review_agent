# Consistency Flow Model Achieves One-step Denoising Error Correction Codes

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Error Correction Codes (ECC) are fundamental to reliable digital communication, yet designing neural decoders that are both accurate and computationally efficient remains challenging. Recent denoising diffusion decoders with transformer backbones achieve state-of-the-art performance, but their iterative sampling limits practicality in low-latency settings. We introduce the Error Correction Consistency Flow Model (ECCFM)} an architecture-agnostic training framework for high-fidelity one-step decoding. By casting the reverse denoising process as a Probability Flow Ordinary Differential Equation (PF-ODE) and enforcing smoothness through a differential time regularization, ECCFM learns to map noisy signals along the decoding trajectory directly to the original codeword in a single inference step. Across multiple decoding benchmarks, ECCFM attains lower bit-error rates (BER) than autoregressive and diffusion-based baselines, with notable improvements on longer codes, while delivering inference speeds up from 30x to 100x faster than denoising diffusion decoders.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the use of diffusion models for the task of error correction code (ECC). While prior works have demonstrated promising results using diffusion models for ECC, they suffer from high computational costs. To improve efficiency, the authors adopt the consistency model, which enables one-step denoising and significantly reduces computational overhead. Experimental results show that the proposed framework achieves a lower bit-error rate compared to previous methods.

### Strengths
* The paper is easy to follow.
* The work focuses on an interesting and important application.
* Experimental results indicate that the proposed method achieves promising performance with improved efficiency.

### Weaknesses
* Notation
    * The definition of the function f at Line 223 and in Equation (3) appears problematic. The output of f should be a prediction of the codeword rather than a probability distribution. The correct formulation should align with Algorithm 1, e.g., $d(f_\theta(x_r, r), x_0)$ instead of $d(f_\theta(x_r, r), \delta(x - x_0))$.
    * The notation $L_{\text{Consistency}}$ is inconsistent with the previous line at Algorithm 1—it should be written as $L_{\text{EC-CM}}$.
* Claims Without Sufficient Support
    * An ablation study on the use of soft syndrome is missing. Since this is a key contribution of the paper, a comparison between using soft and hard syndromes should be included to substantiate the claimed advantage. Additionally, it would be valuable to report results when incorporating an explicit soft-syndrome loss in the overall objective to validate the design choice.
    * The paper claims that the proposed objective in Equation (3) provides a stronger learning signal than the conventional objective in Equation (2). However, no experimental evidence is provided to support this claim. It remains unclear whether the proposed objective improves performance, convergence speed, or both.
* Limited Insight
    * While I typically avoid judging contributions solely based on novelty, this paper seems to offer limited conceptual insight. The motivation—to enhance the efficiency of diffusion-based denoisers for ECC—is clear, but the proposed solution (adopting the consistency model) is rather straightforward, as the efficiency benefit of consistency models is already well established. Beyond combining existing techniques such as the consistency model and soft-syndrome mechanism to achieve better empirical results, the paper provides limited generalizable insights for broader applications.
* Inaccurate statement
    * The description from Lines 211–214 appears inaccurate. The Boundary Condition and Self-Consistency are not “naturally inherent” properties of ECC data. Instead, they are intrinsic properties of the consistency model framework, imposed on the function $f_\theta$ through its parameterization and training objective. These properties are data-agnostic, whereas the text incorrectly attributes them to ECC itself.

### Questions
* During inference, how is the value of $\sigma$ in Equation (7) determined?

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
4

### Summary
The authors address the long-latency problem of DDECC by introducing a consistency flow model. In addition, they employ a soft-syndrome formulation to replace the hard-syndrome approach. The results are meaningful, achieving better performance than the baseline, CrossMPT, while maintaining similar inference time.

### Strengths
The motivation of the paper is clear, and the methodology is well adapted from the machine learning field. The modification of the model for the channel coding context is also well derived. The simulation results overall appear to be accurate and convincing.

### Weaknesses
Most parts of the paper focus on the training method. However, I am curious about the decoding architecture. The authors mention that CrossMPT is used — does this mean the architecture is exactly the same as CrossMPT? How does e_T (the second parameter of f_theta) affect the decoding process? Do we need multiple models depending on the value of this second parameter to perform decoding? If so, this could be a drawback of the proposed approach. Please clarify this point.

In the main body of the paper, it would be valuable to include simulation results for ECCFM with the ECCT architecture, not only with the CrossMPT architecture. For a fair comparison with DDECC, both ECCFM and DDECC should share the same underlying architecture (ECCT). Although I noticed that the authors included results with the ECCT-based architecture in the Appendix, it would be better to include them in the main text.

The graphs in Figure 4 (particularly the FER graph for Polar code) appear somewhat abnormal. Increasing the SNR step size resolution from 1 dB to 0.5 dB and raising the maximum number of testing trials from 10^7 to 10^8 would improve the reliability of the results.

### Questions
The sentence above Eq. (6) mentions the “soft-syndrome error condition for each row j,” but the formulation of the soft-syndrome error condition sums over all j, losing the dependency on j. Could the authors clarify this inconsistency?

In Eq. (8), what is the specific reason for including the soft-syndrome loss in the total loss term? Is it primarily for stabilizing training, or does it also contribute to performance improvement?

Many com fair comparison, the number of training epochs should be aligned.parative works use 1000 epochs, while this paper uses 1500. For a

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduced a new architecture-agnostic training framework for high-fidelity one-step decoding. It seems that this work integrates the consistency model framework to transformer-based decoders well.

### Strengths
- This paper is the first to apply the consistency model framework to error-correcting codes (ECC), achieving state-of-the-art performance.
- The proposed approach replaces the reverse process of the diffusion model with consistency model framework, effectively improving inference efficiency and reducing overall latency compared to DDECC.
- For the noise condition, this paper employs soft-syndromes, whereas DDECC uses hard syndromes, resulting in smoother trajectories and more stable training.

### Weaknesses
- In the modifying the loss function, the authors applied triangle inequality. However, since binary cross entropy (BCE) is not a distance metric, it is unclear whether applying the triangle inequality is theoretically valid in this context.
- As the experiments were conducted only on the ECCT architecture, it may be inappropriate to claim the model-agnostic properties. Additional results using other architectures, such as CNN would strengthen this claim.
- When adopting the consistency model (CM) framework, it would be helpful to quantify how much the training cost was reduced compared to other models. A more detailed analysis—such as reporting FLOPs or the number of parameters—would also be beneficial.
- The texts in Figure 5,7,8,9, and 10 are too small and difficult to read.
- Equation (6) should add a minus sign at the front, since it represents the binary cross-entropy between the estimated syndrome and the all-zero syndrome.
- In Equation (7), the “+” following the 1/2 should be a “–”. This is because, under BPSK modulation, a valid codeword satisfies the parity condition with an even number of ones, resulting in a soft syndrome value of 0. If we follow equation (7), the soft syndrome becomes 1 when the codeword is valid, which contradicts the statement in the paper.

### Questions
- The paper states that ECCFM employs a Transformer architecture with cross-attention. However, it is unclear which specific model was used. Is this architecture distinct from CrossMPT or a variant of it? If it differs from CrossMPT, please include the performance of the neural decoder used in ECCFM for comparison.
- In Equation (7), the “+” following the 1/2 should be a “–”. Under BPSK modulation, a valid codeword satisfies the parity condition with an even number of ones, leading to a soft syndrome value of 0. According to the current formulation in Equation (7), a valid codeword yields a soft syndrome of 1, which contradicts the intended parity-check behavior.
- In Table I, the results are presented only as numerical values. It would be beneficial to include corresponding figures for representative cases to enhance interpretability and comparison.

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
2

### Summary
This paper introduces the Error Correction Consistency Flow Model (ECCFM), a framework that enables single-step decoding for error correction codes. By using an "soft syndrome" condition, it matches the state-of-the-art accuracy of slow diffusion-based decoders while delivering massive 30-100x speedups, making high-performance neural decoding practical for low-latency applications.

### Strengths
- The method innovatively introduces a 'soft syndrome' to solve the critical problem of non-smooth trajectories when applying consistency models to ECC decoding.
- It uniquely achieves both state-of-the-art decoding accuracy and massive inference speedups of 30-100x over diffusion-based methods.
- The framework has high practical value for low-latency applications and is model-agnostic, making it widely applicable to different network architectures.
- Its claims are substantiated by rigorous and comprehensive experiments across various standard codes with fair comparisons to strong baselines.

### Weaknesses
- The study's evaluation is confined to the ideal AWGN channel, leaving its effectiveness in more realistic fading channels unexplored.
- The training process is potentially sensitive and highly dependent on the construction of a smooth trajectory, which may require careful tuning for different codes.
- Its core idea is a clever adaptation of consistency models from another field, rather than a fundamentally new theoretical invention. I'm more familiar with consistency models than with error-correcting codes, so it's hard to assess the novelty.

### Questions
- How robust is the core 'soft syndrome' method under extremely low SNR conditions, where unreliable Log-Likelihood Ratios could disrupt the smoothness of the decoding trajectory and lead to performance degradation?
- Have the authors experimented with multi-step sampling for ECCFM, and if so, does it provide a meaningful improvement in Bit Error Rate, offering a flexible trade-off between decoding latency and accuracy?
- What are the primary technical challenges in extending the ECCFM framework to more complex channel models like Rayleigh fading, and would it require a fundamental redefinition of the 'soft syndrome' to maintain a learnable trajectory?

### Soundness
2

### Presentation
3

### Contribution
2
