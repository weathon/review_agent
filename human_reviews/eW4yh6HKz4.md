# CBQ: Cross-Block Quantization for Large Language Models

- Avg Score: 7.60
- Decision: Accept (Spotlight)
- Scores: 6, 6, 8, 10, 8

## Abstract
Post-training quantization (PTQ) has played a pivotal role in compressing large language models (LLMs) at ultra-low costs. Although current PTQ methods have achieved promising results by addressing outliers and employing layer- or block-wise loss optimization techniques, they still suffer from significant performance degradation at ultra-low bits precision. To dissect this issue, we conducted an in-depth analysis of quantization errors specific to LLMs and surprisingly discovered that, unlike traditional sources of quantization errors, the growing number of model parameters, combined with the reduction in quantization bits, intensifies inter-layer and intra-layer dependencies, which severely impact quantization accuracy. This finding highlights a critical challenge in quantizing LLMs. To address this, we propose CBQ, a cross-block reconstruction-based PTQ method for LLMs. CBQ leverages a cross-block dependency to establish long-range dependencies across multiple blocks and integrates an adaptive LoRA-Rounding technique to manage intra-layer dependencies. To further enhance performance, CBQ incorporates a coarse-to-fine pre-processing mechanism for processing weights and activations. Extensive experiments show that CBQ achieves superior low-bit quantization (W4A4, W4A8, W2A16) and outperforms existing state-of-the-art methods across various LLMs and datasets. Notably, CBQ only takes 4.3 hours to quantize a weight-only quantization of a 4-bit LLAMA1-65B model, achieving a commendable trade off between performance and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces Cross-Block Quantization (CBQ), a novel post-training quantization (PTQ) framework targeting large language models (LLMs). CBQ's core innovation is its cross-block dependency mechanism, which introduces dependencies across multiple blocks to mitigate error accumulation. Additionally, a coarse-to-fine preprocessing strategy and LoRA-Rounding are proposed to handle weight and activation outliers. The authors validate CBQ on various LLMs and demonstrate that it achieves state-of-the-art performance in low-bit settings, like W4A4, across a range of datasets.

### Strengths
Innovation: The idea of using cross-block dependencies to reduce quantization errors is both intuitive and impactful, as it tackles a recognized issue in PTQ for LLMs.

Experimental Scope: CBQ is tested extensively across multiple LLM architectures and low-bit settings, showing clear performance gains over prior methods.

Significance: Reducing computational overhead for LLMs without retraining is valuable for real-world applications, and CBQ provides a viable solution for efficient model deployment.

### Weaknesses
Baseline Comparisons: Although the results on LLAMA models are promising, the exclusion of more recent PTQ methods like BiE[1] limits the impact of the results. Including such baselines could provide a more complete picture of CBQ’s effectiveness.

Broader Applicability: The focus on language models limits the generalizability of CBQ. Expanding evaluations to other architectures, such as vision transformers and VLMs (vision language models), could strengthen the paper’s applicability.

[1] BiE: Bi-Exponent Block Floating-Point for Large Language Models Quantization. ICML 2024

### Questions
The paper introduces a coarse-to-fine preprocessing (CFP) strategy for handling weight and activation outliers, yet it is unclear how this approach compares in complexity and execution time against other established methods like OMSE or SmoothQuant under similar quantization settings. Could the authors clarify the computational trade-offs of CFP compared to these methods? Additionally, a broader comparison in terms of both quantization accuracy and processing efficiency would strengthen the evaluation of CFP's effectiveness.

Can the authors elaborate on the expected quantization efficiency for models larger than 100 billion parameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors find that optimizing for inter-layer dependencies is crucial, especially for lower-bit quantization. To address this challenge, they propose the use of cross-block loss where the loss is taken over a window of blocks and not just a single block. For the cross-block loss, in addition to using the L2-distance, the authors use KL-divergence between the activations. Inspired by AdaRound, the authors use LoRA parameters for accurate rounding. For outlier detection, the authors present a two step process. They use a quantile-based thresholding to identify an initial set of outliers following which they divide these coarse outliers into two sets by maximizing the inter-set variance and minimizing the intra-set variance.

### Strengths
- The paper clearly motivates the proposed cross-block quantization.
- CBQ shows substantial performance improvements over OmniQuant, especially in the 2-bit setting.
- The authors provide a thorough ablation study on several aspects of the proposed objective.

### Weaknesses
- The addition of LoRA parameters should take some hit in latency. The authors should present the latency numbers, especially for W2A16 and W4A4.
- Given that the paper focuses on two-bit quantization, comparison to several recent rotation-based methods such as FrameQuant [1] and QuIP [2] are omitted from the related works section and the results. The authors should compare against these methods for 2-bit quantization.
- In practice, most quantization frameworks use sub channel quantization for weights, the authors should report weight-only quantization numbers with sub-channel quantization for all bit-widths.

### Questions
- Majority of the experiments conducted in the paper are on the OPT and Llama-1 family of LLMs. It would be interesting to see some of the results presented in the paper on SOTA open-source LLMs like Llama-3 [3] and Gemma-2 [4]. 
- How exactly is W6A6 implemented, could the authors briefly discuss about its latency benefits by providing some numbers?

I am open to discussions and willing to reconsider my score.

[1] Adepu, Harshavardhan et al. “FrameQuant: Flexible Low-Bit Quantization for Transformers.” ICML 2024.

[2] Chee, Jerry et al. “QuIP: 2-Bit Quantization of Large Language Models With Guarantees.” NeurIPS 2023.

[3] Dubey, Abhimanyu et al. “The Llama 3 Herd of Models.” ArXiv abs/2407.21783 (2024): n. pag.

[4] Riviere, Gemma Team Morgane et al. “Gemma 2: Improving Open Language Models at a Practical Size.” ArXiv abs/2408.00118 (2024): n. pag.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Cross-Block Quantization (CBQ), a novel method designed to optimize post-training quantization (PTQ) for large language models (LLMs) by addressing inter-block dependencies and introducing a coarse-to-fine outlier handling strategy. CBQ's approach aims to preserve model accuracy at low-bit configurations by leveraging cross-block reconstructions and refining outlier quantization through LoRA-Rounding. The authors present experimental results demonstrating CBQ’s strong performance across different quantization bit widths and various LLM architectures, with promising efficiency for real-world deployment.

### Strengths
1. By focusing on inter-block dependencies, CBQ takes a proactive approach to minimize error accumulation, which is often a critical challenge in low-bit quantization. This dependency handling shows clear improvements in model accuracy.
2. The design of CBQ, particularly the coarse-to-fine outlier preprocessing and adaptive rounding, ensures flexibility across different LLM sizes, making it a practical choice for varied deployment needs.
3. The authors have included a wide range of experiments showing CBQ’s effectiveness in diverse quantization configurations, including settings that emphasize computational efficiency, which makes it relevant for both research and applied contexts.

### Weaknesses
1. While the paper demonstrates that overlapping blocks in CBQ contributes to performance, it lacks an in-depth analysis of how varying the overlap size impacts memory efficiency, latency, and overall quantization stability. Providing such details would clarify practical deployment considerations.
2. The coarse-to-fine preprocessing for outliers appears effective; however, an assessment of its necessity relative to simpler methods could be useful. It is unclear whether this specific strategy is essential or if comparable results could be achieved with simpler preprocessing.

### Questions
1. What would be the impact of reducing the overlap size in cross-block dependencies? Could reducing overlap compromise quantization efficiency, or are there scenarios where this would be advisable?
2. Could the authors clarify if the coarse-to-fine preprocessing provides unique benefits over standard outlier suppression methods? A comparative analysis would help isolate its specific contribution.
3. How do the authors envision CBQ scaling with models that incorporate multimodal inputs or models beyond text-based LLMs? Would additional adaptations be required?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper introduced CBQ  post-training quantization PTQ technique designed for compressing large language models under low-bit precision. The authors identified the primary challenge in ultra-low-bit quantization: the dependencies within and between layers that amplify quantization errors, especially as model size and parameter counts increase. To address this, CBQ incorporates a cross-block reconstruction strategy, which leverages both intra-layer and inter-layer dependencies by optimizing multiple transformer blocks within a sliding window approach. Additionally, the method employs a LoRA-Rounding technique to manage intra-layer dependencies and reduce computational costs, while an adaptive coarse-to-fine preprocessing strategy effectively handles outliers in weights and activations.

### Strengths
I have carefully read this work, I think the result of this work was convincing, and the approach was solid.

### Weaknesses
N/A

### Questions
I have read this paper multiple times before, and I fully agree with the authors' approach. I have no further questions.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Cross-Block Quantization (CBQ), a novel post-training quantization (PTQ) technique developed for large language models (LLMs). CBQ tackles critical inter- and intra-layer dependencies that compromise quantization accuracy at ultra-low bit settings, establishing long-range dependencies through cross-block reconstruction and managing intra-layer dependencies via adaptive LoRA-Rounding. The approach also incorporates a coarse-to-fine preprocessing method to optimize handling of weights and activations.

### Strengths
1. The paper offers an interesting insight: full model quantization introduces inter-layer correlations.

2. Experimental results on LLAMA1, LLAMA2, and OPT demonstrate substantial improvements in W2 and W4 settings, alongside a significant reduction in PTQ scenario.

### Weaknesses
1. The writing could be improved, especially regarding clarity in notation. While the concepts are interesting, it is difficult to follow due to unclear notation. For example, symbols like \(i\), \(j\), and \(K\) in Equation 3, as well as \(V\) in Equation 11, are not clearly defined when first introduced. It would help if each symbol had an explicit definition upon first use. Additionally, the term "scales" in the phrase "comparisons of the scales between adjacent layers..." lacks clarity. Specifying what "scales" refers to in the context of adjacent layers would enhance readability.

2. The range of models tested is limited. While the authors include results on LLAMA1, LLAMA2, and OPT, these are relatively dated models, and results on newer models like LLAMA3, Mistral, and Falcon should be included.

3. While the authors suggest that LoRA-Rounding was introduced to reduce computation, they do not evaluate this aspect in their experiments. To make this claim more compelling, the authors could include specific metrics or experiments comparing the computational efficiency of LoRA-Rounding to alternative approaches.

4. The authors should compare their method with other W4 approaches, such as AWQ. Comparing with AWQ would be valuable because of its relevant strengths or similarities, which could provide a more comprehensive evaluation of the method’s performance in a competitive context.

### Questions
1. Will the code be released?

2. LoRA-Rounding needs clearer explanation. Is *V* the rounding mask? On what type of matrix is it applied, and does each matrix in the model have its own *V*?

3. Has CBQ been tested for improvements in few-shot scenarios?

4. Figure 1 is intriguing; does this phenomenon appear in other models, like LLAMA3, Mistral, or Falcon?

### Soundness
3

### Presentation
3

### Contribution
3
