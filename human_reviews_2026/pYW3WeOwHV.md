# Optimal Formats for Weight Quantisation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Weight quantisation is an essential technique for enabling efficient training and deployment of modern deep learning models. However, the recipe book of quantisation formats is large, and formats are often chosen empirically. In this paper, we propose a framework for systematic design and analysis of quantisation formats. By connecting the question of format design with the classical quantisation theory, we show that the strong practical performance of popular formats comes from their ability to represent values using variable-length codes. We frame the problem as minimising the KL divergence between original and quantised model outputs under a model size constraint, which can be approximated by minimising the squared quantisation error, a well-studied problem where entropy-constrained quantisers with variable-length codes are optimal. We develop nonlinear quantisation curves for block-scaled data across multiple distribution families and observe that these formats, along with sparse outlier formats, consistently outperform fixed-length formats, indicating that they also exploit variable-length encoding. Finally, by using the relationship between the Fisher information and KL divergence, we derive the optimal allocation of bit-widths to individual parameter tensors across the model’s layers, saving up to 0.25 bits per parameter when applied to large language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explores various tweaks to scalar quantization formats and proposes minor improvements over existing practices.
However, I am not sure what is the main takeaway here.

### Strengths
The paper does a thorough evaluation of various settings of scalar quantization.

### Weaknesses
- The overall message of the paper is unclear.
- Concepts are sometimes not introduced or introduced in inappropriate places (Lloyd-max, Huffman coding).
- Some proposed schemes (i.e. huffman compression) are impractical and cannot be used in efficient GPU kernels.
- Saying that block schemes perform variable-length encoding seems far fetched.
- Variable bit allocation could be explored more and compared to previous works like Evopress.

### Questions
What is the main takeaway for people who work with scalar quantization here?

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
2

### Summary
The paper proposes a theoretical framework for choosing the optimal data format for quantization. It first demonstrates that it is sufficient to minimize the squared error between the quantized and non-quantized tensors to find the optimal data format, then compute the optimal format for some known distribution. For unknown distributions, it proposes to fit the experimental distribution with some known ones, using scaling or k-means.
Under this framework, the paper show that variable-length code format consistently outperform fixed-length ones. It is also shown that optimal data format choice can also save up to 0.25 bit per parameter for LLMs.

### Strengths
- The paper advances the formalization of quantization data format selection, which is often tackled only empirically;
- Mathematical background seems solid, the SoTA seems adequately cited;
- Supplementary materials is rich and support some asumptions and approximations made in the paper.

### Weaknesses
- I found the paper hard to read and follow. Overall structure could be improved. The figures are way out of place with the text. Some figures mentioned in the main text are missing (figure 8, 29, 33...) only to be found in the supplementary material.
- Overall, it seems to me that the main paper is not entirely self-supporting without the help of supplementary material.
- Actual quantization results on LLMs models are completely absent from the paper, and again, can be found only in supplementary material.
- The reduction of the optimal data format problem finding to the minimization of the squared error between quantized and unquantized tensors relies on a lot of approximations. These approximations are further accumuled with the need to fit unknown distributions. Overall, I find it hard to properly appreciate the validity of these, even though some of them are addressed in supplementary material.

### Questions
I would appreciate if more insight could be provided regarding the various hypothesis made in the paper. I think that there is both too much information and too little information: the list of all known distribution's optimal format could be shortened. The related work comes very late in the paper with questionable impact. What do the authors think about perhaps simplifying these sections and bringing more insights on the main propositions and demonstrations notably from the supplementary materials?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the design of optimal quantization formats for neural network weight compression. The authors frame the problem as minimizing KL divergence between original and quantized model outputs under a memory constraint, which they approximate via Fisher-information-weighted squared quantization error. This theoretical reduction enables the application of classical quantization theory techniques to neural network weight compression. They derive optimal element-wise quantizers based on the cube root density rule and extend these to block-scaled formats. A key insight is that effective quantization formats exploit variable-length encoding through either block absmax scaling, sparse outlier storage, or explicit lossless compression. The authors also propose a Fisher-information-based scheme for optimal bit-width allocation across layers. Experiments on multiple LLM families (Llama 3, Qwen 2.5, Gemma 3, Phi 4) validate these insights.

### Strengths
- The paper establishes a principled theoretical framework for analyzing quantization formats by reducing the problem to Fisher-information-weighted squared quantization error, enabling the application of classical quantization theory to neural network weight compression. This is a significant, principled contribution that enables systematic format design rather than ad-hoc heuristics.
- The power of this theoretical framework is demonstrated by the authors' ability to directly leverage the rich domain of classical quantization theory to derive concrete technical results: cube root density quantizers for block-scaled Normal, Laplace, and Student-t distributions; the signmax scaling scheme; and a variable bit allocation scheme based on Fisher information. This connection to established theory provides both rigor and a pathway for future advances.
- The variable-length encoding insight provides a unifying explanation for why seemingly disparate techniques (block scaling, sparse outliers, compression) succeed. This is valuable both conceptually---explaining _why_ current methods work---and practically---suggesting that future format designs should explicitly consider variable-length encoding mechanisms.
- The experimental evaluation is comprehensive and rigorous, covering 11 models across 4 families, multiple formats, both direct-cast and QAT settings, and extensive ablations (block size, scale format, symmetric/asymmetric variants). Validation on synthetic data before real models strengthens confidence in the approach.
- The paper is exceptionally well-written with clear progression from problem formulation to theory to empirical validation. Figures effectively communicate key insights and extensive appendices provide implementation details without cluttering the main narrative.
- Quantization is critical for sustainable model training and inference. This paper makes important contributions to our theoretical understanding of quantization---a key step toward developing better methods and providing principled guidance for practitioners.

### Weaknesses
- The authors only empirically investigate transformer LLMs, and even among these, Gemma models exhibit behavior that deviates from the theoretical predictions. This raises concerns about the generality of the framework. If discrepancies arise within transformer LLMs alone, it is unclear how well the insights would extend to other architectures such as CNNs, GNNs, or state-space models.
- The theoretical framework relies on three approximations: second-order Taylor expansion of KL divergence, diagonal Fisher approximation, and constant-per-tensor Fisher. The authors provide extensive empirical validations of their assumptions (Figures 10-12) across different models, but it remains unclear whether these approximations hold more generally outside the specific experimental conditions tested.
- As the authors note, even if the cube root density quantizers are theoretically optimal, the practical utility is limited by optimized implementations and hardware support. As the paper's focus is on theoretical insights rather than hardware efficiency, this is not a critical flaw, but it does limit immediate applicability.

### Questions
The following questions are intended to help better understand the scope of the work and to think about future directions. These are challenging topics, and a lack of definitive answers is perfectly fine and will not be held against the work.

1. Can you provide more insight into why Gemma models show different behavior compared to Llama, Qwen, and Phi families? What additional analysis or probing have you conducted to understand the source of this discrepancy?
    - Gemma 3 is architecturally distinct from the other models evaluated---it uses a 5:1 local-to-global attention layer split (with different RoPE base frequencies for local and global attention layers) whereas the other models use standard global attention throughout. Do you believe these architectural differences contribute to the observed discrepancies?
    - What do you think the implications of this discrepancy are for the generality of your theoretical framework?
2. The work focuses exclusively on transformer LLMs. How do you anticipate the framework and its underlying assumptions would hold up for fundamentally different architectures, such as CNNs, state-space models, or GNNs? Are there specific architectural features (e.g., different weight distributions, inductive biases) that you believe would make the framework more or less applicable?
3. Can you characterize the regime where your approximations are valid? For instance, under what conditions (model architectures, weight distributions, quantization bit-widths) can practitioners expect any of the three key approximations to break down?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a theoretical framework for designing quantization formats by minimizing the KL divergence between quantized and reference models under a memory constraint. By connecting modern quantization with classical rate–distortion theory, the authors show that efficient schemes perform well because they implicitly use variable length encoding. They derive optimal elementwise quantizers for common distributions, introduce new scaling methods such as RMS, absmax, and signmax, and present a Fisher information based rule for allocating bits across tensors. Experiments on large language models including LLaMA 3, Qwen 2.5, Gemma 3, and Phi 4 show that formats exploiting variable length encoding through block scaling, sparse outlier storage, or compression consistently outperform fixed length ones. The study provides a principled explanation for the effectiveness of formats such as NF4 and SF4 and identifies uniform quantization with lossless compression as the theoretical optimum.

### Strengths
The paper provides a solid theoretical perspective by linking neural network quantization with classical information theory, offering useful insights into why certain formats perform well. It introduces new scaling schemes and a Fisher information based bit allocation rule that appear to improve efficiency across model tensors. Experiments on several large language models support the proposed framework and suggest its potential practical value.

### Weaknesses
I have the following concerns about the paper:

1. Equations (1) and (2) require stronger justification: Minimizing the KL divergence does not necessarily guarantee that the model’s accuracy will be preserved, and the validity of Equation (2) needs a clearer theoretical explanation.

2. Additional background is needed: to help readers follow the technical development. For instance, the sections around lines 143–146 and 153–157 would benefit from more context and introductory material.

3. Experimental evaluation is insufficient: Given the extensive prior work in this area, it is important to include comparisons with existing quantization methods such as SmoothQuant and QuaRot to better demonstrate the advantages of the proposed approach.

4. This work has limited focus on practical hardware efficiency, as the proposed non-linear quantization formats may be difficult to implement or accelerate on existing hardware.

### Questions
How does minimizing KL divergence ensure preservation of task-level accuracy, and could the authors provide theoretical or empirical evidence linking KL divergence to model performance?

What assumptions are required for Equation (2) to hold, and how sensitive are the results to violations of these assumptions?

Can the authors expand the background discussion to better contextualize the derivations around lines 143–157?

### Soundness
2

### Presentation
2

### Contribution
2
