# DBellQuant: Breaking the Bell with Double-Bell Transformation for LLM Post Training Binarization

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Large language models (LLMs) demonstrate remarkable performance but face substantial computational and memory challenges that limit their practical deployment. Quantization has emerged as a promising solution; however, its effectiveness is often limited by quantization errors arising from weight distributions that are not quantization-friendly and the presence of activation outliers. 
To address these challenges, we introduce DBellQuant, an innovative post-training quantization (PTQ) framework that achieves nearly 1-bit weight compression and 6-bit activation quantization with minimal performance degradation. DBellQuant uses learnable transformation to map single-bell weight distribution to dual-bell distribution to reduce binarization error and smooth activations using inverse transformation. DBellQuant sets a new state-of-the-art by preserving superior model performance under aggressive weight and activation quantization. For example, on the Wikitext2 dataset, DBellQuant achieves a perplexity of 14.39 on LLaMA2-13B with nearly 1-bit weight and 6-bit activation quantization, significantly outperforming BiLLM’s 21.35 without activation quantization, underscoring its potential in compressing LLMs for real-world edge applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper’s key strength lies in pioneering near 1-bit weight and 6-bit activation post-training quantization (PTQ) for large language models (LLMs) with minimal performance loss, aligning with practical deployment needs for low memory and latency. However, it has critical weaknesses—including reproducibility issues from unspecified algorithm details, unclear connections to prior work, unaddressed causal/theoretical gaps in core mechanisms, lack of significant innovation, and minor errors—leading to a "Reject" score, though revision of these issues could warrant reconsideration.

### Strengths
1. This work achieves near 1-bit weight quantization combined with 6-bit activation quantization in the Post-Training Quantization (PTQ) scenario for the first time, with minimal performance loss. From a research direction perspective, it aligns with practical deployment needs of Large Language Models (LLMs), such as low memory usage and low latency.

### Weaknesses
1. There is a spelling error in Tables 4 and 5: the term "Activation" is misspelled as "Avtivation".  

2. The iterative steps of the core LTDB algorithm (Section 3.2) are only provided as simplified pseudocode in Appendix A.4. Critical details such as the optimizer type for gradient updates (e.g., Adam/SGD) and learning rate scheduling strategy are not specified. For the loss function coefficients λ_DTMD and λ_DTNP (Section 3.3), only their existence is mentioned, while specific values are not provided. For instance, Appendix A.5 only vaguely refers to a "loss coefficient of 100" without clarifying which loss term it corresponds to. These omissions severely hinder the reproducibility of the proposed method.  

3. The "Related Work" section (Section 2) merely lists existing methods without highlighting the connection between core limitations of prior work and the research gap addressed by this paper. For example, the essential differences between SmoothQuant’s "scaling factor redistribution" and the proposed "T-transformation" are not clearly compared or analyzed.  

4. In the method section (Section 3), the description of "how the T-matrix is integrated into LayerNorm weights" (Section 3.2) is only mentioned in a single sentence. No formulas or schematic diagrams are provided to illustrate the integration details, leaving this critical implementation step unclear.  

5. The paper only observes that "95% of the values of T⁻¹ are less than 1, thus compressing the activation range" (Section 3.4). However, it fails to explain a key logical question: why the inverse transformation of the T-matrix (designed for weight quantization) can恰好 suppress activation outliers? No causal relationship or theoretical reasoning is provided to support this observation.  

6. The reason why the DTMD loss causes T-matrix shrinkage is not analyzed. The paper only states that "DTMD causes T to shrink and needs to be compensated by DTNP" (Section 3.3), but does not explain why the gradient updates of DTMD tend to reduce the values of T (e.g., mathematical derivation of gradient direction). Additionally, no comparative experiments (e.g., using only DTNP vs. combining DTMD+DTNP) are conducted to verify the necessity of using two loss functions.  

7. The work lacks significant innovation. Adjusting weight distribution to adapt to quantization is not a new idea—QAT (Quantization-Aware Training) methods (e.g., BitNet) have already shaped weights into a double-bell distribution through training. This paper merely migrates this concept to the PTQ scenario. Moreover, there is no novel design in the "distribution transformation method": the T-transformation is essentially element-wise scaling, which is similar to the "weight-activation scaling factor redistribution" idea of SmoothQuant, with only the addition of learnability.  

Based on the above weakness, I would assign a Reject score. I look forward to the authors addressing the aforementioned problems in future revisions, and I would be happy to reconsider and raise my score accordingly.

### Questions
See "Weaknesses"

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DBellQuant, a post-training quantization (PTQ) framework that simultaneously achieves near 1-bit weight binarization and 6-bit activation quantization for large language models (LLMs). The core idea is to apply a learnable, per-channel transformation T to reshape the single-bell (Gaussian-like) weight distribution into a dual-bell form—arguably more amenable to binarization—while applying the inverse transformation $T^{−1}$
  to activations to preserve functional equivalence and suppress outliers. The method introduces a lightweight optimization algorithm (LTDB) with a custom dual-target loss and early stopping to learn $T$.

### Strengths
1) The authors conduct thorough experiments on multiple LLMs, demonstrating the effectiveness of DBellQuant across different model sizes and architectures. 

2) The authors perform extensive ablation studies to validate the effectiveness of their proposed Learnable Transformation for Dual-Bell (LTDB) algorithm and the impact of various hyperparameters.

### Weaknesses
1) While the transformation of weight distributions into dual-bell shapes is novel, the overall framework of DBellQuant can be seen as a combination of existing techniques (e.g., learnable transformations, activation smoothing). The authors do not sufficiently justify why this specific combination leads to significantly better results compared to other potential combinations or state-of-the-art methods. 

2) More directly, the paper compares against BiLLM and ARB-LLM but omits comparison with rotation-based PTQ methods (e.g., QuaRot, SpinQuant, DuQuant), which also aim to suppress outliers and enable low-bit quantization via learned or fixed orthogonal transforms. These works are cited but not evaluated.

3) The paper does not report the practical speedup or memory savings after LLM binarization using DBellQuant.

### Questions
How about the results of DBellQuant on reasoning large models and multi-modal large models?

### Soundness
2

### Presentation
3

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
This work connects the bimodal distribution with 1-bit quantization, which is somewhat intuitive. On this basis, a corresponding smooth coefficient is trained for each weight to form a bimodal distribution, and finally achieves SOTA results under the W1A6 setting.

### Strengths
This work believes that the bimodal distribution is beneficial for binarization, which is somewhat intuitive.
The writing is clear and easy to understand.

### Weaknesses
1. 
There is no strict theoretical connection between the bimodal distribution and 1-bit quantization. For example, in extreme cases, an entire channel may belong to a single peak, which limits the rigor of this work.

2.
After applying Smooth, the quantization error of the weights cannot truly reflect the quantization error because activations also have differences between value channels.

3.
Only results on the LLaMA-7B model are reported, lacking results on larger-scale models such as 70B. This is not conducive to demonstrating the universality of the method. Additionally, reporting results on models like Qwen and Mixtral would enhance the universality of the method.

4.
The design of the training loss is also overly complex, introducing too many uncertain factors such as hyperparameter adjustments and early termination of training.

5.
The 1-bit quantization in arb-llm involves too many smooth operations, which actually require complex dequantization and cannot utilize low-precision tensor cores, often providing no help in reducing computational load. Furthermore, it lacks detailed latency statistics (especially for large-batch prefill). 6-bit quantization often fails to achieve acceleration (generally, 4/8-bit quantization is more conducive to acceleration).

### Questions
Activated 6-bit quantization is often detrimental to memory alignment. It can provide activated 4-bit results. Are these results usable?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DBellQuant, a post-training quantization (PTQ) framework targeting highly compressed large language models by applying simultaneous near-1-bit weight binarization and 6-bit activation quantization with minimal performance loss. The central innovation is the Learnable Transformation for Dual-Bell (LTDB) algorithm, which transforms unimodal (single-bell) weight distributions into dual-bell forms, making weights more amenable to binarization while inversely transforming activations to reduce outlier effects and facilitate low-bit quantization. The framework is empirically shown to outperform state-of-the-art PTQ methods, such as BiLLM and ARB-LLM, on several LLM families and benchmarks, preserving accuracy and language modeling ability under aggressive quantization.
This paper presents a valuable contribution to LLM post-training quantization through the DBellQuant framework, whose core LTDB algorithm reshapes weights into dual-bell distributions to enable high-compression quantization (near-1-bit weights, 6-bit activations) with minimal performance loss, outperforming SOTA methods like BiLLM and ARB-LLM. However, it suffers from relatively modest overall novelty, an unresolved 4-bit activation limitation, and several presentation and clarity issues. Nonetheless, the approach demonstrates genuine promise for efficient LLM deployment and could serve as a solid foundation for future work.

### Strengths
(1) Principled Foundation and Theoretical Motivation: The authors offer a thorough mathematical argument for why dual-bell weight distributions are better suited for binarization. This is explicitly elaborated in Section 3.1 and Appendix A.16, where they present formal proofs that demonstrate the soundness of the transformation and clarify its impact on the target weight distributions.
(2) Solid Empirical Validation: Experiments span multiple LLMs (LLaMA, OPT families, etc.) and metrics (perplexity on WikiText2 and C4; QA accuracy on ARC, PIQA, etc.). Quantitative results consistently demonstrate that DBellQuant outperforms strong post-training quantization baselines such as BiLLM and ARB-LLM. For instance, Table 1 and Table 2 directly show measurable improvements, especially at aggressive quantization levels.
(3) Ablation and Robustness Studies: The impact of block size, activation bits, and loss design choices are all explored systematically (Tables and commentary in ablation section). These experiments reinforce the reliability of the findings and offer valuable insights into the design choices.

### Weaknesses
(1) Limited Novelty Relative to Prior PTQ/Quantization Methods: Although the transition from single- to double-bell distributions is well-motivated and the learnable transformation is elegant, many individual components (e.g., smooth-scaling, block-wise transformations, loss design) are iterative extensions or combinations of well-established quantization techniques.
(2) Insufficient Depth in Activation Quantization Analysis: While it claims that the inverse transformation effectively mitigates activation outliers--supported by visual illustrations (Figures 2, 5) and perplexity improvements--it falls short of providing rigorous statistical or theoretical validation, such as quantifying reductions in kurtosis, and omits direct comparisons of activation distributions before and after smoothing. Moreover, the method is currently limited to 6-bit activations, with model collapse at 4 bits, undermining its claim to state-of-the-art status in extreme compression; this inability to support 4-bit activations represents a barrier for edge deployment scenarios.
(3) Limited Discussion of Learned Transformation Properties: While Theorem 1 (Section 3.2) establishes existence, the main text offers insufficient insight into the practical challenges of learning such transformations in high-dimensional settings, especially with constraints on invertibility and computational overhead. 
(4) Editing Issues: The manuscript contains several typographical errors (e.g., “doubel-bell,” “algrithm”) and occasional awkward or imprecise phrasing that detracts from readability and professionalism.

### Questions
(1) Could the authors provide more analysis on why DBellQuant fails when pushing activation quantization below 6 bits? Additionally, since the current implementation only supports INT8 activation quantization, it would be valuable to explore whether integrating emerging low-precision floating-point formats (e.g., MXFP6, MXFP4, or NVFP4), which are designed to maintain representational capacity at very low bitwidths while supporting efficient hardware execution, could mitigate these limitations.
(2) Did the authors explore combining DBellQuant with other methods that promote incoherence or regularization (e.g., Hadamard-based transforms, SpinQuant and Atom) and how would these interact? 
(3) Regarding outlier robustness: does LTDB effectively handle strongly outlying weights or highly non-Gaussian channels, or does it implicitly assume that the input weights are already approximately Gaussian?
(4) Could the authors offer additional discussion on how well DBellQuant generalizes to LLM architectures beyond OPT and LLaMA, or are there theoretical constraints limiting its generality?

### Soundness
3

### Presentation
2

### Contribution
3
