# Hybrid Mamba–Transformer Decoder for Error-Correcting Codes

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We introduce a novel deep learning method for decoding error correction codes based on the Mamba architecture, enhanced with Transformer layers. Our approach proposes a hybrid decoder that leverages Mamba’s efficient sequential modeling while maintaining the global context capabilities of Transformers. To further improve performance, we design a novel layer-wise masking strategy applied to each Mamba layer, allowing selective attention to relevant code features at different depths. Additionally, we introduce a progressive layer-wise loss, supervising the network at intermediate stages and promoting robust feature extraction throughout the decoding process. Comprehensive experiments across a range of linear codes demonstrate that our method significantly outperforms Transformer-only decoders and standard Mamba models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a hybrid Mamba-Transformer architecture for decoding error correcting codes in wireless transmissions. This scheme is ablation tested by removing components - only the Mamba, the transformer are tested against the use of both. The unique characteristics of SSMs allowing retention of very long context complements the transformer's ability to fit to prior training data in a shorter context window with high fidelity, allowing high decode fidelity here. The approach is contrasted against prior art using machine learning methods for error correcting code decoding.

### Strengths
The paper's approach outperforms the state of the art and is shown to be the optimal design given the components the authors have used - ablations demonstrate that removal of components from this system degrades performance. The bit-error based binary cross entropy loss ensures decoder training remains stable.

### Weaknesses
The given architecture seems much more complex than prior art, and given that channel noise decoding and wireless systems are real-time systems, I would ask what the latency is and whether this high-performance algorithm is suitable for the applications the authors have envisioned. 

I also believe the paper would benefit from a hyperparameter sweep of the network, as it is the experiments seem to have been conducted for a given parameter set but we do not know if that is the optimal one. A balance of latency and quality would be necessary for these applications, and a Pareto plot of architecture/parameter count normalized by latency plotted against quality may be illustrative.

### Questions
Would the authors say that this method remains unproven for real applications, given the complexity of transformers? Where do they see these approaches being used for real world deployments? Server-side decoder deployments may be unrealistic given latency and transmission requirements for wireless systems, is this intended for the edge?

### Soundness
3

### Presentation
4

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
This paper presents ECCM, a hybrid decoder that integrates Mamba and Transformer architectures for error correction code (ECC) decoding, marking the first use of Mamba’s state-space model in this domain. By combining Mamba’s efficient sequential processing with Transformer’s global attention, the method improves computational efficiency and captures long-range bit dependencies. The authors introduce a new masking strategy for Mamba layers, alternate Mamba and Transformer blocks, and apply progressive layer-wise loss, showing improved performance.

### Strengths
The paper is novel as it introduces the first use of Mamba architecture for ECC decoding. The approach shows consistent improvements across various code families, with experiments clearly validating the impact of each component including the Mamba architecture, masking strategy, and layer-wise loss, supporting the authors’ design choices.

### Weaknesses
Despite the interesting idea, the paper has some clarity issues that make the method difficult to understand and implement correctly. These issues are detailed further in the questions section.

### Questions
1. In Sec: 4.2 and 4.3 the authors state that if s_i=s the sample stops, and later outputs are “not calculated” and “not summed” in the loss. Could you add a short pseudocode block showing how this is implemented per sample within a batch during training and testing? And, could you please add a brief layer-usage summary showing where samples stop (i_"last" ) in particular, how many samples terminate before the final block?
2. For Table 2, could you explicitly state the number of blocks used in the ‘Transformer only’ and ‘Mamba only’ settings?
3. The authors train the Hybrid model with Adam at 2.5×10^-4and cosine annealing down to  10^-10(1000 batches/epoch). Could you briefly justify diverging from ECCT/CrossMPT LR/schedules? For Table 2, could you confirm that the ‘Transformer-only’ used the same LR/schedule? For comparability, please add a control run using the ECCT/CrossMPT learning rate and schedule.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Error-Correcting Code Mamba-Transformer (ECCM), a hybrid neural decoder architecture for linear block codes. The design alternates between Mamba (State-Space Model) layers and standard Transformer attention layers, aiming to leverage the linear-time complexity of Mamba for sequential modeling while retaining the global contextual modeling capabilities of the Transformer.

The authors claim three primary contributions: 1) The **hybrid Mamba-Transformer architecture** itself, 2) a new **layer-wise mask for Mamba layers,** **$f(H)$,** derived from parity check matrix, 3) a **progressive layer-wise loss** **function** combined with a syndrome-based early-exit mechanism to stabilize training and reduce inference latency.

Experiments are conducted on BCH, Polar, and LDPC codes. The results show that the proposed ECCM model achieves performance that is on par with, or in some cases (notably BCH) superior to, existing Transformer-only decoders, while offering improved computational complexity.

### Strengths
1. **Novel and Timely Architecture:** The integration of Mamba, a recent and efficient SSM, into the ECC decoding problem is a new architecture for model-free channel decoders. The hybrid approach achieves strong decoding performance. 
2. **Strong Empirical Performance:** The model achieves highly competitive, and in the case of several BCH codes, state-of-the-art performance compared to strong Transformer-based baselines like AECCT and CrossMPT. This validates the practical efficacy of the paper’s approach.

### Weaknesses
1. **Inconsistencies, Typos, and Presentation Issues:** The paper suffers from many typographical errors and presentation issues that reduce clarity and undermine its professional quality.
    - **Formulation and Notation Errors**
        - Eq (30) & (31): The calculation of error vector $z$ and the estimated codeword $\hat{c}^{i}$ should reference the original received vector $y[]$, not the model input $y_{in}[]$ since $y_{in}=[|y|, s]$.
        - Eq (28): The output dimension $N$ of $o^{i}$ should be $n$, given its relationship to the error vector $z$.
        - Eq (4): The notation uses a semicolon ($;$), which conventionally denotes vertical concatenation, whereas the operation required by the stated dimensions is horizontal concatenation.
        - Eq (31): The iterator should use a different variable than $n$ (e.g., $m$), and the upper value should be $n$ (code length).
        - Eq (28) and (29): The paper used the following rule for hard-decision vector: positive y → 0 (binary) and negative y → 1 (on page 3). Because of this rule, it seems that (28) should be $s^i = H \cdot (o^i < 0.5)$ instead of $s^i = H \cdot (o^i > 0.5)$.
    - **Figure-Text Inconsistency**
        
        In Figure 1(b), the Mamba block diagram is misleading. It correctly shows a SiLU activation for the $z$ path, but omits the SiLU activation for the $u$ path after the Conv block. This contradicts the text, as Eq (7) explicitly states $u_{conv}^{i}=SiLU(Conv^{i}(u^{i}))$.
        
    - **Poor Presentation**
    In Figure 1(b), the final output label ($y^i$) is poorly rendered and overlaps with the output arrow, demonstrating a lack of attention to detail and reducing the diagram's readability.
Definition discrepancy in Mamba mask $f(H)$: The mathematical formulation in Eq (13) and (14) is misleading. The paper’s Mamba mask $f(H)$ is defined with dimensions $(n-k) \times (2n-k)$, yet the paper applies it using indices $d $ and $s $ (model/state dimensions), as seen in $f(H)[l, d]$ and $f(H)[l, s]$. This indexing is dimensionally inconsistent with the mask's definition.
3. **Lack of empirical runtime metrics:** The conclusion states that ECCM achieves *“low and improving inference speed”*. However, the main text provides only the big-O complexity (Section 7.2) without any empirical latency, throughput, or FLOPs.

### Questions
1. **Runtime evidence:** Were empirical measurements (e.g., samples/sec, latency, GPU memory) taken to support the “*low and improving inference speed*” claim? If so, please include them or provide a runtime benchmark appendix.
2. **Clarification on the SCL Baseline for Polar Code Comparison:** In Appendix 12.2 and Table 4, the paper compares ECCM and CrossMPT against a classical "SCL (L=32)" baseline. Based on this, the authors conclude that ECCM "meaningfully closes the distance to the strong SCL baseline". However, the baseline is not specified as being CRC-Aided (CA-SCL), which is the standard high-performance SCL decoder configuration in modern literature. Could the authors clarify whether the "SCL (L=32)" baseline used in Table 4 includes a CRC? If it does not, how can the claim of closing the gap against a "strong SCL baseline" be justified, given that a ‘plain’ SCL is a significantly weaker baseline than the standard CA-SCL?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a hybrid deep learning-based decoder for error-correcting codes, combining the Mamba architecture with Transformer layers. The proposed model aims to leverage Mamba's efficient sequential modeling while maintaining the global contextual modeling of Transformers. The method is evaluated on a range of binary linear block codes, including BCH, Polar, LDPC, and MacKay codes. The authors report improved decoding performance and inference speed relative to previous neural network-based decoders, such as Transformer-only architectures and the CrossMPT model.

### Strengths
The combination of Mamba layers with Transformer layers is novel in the context of ECC decoding. This hybrid architecture appears to improve both decoding performance and computational efficiency relative to previous machine learning-based approaches.

### Weaknesses
I also reviewed this paper for NeurIPS (Reviewer ID: u6gT). This submission appears essentially identical to the NeurIPS version, which was, in my view, justifiably rejected.

While the paper is technically correct (though it could have been written more clearly) and the proposed hybrid architecture is interesting, its contribution remains limited in both novelty and impact. The literature on machine learning-based decoders for error-correcting codes is now very extensive, and this work does not meaningfully advance the state-of-the-art, either in methodology or insight.

Overall, the paper is solid but rather incremental. It would be more appropriate for a specialized or mid-tier venue rather than a top-tier conference like ICLR, which typically expects stronger conceptual contributions or more significant performance improvements (the proposed decoder does not beat state-of-the-art decoders).

### Questions
I do not have any questions. The paper is technically correct, but in my opinion, the contribution is too limited to warrant acceptance.

### Soundness
2

### Presentation
2

### Contribution
2
