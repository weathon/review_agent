# Accelerating Error Correction Code Transformers

- Avg Score: 6.00
- Decision: Reject
- Scores: 3, 8, 8, 5, 6, 6

## Abstract
Error correction codes (ECC) are crucial for ensuring reliable information transmission in communication systems. Choukroun & Wolf (2022b) recently introduced the Error Correction Code Transformer (ECCT), which has demonstrated promising performance across various transmission channels and families of codes. However, its high computational and memory demands limit its practical applications compared to traditional decoding algorithms. Achieving effective quantization of the ECCT presents significant challenges due to its inherently small architecture, since existing, very low-precision quantization techniques often lead to performance degradation in compact neural networks. In this paper, we introduce a novel acceleration method for transformer-based decoders. We first propose a ternary weight quantization method specifically designed for the ECCT, inducing a decoder with multiplication-free linear layers. We present an
optimized self-attention mechanism to reduce computational complexity via codeaware multi-heads processing. Finally, we provide positional encoding via the Tanner graph eigendecomposition, enabling a richer representation of the graph connectivity. The approach not only matches or surpasses ECCT’s performance but also significantly reduces energy consumption, memory footprint, and computational complexity. Our method brings transformer-based error correction closer to practical implementation in resource-constrained environments, achieving a 90% compression ratio and reducing arithmetic operation energy consumption by at least 224 times on modern hardware.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a method to enhance the efficiency and practicality of transformer-based error correction code (ECC) decoders, specifically targeting the error correction code transformer (ECCT) model introduced in prior work. The authors propose several modifications to ECCTs, including ternary weight quantization, a novel attention mechanism called head partitioning self-attention (HPSA), and a spectral positional encoding inspired by the Tanner graph's eigenspace. 
Jointly, these improvements lead to reduce memory, energy, and computational requirements while maintaining or improving the performance of the original ECCTs.

### Strengths
The paper proposes interesting methods like HPSA and adaptive absolute percentile quantization, which could be valuable outside of coding applications.

### Weaknesses
While the proposed approach introduces certain novel elements, I believe the paper falls short in critical areas, particularly regarding contribution scope and experimental results. Therefore, I recommend rejection for the following reasons:

1. Limited contribution: The paper proposes interesting methods like HPSA and adaptive absolute percentile quantization, which could be valuable outside of coding applications. However, since this paper is fundamentally a paper in coding theory, these contributions remain peripheral to the main focus (furthermore, per se they are not enough to warrant publication at ICLR).
And within the realm of coding theory, the contribution of the paper is limited. 

While relevant for the coding community, the contribution lacks the depth and broader impact expected at a major machine learning conference or in a top coding journal. A more appropriate fit would be a coding conference such as ISIT or ITW.

Furthermore, although ML-based decoding of conventional codes is a growing research area, with already many published papers, this paper does not provide a method that competes with conventional state-of-the-art codes/decoders. 


2. Weak comparisons and benchmarking: The experimental results are insufficient to substantiate the relevance of the proposed decoder.

Specifically, the authors compare the performance of their transformer-based decoder to that of belief propagation decoding for various codes, including polar codes and BCH codes. BP decoding of BCH codes and polar codes is known to be highly suboptimal. There are well-known, more efficient decoding algorithms for these codes (note that the considered BCH codes are very short). Hence, the current results provide no hint on whether the proposed decoders are good or not. 

To establish relevance, the paper would need to demonstrate comparable or superior performance to state-of-the-art decoding techniques in terms of accuracy or complexity. However, even if such comparisons were provided, in my opinion the contribution is still insufficient for a high-profile venue.

3. Use of non-standard metrics: The paper presents performance using the negative natural logarithm of the BER. Is there any reason for not adhering to standard metrics such as bit error rate curves or block error rate (BLER) curves (the latter being preferable)?  
The use of this unconventional (to day the least) metric complicates unnecessarily the interpretation of the results. I do not think it is wise to come up with an awkward way of showing results, when there is a established way to present results for error correcting codes.

Furthermore, reporting performance for only a few specific values of $E_b/N_0$ provides an incomplete view; it's unclear whether phenomena like error floors might occur at higher values.

Including conventional BLER curves over a range of $E_b/N_0$ values would make the results easier to interpret and align with community standards. Without such curves, it's challenging to assess the effectiveness of the proposed decoder and, in my opinion, the paper should be accepted in any venue.

### Questions
1. As mentioned above, the authors should report BLER curves.

2. The authors provide results for a number of LDPC codes. However, there are no details regarding these codes. Are them the best-available LDPC codes in the literature, or some designed by the authors? 

3. Related to Comment 2 and my comments in "Weaknesses," the authors should rework their results section and compare the performance of the proposed approach with that of state-of-the-art codes/decoders

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper has three main thrusts that I can see, and it targets these towards specifically, acceleration and compression of small transformer model architectures used for error-correction code decoding in communications systems. This is an incremental but significant upgrade to earlier work on ECC transformer decoders. The main thrusts are:
1) Quantization: The paper uses an adaptive layer-based quantization scheme that combines maximum-based quantization with distribution-based quantization schemes to provide a ternary quantization for the network, greatly accelerating its computation (claimed) while also compressing model size.
2) Head Partitioning: Masking attention heads based on the Tanner graph node connections at varying levels of relation (first- and second-ring masks) allows greater sparsity in attention computation while also preserving performance.
3) Positional Encoding: A soft positional encoding is also introduced as an inductive bias in the tokenizer/learnable encoder (from what I can see) that further preserves signal positional information and compensates for information loss in the attention head masking.

### Strengths
1) Solid presentation, all major thrusts are clearly presented and detailed. The paper also provides diagrams and examples to ease understanding and reading in what can be presented as a very dense, mathematical topic that a reviewer would have trouble grasping or reading through.
2) Analysis and experimentation include results on hardware for energy consumption reduction, a key metric in model compression. It also includes ablation studies for different encoding methods and acceleration due to model compression. All in all it analyzes most of the ablations relevant to the topic. I particularly commend the hardware results for different transistor nodes.
3) Contributions while building clearly off of a prior design and prior work, are cleverly tuned to the problem domain and a very clear niche. In particular the use of the Tanner graph and selective masking shows adaptation of the transformer compression problem to the given domain.

### Weaknesses
1) Concluding statements mention that the paper shows a general approach for the compression problem in small transformers - I would say that this is a bit of a stretch - the Tanner subgraph masking and the adjustments to positional encoding are not applicable to all problem domains, leaving the model quantization formulation the only really general cross-domain contribution. That formulation has not been benchmarking in the experimental section against prior art, and its acceleration alone (as opposed to the entire framework) has not been tested in the main document. I would say that ablation results for each component of the framework should be added in order to make this general claim.

2) A key contribution is replacing GeLU with ReLU in the bullet points of the front matter. This is absolutely not novel, as has been examined in [1] just to name a single source. I would strongly advise omitting this claim or qualifying it to the given problem domain.
[1] https://arxiv.org/abs/2310.04564 [Addressed]

3) While this work builds on the prior ECCT work, I would want a presentation of the model architecture in the main draft and clearly marked so that the scale of the 'small transformer' is visible, rather than having to dig through the ECCT paper and assume that this is the paper used. [Addressed]

4) The work assumes dedicated hardware for the transformer decoder in Section 5 when calculating complexity. This may not be a realistic assumption in practice due to the specialized and expensive design process for accelerator ASICs. I would want to see complexity calculations on a common edge accelerator platform such as an NVIDIA Hopper architecture or a Coral edge TPU, or a citation for the architecture-level simulator presumably used to generate these numbers (e.g. ScaleSim). Numbers for hardware without that backing are unreliable.

5)  The sparsity induced by masking is structured, and therefore may be applicable to standard accelerators. Why is there no mention of this? I do not see an analysis of the benefits of the structured sparsity from the masks in Section 4.2 compared to unstructured sparsity, which is a major plus point that has not been exploited.

### Questions
1) See point 5 in Weaknesses - is the sparsity structured, and if so, does your assumed hardware architecture exploit that?

2) See (4) in Weaknesses - Where is this assumption coming from and what edge platform architecture was used? Is there a citation or a cycle-accurate simulation of the architecture? Analytical complexity numbers that assume optimal hardware are unrealistic.

3) What are the authors' thoughts on the contribution of each component of this framework to the overall acceleration? Is there an ablation study that can be presented in the main doc? If not, a small table and rough numbers would be good so that we can see how much general approaches like quantization contribute and how much the domain-specific tweaks such as Tanner graph masking contribute.

See discussion - the authors have conditionally addressed much of the questions above and paper weaknesses. I am editing the review to reflect this.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a novel transformer based neural error correction architecture. It builds on the idea of the Error Correction Code Transformer (ECCT) by performing ternary quantization of the parameters in linear layers, by modifying the masking mechanism in the attention operation and also by introducing positional embeddings. The architecture is empirically validated against the original ECCT and the more traditional Belief Propagation algorithm and is shown to outperform both in terms of accuracy.

### Strengths
Through it's experimental results, the paper convincingly demonstrates the benefit of ternary weight quantization, Spectral Position Embedding and tailoring the attention mask for bipartite graph message passing over the previous state of the art transformer based method - the error correction code transformer (ECCT).

### Weaknesses
As an outsider to the field of error correction, I found that the description of the problem domain was a bit too brief. The paper would benefit from expanding the problem setting section. It would also benefit the paragraph describing the ECCT architecture if the equations weren't inlined and that the sequence of transformations defining the architecture were serialized.

### Questions
Merely out of curiosity: have the authors considered non transformer based architectures for this task? Presumably denoising diffusion models might have a role to play given the nature of the problem?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the challenges of high computational complexity and memory requirements faced by Error Correction Code Transformers (ECCT) in practical applications by introducing an innovative acceleration method. The approach centers on three technical innovations: 
- a specially designed ternary weight quantization method (AAP) that enables multiplication-free linear layers.
- an optimized self-attention mechanism (HPSA) based on code-aware multi-head processing that significantly reduces computational complexity.
- a positional encoding scheme (SPE) through Tanner graph eigendecomposition that provides richer graph connectivity representation without affecting inference runtime. 

Experimental results demonstrate that this method not only matches or surpasses the original ECCT's performance but also achieves a 90% compression ratio while reducing arithmetic operation energy consumption by at least 224 times on modern hardware, making transformer-based error correction more practical in resource-constrained environments.

### Strengths
This paper presents contributions to enhancing ECCT's practical applicability through a comprehensive optimization approach. The key strengths lie in its multi-faceted architectural improvements and efficient implementation strategies.

1. The paper introduces three well-designed technical innovations that directly address ECCT's structural limitations. The Head Partitioning Self Attention (HPSA) mechanism optimizes the self-attention computation specifically for bipartite graph message passing, while the Spectral Positional Encoding (SPE) leverages Tanner graph eigendecomposition to provide richer structural information without runtime overhead. These architectural improvements demonstrate a deep understanding of ECCT's underlying structure and successfully enhance its performance.

2. Except for two structural improvements of ECCT, the paper makes a step further to explore the possibility of low-bit quantization of ECCT. The paper uses a ternary weight quantization method (AAP) specifically designed for ECCT's compact architecture, achieving multiplication-free linear layers while maintaining model accuracy. 

The comprehensive experimental validation demonstrates that these improvements maintain ECCT's decoding performance while reducing its computational complexity to a level comparable with traditional Belief Propagation methods. This work contributes to making transformer-based error correction more practical in resource-constrained environments and helps narrow the efficiency gap between neural decoding techniques and traditional algorithms.

### Weaknesses
The weaknesses of this work are manifested in the following three aspects:

1. Limited Scope and Generalization.
The method's exclusive focus on ECCT architecture raises concerns about its broader impact. As ECCT has not gained significant attention in academia nor found applications in industry, optimizations specific to this architecture may have limited practical value.

2. Lack of Technical Novelty in Quantization.
The proposed AAP quantization method, while showing good performance, primarily combines existing techniques - weight distribution analysis and learnable quantization steps. The approach does not present substantial innovation in quantization methodology, making it more of an implementation enhancement than a theoretical advancement.

3. Insufficient Validation of Acceleration Claims.
Despite positioning itself as an acceleration method, the paper's experimental validation concentrates on compression ratios, sparsity metrics, and energy consumption reduction. While these metrics demonstrate improved efficiency, the absence of direct inference speed measurements and real-world acceleration benchmarks leaves the actual acceleration effects unverified. This gap between claimed acceleration and demonstrated results weakens the paper's central premise.

Suggestions:

1. Enhanced Validation of Quantization Method

The paper would benefit from more comprehensive experiments comparing the proposed AAP method with existing quantization approaches. Specifically, comparative studies with methods like GPTQ and OmniQuant that focus solely on weight distribution or learnable scales would better demonstrate the advantages of combining these techniques. Such experiments would strengthen the paper's claims about AAP's effectiveness and innovation in quantization methodology.

2. Demonstration of Actual Acceleration Effects

The paper needs to provide concrete evidence of inference speedup to support its acceleration claims. We suggest adding experiments that measure and analyze the actual inference acceleration achieved after implementing all proposed optimizations. If practical acceleration is currently limited by implementation constraints, this limitation should be explicitly acknowledged, and the path toward achieving actual speedup in future work should be discussed. This transparency would better align the paper's claims with its demonstrated results.

### Questions
1.In table2, why are the results of ECCT+AAP better than AECCT for datasets BCH and LDPC? Does it mean HPSA and SPE bring degradation for quantized ECCT? 

2.Are the energy consumption saving rate(224 times on 7nm chips and 139 times on 45nm chips) tested on real chips, or just an estimation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Accelerated Error Correction Code Transformer (AECCT), which reduces the memory footprint and computational complexity of Error Correction Code Transformers (ECCT). The proposed technique introduces three distinct novel components:
1. Adaptive Absolute Percentile (AAP) Quantization: This method compresses model weights to ternary values, reducing memory and energy use while maintaining model accuracy. Unlike traditional quantization, AAP dynamically adjusts sparsity to retain essential features.
2. Head Partitioning Self-Attention (HPSA): This self-attention mechanism reduces computational complexity by dividing attention heads to focus separately on first-ring (neighbor) and second-ring (more distant) connections within ECC’s Tanner graph. HPSA achieves higher sparsity, thus lowering computational costs compared to ECCT’s Code-Aware Self-Attention (CASA).
3. Spectral Positional Encoding (SPE): By embedding structural information from the Tanner graph’s Laplacian eigenspace, SPE enriches the model’s positional representation, enhancing the decoding ability without increasing runtime.

Using these components, the authors demonstrate a 90% compression ratio and reduce energy consumption by over 224 times compared to ECCT, approaching the efficiency of traditional algorithms like Belief Propagation (BP), making practical deployment possible.

### Strengths
1. The paper is well written and easy to follow, and the contributions are relevant.
2. The techniques introduced demonstrate significant compression without performance degradation compared to ECCTs.
3. The technique is tested on 3 different ECC types - Polar, Low-Density Parity Check (LDPC), and Bose–Chaudhuri–Hocquenghem (BCH) codes - demonstrating robustness and applicability across various error correction scenarios.
4. The authors conduct ablation studies evaluating the impact of each of the three components.

### Weaknesses
1. Two out of three proposed techniques (HPSA and SPE) may have limited applicability outside of ECC.
2. It is not clear how easy it will be to implement these techniques in existing hardware. Ternary quantization has been explored by previous work. Beyond quantization, does HPSA and SPE introduce additional implementation complexities on existing hardware? 
3. The energy efficiency numbers are estimates, the paper does not provide any real measured numbers on actual hardware.

### Questions
1. Discuss any potential challenges or modifications needed to implement HPSA and SPE on current hardware platforms used for error correction.
2. Compare the implementation complexity of HPSA and SPE to that of the original ECCT approach.
3. If possible, provide estimates or examples of how these techniques might impact hardware design or utilization in practical ECC systems.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces simplifications to the Error Correction Code Transformer (ECCT) neural decoder model to reduce the memory and compute requirements. The authors claim that conventional quantization methods result in a significant drop in performance and instead propose alternative approaches to reduce the complexity. 

First is Adaptive Absolute Percentile (AAP) quantization, to optimize the percentile of scaling factor (percentage of absolute values). 

Second it head partitioning self-attention to limit the attention to either between similar nodes (check/variable) or between different nodes (check<->variable), dividing the attention heads into two groups. This significantly reduces the number of active elements in the attention mask.

Finally, uses spectral positional encoding (SPE) to introduce an inductive bias about the tarnner graph into the model.

### Strengths
1. Tackles an important problem of reducing the complexity of neural decoders. 

2. The performance improvements in terms of memeory and compute are non-trivial and very impressive. 

3. All the three main ideas proposed are novel and interesting. Specifically, splitting the masking based on the node structure is a clever example of leveraging domain structure. 

4. Set of experiements are exhaustive and includes sufficient number of results as well as ablation studies to show the merits of proposed ideas.

### Weaknesses
1. While a lot of interesting ideas were discussed in the paper to improve the efficiency of ECCT, I am not fully convinced about the practical relevance of a "universal decoder" architecture that simply utilizes a parity check matrix for all codes. In coding theory, each well-known family of codes has highly specific representations and special propoerties that cannot be fully captured by a simple parity check matrix. For instance, in Polar codes, the reliability sequence and the sequential decoding of information bits plays a crucial role in deciding the performance of the code. Claiming better performance for all family of codes w.r.t BP decoder is a bit misleading. Again, I appreciate the line of ECCT and related works from an academic curiousity point of view, but in my opinion, the merits and drawbacks should be highlighted more clearly without any overpromising. 

2. One common troubling trend in comparing the performance of neural decoders is the choice of weak classical baselines. While I understand the difficulty of a common decoder architecture outperforming the highly specialised decoders for each of the class of codes, the comparison should neverthless be done for completeness. For instance, it is known that BP is not optimal for many codes. For instance, when comparing with BCH codes, Berlekamp-Massey decoding algorithm should be used as the classical non-learning baseline and similarly for Polar codes, successive cancellation list decoding should be used. I strongly suggest authors to include the best classical baselines for each code to clearly show the gap with respect to the SOTA neural decoder, whether it is positive or negative.

3. I beleive authors chose the format of the negative log BER for presenting the results based on the original ECCT paper. But this format is not very informative and very uncommon in information and coding theory literature. While it is easy to identify which scheme is doing better, the gap between the performances is hard to interpret as the scale is not very intuitive. Rather, the standard format of SNR gap in dB for a target BER/BLER is more informative.

4. "Our analysis shows that AAP outperforms AP quantization" --> I see from ablation studied in Table 2 that the performace difference between ECCT + AP vs ECCT + APP is very small (0.01 - 0.1). Again, this scale of -log(BER) is very non-intuitive but in my opinion, this is negligible difference in performance and brings into question the value of doing AAP over AP. 

5. While the head partitioning idea is interesting, is it also similar to the idea proposed in the paper "CrossMPT: Cross-attention Message-Passing Transformer for Error Correcting Codes", specifically the second-ring MP. 

6. The details about SPE are unclear and I do not understand the "spectral" positioning and the "Laplacian eigenspace" ideas discussed. Rewriting the seubsection to simplify the discussion of these ideas would be helpful.

### Questions
Covered as part of weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
