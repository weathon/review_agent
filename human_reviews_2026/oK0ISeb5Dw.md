# Otters: An Energy-Efficient Spiking Transformer via Optical Time-to-First-Spike Encoding

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4, 6

## Abstract
Spiking neural networks (SNNs) promise high energy efficiency, particularly with time-to-first-spike (TTFS) encoding, which maximizes sparsity by emitting at most one spike per neuron. However, this energy advantage is often unrealized because inference requires evaluating a temporal decay function and then multiplying the result by the synaptic weights. 
This paper challenges this costly approach by repurposing a physical hardware `bug', namely, the natural signal decay in optoelectronic devices, as the core computation of TTFS. We fabricated a custom indium oxide optoelectronic synapse that demonstrates how its intrinsic physical decay directly implements the required temporal function. By treating the device's analog output as the fused product of the synaptic weight and temporal decay, optoelectronic synaptic TTFS (named Otters) eliminates these expensive digital operations. 
To use the Otters' paradigm in complex architectures such as the transformer, which are challenging to train directly due to sparsity, we introduce a novel quantized neural network-to-SNN conversion algorithm. 
This complete hardware-software co-design enables our model to achieve state-of-the-art accuracy across seven GLUE benchmark datasets and demonstrates a 1.77$\times$ improvement in energy efficiency over previous leading SNNs, based on a comprehensive analysis of compute, data movement, and memory access costs using energy measurements from a commercial 22nm process. 
Our work thus establishes a new paradigm for energy-efficient SNNs that translates fundamental device physics directly into powerful computational primitives. All codes and data are open source.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents an interesting hardware-software co-design for ultra-low-energy Spiking Neural Networks (SNNs). The core innovation is the OTTERS optoelectronic synapse, which repurposes the natural signal decay of a custom-fabricated Indium Oxide (In2O3) thin-film transistor (TFT) to physically implement the temporal decay function required by Time-to-First-Spike (TTFS) encoding. This strategy eliminates the costly digital computation that has plagued traditional TTFS approaches.

### Strengths
Repurposing the natural decay of a fabricated In2O3 optoelectronic TFT to implement TTFS temporal decay is original and elegant, it turns a device “bug” into a computational primitive.

Practical engineering choices are well motivated to avoid run-time multiplications and to exploit sparsity in TTFS.

The work is novel, impactful, and supported by extensive experiments and a rigorous analytical model.

### Weaknesses
The energy numbers rely on an analytical model mixing measured device values (ADC, TFT) with literature energy numbers for ACC/NoC/etc. However, the headline claims (Otters 4.06 mJ per attention block, etc.) would require clearer provenance.

Table 1 and Table 2 compare to SNN and QNN baselines, but it is unclear whether baselines were re-run (or taken from the original work) with identical data preprocessing, same batch length/sequence length, or whether SOP/spike rates were taken from original papers.

### Questions
In the energy estimations, which terms dominate (analog read, ADC, spike movement, threshold reads, K/V write)? Provide a breakdown per term with uncertainties and explicitly list all constants used and their sources.

In the baseline comparisons, clarify experimental protocol and, where feasible, re-evaluate key baselines under the same measurement assumptions or provide scripts to reproduce the energy comparison.

Minor:
- In Table 2, for the last column Energy Ratio, I suggest using an upward arrow (or statement in the caption) to clearly indicate that the higher is better.

- Please add the date for the ICML paper of Xing et al. line 588 (2024), and clarify which paper by Xingrun Xing is referred to by "Xing et al." since you cite two papers in the same year with same first author.

### Soundness
3

### Presentation
3

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
This submission implements spiking neural networks (SNNs) on optoelectronic devices to mitigate the high computational cost associated with evaluating temporal decay functions and their subsequent multiplications with synaptic weights. Specifically, the authors fabricate a custom indium oxide–based optoelectronic synapse capable of directly realizing the required temporal function. Furthermore, they introduce a quantized neural network–to–SNN conversion algorithm to enhance the efficiency and practicality of the proposed system.

### Strengths
The work elegantly bridges neuromorphic hardware and algorithmic innovation by exploiting the intrinsic physical decay of optoelectronic devices to implement TTFS encoding. This approach offers a conceptually novel solution to the costly digital decay function in TTFS.

The authors provide a rigorous and realistic energy model covering compute, data movement, and analog operation costs, grounded in commercial 22nm process measurements.

The paper contributes a tangible step toward neuromorphic computation for language models, merging physical device dynamics with algorithmic efficiency, an area of high impact given the growing energy concerns of LLMs.

### Weaknesses
**i**. While the concept is elegant, it remains unclear how scalable the proposed In₂O₃ optoelectronic synapses are for large-scale deployment or integration into full neuromorphic hardware systems. For instance, what is the device size of otters for BERT?

**ii**. The transition from quantized neural networks (QNNs) to spiking neural networks (SNNs) is not a new research question, as several prior works have explored QNN-to-SNN conversion frameworks. However, this submission does not provide a sufficient comparative analysis with existing methods, like [1][2]. A more detailed discussion is needed to highlight the unique technical contributions, methodological distinctions, and advantages of the proposed conversion algorithm over established approaches. 

**iii**. The experiments are conducted only using BERT as the base architecture. It remains unclear whether the proposed Otters framework can generalize to other Transformer variants or model families.

**iv**. The reported energy savings appear to stem primarily from quantization effects rather than the proposed optoelectronic TTFS mechanism itself. In particular, the ablation study compares a 4-bit SNN with a 1-bit Otters, which inherently favors the latter due to lower bit precision. This comparison seems unfair and may overstate the true contribution of the proposed hardware-software co-design. A more controlled comparison at equal bit-widths would provide a clearer and more convincing evaluation of Otters’ energy efficiency.

[1] Liu, Fuqiang, and Chenchen Liu. "Towards accurate and high-speed spiking neuromorphic systems with data quantization-aware deep networks." Proceedings of the 55th Annual Design Automation Conference. 2018.

[2] Ajay, B. S., and Madhav Rao. "MC-QDSNN: Quantized Deep evolutionary SNN with Multi-Dendritic Compartment Neurons for Stress Detection using Physiological Signals." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2024).

### Questions
1. Since the experiments are conducted through simulation rather than on real optoelectronic hardware, it remains unclear how much of the performance degradation shown in Table 1 originates from the Otters implementation itself, as opposed to the quantization step during the QNN-to-SNN conversion. A clearer analysis quantifying the accuracy gap relative to standard BERT, after isolating the impact of quantization, would help clarify the true effect of the proposed Otters mechanism on model accuracy.


2. Can you provide the reference to the cost computation functions in A 3.1. Without proper references, these cost formulations appear insufficiently justified, as they seem to assume idealized conditions rather than reflecting the complexities of realistic hardware environments.

### Soundness
3

### Presentation
4

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
This paper proposes to use opto-electrical neurons for low-power SNN acceleration using time to first spike encoding that naturally aligns with the opto electrical neuron's characteristics.  The energy analysis and accuracy analysis is quite reasonable.  The conversion approach and quantization techniques as well as the noise-aware training all appear well explained and reasonable.

### Strengths
I think the novelty of the approach and the write-up and analysis are solid. Well explained paper and concept.  I was very happy to see the energy analysis use estimates from Horowitz which are realistic and capture the cost of ADCs and data movement.

### Weaknesses
It seems the opt-electrical device is not small compared to a digital SNN accumulator.  It seems to me that the scale of the hardware may be the biggest challenge in making such an approach practical and yet there seems to be no description of this challenge in the paper, even in the future work. 

I think even for a conference like ICLR and perhaps especially for a conference like ICLR where the audience may not realize the fundamental difference in hardware scale, this should be mentioned and scaling efforts for this technology should be at least outlined.  Otherwise, this is yet another pipedream in the field of energy-efficient SNN hardware. More specifically, how many such devices might be needed to have the same order of magnitude in compute latency and what, with current technology, would the size of such a chip be.  What are the challenges of heterogeneously integrating such devices with SOTA digital hardware?

### Questions
Please see weaknesses. 

I also heard of other SOTA SNNs such as QKFormer. Can you comment on how this work relates to yours? See https://arxiv.org/abs/2403.16552.

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
This paper introduces a hardware–software co-design framework named Otters, which leverages the natural signal decay in indium oxide optoelectronic transistors to physically implement TTFS computation. By interpreting the analog output as a fused representation of synaptic weight and temporal decay, the system eliminates costly digital multiplications. Furthermore, the authors propose a QNN-to-SNN conversion algorithm, integrating it into Transformer architectures. Experimental results on the GLUE benchmark demonstrate SOTA accuracy and a 1.77× improvement in energy efficiency compared to previous SNNs, validated through comprehensive measurements on a 22nm commercial process.

### Strengths
1. This paper presents a novel physical computing paradigm that leverages the inherent signal decay of optoelectronic devices as the temporal decay mechanism for TTFS computation, thereby eliminating the need for energy-intensive digital multiplications.

2. It introduces a theoretically grounded QNN-to-SNN conversion framework, establishing the functional equivalence between quantized neural representations and spike-based computation, which is significant for bridging conventional deep learning and neuromorphic systems.

3. The proposed Otters architecture, validated through a hardware–software co-design, achieves state-of-the-art performance on GLUE and demonstrates energy efficiency advantages.

### Weaknesses
1. The experimental evaluation is limited to Transformer architectures, which are not the primary application domain for SNNs. Since SNNs are particularly suited for edge computing and low-power vision tasks, extending the analysis or providing validation on CNN-based models [1] would substantially strengthen the work.

2. The proposed QNN–SNN mapping depends on a pre-trained quantized model, which introduces additional training overhead. The authors should clarify whether this preprocessing step yields tangible performance benefits and discuss whether the Otters framework provides practical deployment advantages over conventional QNNs on real hardware.

3. The experimental comparison is limited. Including results against other TTFS-based conversion methods [2] would better substantiate the claimed efficiency gains.

4. The mathematical expressions lack consistency in formatting and punctuation. All equations should be standardized, properly numbered, and punctuated to ensure clarity and formal correctness.

[1] Deep residual learning for image recognition. CVPR 2016.

[2] TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer. ICML 2025.

### Questions
1. The reported energy analysis omits Softmax operation and Layer Normalization [3], both essential operations in Transformer computation. Incorporating these operations would provide a more comprehensive and realistic assessment of the total energy consumption.

2. Since the paper emphasizes energy efficiency, providing inference latency results is necessary to validate the method’s real-time performance and practicality. 

3. It remains unclear whether the proposed Otters-based TTFS framework generalizes to other datasets or CNN-based architectures. Demonstrating such experiments would enhance the paper’s impact and generality. 

[3] Spatio-temporal approximation: A training-free snn conversion for transformers. ICLR 2025.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose to transform the intrinsic “signal decay” of optoelectronic synapses into a computational resource, thereby achieving TTFS coding in SNNs with extremely low energy consumption. Specifically, it uses the physical decay curve of customized indium oxide transistors to replace digital computation, and then proposes a novel QNN-to-SNN conversion algorithm that maps a trained quantized Transformer into an equivalent spiking network. Finally, this paper adds 1-bit KV projection to further reduce energy consumption.

### Strengths
1. It proposes a new “physical computation” paradigm that achieves in-memory computation by integrating synaptic weight multiplication and temporal decay into a single optoelectronic physical process, thereby improving energy efficiency from the ground up.

2. It achieves the highest accuracy among SNNs on the GLUE benchmark and demonstrates a 1.77x energy-efficiency improvement over the previous SOTA model, which validate the effectiveness of the proposed framework.

### Weaknesses
1. Since the paper’s contributions involve algorithm–hardware co-design, it should provide a more detailed breakdown of energy consumption to clarify the contribution of each component, such as $(E_{\text{analog}})$.

2. The results are presented only for the BERT-Base model, which raises concerns about the generalizability of the proposed method.

3. For a fairer comparison, the paper should include both performance and energy consumption results of other models with 1-bit KV to more clearly demonstrate the energy savings achieved by the new hardware paradigm.

### Questions
check the weakness

### Soundness
2

### Presentation
3

### Contribution
3
