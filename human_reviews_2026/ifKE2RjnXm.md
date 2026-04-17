# CHEBYUNIT: HARDWARE-ACCELERATED ENERGY-EFFICIENT FPGA WITH LOW COMPUTATION COMPLEXITY FOR ARTIFICIAL INTELLIGENCE ACCELERATION

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Multi-Layer Perceptrons (MLPs) achieve high accuracy but are hindered by a large number of parameters, leading to significant memory and power consumption. While Kolmogorov-Arnold Networks (KANs) address this by using learnable functions instead of weight matrices, their B-spline implementations are complicated in hardware designs. To overcome this limitation, we propose a novel hardware framework for Chebyshev-KANs, leveraging the recursive properties and numerical stability of Chebyshev polynomials. Our core component, the ChebyUnit, efficiently generates polynomial bases and reuses coefficients from on-chip storage to perform lightweight inner product operations in a streaming fashion. This approach significantly reduces external memory access called Double Data Rate(DDR) traffic and resource utilization while maintaining high throughput. Our Verilog implementation on a Xilinx ZCU102 Field-Programmable Gate Array(FPGA) demonstrates over 90% reductions in Look-Up Table(LUT), Flip-Flop(FF), and Digital Signal Processing(DSP) utilization compared to a baseline high-level synthesis (HLS) design, all while preserving excellent approximation accuracy. These findings confirm the practical efficiency of Chebyshev-KANs, positioning them as a promising solution for interpretable and energy-efficient neural networks, particularly in resource-constrained edge AI applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Kolmogorov–Arnold Networks (KANs) as a means to improve the computational efficiency of MLPs by replacing traditional weight matrices with learnable functions. While B-spline implementations in KANs are often complex and difficult to realize in hardware, this paper presents a customized hardware framework for Chebyshev-KANs, which effectively exploits the recursive properties and numerical stability of Chebyshev polynomials. The experimental results are promising and demonstrate the potential of the proposed approach.

### Strengths
1) The idea of using Kolmogorov–Arnold Networks (KANs) to improve MLP computational efficiency is novel and insightful. 

2) The proposed hardware implementation is efficient.

### Weaknesses
1) While the idea of replacing MLPs with KANs is novel, the paper lacks sufficient evidence demonstrating the feasibility of such replacement in typical neural network models, particularly from the perspective of maintaining model accuracy. The workloads evaluated in this paper are mostly non-neural-network benchmarks, which are insufficient to convincingly support the claim of KANs as effective MLP substitutes.

2) In addition, the hardware implementation of KANs appears relatively straightforward, and the paper does not clearly identify or address significant challenges from a hardware design perspective.

### Questions
See the weaknesses.

The performance of neural network models on typical tasks such as classification and object detection needs to be clarified when MLPs are replaced with KANs.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an efficient field-programmable gate array (FPGA) implementation of Chebysev Kolmogorov-Arnold networks (Chebyshev-KANs). Compared to standard KANs, this approach significantly reduces external DDR memory access and resource utilization while maintaining high throughput. A Verilog implementation on a Xilinx ZCU102 FPGA demonstrates over 90% reductions in LUT, FF, and DSP utilization compared to a baseline KAN high-level synthesis (HLS) implementation, while preserving high approximation accuracy to an artificial dataset.

### Strengths
* There is a good correspondence between the Chebyshev-KAN model and the targeted hardware
* Evaluations indicate significantly better resource utilization and latency than an HLS implementation of a B-spline-based KAN for an artificial dataset.

### Weaknesses
* Limited evaluation with respect to existing MLP FPGA implementations from frameworks like FINN and hls4ml or dedicated FPGA ML algorithms like Logic-Nets, PolyLUT, NeuralLUT, and Differentiable Weightless Neural Networks (DWNs)
* Limited evaluation on different datasets, especially real as opposed to synthetic ones.
* Lack of information on training details and accuracy of models.
* Lack of description and citations of recent related work on dedicated FPGA ML algorithms in Section 2.3

### Questions
* L157: While there are ML accelerator systems that connect host CPUs and FPGAs, there are also several that focus on pure dataflow / FPGA implementations, like those from FINN and hls4ml.
* L349: What loss function is used for training? How large is the training dataset? How long does training take? What hardware is used, etc.? Please add details.
* L381: Do you suspect the HLS implementation of KANs (B-spline) you use is suboptimal in some way? Could this be discussed?
* L388: Could you describe and cite the Moons, Wine, Dry Bean, and Mushroom datasets?
* Table 1: Could you add some accuracy metrics to the table?
* Tables 1 and 2: Could you add additional comparisons to MLPs (e.g. FINN or hls4ml implementations) and dedicated FPGA ML algorithms like Logic-Nets, PolyLUT, NeuralLUT, and Differentiable Weightless Neural Networks (DWNs)?
* Figs. 4 and 5: Can the overall approximation accuracy be quantified? Can you also simply plot the difference between the target and learned function in one figure? As it is, it is impossible to tell quantitatively how well the model is performing.
* Please proofread the paper for typos and grammatical errors, for example
  * L111: discreet
  * L369: the dominant factor influencing precision is the fixed-point format to influence the precision of the hardware results is the fixed-point format

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
This paper introduces a hardware accelerator named CHEBYUNIT based on Chebyshev polynomials. The work addresses the limitations of traditional MLPs and B-spline-based KANs in terms of parameter count, memory, and hardware complexity, especially for edge AI applications. Experimental results tested on FPGA shows good results, especially in terms of resource usage.

### Strengths
* The paper provides a clear breakdown of the CHEBYUNIT, including basis generation, coefficient storage, and parallelization.
* The use of Chebyshev polynomials for KANs feels novel, as it leverages their recursive simplicity and numerical stability for efficient hardware implementation, addressing the issue of .
* Experimental results look promising, achieving huge reduction in LUT, FF, and DSP usage.

### Weaknesses
* The datasets used for evaluation could be less limited and including more complex or demanding datasets.
* It's unclear how weight reuse proposed to address the memory bandwidth bottleneck would scale on large models. 
* While the experimental results look promising, the baseline results comes from HLS instead of FPGA.

### Questions
* For the manual tuned fixed point precision setting, how is it tuned and is there a systematic study or analysis on its impact, especially for different tasks?
* How well does the method would scale, for more complex models? Given that there are limitations stated related to memory bandwith. And how would the weight reuse handle it for more complex models?
* Is there an accuracy comparison on the final classification accuacy against software accuracy?
* Quantization wise, would it affect the proposed method more than B-splines?

### Soundness
2

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
3

### Summary
This paper aims to tackle issues related to the hardware complexities and inefficiencies associated with the classically implemented B Spline component of the KAN architecture. Specifically, the paper explores the creation of an FPGA accelerator, not for the B Spline implementation but rather in the case of Chebyshev polynomial basis functions. This is done through the creation and adoption of the novel ChebyUnit.


The ChebyUnit is introduced as this architecture’s answer to a modular activation unit. This is the key to the paper's design.

### Strengths
The foremost standout quality of this paper was its clarity and well considered introduction to the theoretical basis of the challenges addressed. The authors insightfully walk the reader through the evolving motivation behind the pursuit of their accelerator from the benefits associated with a KAN over MLPs, the intuition behind the hardware friendly aspects of the Chebyshev polynomial implementation over the B Spline, and how certain design choices could be made as a result of adopting Chebyshev polynomials. Similarly, the well articulated motivation, implications, and benefits for nearly every design choice contributed greatly to making this work approachable.

Beyond this, a notable strength was the clear effort made to make the design modular and well explained so that this modularity could be more easily made use of making this work a potent and effective starting point for future work. This aspect becomes all the more valuable with the comparatively lower maturity of this idea in contrast to accelerators catering to the B Spline implementations. 

Finally, the notable hardware utilization and latency improvements presented in the evaluation were quite compelling adding to the appeal of this approach and synergizing with the aforementioned points to make this paper.

### Weaknesses
While this work focuses on introducing and explaining the design, the evaluation presents as considerably more limited especially when compared to the expansive introduction just prior. This was addressed in part by the authors in the conclusion however, to a reader, this admission does not excuse its absence outright.

If considering a narrow aim of this paper is to introduce the intuition of this approach, the design and its value, and to position it as a potent alternative to B Spline implementations then perhaps this evaluation could be reasoned to be sufficient. However I feel a more comprehensive evaluation could serve to elevate this paper greatly and position it on the radar of many more individuals. Given factors like the theoretical elegance of KANs, growing interest in AI explainability, and the strength of the explanations and results presented, more evaluation could serve to amplify the paper's impact, mirror the comprehensive presentation of value evident in initial sections, and establish it as a key motivator for future work.

I appreciate there are some of the barriers associated with this as the infrastructure has not yet been built out for some evaluations to take place. But as a naive example, if compared to more classical NN FPGA accelerators, one could imagine a higher interest from those working on other network architectures. Another could be expanding on the behavior of different quantization techniques.

### Questions
No particular questions.

### Soundness
3

### Presentation
3

### Contribution
2
