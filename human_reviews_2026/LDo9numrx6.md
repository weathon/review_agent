# A Unified Total Variation Framework for Membrane Potential Perturbation Dynamic

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Membrane potential perturbation dynamic (MPPD) is an emerging approach to capture perturbation intensity and stabilize the performance of spiking neural networks (SNN). It discards the neuronal reset part to intuitively reduce fluctuations of dynamics, but this treatment may be insufficient in perturbation characterization. In this study, we prove that MPPD is total variation (TV), which is a widely-used methodology for robust signal reconstruction. Moreover, we propose a novel TV-$\ell_1$ framework for MPPD, which allows for a wider range of network functions and has better denoising advantage than the existing TV-$\ell_2$ framework, based on the coarea formula. Experiments show that MPPD-TV-$\ell_1$ achieves robust performance in both Gaussian noise training and adversarial training for image classification tasks. This finding may provide a new insight into the essence of perturbation characterization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reframes a heuristic SNN stabilization mechanism (MPPD) into a rigorous TV theory, generalizes it via a new ℓ₁-based variational framework, and validates its advantage in both adversarial and noisy environments.

#### **1. Research Problem**

* Spiking Neural Networks (SNNs) are vulnerable to adversarial and noisy perturbations that destabilize their dynamics. The existing *Membrane Potential Perturbation Dynamic* (MPPD) technique empirically improves robustness but lacks solid theoretical grounding.
* The paper tries to reveal mathematical nature of MPPD, and how can it be formalized to enhance SNN robustness in a principled way.

#### **2. Proposed Method**

* The authors prove that MPPD is mathematically equivalent to Total Variation (TV)
* Based on this equivalence, they introduce the MPPD–TV–ℓ₁ framework, extending prior ℓ₂-based formulations (MPPD–TV–ℓ₂).


#### **3. Theoretical Contributions**

* Rigorous proof that MPPD is TV under measurable perturbations.
* Establishment of a new TV–ℓ₁ regularization theory for SNNs, encompassing:

  * The coarea formula specific to SNN membrane dynamics.
  * A dominated TV property showing layer-wise boundedness of perturbations.
  * A closed-form subgradient that enables backpropagation through non-smooth TV terms.

#### **4. Experimental Contributions**

* Across CIFAR-10 and CIFAR-100, the MPPD–TV–ℓ₁ model outperforms both the ℓ₂ variant and other baselines under Gaussian noise and adversarial attacks (FGSM, PGD, CW, AutoAttack).
* Demonstrates higher resistance to increased attack intensity and step size, confirming TV–ℓ₁’s superior denoising behavior.

### Strengths
### **1. Theoretical Contributions**
Overall, this paper provides a theoretically sound bridge between signal variation analysis and adversarial robustness in SNNs, offering a new mathematical perspective for neuromorphic robustness theory with clear derivations.

* Introduces a novel reinterpretation of membrane potential perturbation dynamics (MPPD) as a form of total variation (TV), unifying biological spiking dynamics and variational regularization theory in an original analytical framework.
* Transforms prior MS-MPPD regularization (previously heuristic) into a mathematically grounded TV–ℓ₂ model and further generalizes it into the TV–ℓ₁ formulation, expanding the functional space of admissible membrane potentials and enabling sharper perturbation modeling.
* Mathematical Rigorly establishes formal results including the Coarea formula for spiking potentials, the Dominated TV Property linking layerwise stability to weight norms, and a closed-form subgradient for optimization without additional computational cost.



### **2. Experimental Contributions**
Experimental result supports that total variation regularization can serve as a universal principle for temporal–spatial robustness in SNNs.

* Conducts extensive controlled experiments on CIFAR-10 and CIFAR-100 using both VGG11 and WRN16 backbones under Gaussian and adversarial training, comparing six state-of-the-art SNN methods.
* Demonstrates consistent and often substantial gains in adversarial accuracy, validating that MPPD-TV–ℓ₁ effectively suppresses perturbations across attack intensities and steps.
* Shows that the closed-form subgradient introduces no measurable computational overhead while maintaining compatibility with mainstream deep learning frameworks.

### Weaknesses
Overall, this paper provides rigorous theoretical analysis and effective experimental results to demonstrate the theoretical foundations of MPPD and offer a more complete version. I particularly appreciate the paper's rigorous treatment of pulse discontinuities, which is rare but meaningful in the SNN field.

1. My main concern is that MPPD has not yet become a mainstream method for SNNs, and this paper is almost entirely based on this premise, which limits its broader impact. I am not sure how interested most people in SNN field are in it. For example, can the techniques for handling spike discontinuities in the paper be generalized to other SNN research?

2. Please add some missing citations. Some related work also focuses on smoothing membrane potential perturbations under adversarial attacks with uses different methods, such as dynamic thresholding (https://arxiv.org/pdf/2308.10373) and stochastic gating (https://ojs.aaai.org/index.php/AAAI/article/view/27804).

3. In Theorem 4, "assume every node i in layer L uses the same set of preceding nodes in layer L-1" which does not hold for sparse, or skip-connected SNN architectures. Relaxing this constraint would enhance generality.

### Questions
1.Can the techniques for handling spike discontinuities in the paper be generalized to other SNN research?
2.Please add some missing citations.

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
This work is build on the concept of Membrane Potential Perturbation Dynamics (MPPD) as a method for enhancing the robustness SNNs, particularly in the face of adversarial perturbations. The authors propose that MPPD can be framed as a Total Variation (TV) model and further develop a novel MPPD-TV-L1 framework, which they show improves the robustness of SNNs in adversarial environments. The proposed approach demonstrates superior performance over existing TV-L2 models on image classification tasks using the CIFAR-10 and CIFAR-100 datasets.

### Strengths
1: This work proves that MPPD is equivalent to TV, providing a strong mathematical foundation that underpins the proposed method.

2: The experimental setup is comprehensive, involving state-of-the-art methods and adversarial training schemes. 

3: The motivation that extend the existing TV-L2 framework to TV-L1 is well-articulated.

4: The proposed framework has clear practical implications for improving the security and reliability of SNNs.

### Weaknesses
A major concern is the incremental novelty of this work. The MPPD-TV-L2 framework was already proposed in Ding et al. (2024), and this work introduces the MPPD-TV-ℓ1 framework. Furthermore, as shown in Figure 1, in the case of AT+Reg, the MPPD-TV-ℓ1 shows only minor (or no) improvement. This suggests that MPPD-TV-ℓ1 has a similar effect to the regularizer (Ding et al., 2022), which somewhat weakens the novelty and necessity of this work.

The writing and clarity of the paper can be improved. For example, the title may mislead readers into thinking that the paper proposes MPPD (which it does not). Additionally, there is no punctuation for $\epsilon$ in Equation (3), and the term "MS-MPPD" looks awkward in Equations (4) and (5).

### Questions
See the weaknesses section for my major concerns.

In Table 1, the cases where MPPD-TV-L1 performs worse than MPPD-TV-L2 all occur with the FGSM and the APGD_DLR. It would be helpful if the authors could provide a theoretical or intuitive explanation for this behavior.

Could you include a comparison that shows the performance of ANNs in handling adversarial perturbations, to better highlight the relative robustness (if any) of SNNs?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a theoretical analysis of the Membrane Potential Perturbation Dynamic (MPPD) in Spiking Neural Networks (SNNs). The authors' main contribution is framing MPPD as a form of Total Variation. This provides a theoretical foundation for MPPD.

### Strengths
The primary strength is the formal theoretical analysis offered for the MPPD approach, addressing a known gap in the field.
​​
The method appears to achieve good performance, suggesting the theoretical insight translates into practical benefits.

### Weaknesses
A major issue is that Table 1 does not report the "clean" accuracy (performance without noise), making it impossible to evaluate the true cost of the denoising improvement.

​​The choice of the key parameter ζ in Section 4.1 is not explained. It is unclear if it was set arbitrarily, tuned for this work, or copied from another paper. If compared papers in Table 1 used different ζvalues, the comparison is misleading and should be noted.

The preliminary discussion describes previous MPPD work in a discrete setting, but the proposed method uses a continuous formulation. The paper does not justify this shift or explain how the continuous form is compatible with or translates to the discrete SNN simulation.

The proposed TV loss might not fully capture the original MPPD behavior. Specifically, when a reset mechanism is involved, small perturbations that are insufficient to evoke a spike may be excluded from the loss calculation, potentially making the model less sensitive to certain types of noise.

​​In Equation 2.4, using a shorter minus sign (e.g., \text{-}) would improve visual alignment and readability.

​​The statement that previous work "lacks reliable explanations and theoretical foundation" (Line 55) is too strong, especially if the authors' own prior work is in the same theoretical domain. This should be phrased more precisely.

### Questions
Why was the continuous formulation chosen for the proposed method when the preliminary MPPD description is discrete? How is the continuous formulation implemented or made compatible with the discrete-time dynamics of an actual SNN?

What are the clean (noise-free) accuracy scores corresponding to the results in Table 1? This is critical for assessing the performance-robustness trade-off.

How does the proposed TV loss account for sub-threshold membrane potential perturbations that do not lead to a spike reset? Does excluding these perturbations limit the model's sensitivity to low-intensity noise?

Minor: The use of bold font in the main text seems arbitrary and should be applied more consistently.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper establishes a theoretical foundation for Membrane Potential Perturbation Dynamics (MPPD) in spiking neural networks (SNNs), proving that MPPD corresponds to Total Variation. The authors propose a new framework that improves robustness to adversarial perturbations compared to the existing MPPD model. Experimental results on CIFAR-10 and CIFAR-100 demonstrate that MPPD achieves superior accuracy and robustness under various adversarial attacks.

### Strengths
- The paper provides a clear mathematical link between MPPD and total variation, offering the first formal theoretical explanation for an empirically effective mechanism in SNN robustness.
- Experimental results on CIFAR-10 and CIFAR-100 demonstrate that MPPD achieves superior accuracy and robustness under various adversarial attacks.

### Weaknesses
- The experiments are limited to CIFAR-10 and CIFAR-100. These are small-scale image datasets. Experiments on more large-scale datasets and neuromorphic datasets are encouraged.
- Although the paper mentions efficiency, there is no detailed analysis of training time and gradient stability.
- The paper does not report clean test accuracy alongside adversarial robustness results.

### Questions
In the abstract, the authors state that “this finding may provide a new insight into the essence of perturbation characterization.” Could the authors clarify what specific insights are being referred to here?

### Soundness
3

### Presentation
3

### Contribution
2
