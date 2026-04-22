# Gradients Through Logarithmic Lens: Reformulating Optimization Dynamics

- Avg Score: 4.40
- Decision: Reject
- Scores: 8, 2, 6, 2, 4

## Abstract
Optimization in deep learning remains a fundamental challenge, and developing techniques that improve training efficiency and enhance model performance is essential. We present a method for producing effective optimization frameworks, introducing the activation function LogLU (***log**arithmic **l**inear **u**nit's*) and the optimizer ZenGrad (***zen** represents smooth, **grad**ients*), along with its momentum-based variant, M-ZenGrad, all of which incorporate the logarithmic formulation. We conducted extensive evaluations on benchmark datasets spanning vision and language tasks, demonstrating that each component individually enhances performance while collectively showcasing the advantages of the logarithmic approach. Additionally, ablation studies analyze the contribution of each method, and careful hyperparameter tuning ensures robust and optimal performance, indicate the effectiveness of our logarithmic optimization framework across diverse tasks and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a "logarithmic lens" for deep learning optimization, presenting a new activation function, LogLU, and an optimizer, ZenGrad (with a momentum variant, M-ZenGrad). Both components use logarithmic formulations to, respectively, stabilize gradient flow and adaptively scale the learning rate. The authors provide theoretical proofs for stability and report strong empirical results across vision and language tasks.

### Strengths
- The paper presents a novel and cohesive framework by applying the logarithm to both activation functions and optimizers, supported by theoretical analysis.

- The proposed LogLU activation and ZenGrad optimizer are independent contributions, allowing them to be adopted separately, which increases their potential impact.

### Weaknesses
- The paper fails to adequately justify why logarithmic scaling is superior to the square-root scaling used in well-established optimizers like AdaGrad or Adam.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces LogLU, a logarithmic activation function designed to bound gradient magnitudes, and ZenGrad, an optimizer whose step size decays with the logarithm of the accumulated squared gradients. The authors claim these “logarithmic” mechanisms improve stability and generalization, reporting strong gains across diverse tasks, including large improvements on ImageNet with ViT-S/16.

### Strengths
- S1: Simple, interpretable design ideas (bounded activation and logarithmic learning-rate scaling) that connect well to stability intuitions.
- S2: Empirical coverage spans several architectures and tasks, at least demonstrating ease of integration.
- S3: The proposed methods are computationally lightweight and compatible with standard frameworks.

### Weaknesses
Theory:
There are big gaps in the theoretical results presented in this paper. Theorem 3.3 uses a first-order approximation in place of a descent inequality to claim $V(w_{t+1}) \leq V(w_t)$. Without $L$-smoothness and a step-size bound that controls the second-order remainder (and without convexity/PL-type structure) the claimed Lyapunov decrease and ''convergence towards the global minimum'' are not justified. A correct result would either (i) assume L-smoothness and derive a proper descent inequality (with PL for global claims), or (ii) restrict to convergence to stationary points.

Experiments:
The ViT-S/16 ImageNet jump to 78.8% (Table 1) vs. AdamW/Lion (~70–72%) appears exceptionally large. To better understand whether this is purely attributable to the potentially better optimizer, the authors should provide more details on the experimental setup. They claim that each optimizer is carefully tuned, but the tuning section adjusts only LR and WD. A full recipe (augmentations, EMA, label smoothing, mixup/cutmix, LR schedule/warmup, batch/global batch, precision, clipping/LLRD) would be important to judge these results better. Additionally,  wall-clock results as well as a few key baselines (Adagrad/RMSProp) and Softplus for the activation study are absent.

### Questions
Q1: What is the exact recipe used in the ViT experiments?
Q2: Have the authors tried different learning rate schedules for each optimizer?
Q3: Why include an ablation on ln vs log-10 when it is just a constant factor?

### Soundness
1

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
4

### Summary
This manuscript challenges gradient descent optimization techniques. The authors present a new activation function of LogLU, which is a bounded gradient norm. The improved property of LogLU is further enhanced with ZenGrad. Experiments on ImageNet, CIFAR, Pascal VOC, and WikiText-2 demonstrate improvements.

### Strengths
- ViT-S/16 training on ImageNet-1K exhibits clear and impressive results.
- Both of the two contributions, LogLU and ZenGrad, are valuable. Specifically, I appreciate Table 4, where LogLU itself brings clear gain compared with the ReLU and GELU.
- Source code is available, which eases deployment in practice.

### Weaknesses
- I think the inclusion of the logarithmic term is intended for the bounded gradient norm, such as Lemma 3.1 and others, but it would be better to further clarify the connection of LogLU.
- How about existing activation functions, such as ReLU and GELU? Compared with them, does LogLU bring improved properties such as a bounded gradient norm?
- Although ViT-S/16 training on ImageNet-1K exhibits clear and impressive results, training 90 epochs for ResNet-18 on ImageNet-1K does not look like full convergence; top-1 accuracy of 60~70% is not strong enough. Also, ViT-S/16 results might be too impressive a gain; the authors should clarify the hyperparameter setup.
- This is not a request for rebuttal, but I would like to see more experiments on other models such as ViT-B and ViT-L in the future.
- The use of smaller \epsilon, such as $10^{-8}$, looks unstable in early training when P is close to zero.
- Please check the following mathematics.
    - Check the sign of z^2/2 at Line 613.
    - For the total loss at Line 624, is it correct to adopt $\log(…)$ and not $-\log(…)$?
    - For Lemma 3.1, because $P_t$ at Eq. 2 applies square, I think its upper bound should be $tG^2$.
    - The denominator at Eq. 3 uses $P_t$, whereas the denominator at Line 183 uses $P_{t+1}$. This requires explanation.
- Writing should be improved.
    - “indicates the effectiveness” → “indicate the effectiveness” or “indicating the effectiveness” in abstract.
    - “We evaluate” → “we evaluate” at Line 270.
    - “we evaluate” → “We evaluate” at Line 324.
    - “M-ZenGrad ,across” → “M-ZenGrad, across” at Line 353.
    - “COMPARISION” → “COMPARISON” at Section 4.5.
    - “Additionally, Convergence” → “Additionally, convergence” at Line 194.
    - “it..” → “it.” at Line 376.
    - “ZenGrad and M-ZenGrad optimizer, comparing its performance” → “ZenGrad and M-ZenGrad optimizers, comparing their performance”
    - “In our experiments” → “in our experiments” at Line 413.
    - “to Keep the” → “to keep the” at Line 414.
    - “MZenGrad” → “M-ZenGrad” at caption of Figure 5.

### Questions
Please see the weaknesses above. My score is based on the assumption that all typos are corrected in the revised manuscript.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LogLU, a piecewise activation function that applies identity mapping for positive inputs and logarithmic transformation for negative inputs, alongside ZenGrad and M-ZenGrad optimizers that scale learning rates using logarithmic functions of accumulated squared gradients. The authors provide theoretical analysis establishing gradient bounds, Lipschitz continuity, and convergence properties, and conduct experiments across image classification (ImageNet, CIFAR-10/100), segmentation (Pascal VOC), and language modeling (WikiText-2) tasks to demonstrate that their logarithmic formulations can improve training stability and performance compared to standard methods like AdamW, Lion, and SGD.

### Strengths
1. Solid theoretical foundation with formal analysis. The paper provides rigorous mathematical proofs for key properties including Proposition 2.1 showing LogLU gradients are bounded in (0,1], Proposition 2.2 establishing Lipschitz continuity with constant L=1, and Theorem 3.3 proving Lyapunov stability and convergence for ZenGrad. These theoretical guarantees provide formal justification for why the logarithmic formulations should improve optimization dynamics.
2. Comprehensive experimental evaluation across diverse tasks and careful ablation studies. The authors test their methods on multiple domains including vision (ImageNet-1K with ResNet-18 and ViT-S/16, CIFAR-10/100 with ResNet-32, Pascal VOC segmentation) and language (WikiText-2), with both training from scratch and fine-tuning protocols. Section 4.4 and Figure 5 provide thorough ablation studies examining learning rate, epsilon values, momentum coefficients, and logarithmic base choices, demonstrating attention to experimental rigor.

### Weaknesses
1. Limited novelty and insufficient differentiation from existing methods. The LogLU activation is essentially a variant of Softplus (log(1+exp(x))) with a piecewise definition, and the ZenGrad optimizer closely resembles AdaGrad with logarithmic damping instead of square root damping. The paper does not adequately explain why the logarithmic formulation offers fundamental advantages over existing adaptive methods. 

2. Experimental results are inconsistent and sometimes show marginal or negative improvements. Table 1 shows highly variable performance across datasets and models. For ResNet-18 on ImageNet, ZenGrad achieves 67.78±0.282% while M-ZenGrad reaches 69.29±0.254%, which is only marginally better than baseline methods (AdamW: 66.21±0.482%, Lion: 66.15±0.361%) and still substantially worse than SGD's 69.22±0.32% reported in Table 5. More concerningly, on CIFAR-100 (Table 3), LogLU with many optimizers actually performs worse than other activations (e.g., LogLU+ZenGrad: 72.37±0.21% vs Mish+M-ZenGrad: 73.54±0.17%). 

3. Presentation issues and missing critical experimental details undermine reproducibility. The paper suffers from unclear writing and incomplete information. The motivation for the specific logarithmic formulation remains vague (lines 118-127 discuss benefits but don't explain why logarithm specifically). The choice of $\epsilon$ outside versus inside the logarithm in Equation 3 is stated as preventing instability but lacks theoretical or empirical justification.

### Questions
1. What is the fundamental advantage of logarithmic scaling over existing adaptive damping mechanisms?


2. Why does your method show highly inconsistent performance, and under what conditions should practitioners expect it to work?

3. Why is $\epsilon$ placed outside the logarithm in Equation 3, and what is the empirical justification for this design choice?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel optimization framework based on logarithmic transformations, proposing the LogLU activation function, the ZenGrad optimizer, and its momentum variant M-ZenGrad, all designed to improve gradient flow and training stability in deep neural networks. Theoretical analysis establishes key properties such as bounded gradients and Lipschitz continuity for LogLU, and convergence for ZenGrad, while extensive experiments across vision and language tasks demonstrate consistent performance gains over standard methods. Ablation studies confirm the robustness of the proposed components and their effectiveness in combination.

### Strengths
1. The paper provides rigorous theoretical analysis, proving that bounded LogLU gradients, Lipschitz constant = 1, ZenGrad step-size bounds and Lyapunov convergence guarantees for both convex and non-convex cases. 

2. The authors conduct extensive experiments on diverse benchmarks (ImageNet, CIFAR, Pascal VOC, Wikitext-2) and architectures (ResNet, ViT, U-Net, GPT-style), demonstrating consistent and often superior performance of their methods compared to established baselines like AdamW and SGD.

3. The design choices (e.g., the logarithmic scaling in ZenGrad, the negative log compression in LogLU) are well-motivated, and thorough ablation studies (on hyperparameters, logarithmic bases, and component combinations) effectively validate the contributions of each part of the framework.

### Weaknesses
1. The use of logarithmic or sub-linear functions for gradient scaling (as in ZenGrad) or activation (as in LogLU) builds heavily on existing concepts like adaptive learning rates (e.g., AdaGrad) and smooth, bounded activation functions, potentially limiting the perceived novelty of the core mechanisms.

2. The paper does not report or discuss the computational overhead (e.g., training time, memory usage) of the proposed methods compared to standard optimizers, which is a crucial practical consideration for adoption, especially given the additional logarithmic computations and gradient accumulation.

3. Lack  of hyper-parameter sensitivity analysis. Although ablations are thorough, ZenGrad needs 5–10× higher learning rate than AdamW; the paper does not clarify how brittle this scaling is across very deep or heterogeneous architectures.

### Questions
ZenGrad’s denominator accumulates every squared gradient element-wise, so its memory footprint grows linearly with the number of parameters; how does this scale to billion-parameter models?

### Soundness
3

### Presentation
3

### Contribution
2
