# Incorruptible Neural Networks: Training Models that can Generalize to Large Internal Perturbations

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 4

## Abstract
Flat regions of the neural network loss landscape have long been hypothesized to correlate with better generalization properties. A closely related but distinct problem is training models that are robust to internal perturbations to their weights, which may be an important need for future low-power hardware platforms. Several methods have been proposed to guide optimization toward improved generalization, such as sharpness-aware minimization (SAM) and random-weight perturbation (RWP), which rely on either adversarial or random perturbations, respectively. In this paper, we explore how to adapt these approaches to find minima robust to a wide variety of random corruptions to weights. First, we evaluate SAM/RWP across a wide variety of noise settings, and in doing so establish that over-regularization during training is key to finding optimally-robust minima. At the same time, we also observe that large perturbations lead to a vanishing gradient effect caused by unevenness in the loss landscape, an effect particularly pronounced in SAM. Quantifying this effect, we map out a general performance trend of SAM and RWP, determining that SAM works best for robustness to small perturbations, whereas RWP works best for large perturbations. Lastly, to overcome the deleterious vanishing gradient effect during training, we propose a dynamic perturbation schedule which matches the natural evolution of the loss landscape and produces minima more noise-robust than otherwise possible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the model's robustness to weight perturbations, which may appear on real-world hardware devices. Building on existing SAM and RWP studies, this work examines the weight loss landscape during the test phase and observes that over-regularization is key for achieving optimal robustness. The paper further proposes a dynamic perturbation strategy to mitigate the negative impact of overly large perturbations on model performance.

### Strengths
1. It investigates the model's robustness to weight perturbations, which corresponds to the flatness of the weight loss landscape during the test phase, distinguishing it from previous research on SAM and RWP.  
2. The paper is clearly written, the figures and tables are easy to understand, and the overall flow of the text is good.  
3. The experiments in this paper are comprehensive. The paper conducts a systematic evaluation of SAM and RWP under various noise settings, and its proposed dynamic perturbation schedule is validated across multiple experimental setups, demonstrating that it can produce minima with significantly higher noise robustness than other methods.

### Weaknesses
1. The motivation is ambiguous. This paper associates the model's robustness to weight perturbations with AIMC hardware errors, which is a relatively novel scenario. However, it is not clear that the methods and experiments in this paper can be reliably transferred to real AIMC hardware.  
2. The practical value of this article is unclear. The paper does not clearly demonstrate the effectiveness of its proposed method in real-world hardware deployment scenarios, such as evaluation using a target hardware's specific dataset or error profile. This leaves the practical application value of the work ambiguous.

### Questions
1. Could the authors provide more sufficient experiments to validate the phenomena and methods presented in this paper, such as broader noise modeling (hardware-specific noise modeling) or more complex network architectures (lightweight designs)?  
2. Are there more application scenarios for examining the weight loss landscape during the test phase, especially those related to deep learning and neural network tasks? Relevant discussion would make the contribution of this paper more explicit.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work empirically studies SAM and RWP against weight perturbations, and design a  dynamic scheme for enhancement.

### Strengths
It presents in-depth comparisons between SAM and RWP, providing more extensive evaluations on top of the existing findings.

### Weaknesses
1. This work seems an empirical study, where no theoretical analyses or analytical discussions are provided, making limited contributions to the fundamentals in this topic.

2. As being positioned from a rather empirical perspective, it would be more solid to expand the evaluations, i.e., Transformer-based architectures, tasks/datasets beyond imaging processing, and so on. 

3. By intuitive speculations, results from sec 4.2 and 4.3 are not difficult to conceptualize. Despite the observations presented in this work, it is still taking efforts to determine the perturbation strength (we still need a lot tuning in the experiments). Hence, from the reviewer's perspective and understanding, it is a bit obscure to truly seize the fundamental and key contribution here.

4. For the dynamic schedule, it is rather heuristic and remains with efforts to do quite some tuning, which hinders its contributions practical-wisely. 

5. In general, this work is a bit tedious in writing and wordy, where all analyses, investigations, results explanations are all by text. It would be good with some analytical discussions and other way of presenting. The reviewer recommend the authors to clarify the novelty, and its soundness and practical impact to the field with more in-depth analyses, better with enhance evaluations under more varied settings in model structures, datasets, and tasks, rather than the current plain comparisons between SAM and RWP.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the weight robustness of neural networks trained using Sharpness-Aware Minimization (SAM) and Random Weight Perturbation (RWP). The authors find that using stronger perturbations during training than those at test time leads to optimally noise-robust models. They also identify a vanishing-gradient effect in SAM when strong perturbations are applied. To address this, they introduce a dynamic perturbation schedule that gradually increases perturbation magnitude throughout training, aligning with the evolving loss landscape and resulting in models with significantly improved robustness to weight noise.

### Strengths
The demonstrations in Figures 3 and 4 are clear and informative. They effectively illustrate how the loss, gradient norm, and sharpness evolve during training, revealing how SAM and RWP interact with the geometry of the loss surface. The experiments are well-controlled, with systematic variations in hyperparameters to isolate specific effects. These analyses lead to meaningful conclusions about the vanishing-gradient phenomenon in SAM and the differing robustness characteristics of SAM and RWP.

### Weaknesses
1. The motivation for this work is not clearly articulated. As stated on page 1, paragraph 2, the authors refer to analog in-memory computing (AIMC) and “hardware errors” as the practical motivation. However, the discussion remains vague. The paper briefly claims that prior works lack “a broad understanding of noise-robustness in neural networks” and “a connection of these efforts to existing flatness-finding approaches such as SAM and RWP,” but these statements are overly general and do not convincingly establish the importance or novelty of addressing these issues.
Moreover, regarding “hardware errors,” it is unclear what specific form these errors take. Are they modeled as random Gaussian perturbations, or do they follow structured patterns such as quantization errors, conductance drift, or loss of precision? Since the type and distribution of hardware noise fundamentally determine the robustness objective, this assumption is crucial and should be explicitly defined and justified. Without a clear characterization of the error model, the practical relevance of the results to real AIMC systems remains uncertain.

2. On page 4, the authors state that all experiments are conducted using ResNet-18 on the CIFAR-100 dataset. While this setup is reasonable for preliminary investigation, it represents a relatively small-scale benchmark. Consequently, the conclusions about over-regularization, vanishing gradients, and noise robustness may not generalize to larger or more complex architectures and datasets.

3. The paper lacks theoretical justification or analytical evidence supporting the observed phenomena. The findings are entirely empirical, relying on trends specific to this experimental configuration. To substantiate the general claims about the relationship between perturbation strength, gradient behavior, and robustness, additional experiments on larger-scale datasets (e.g., ImageNet) or complementary theoretical analysis would be necessary. Without such validation, the conclusions remain suggestive rather than definitive.

### Questions
1. Theoretical background:
While the paper includes a Preliminaries section, it does not provide a complete theoretical framework or formal theorems supported by clear assumptions. I would expect at least a subsection devoted to theoretical analysis that establishes or justifies the observed phenomena, such as the relationship between perturbation strength, flatness, and gradient behavior. I suggest that future revisions include a more rigorous theoretical component to substantiate the empirical findings.

2. Experimental coverage:
In the absence of sufficient theoretical support, the experimental evaluation should be broadened to ensure the robustness and generality of the conclusions. Specifically, experiments should include a wider range of architectures, datasets, or even different tasks to demonstrate that the proposed methods and observations hold beyond the ResNet-18 and CIFAR-100 setting.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates model robustness in the weight space by introducing two loss functions, SAM and RWP. It yields two key findings: (1) over-regularized training with strong perturbations yields the most robust weights; (2) strong perturbations in the weight space could induce a vanishing gradient effect due to increased sharpness in the loss landscape.

### Strengths
**S1.** Investigating the impact of weight perturbations at test time is of significant interest, as it represents a practical challenge in real-world systems affected by hardware noise.

**S2.** The authors conduct extensive experiments analyzing the loss landscape, sharpness, backbones, and the effects of varying $\sigma$ values.

### Weaknesses
**W1.** Since model parameters are often not externally accessible, I think the premise of manually perturbing weights has limited practical relevance. However, I agree with the argument in the introduction that hardware can inject noise into model weights. Therefore, investigating robustness against real hardware noise patterns is crucial. In this work, the perturbations are distributed as zero-mean isotropic Gaussian. How this assumption aligns with real-world noise patterns?

**W2.** As shown in Tables 1 and 2, model robustness varies significantly with different configurations of $\sigma_{test}$, $\sigma_{train}$, and $\rho$. This implies that a model must be retrained specifically for different test-time noise profiles (i.e., different hardware), which is computationally expensive. A more critical issue arises when the hardware noise characteristics are unknown, making it impractical to determine the appropriate training parameters.

**W3.** The experimental validation is limited to relatively shallow networks (e.g., ResNet-18, WRN-16). It remains unclear whether the observed benefits of SAM and RWP would scale to deeper architectures, which is critical for assessing the general applicability of the proposed methods.

**W4.** The practical impact of this work would be significantly strengthened by validation on real hardware. The current approach, which relies on simplifying Gaussian assumptions to simulate noise, may not fully capture the complex characteristics of actual analog systems, thereby limiting the persuasiveness of the findings.

### Questions
Please address **W1-W3** in the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3
