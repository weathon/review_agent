# Darwinian Optimization: Training Deep Networks with Natural Selection

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
In conventional deep learning training paradigms, all samples are usually subjected to uniform selective pressure, which fails to adequately account for variations in competitive intensity and diversity among them. This often leads to challenges such as class imbalance bias, insufficient learning of hard samples, and improper handling of noisy samples. Drawing inspiration from the principles of species competition and adaptation in natural ecosystems, we propose a bio-inspired optimization method for deep networks, termed Natural Selection (NS). NS introduces a competition mechanism by first assembling a group of samples into a composite image and then downscaling it to the original input size for model inference. Each sample is then assigned a natural selection score based on the model's predictions on this composite image, reflecting its competitive status within the group. This score is further used to dynamically adjust the loss weight of each sample, facilitating an adaptive network optimization process driven by competitive interactions among training samples. Experimental results on 12 public datasets consistently demonstrate that NS improves performance without being tied to specific network architectures or task assumptions. This study offers a novel perspective on deep network optimization and holds instructive significance for broader applications. The code will be made publicly accessible.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Quite interesting work; A novel Darwinian perspective to optimization dynamics in NN. The paper presents a novel bio-inspired optimization method called Natural Selection (NS) that introduces explicit competition among training samples. By computing competitive scores through image stitching and dynamically adjusting sample loss weights, the method as empirically shown achieves consistent improvements across diverse computer vision tasks.

### Strengths
I quite enjoyed reading this paper, which offers a refreshing biology-inspired perspective on optimization for deep networks.

1) Novel research problem: The authors adequately point out the uniform handling of samples in current optimization theory and raise an intuitive question about whether introducing an explicit competition mechanism could be beneficial.
2) Clear visualization: Figure 1 provides an effective visual explanation of their core idea, making the concept accessible.
3) Well-articulated analogy: The authors provide an easy-to-understand mapping between network optimization and ecosystem evolution, along with a succinct research question that is well-formulated in Section 3.1.
4) Thorough empirical validation: The experimental evaluation is comprehensive across multiple tasks. I especially appreciate the thoroughness of the analysis, for example in the long-tailed classification experiments, which demonstrates the method's effectiveness across diverse scenarios (especially following the observations in line 329-339).

### Weaknesses
Weaknesses 
1) Computational overhead: 10-13% training time increase (Table 6) is non-trivial for large-scale applications. 
2) Group size limitations: Why groups of 2 or 4? The authors should include some comment/explanation or intuition around this or include a systematic exploration of other group sizes.
3) Hyper-parameter sensitivity: The base value $\sigma$ seems to vary significantly across tasks (0.6 to 2.5) as per Appendix, suggesting the method may require careful tuning per dataset. Is there a way to tackle this? 
4) I think naively combining images through simple concatenation may introduce artificial edge patterns that don't reflect natural competition. This may potentially cause the NS scores to reflect stitching artifacts rather than genuine sample competitiveness. I think that could explain some of the task-dependent behavior and hyper-parameter sensitivity observed across experiments? Do they authors saw/infer something similar as well? 

While I don't think this is a weakness - I do think that the authors should comment briefly on this in the paper: Could a variant of NS that does backpropagate through the competition mechanism learn better competitive evaluations over time? Or would this introduce instability and other issues?

### Questions
1) Do the authors have some idea how can they tackle the computation overhead limitation? 
2) How is the optimal group size determined across different tasks?Is group size a sensitive hyper-parameter? What principles guide the selection—does it depend on batch size, number of classes, or dataset characteristics?
3) How does NS behave in the presence of outliers or anomalous samples? Based on the image classification results (Section 4.2), the Winner-Strengthening strategy performs better on ImageNet, which has label noise. Does favoring winners risk marginalizing legitimate hard samples or minority patterns that appear outlier-like?
4) Do the authors believe this approach will transfer effectively to text-based or other non-vision tasks? The stitching mechanism is naturally suited to images, but how would competition be simulated for text (concatenation of sentences/tokens?)

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Darwinian Optimization, a bio-inspired training paradigm for deep neural networks based on the principle of natural selection. By explicitly modeling competition among samples through a Natural Selection (NS) score, the method dynamically adjusts per-sample loss weights to mimic ecological adaptation, enabling more balanced, efficient, and generalizable optimization of deep networks.

### Strengths
The key strength of this paper lies in introducing a biologically inspired optimization framework that embeds the principle of natural selection into deep network training. By modeling explicit inter-sample competition, it breaks away from uniform loss optimization, enabling adaptive and ecologically balanced learning that enhances both generalization and robustness. Moreover, the method is lightweight, architecture-agnostic, and easily integrable into existing training pipelines, demonstrating strong universality and scalability.

### Weaknesses
1.The analogy between biological evolution and sample competition remains conceptual without formal mathematical grounding.
2.The paper lacks detailed ablation studies isolating the effects of design components such as stitching, σ, and ρ.
3.Although the method achieves stable performance improvements across multiple datasets and network architectures, the gains appear to be relatively modest.
4.The “evolutionary rethinking” section repeats background rather than deepens analysis.
5.How should we interpret the phrase in the abstract: “thereby forming an optimization process that more closely mimics a Darwinian ecological equilibrium”?

### Questions
See the weakness section.

### Soundness
3

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
The paper proposes Natural Selection (NS) as a novel optimization method, drawing inspiration from species competition and adaptation in natural ecosystems. NS introduces a dynamic competition mechanism among training samples. This bio-inspired approach is designed to mitigate classic deep learning training challenges, including class imbalance bias, insufficient learning of hard samples, and instability due to noisy data by applying non-uniform selective pressure.

### Strengths
- The paper is well-written, and its central perspective of connecting Darwinian theory of species to deep learning provides a novel perspective.
- The experiments cover a wide range of factors, including diverse architectures, multiple datasets, class imbalance and domain adaptation scenarios.

- The approach shows strong performance and maintains its efficacy even under challenging class imbalance conditions.

### Weaknesses
- The average empirical improvements reported in Tables 1, 2, and 3 for NS are marginal especially in the large-scale data like Imagenet 1K. 

- Furthermore, the number of experimental seeds used is not specified, making the reported gains difficult to interpret. Statistical significance analysis is needed to confirm that these marginal improvements are meaningful and not due to random variation.

-  Interpreting the results is difficult because the authors label their overall method as "NS" while the actual techniques applied in the experiments are "Winner Strengthening" or "Loser Focussing are not specified clearly. 

- The abstract description of the core mechanism ("stitching and scaling a group of samples") is vague. Without explicit mathematical details, it is difficult to assess the computational complexity. Provide a detailed comparison of the per-iteration computational overhead of NS versus comparisons. 

- The current set of empirical comparisons is heavily biased toward loss function modification methods (e.g., Focal Loss, GCE, PolyLoss). The evaluation lacks crucial baselines from the highly effective data-level rebalancing category, such as those based on sampling. This gap makes it impossible to assess if the proposed method's gains are simply equivalent to or outperformed by simpler, established sampling techniques.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3
