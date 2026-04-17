# Progressive Alignment for Robust Domain Adaptation

- Decision: Reject
- Scores: 4, 4, 4, 6, 2

## Abstract
Unsupervised Domain Adaptation (UDA) has advanced knowledge transfer between labeled source and unlabeled target domains, yet existing methods fall short in real-world scenarios where adversarial attacks threaten model reliability. Robustness against such attacks is essential but remains critically underexplored in UDA. Existing methods often treat domain alignment and adversarial defense as separate steps, causing unstable training, noisy pseudo-labels, and incomplete feature alignment ultimately limiting their effectiveness. Addressing both domain shift and adversarial robustness simultaneously is vital for deploying trustworthy models in dynamic, adversarial environments. In this work, we propose a robust UDA method from the perspective of multi-source and multi-target domain adaptation, treating clean and adversarial samples across both source and target as distinct domains. We aim to align both clean and adversarial domains across source and target within the adaptation framework. Therefore, we use progressive domain alignment strategy that explicitly aligns clean target features with multi-source domains through classifier discrepancy minimization, and implicitly aligns adversarial target features by enforcing classifier agreement on pseudo-labels. We find that this strategy effectively handles both domain shift and adversarial perturbations, leading to improved generalization and robustness. We demonstrate the effectiveness of our approach through extensive experiments on four benchmark datasets, accompanied by component-wise ablations. Our method achieves standard accuracies of 62.0\%, 88.4\%, 82.5\%, and 73.7\%  and the corresponding robust accuracies under PGD-20 attack with $\epsilon = 2/255$ are 49.4\%, 78.3\%, 77.3\%, and 72.1\% on the \textit{Office-Home}, \textit{PACS}, \textit{VisDA}, and \textit{Digit} benchmark datasets, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel approach to adversarially robust unsuperivsed domain adaptation, which treats the clean and adversarial samples from source and target data as four distinct domains, and handles the alignment with adversarial source domain, clean target domain, and adversarial target domain successively: (1) warm-start training adopts adversarial training on source data; (2) explicit alignment utilizes MCD for clean target domain; (3) implicit alignment adapts the model to adversarial target domain with a curriculum learning strategy from high-confidence to low-confidence adversarial samples, and uses a double consistency criterion for more reliable pseudo-labels. Experimental results suggest the superiority of the proposed method.

### Strengths
1. The proposed progressive learning framework and the implicit alignment method are novel for robust UDA.
2. Figures 3 and 4 provide insights into the training process and demonstrate the contribution of different stages.
3. The effectiveness of the proposed method is validated by extensive experiments.

### Weaknesses
1. The presentation of the paper requires serious revision to ensure formalness and clarity. 
  - $\mathcal{H}$ is undefined in Line 145.
  - Eq. 9-12 are ambiguous due to the reuse of $\mathbf{Z}$ to represent the outputs for different samples.
  - Incorrect citation styles (should use `\citep` for most cases).
  - Missing citations at Lines 709-710.
  - Missing spaces around periods and parentheses (e.g., Line 344) and other typos.
2. The proposed method uses the Target Consistency Rate to determine the convergence of explicit alignment instead of the discrepancy loss used in the original MCD. The superiority of this convergence condition is not empirically verified.
3. Judging from Algorithm 1 (Sec. C), the adversarial source samples are pre-computed before warm-start and fixed during training, instead of being computed based on the latest model parameters. This can significantly deteriorate the effectiveness of adversarial training. The implicit alignment stage may also suffer from the issue of offline adversarial samples, even though the adversarial test samples are updated every $n$ epochs.
4. It is unclear which baseline results are replicated in this paper and which are quoted from the original papers. For quoted results, the comparison may be unfair because the test target data are split for this paper (Lines 811-815) and could differ from those in previous studies. The consistency in experimental settings of the compared methods requires further clarification.
3. The provided source code (anonymous link at Line 845) is buggy, with undefined variables and indentation errors.

### Questions
1. Are the results in Section D.2 produced by training the models with $\epsilon=8/255$?

### Soundness
2

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
3

### Summary
This paper addresses adversarial robustness in the context of unsupervised domain adaptation (UDA). The authors propose a robust UDA framework that treats adversarial examples as samples from distinct source and target domains separate from the clean data. This formulation effectively converts the standard UDA problem into a multi-source, multi-target domain alignment setting. A progressive alignment strategy is then introduced to train the model to align these domains sequentially.

### Strengths
The paper proposes an interesting and novel perspective on integrating adversarial robustness and domain adaptation. By incorporating adversarial examples directly into the UDA framework, the method unifies what is often a two-stage training process (UDA followed by adversarial training) into a single, coherent pipeline.

### Weaknesses
Further clarifications needed:

- In Line 139, adversarial examples are introduced as belonging to additional domains. However, adversarial perturbations are typically generated with respect to a given classifier. Could the authors clarify which classifier is used to generate these adversarial examples within the “adversarial source” and “adversarial target” domains?
- As the paper studies UDA under adversarial perturbations, the precise **learning objective** and **evaluation metrics** need to be formally defined. What exactly is the target performance criterion (e.g., robust accuracy on target domain under specific attack)? Is the clean accuracy a part of performance criterion as well?
- Regarding loss functions (2) and (3), please elaborate on the challenges of directly optimizing Equation (2). Explaining this difficulty would clarify the motivation for adopting the surrogate objective in Equation (3).
- The paper maximizes the discrepancy between two classifier heads for unlabeled target data (i.e., objective function (4)). Could the authors explain the intuition behind this design? Why should increasing the discrepancy between classifiers help improve adversarial robustness or facilitate domain alignment?
- Line 244-248. Which model is used to generate adversarial examples? Why are _“weak” adversarial examples_ (whose predictions match the clean examples) retained for training? In what sense are these examples considered _“reliable”_? Reliable with respect to what?
- The proposed progressive training strategy begins with “weak” adversarial examples and gradually introduces “stronger” ones. Could the authors justify why such progressive inclusion is necessary? Providing a reasonable explanation would greatly improve the soundness of this design choice.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the robust unsupervised domain adaptation (UDA) problem setting. The paper proposes a method from the perspective of multi-source and multi-target domain adaptation, treating clean and adversarial samples across both source and target as distinct domains. The proposed method leverages progressive domain alignment strategy that explicitly aligns clean target features with multi-source domains through classifier discrepancy minimisation, and implicitly aligns adversarial target features by enforcing classifier agreement on pseudo-labels.

### Strengths
1. The paper addresses robust UDA setting which is an important and practical problem setting.
2. The proposed method is well motivated and the overall paper is well written.

### Weaknesses
1. Limited Novelty: The methodologies in the two main modules: Explicit Alignment and Implicit Alignment, appear closely related to prior work. In particular, Explicit Alignment is similar to Saito et al. (2018b), and Implicit Alignment resembles Han et al. (2020). Please clarify what is fundamentally new in your formulation and why these differences matter empirically or theoretically.

2. Methodology needs clearer explanation. In the warm-start stage, why is the loss on adversarial samples defined identically to the loss on clean samples? By design, adversarial examples disrupt effective feature learning, optimizing both sets in the same way makes it unclear how the latent features become robust. Likewise, the Implicit Alignment module requires more detail: from Eq. (10), the pseudo-label is the argmax agreed upon by both classifiers. In that case, wouldn’t the loss in Eq. (12), computed between the same pseudo-label and the logits, tend to be trivially small? 
 
3. Computational complexity analysis is missing: As the proposed method contains multiple stages, a detailed analysis of training and inference complexity is necessary to assess practical feasibility.

4. Missing baselines: In Table 1, several standard UDA methods are missing [R1][R2][R3]. It is important to compare the proposed method with these to understand the effectiveness of the proposed method.


[R1] Zhang, Yuchen, et al. "Bridging theory and algorithm for domain adaptation." International conference on machine learning. PMLR, 2019.

[R2] Rangwani, Harsh, et al. "A closer look at smoothness in domain adversarial training." International conference on machine learning. PMLR, 2022.

[R3] Zhang, Xinyu, Meng Kang, and Shuai Lü. "Low category uncertainty and high training potential instance learning for unsupervised domain adaptation." Proceedings of the AAAI conference on artificial intelligence. Vol. 38. No. 15. 2024.

### Questions
1. Explain clealry the differences between the proposed modules, Explicit Alignment and Implicit Alignment with prior works Saito et al. (2018b) and Han et al. (2020), respectively.

2. Explain the methodology clearly. Specifically, In the warm-start stage, why is the loss on adversarial samples defined identically to the loss on clean samples? By design, adversarial examples disrupt effective feature learning, optimizing both sets in the same way makes it unclear how the latent features become robust. Likewise, the Implicit Alignment module requires more detail: from Eq. (10), the pseudo-label is the argmax agreed upon by both classifiers. In that case, wouldn’t the loss in Eq. (12), computed between the same pseudo-label and the logits, tend to be trivially small? 

3. Provide a detailed computational complexity analysis to assess the practical feasibility of the proposed method.

### Soundness
2

### Presentation
3

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
The paper aims to address the adversarial robustness problem in an unsupervised domain adaptation setting. The authors argue that solving both the domain shift and adversarial shift problems jointly, rather than treating them as separate (decoupled) problems, is the right approach. To this end, they reframe the task as a multi-source, multi-target domain adaptation problem, which accounts for both the domain shift and the adversarial gap. The proposed method consists of three phases.

First, a model with two classifier heads is trained on labeled source domains using both normal and adversarial examples. Second, the classifiers and feature extractor are trained alternately in an adversarial manner. Finally, a pseudo-labeling process is used to obtain labels, after which adversarial examples from the target domain are generated and explicit domain adaptation is performed.

Multiple experiments are conducted to demonstrate the performance of the proposed approach. For the experiments, the authors assume access to two labeled domains.

### Strengths
Overall, this is a clear and easy-to-follow paper. The authors have identified gaps in prior work, reframed the problem and proposed a method to address them. The proposed approach is sufficiently novel and effectively bridges the identified gap.

### Weaknesses
The experimental results are somewhat limited. It would be useful to include additional experiments using transformer-based backbones, such as ViT. Moreover, evaluating the method with standard adversarial attacks, such as AutoAttack [1], would strengthen the results, as PGD-based attacks are known to be prone to certain issues.

Another weakness of the work is practical utility. Authors, in the introduction and abstract mention existing works limited in real-world domain, such as 
>yet existing methods fall short in real-world scenarios where adversarial attacks threaten model reliability. 

Yet, there is no discussion around this point in the paper. 

[1] https://robustbench.github.io/

### Questions
In Section 2.2, it is unclear how the authors ensure that adversarial training actually contributes to the overall learning.

The authors assume access to multiple labeled source domains for their experiments, whereas most prior works typically consider only a single labeled source domain. It is therefore unclear how a fair comparison with these prior works can be justified.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the critical challenge of adversarial robustness in Unsupervised Domain Adaptation (UDA) by proposing a novel formulation that treats clean and adversarial samples from both source and target domains as four distinct distributions to be aligned. The authors introduce a progressive alignment strategy that first explicitly aligns clean target features with the multi-source domains and then implicitly aligns adversarial target features by enforcing classifier consistency on refined pseudo-labels. Some experiments on four benchmarks validate that the performance of the proposed method outperforms baselines.

### Strengths
Pros:
* The authors conducted a bunch of experiments to validate the effectiveness of their method across four benchmarks.

### Weaknesses
Cons:
* Please carefully use \citet and \citep. You should only use \citet when the reference is grammatically part of the sentence, usually as the subject.
* In Lines 171-172, why does minimizing the loss in Eq. (2) produce consistent predictions across two classifiers? And since (2) is upper bounded (instead of lower bounded) by the average of individual losses, it doesn’t mean each individual loss is also minimized due to (2). 
* What is the motivation for introducing two classifiers, H1 and H2? The authors use the discrepancy between these two classifiers as the divergence between source and target domains. But this doesn’t make sense to me. And why don’t you just use some widely used divergence metric? Because of this, I don’t think the minimax process in Section 2.2 could align the source and target domain.
* Very limited innovation compared to prior work and methods. This method wants to jointly align the source and target, clean and adversarial, while there are many prior studies [1,2,3] proposing similar frameworks and ideas. 

[1]: Exploring Adversarially Robust Training for Unsupervised Domain Adaptation. https://arxiv.org/abs/2202.09300

[2]: Adversarially robust unsupervised domain adaptation. https://www.sciencedirect.com/science/article/pii/S000437022500102X

[3]: Adversarial Feature Alignment: Balancing Robustness and Accuracy in Deep Learning via Adversarial Training. https://arxiv.org/abs/2402.12187

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
