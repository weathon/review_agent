# $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging.
While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks.
In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint.
However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks.
In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity.
Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$-zero finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a $\ell_0$-norm attack, called sigma-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization. The attack can find minimum $\ell_0$-norm adversarial examples. The experiments show that sigma-zero exhibits good performance in different settings.

### Strengths
1. The paper is easy-to-follow

2. The method is simple but effective

3. The experiments are relatively comprehensive

### Weaknesses
1. Despite the effectiveness of sigma-zero on different models, the evaluation on $\ell_0$-robust models is missing. Please consider including the results of sAT / sTRADES [1], e.g., those trained on CIFAR-10, $k_{train}=6\times20$ in pixel space.

2. Due to the early-stopping mechanism widely adopted in various attacks, when the batch size is large, white-box attacks could run even faster than black-box attacks with the same iteration budget. Thus, I am still curious about the results of sigma-zero, sPGD and Sparse-RS with the same budget, e.g., N=10000. You can report the same metrics on a subset of representative models as in Table 13.

I will adjust my score if the authors can address my concerns.

[1]  Xuyang Zhong, Yixiao Huang, and Chen Liu. Towards efficient training and evaluation of robust models against l0 bounded adversarial perturbations.

### Questions
1. Typo on the title of Table 2: "Columns $q_{100}$ and $s_{100}$ ... at $k=100$", but $k$ is 24 in the below.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a $\ell_0$-norm adversarial attack that efficiently breaks various models trained on MNIST, CIFAR10, and ImageNet with 100% success rate. The proposed method is significantly faster than prior attacks at a high success rate. The proposed attack is composed of three main components:
1) A differentiable relaxation of the $\ell_0$ loss (Eq. 7).
2) An adaptive projection to project near-zero components to zero. The threshold for projection, $\tau$, is adapted dynamically such it is increased when the sample is adversarial and decreased otherwise.
3) Cosine annealing of the learning rate.

### Strengths
- The proposed method is highly effective against a diverse set of models on multiple datasets and all attack budgets (Table 1-3, Figure 2). Particularly, on ImageNet $\sigma$-zero improves the attack success rate at 100 by nearly 10% across all models (Table 3).
- Compared with BBadv that often achieves similar ASRk values, the proposed method is 10-200x faster.

### Weaknesses
- I recommend moving the ablations on the components of the proposed method (Table 5) to the main body of the paper. For completeness, consider adding the missing row with normalization/adaptive $\tau$ but without projection and dropping the column approximation as all rows rely on it.

### Questions
- Eq. 4 defines the regularized objective with a fixed regularization coefficient of $1/d$. How important is this normalization? Do you have any theories or ablations?
- BBadv is said to be slow, particularly it relies on adversarial initialization. Would $\sigma$-zero benefit in any way if it is also initialized the same way if we ignore the extra runtime?

Minor:
- Line 321: The sentence starts with “Despite BBadv does not suffer …” needs grammar correction, e.g., substituting “Despite” with “Although”.
- Consider annotating Figure 2 with subfigure captions.
- How is table 4 ordered? It seems to be random.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces σ-zero, a novel attack aimed at creating sparse adversarial examples under the ℓ0 norm constraint. By using a differentiable ℓ0 norm approximation and adaptive projection operator, the attack achieves high success rates across multiple benchmarks (MNIST, CIFAR-10, and ImageNet) and models. It outperforms existing sparse attacks in both efficiency and effectiveness.

### Strengths
1. **Innovative Approach**: σ-zero combines a unique differentiable approximation of the ℓ0 norm with adaptive projections, addressing challenges in sparse adversarial attack optimization with a gradient-based method.
2. **Robust Evaluation**: The experimental setup is thorough, involving extensive model types and datasets, and a range of comparison attacks. Evaluation metrics like success rate, median perturbation size, runtime, and query count provide a clear assessment of σ-zero's performance relative to existing methods.
3. **Efficiency and Robustness**: σ-zero demonstrates effective performance without the need for adversarial initialization or hyperparameter tuning, potentially broadening its applicability.

### Weaknesses
1. **Limited Innovation**  
   The approach presented shows minimal originality and bears a strong resemblance to existing methods, with only slight modifications. These adjustments lack ingenuity, as they do not introduce new insights or significant advancements over prior work. The lack of a novel perspective limits the contribution of this study to the field.

2. **Lack of Theoretical Understanding and Analysis**  
   A major shortcoming of this work is the absence of a robust theoretical foundation. The paper lacks in-depth analysis to demonstrate the underlying principles or broader significance of the method. Without a theoretical understanding, the impact of the work remains superficial, leaving readers without a clear sense of the method’s potential applicability or contribution to the advancement of adversarial robustness.

3. **Limited Application Scope**  
   All experiments focus exclusively on white-box attacks, which limits the generalizability of the findings. While white-box evaluations are useful for testing model robustness, the absence of analysis in other contexts raises concerns about the method’s broader applicability. Expanding to include black-box or transfer-based attacks, for instance, could improve the study’s relevance and highlight potential real-world applications.

4. **Simplistic Model Choices in Experiments**  
   Although the experiments involve several models, these are relatively simple in structure. The paper does not consider more complex architectures, such as ResNet-101 or models of similar or higher complexity, which are frequently used in evaluating ImageNet-related tasks. Given that larger and more sophisticated models are common in real-world applications, the study’s reliance on simpler architectures limits the robustness and scalability of its findings.


**Miner Comments:** 
1. There is an indexing error in Table 2 that needs to be addressed.
2. Footnote 1 states, 'when the source point $x$ is already misclassified by $f $, the solution is simply $\delta^* = 0$. This is quite puzzling, as attacks are typically conducted on samples that can be correctly classified. If the attack is performed on samples that are already misclassified, I question the persuasiveness of the higher ASR observed in the experimental results.

### Questions
**Q1**: In the introduction, you mention that “evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional ℓ2- and ℓ∞-norm attacks.” Could you specify what types of model weaknesses can be exposed by ℓ0 attacks that may remain undetected under conventional ℓ2 and ℓ∞ attacks? Examples or specific scenarios would be helpful for understanding.

**Q2:** Have you considered any step size decay methods other than cosine annealing in your algorithm? Exploring more direct approaches might further reduce computational complexity.


**Q3:** Does your method enhance the improvement or understanding of attacks using other norms? Based on your assertion that 'ℓ0-norm attacks, which perturb only a minimal fraction of input values, can identify the most sensitive features affecting the model's decision-making,' could your method be integrated into attacks with other norms to avoid targeting these sensitive features, thereby reducing the visual damage of adversarial examples?


**Q4:** I appreciate your clarification that 'the goal of ℓ0-norm attacks is not to be indistinguishable to the human eye—a common misconception regarding adversarial examples—but rather to demonstrate whether and to what extent models can be deceived by altering only a few input values.' However, in the visualizations of adversarial examples provided in your appendix, the perturbations lack semantic information comprehensible to humans. How can we understand and apply the adversarial perturbations generated by your method in this context?

### Soundness
2

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
3

### Summary
The paper proposes a novel $\ell_0$-norm attack by leveraging a differentiable approximation of the $\ell_0$-norm constraint. The approach can be applied to finding both minimum $\ell_0$-norm and fixed budget adversarial examples. The authors conduct extensive evaluations on diverse datasets (e.g. CIFAR-10, ImageNet) and models. The results show that the proposed attack is effective and efficient in reducing the number of queries and memory usage.

### Strengths
- The problem is well-motivated as $\ell_0$-norm attacks are less studied due to the non-convexity nature.
- The paper is easy to follow and well-organized
- The evaluations are comprehensive and indeed show that the proposed algorithm is more effective and efficient.

### Weaknesses
- I appreciate the authors' efforts in evaluating the method. However, I was a bit confused by the inconsistency in some parts of the evaluations. For example, the memory usage is shown for minimum-norm attacks while it's missing in the bounded-norm attack results; The mean runtime for bounded attacks is shown in Tables 2 and 3 while it's missing in Tables 10 - 12; In Tables 10 - 12, the budget level that the number of queries corresponds to is also missing.

### Questions
- To enable a clearer comparison between different attacks, could the authors provide ASR under varying iterations and perturbation budgets for fixed-budget attacks, similar to Figure 2? Showing results on one model would suffice and help better understand the performance under different conditions.
- The transferability of the proposed attack across different model and architectures is unclear. Would it be possible to evaluate the method on a different model/architecture than the one used for generating adversarial examples? Such an evaluation would strengthen the applicability of the attack across varied settings.
- In the minimum-norm attacks, ASR of the three datasets are reported on the same budgets: $k = 24, 50,\infty$. However, in the fixed-budget comparisons, the results of ImageNet are reported at budgets $k = 100, 150$ while the other two datasets are evaluated at $24, 50, 100$. Could the authors clarify the reasoning behind this difference?
- Minor writing issues
  - The bounded-norm attacks are not cited at the first appearance in Section 3.1
  - In Table 2’s caption, $q_{100}$ and $s_{100}$ are mentioned, but it seems these should be $q_{24}$ and $s_{24}$, based on the table’s content.

### Soundness
3

### Presentation
3

### Contribution
3
