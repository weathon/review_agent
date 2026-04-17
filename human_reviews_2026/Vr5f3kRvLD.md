# IDER: IDempotent Experience Replay for Reliable Continual Learning

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 10

## Abstract
Catastrophic forgetting, the tendency of neural networks to forget previously learned knowledge when learning new tasks, has been a major challenge in continual learning (CL). To tackle this challenge, CL methods have been proposed and shown to reduce forgetting. Furthermore, CL models deployed in mission-critical settings can benefit from uncertainty awareness by calibrating their predictions to reliably assess their confidences. However, existing uncertainty-aware continual learning methods suffer from high computational overhead and incompatibility
with mainstream replay methods. To address this, we propose idempotent experience replay (IDER), a novel approach based on the idempotent property where repeated function applications yield the same output. Specifically, we first adapt the training loss to make model idempotent on current data streams. In addition, we introduce an idempotence distillation loss. We feed the output of the current model back into the old checkpoint and then minimize the distance between this reprocessed output and the original output of the current model. This yields a simple and effective new baseline for building reliable continual learners, which can be seamlessly integrated with other CL approaches. Extensive experiments on different CL benchmarks demonstrate that IDER consistently improves prediction reliability while simultaneously boosting accuracy and reducing forgetting. Our results suggest the potential of idempotence as a promising principle for deploying efficient and trustworthy continual learning systems in real-world applications. Our code is available at https://github.com/YutingLi0606/Idempotent-Continual-Learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Idempotent Experience Replay (IDER), a method for class-incremental learning (CIL) that exploits the mathematical property of idempotence to improve accuracy, forgetting, and uncertainty calibration. The method consists of two components: 1) a standard idempotent module that trains the current model to be idempotent on the current task; and 2) an idempotent distillation module that enforces idempotence between the previous and current model checkpoints. Experiments on standard datasets demonstrate improvements in both accuracy and calibration.


**Recommendation**
I lean toward rejecting the paper in its current form. While the core idea of applying idempotence to continual learning is novel and the empirical results are encouraging, the paper suffers from significant theoretical and experimental gaps which prevent a rigorous and clear assessment of the paper.

### Strengths
1. **Novel application of idempotence to continual learning:** while idempotence has been explored in deep learning for other tasks, this appears largely under-explored in continual learning settings.
2. **Plug-and-play design with rehearsal-based approaches:** the method can be integrated with existing rehearsal-based approaches (e.g., ER) with minimal overhead, making it practically appealing in those cases.
3. **Experimental evaluation on general-CIL:** the method has also been tested on more realistic conditions (i.e., general class-incremental learning), making it interesting from a practical point of view.

### Weaknesses
1. **Limited theoretical justification:** while the paper takes inspiration from prior work on idempotence, the theoretical connection between idempotence and mitigation of both catastrophic forgetting and uncertainty miscalibration is not clear. Although being demonstrated empirically, it is unclear why enforcing idempotence specifically helps with continual learning problems in the first place. The intuition lacks formal analysis. Can you provide explanations on why idempotence should help with catastrophic forgetting and uncertainty calibration?
2. **Equation 6:** the idempotent distillation loss uses $f_{t-1}(x, f_t(x, 0))$ rather than $f_t(x, f_t(x, 0))$ to avoid pulling predictions toward incorrect output, but this breaks the true idempotence property since $f_{t-1}$ and $f_t$ are two different functions. What happens if you use $f_t(x, f_t(x, 0))$ instead of $f_{t-1}(x, f_t(x, 0))$ in Equation 6?
3. **Missing evaluation with calibration methods for CL:** one of the main focus of the paper is reliability (i.e., reducing the calibration error). However, the paper does not consider recent work on uncertainty calibration for class-incremental learning [1,2].
4. **Limited comparison with other CL methods:** apart from rehearsal-based methods, it would be interesting to include comparison with, e.g., regularisation-based or parameter-isolation approaches to demonstrate broader applicability. 
5. **Incomplete experimental analysis:**
    1. Hyperparameter sensitivity: no analysis of sensitivity to $\alpha$ and $\beta$ introduced in Equation 8.
    2. Probability $P$: the probability is not reported nor ablated.
    3. Backbone architecture: only ResNet18 is tested. What will happen when using a different architecture? Again, this would demonstrate broader applicability.
    4. Limited experiments on GCIL (Table 2): it would be interesting to see the results on the other considered datasets.
    5. Limited calibration results (Table 3): as anticipated above, the paper is missing recent work on calibration in continual learning. Furthermore, why is Tiny-ImageNet not included? Finally, since the results from NPCL were copied from the original work, do you assure that the experimental setting is exactly the same? Otherwise, the comparison is not meaningful. 

**Minor issues**
1. No analysis on larger task sequences (e.g., 20 tasks on Tiny-ImageNet).
2. Figures 3, 4, and 5 are difficult to parse.
3. Writing style can be overall improved.

[1] Li, Lanpei, et al. "Calibration of continual learning models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Hwang, Seong-Hyeon, Minsu Kim, and Steven Euijong Whang. "T-CIL: Temperature Scaling using Adversarial Perturbation for Calibration in Class-Incremental Learning." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces IDER (Idempotent Experience Replay), a novel approach for continual learning that addresses the issue of catastrophic forgetting in replay-based methods. The key idea is to make the replay process idempotent, meaning that repeatedly revisiting the same experiences does not alter the model representation undesirably. The method achieves this by enforcing idempotent updates through the standard idempotent module and the idempotent distillation module. Experiments on several continual learning benchmarks demonstrate consistent performance gains over standard replay-based baselines.

### Strengths
The idea of using idempotent property to mitigate issues of poor calibration and recency bias is straightforward and intuitive.

The paper is well-written and easy to follow.

The proposed method is shown to be effective.

### Weaknesses
As stated in Lines 71–74, the paper claims a strong correlation between the idempotence distance and prediction error. However, it remains unclear whether this relationship has been formally analysed. Could the authors provide empirical evidence or theoretical justification for this claim?

To enable idempotence with respect to the second input, the proposed method divides the backbone into two parts. What principles guide this division? Does choosing different partition points (e.g., splitting at shallower or deeper layers) affect model performance or stability? It would be helpful if the authors could provide theoretical insight or experimental analysis on how the division point influences the effectiveness of idempotent learning.

Lines 207–209 describe the probability P that determines whether the second input is set to the ground-truth label or the empty signal. Is there any hyperparameter analysis that explores the sensitivity of the method to P? How should this value be selected in practice?

The overall loss function combines three components: the Standard Idempotent Module, Idempotent Distillation Module, and Experience Replay. Are there ablation experiments isolating the contribution of each component? Furthermore, the paper states that the proposed method primarily addresses poor calibration and recency bias. However, these issues are often mitigated in modern replay-based methods. Why does this approach not integrate or compare directly with more recent baselines such as L2P [1], HIDE-prompt [2], or VQ-prompt [3]? Has the proposed method been evaluated under online continual learning settings to validate its general applicability?

Table 1 shows improvements when integrating IDER with ER, BFP, and CLS-ER. However, could the proposed method also be compatible with other state-of-the-art baselines such as SCoMMER or SARL? It would be valuable to verify whether the idempotent mechanism consistently enhances these methods as well. 

Regarding the experiments, I noted that the evaluation is primarily conducted on small-scale datasets like the CIFAR series and Tiny-ImageNet. I encourage the authors to validate the method's effectiveness on larger-scale and more diverse datasets. Furthermore, a concrete example illustrating how the proposed method achieves error correction in practice would greatly enhance the paper’s clarity and impact.

[1] Wang et al., Learning to prompt for continual learning. CVPR 2022

[2] Wang et al., Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality. NIPS 2023

[3] Li et al., Vector Quantization Prompting for Continual Learning. NIPS 2024

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript introduces the idempotent property to mitigate catastrophic forgetting, a fundamental challenge in incremental learning. The authors design two complementary loss functions: one enforces idempotence on current data, while the other distills this property from a previous model checkpoint using a memory buffer of historical samples. Extensive experiments are conducted to validate the effectiveness of the proposed approach.

### Strengths
• The manuscript introduce idempotence, a mathematical property, to tackle catastrophic forgetting and poor model calibration in continual learning.
• The proposed IDER is a lightweight framework and functions as a plug-and-play module for performance gains

### Weaknesses
• The paper primarily relies on intuition and empirical success to introduce idempotence. It lacks a rigorous theoretical analysis or hypothesis for why enforcing output stability should directly mitigate catastrophic forgetting at a fundamental level.
• The empirical validation is comprehensive on CIFAR-10, CIFAR-100, and Tiny-ImageNet. However, to firmly establish the method's practicality and generalizability, evaluation on a large-scale dataset, e.g., ImageNet-1K.
• The hyperparameter sensitivity analysis for α and β is relatively brief.
• The experimental comparisons are heavily focused on replay-based methods, which is natural as IDER is a plug-in for this paradigm. However, comparing against strong representatives from other CL fmethods, such as regularization-based methods or memory-free approaches, would more comprehensively position IDER's contribution within the entire field and highlight its unique value.

### Questions
• A simple theoretical proposition or a more in-depth discussion connecting the idempotence loss to established continual learning theory would significantly strengthen the foundation.
• Test IDER in settings like online continual learning or with tasks containing out-of-distribution samples to better probe its limits and robustness.
• Perform a more systematic hyperparameter sensitivity analysis, showing how performance varies with different values of α and β across key datasets.
• Add comparisons with regularization-based and memory-free methods in the main experiments.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper aims to improve accuracy and calibration in continual learning by extending idempotence enforcing loss function into the continual learning setting with experience replay. A function is called idempotent if consecutive applications of it gives the same result as one application. 
In the single task setting, the idempotence enforcing loss function encourages the model $f(x, y)$ to be idempotent with respect to its second input $y$. This work introduces a new loss function to adapt adaptation of this idea in the continual learning setting. This new loss function can be combined with some existing techniques to improve them
The improved calibration and accuracy claims are supported by experimental results.

### Strengths
The paper is well written and organized. Adaptation of the idempotence loss function to the continual learning setting is creative and nuanced. The experimental results are promising.

### Weaknesses
I think giving more intuition about the loss function would be helpful for the reader. For example, in lines 250-252, it is stated that minimizing equation 5 biases $f_t$ towards the wrong label (even though with probability $1-P$ that objective would be minimized in eq 5), but why would $f_{t-1}$ not have the same problems? Expanding that explanation would be great.

### Questions
Please see the weakness section. 
In general providing more intuition, especially on why this loss function helps with calibration would be great.

### Soundness
3

### Presentation
3

### Contribution
4
