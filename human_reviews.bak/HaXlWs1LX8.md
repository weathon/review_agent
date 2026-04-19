# Conflict-Aware Adversarial Training

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Adversarial training is the most effective method to obtain adversarial robustness for deep neural networks by directly involving adversarial samples in the training procedure. To obtain an accurate and robust model, the weighted-average method is applied to optimize standard loss and adversarial loss simultaneously. In this paper, we argue that the weighted-average method does not provide the best tradeoff for the standard performance and adversarial robustness. We argue that the failure of the weighted-average method is due to the conflict between the gradients derived from standard and adversarial loss, and further demonstrate such a conflict increases with attack budget theoretically and practically. To alleviate this problem, we propose a new trade-off paradigm for adversarial training with a conflict-aware factor for the convex combination of standard and adversarial loss, named \textbf{Conflict-Aware Adversarial Training~(CA-AT)}. Comprehensive experimental results show that CA-AT consistently offers a superior trade-off between standard performance and adversarial robustness under the settings of adversarial training from scratch and parameter-efficient finetuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work aims to address the limitations of traditional weighted-average adversarial training, the authors propose a novel framework called Conflict-Aware Adversarial Training (CA-AT). This work identifies that existing weighted-average AT methods face a gradient conflict, particularly with higher attack budgets. The proposed CA-AT mitigates this conflict by introducing a conflict-aware trade-off factor that adjusts the influence of standard gradient and the adversarial gradient based on their alignment.

### Strengths
· This manuscript offers a new perspective on improving adversarial robustness by addressing gradient conflict.

· This work provides a theoretical analysis, supported by empirical experiments, to demonstrate the gradient conflict and present the motivation for CA-AT.

· Extensive experiments across different datasets are conducted to validate the effectiveness of the proposed CA-AT.

### Weaknesses
· Some detailed parameters are not clear, for example, the $\gamma$ and the batch size. According to the study in [1], the batch size is related to the learning rate and they will influence the natural accuracy and robust accuracy.

[1] Bag of Tricks for Adversarial Training, ICLR, 2021.

· The adversarial attacks used to evaluate the adversarial training methods mainly belong to gradient-based methods, more optimization-based attacks can be used to evaluate the effectiveness of the proposed method.

· The main comparison method in this manuscript is vanilla AT. There are many improved adversarial training methods that can be considered for comparison.

### Questions
· How are the hyperparameter $\gamma$ and the batch size set? What effect do they have on the performance of the proposed method？

· How well does the proposed method perform against more optimization-based attacks (e.g., C&W [2], DDN[3])?

[2] Towards Evaluating the Robustness of Neural Networks, IEEE SP, 2017.

[3] Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses, CVPR, 2019.

· How effective is the proposed method compared to the improved AT (e.g., TRADES, MART)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a **Conflict-Aware Adversarial Training (CA-AT)** gradient operation method that effectively alleviates the gradient conflict between the clean loss and adversarial loss. The proposed method is well-supported by theoretical foundations and shows strong experimental results. Findings indicate that, with the gradient conflict mitigation approach, CA-AT improves the balance between standard accuracy (SA) and adversarial accuracy (AA) — the SA-AA frontier — compared to vanilla adversarial training.

### Strengths
1. The paper is well-organized, featuring a logical structure with clear illustrations, algorithms, experimental results, images, and tables, making it easy to understand and follow.

2. The notation and terminology throughout the paper are highly consistent and precise, enhancing clarity and readability.

3. The paper introduces the metric $\mu$ based on the weighted average method to measure the conflict and convergence between gradients, providing theoretical upper bounds. The logical flow is clear, and the conclusions align well with practical observations.

4. In addition to experiments on real datasets, the authors conducted synthetic experiments, which further reinforce the credibility of the proposed method.

5. The experimental section is detailed and well-explained, making it easy for readers to comprehend the setup and results.

6. Apart from the original cross-entropy loss, the authors extensively study the performance of the proposed methods under various other loss functions and test them against different types of attacks. The experimental design is comprehensive and thorough.

### Weaknesses
1. Please provide detailed accuracy results for experiments on ViT and Swin-T in table format.

2. Does CA-AT also perform well on larger datasets such as ImageNet?

3. Additional baselines that achieve similar levels of balance should be included, such as other advanced weighted methods or gradient operations. Using only Vanilla AT as a baseline is insufficient.

### Questions
1. The exact form of $\lambda^*$ is interesting. Could you elaborate on why this form was chosen? What was the inspiration, and what benefits does it offer?

2. Please include additional baselines in the experimental section to allow for a more extensive and objective comparison.

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
The paper proposes to use a dynamic conflict-aware factor to control the trade-off between the standard and adversarial loss. Only the standard gradient is used if the adversarial and standard gradients are close. Otherwise, the adversarial gradient is emphasized. Based on Figure 3 in the paper, I suppose both gradients tend to be similar in the later training phase, and the proposed methods will mainly optimize the standard loss. Extensive experiments are conducted and the results are promising.

### Strengths
- The paper is easy to follow.
- Extensive experiments are conducted to demonstrate the effectiveness of the proposed method empirically.

### Weaknesses
- The proposed factor seems a bit bizarre to me. Take Figure 1 as an example. Any $g_a$ that ends in the dotted line with $arccos(g_a, g_b) > \gamma$ will result in the same $g^*$, which doesn't make sense to me. Imagine $g_a$ equals $g_o$ in Figure 1, and one will still get the same $g^*$.
- One should conduct an ablation study by using traditional $\lambda$-weighted mean of $g_a$ and $g_c$ when $\phi \leq \gamma$ and only $g_c$ when $\phi > \gamma$, as I suspect that this might be the reason for the performance boosts.
- Some other suggestions:
- - Add the legends for Figure 6 (b) and (c).
- - Conduct experiments on ImageNet.
- - Run multiple-trails and report the error bar.

### Questions
Please refer to those in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper explores the issue of gradient conflict in adversarial training, highlighting how traditional methods struggle to balance standard performance and adversarial robustness. The authors introduce Conflict-Aware Adversarial Training (CA-AT), which employs a novel trade-off paradigm based on gradient alignment to address this conflict. Through extensive experiments, CA-AT shows consistent improvements in the trade-off between standard and adversarial accuracy across various datasets and model architectures.

### Strengths
1. The proposed Conflict-Aware Adversarial Training (CA-AT) effectively addresses gradient conflict, achieving a better balance between standard performance and adversarial robustness.

2. Comprehensive experimental results across various datasets and model architectures validate the effectiveness of CA-AT in improving the trade-off between standard and adversarial accuracy.

3. The method demonstrates strong performance in both training from scratch and parameter-efficient fine-tuning, showcasing its versatility.

### Weaknesses
1. The proposed CA-AT aims to manipulate the gradient of the clean examples and the adversarial example, the idea is not novel, either for input gradient alignment [1] or model gradient alignment [2].

2.  I think starting from PGD adversarial training, they just use the adversarial example for training, rather than the combination that CA-AT wants to tackle. 

3. Although CA-AT shows improved performance over Vanilla AT in several experiments, it lacks comparisons with other advanced adversarial training methods.

4. The price of the double propagation should also be considered.


[1] Understanding and Improving Fast Adversarial Training

[2] Quantifying the preferential direction of the model gradient in adversarial training with projected gradient descent

### Questions
1. In line 268, a reference is missing.
2. The authors argue that the weighted-average method does not provide the best tradeoff for the standard performance and adversarial robustness, but in my mind, CA-AT also doesn't, like Vanilla AT can also beat CA-AT  in your FAB experiments.
3. What leads to the large fluctuation during epoch 60-90 in the middle image in  Fig.3(a)?
4. Is the initial learning rate 0.4 the normal setting? As I remember the decayed learning rate schedule is more common.

### Soundness
3

### Presentation
3

### Contribution
3
