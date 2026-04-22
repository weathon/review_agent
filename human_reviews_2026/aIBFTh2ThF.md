# Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Epsilon-Scheduling

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Fine-tuning pretrained models is a standard and effective workflow in modern machine learning. However, robust fine-tuning (RFT), which aims to simultaneously achieve adaptation to a downstream task and robustness to adversarial examples, remains challenging. Despite the abundance of non-robust pretrained models in open-source repositories, their potential for RFT is less understood. We address this knowledge gap by systematically examining RFT from such non-robust models. Our experiments reveal that fine-tuning non-robust models with a robust objective, even under small perturbations, can lead to poor performance, a phenomenon that we dub _suboptimal transfer_. In challenging scenarios (eg, difficult tasks, high perturbation), the resulting performance can be so low that it may be considered a transfer failure. We find that fine-tuning using a robust objective impedes task adaptation at the beginning of training and eventually prevents optimal transfer. However, we propose a novel heuristic, _Epsilon-Scheduling_, a schedule over perturbation strength used during training that promotes optimal transfer. Additionally, we introduce _expected robustness_, a metric that captures performance across a range of perturbations, providing a more comprehensive evaluation of the accuracy-robustness trade-off of diverse models at test-time. Extensive experiments on wide range of configurations (six pretrained models and five datasets) show that _Epsilon-Scheduling_ successfully prevents _suboptimal transfer_ and consistently improves expected robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores finetuning a non-robust pretrained model with adversarial examples. The paper shows that direct adversarial finetuning, even with a small perturbation, leads to poor performance. To this end, the paper proposes epsilon-scheduling, which schedules perturbation strength as finetuning progresses. Experiments show that training curve under scheduling shows quicker clean accuracy convergence, achieves better tradeoff between accuracy and robustness, and leads to more optimal minima.

### Strengths
Robust finetuning of non-robust pretrained model is an important research question. The proposed method is simple, solid, and interesting. The method is well motivated from the observation of "delay time", a novel observation for robust finetuning. Experimental results show effective improvements for most cases. The visualization of training curve and converged minima are helpful for the understanding of the process.

### Weaknesses
1. The evaluation is limited to PGD adversarial training, which is known to be prone to overfitting and diminished clean accuracy in the finetuning setting. More advanced finetuning method like AutoLoRA [1] or CLAT [2] should be attempted to see if the proposed scheduling is also effective.
2. The current performance is highly relevant to baseline, task, and choice of hyperparameters. This makes the heuristic exploration of T1 T2 scheduling hard to be optimal for a new task and backbone. Some criteria to automatically decide when to start and end the epsilon scheduling based on things like gradient, loss, or accuracy would be helpful for better generalizability.
3. The evaluation is limited to finetuning on small-scale tasks, which may be an overkill with the large pretrained model. Having some exploration on ImageNet variants would better show the effectiveness of the proposed method under more complicated settings.

[1] Xu, X., Zhang, J., & Kankanhalli, M. (2023). Autolora: A parameter-free automated robust fine-tuning framework. arXiv preprint arXiv:2310.01818.

[2] Gopal, B., Yang, H., Zhang, J., Horton, M., & Chen, Y. Boosting Adversarial Robustness with CLAT: Criticality Leveraged Adversarial Training. In Forty-second International Conference on Machine Learning.

### Questions
1. Is there an automated way to decide when to start and end the scheduling based on the trianing progress?
2. How would the scheduling help with more recent robust finetuning methods and on larger tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies robust fine-tuning (RFT) of non-robust pretrained models and observes suboptimal transfer, where robust objectives hinder task adaptation and degrade performance. The authors show that RFT initially delays learning, which results in the poor performance in clean performance. To address this, they propose Epsilon-Scheduling, a simple yet effective heuristic that gradually adjusts perturbation strength during training, improving both adaptation and robustness. Extensive experiments demonstrate consistent gains in expected robustness across multiple models and datasets.

### Strengths
The paper is well written and logically presented
The observation that RFT delays task adaptation is interesting, as it explains the low clean performance observed during RFT. 
The proposed mitigation method is straightforward yet effective.

### Weaknesses
I noticed that in Figure 4, scheduler shows lower robustness compared to fixed, while in Table 1 it achieves higher adversarial robustness in most cases. Could the authors clarify this discrepancy or provide insights into why this happens?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how to improve adversarial robustness via robust fine-tuning when the pre-trained model is vanilla/non-robust. Firstly, the authors identify the phenomenon, “suboptimal transfer”, when standard fine-tuning with adversarial training leads to low clean accuracy on downstream tasks. Since this can be attributed to the delayed task adaption by the authors when robust objectives are applied from the start of the fine-tuning, they propose Epsilon-Scheduling, a two-hinge linear schedule that starts with standard fine-tuning before gradually increasing perturbation strength to the target epsilon. Finally, the authors introduce expected robustness, a metric that evaluates model accuracy across the full perturbation range. Epsilon-Scheduling consistently shows improved expected robustness compared to fixed-perturbation RFT with the experimental results.

### Strengths
- **Studying an important and practical problem.** The focus of the paper on non-robust pretrained models is highly relevant. Nearly all publicly available pretrained foundation models are not adversarially trained. Improving robust fine-tuning techniques with those models would help model security in practice. 
- **Interesting insights on task adaptation delay and proposing simple yet effective solution.** The observation that robust fine-tuning delays task adaption (figure 3) is well-introduced. Especially, the strong correlation (>90%) between delay time and suboptimal transfer severity provides valuable understanding of the underlying mechanisms. This makes the idea of the proposed epsilon-scheduling method intuitive. It is an easy approach which generalizes prior linear warmup strategies. 
- **New evaluation metric, strong reproducibility.** Expected robustness helps measure the accuracy-robustness tradeoff than point estimates at a certain epsilon perturbation alone. It is well-grounded in a realistic threat model. Also, an anonymized code repository is provided with detailed hyperparameters and configurations to support reproducibility for future research in the field.

### Weaknesses
We thank the authors for submitting the paper to ICLR 2026! There are a few weaknesses listed below which I believe can make the paper better.
- **Limited theoretical understanding.** The paper is purely based on empirical results. While those observations are compelling, the paper lacks theoretical analysis of why robust objectives delay task adaptation or why Epsilon-Scheduling finds better local minima. Some theoretical perspective can make the claim (such as figure 7) much more grounded.
- **Lack of comparisons with other robust fine-tuning strategies.** There’s no comparison with other methods mentioned in the related works section such as TWINS, AutoLoRA, or RoLi. It would be interesting to see how it competes with other RFT methods and if they could be combined to make further improvements given the proposed scheduling method could be compatible with other architectural changes. 
- **Concerns on the expected robustness metric.** Since it is aiming to measure the trade-off between robustness and accuracy, it would be good to consider cases with very high clean accuracy, when the metric could be overemphasizing clean accuracy. Also, some discussion on how to make decisions when / whether to use this metric vs. traditional worse-case robustness would be informative for practitioners.

### Questions
- Could you provide some theoretical intuition/hypothesis on why robust objectives delay adaptation would occur? For example, is there a connection to the loss landscape geometry? Are there other factors that could be involved – size of the dataset, optimization objectives, etc? 
- Can you provide direct comparisons with TWINS, AutoLoRA, or RoLi with the same experimental setup? Even if they assume robust backbones, understanding relative performance would be valuable. 
- Can you add some error bars/confidence intervals with multiple runs? Since sometimes the values are very close to each other, this will help mitigate noises.

### Soundness
2

### Presentation
3

### Contribution
2
