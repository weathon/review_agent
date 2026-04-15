# Quantifying and Enhancing Multi-modal Robustness with Modality Preference

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Multi-modal models have shown a promising capability to effectively integrate information from various sources, yet meanwhile, they are found vulnerable to pervasive perturbations, such as uni-modal attacks and missing conditions. To counter these perturbations, robust multi-modal representations are highly expected, which are positioned well away from the discriminative multi-modal decision boundary. In this paper, different from conventional empirical studies, we focus on a commonly used joint multi-modal framework and theoretically discover that larger uni-modal representation margins and more reliable integration for modalities are essential components for achieving higher robustness. This discovery can further explain the limitation of multi-modal robustness and the phenomenon that multi-modal models are often vulnerable to attacks on the specific modality. Moreover, our analysis reveals how the widespread issue, that the model has different preferences for modalities, limits the multi-modal robustness by influencing the essential components and could lead to attacks on the specific modality highly effective. Inspired by our theoretical finding, we introduce a training procedure called Certifiable Robust Multi-modal Training (CRMT), which can alleviate this influence from modality preference and explicitly regulate essential components to significantly improve robustness in a certifiable manner. Our method demonstrates substantial improvements in performance and robustness compared with existing methods. Furthermore, our training procedure can be easily extended to enhance other robust training strategies, highlighting its credibility and flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This manuscript concerns the robust multi-modal representation learning, which are positioned well away from the discriminative multi-modal decision boundary. To address this issue, they theoretically discover that larger uni-modal representation margins and more reliable integration for modalities are essential components for achieving higher robustness. Inspired by our theoretical finding, we introduce a training procedure called Certifiable Robust Multi-modal Training (CRMT), which can alleviate this influence from modality preference and explicitly regulate essential components to significantly improve robustness in a certifiable manner. Experiments validate the effectiveness.

### Strengths
1.	The multi-modal robustness learning is meaningful and challenging. The paper is well-written and the proposed method is easy to understand.
2.	The authors theoretically discover that larger uni-modal representation margins and more reliable integration for modalities are essential components for achieving higher robustness.
3.	Experiments on various datasets validate the proposed method.

### Weaknesses
The manuscript claims that they focus on the commonly used joint multi-modal framework, more multi-modal fusion method, and different multi-modal backbones should be compared. For example, the early fusion, and hybrid fusion strategy. On the other hand, different modalities can employ various backbones, the reviewer is curious about the influence of different backbones, and more ablation studies are expected.

In related work and comparison methods, more state-of-the-art multi-modal robustness approaches should be introduced and compared.

How can this setup be extended to three modalities? More explanations and experiments are needed.

### Questions
refer to the weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work employs an orthogonal-based framework that formulates an alternative bound, eliminating the interrelation and explicitly presenting integration.  Building on the theoretical analysis, they introduce a two-step procedure called Certifiable Robust Multi-modal Training (CRMT) to progressively enhance robustness.

### Strengths
(1) Following a more comprehensive analysis, the researchers furnish compelling evidence that demonstrates the constraining effect of multi-modal preference on the robustness of multi-modal systems, which contributes to the vulnerability of multi-modal models to specific modalities.

(2) Building upon their theoretical insights, they present a two-step training protocol designed to alleviate the limitations stemming from modality preference. The suggested approach significantly enhances both the performance and robustness of multi-modal models across different real-world multi-modal datasets.

### Weaknesses
(1) Since FGM and PGD are the two white-box attacks chosen in the adversarial robustness experiments, why not consider the stronger white-box Auto Attack?  It is suggested to add the experiment results about Auto Attack in Section4.

(2) "Robustness against multi-modal attacks" mentioned In Section 4.2, since multi-modal attacks are considered, the experimental results in Table 1 only consider single-mode attacks (#a,#v). Is the method proposed in this paper effective when co-attacks (both modality attacks) are existing?  In [1,2], more effective multi-modal attack methods are proposed than uni-modal(such as #a and #v) attack. Can the proposed method effectively resist these multi-modal attack methods? It is suggested that the relevant experiments should be added to Section 4.2, otherwise the conclusion of "Robustness against multi-modal attacks" is somewhat not convincing.


[1]	Zhang J, Yi Q, Sang J. Towards adversarial attack on vision-language pre-training models[C]//Proceedings of the 30th ACM International Conference on Multimedia. 2022: 5005-5013.
[2]	Lu D, Wang Z, Wang T, et al. Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models[J]. arXiv preprint arXiv:2307.14061, 2023.

### Questions
No

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies adversarial robustness for multi-modal learning by building a new lower bound for the perturbation radius through uni-modal margins and the Lipschitz constraint. Based on the proposed lower bound, a two-step adversarial training framework has been provided to improve the robustness of multi-modal learning. Experimental results on three benchmark datasets were provided regarding multiple attack methods, compared with several strong baselines.

### Strengths
- **New findings**: While discussing a new lower bound with the Lipschitz constraint is nothing new for adversarial robustness, the proposed method provides theoretical and insightful analyses of how the attack on a preferred modality would impact the overall robustness. This is a practical and common problem in multi-modal integration, as one modality often dominates the others. 
- **Good presentation**: The reviewer enjoyed reading the presentation of the proposed method, where each step was well demonstrated with theoretical supports and clearly developed through proper treatments. One minor suggestion is to provide a pseudo `Algorithm` to outline the method, as well as add a `Remark` to better summarize and explain the training steps. 
- **Decent experiment design**: Despite some minor issues, the experiment is overall well-designed and sufficient by 1) comparing with two groups of strong baselines, 2) adopting multiple attack methods (e.g., FGM, PGD, and missing modality), and 3) providing detailed ablation study and model discussions.

### Weaknesses
- **Missing implementation details**: I may have missed something; however, I did not find any implementation details about the multi-modal encoders. What are the backbones used in the experiment? Can the proposed method apply to different backbones? 
- **Unclear model-specific weights/classifiers**: The exact role of introducing model-specific weights $a^{(m)}$ is somewhat unclear to me. How will it be used to guide the orthogonal classifier of each modality? Also, it remains unclear to me how the proposed eventually gets the prediction result upon different modalities's classifiers. 
- **Lacking empirical evidence**: One main motivation of the proposed approach is one modality may be more vulnerable than the others.  While the adversarial accuracy (between uni-modal and multi-modal attacks) could support this observation empirically, it would be more convincing to provide more evidence that can be used to back-up the theoretical results, such as plotting the vulnerability indicator ($\eta$) values, visualizing the perturbation radius over modalities, etc.

### Questions
Please refer to the questions raised in the *Weakness* section. Plus, the reviewer is interested in the following questions:
- Can the proposed method apply to multiple modalities larger than 2?
- What's the selection criterion in choosing datasets for the experiment? 
- Are the provided theoretical results applicable to vision-text data? Any empirical evidence? 
- Could the proposed method be incorporated into the pre-trained multi-modal model (e.g., CLIP or BLIP)?

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors tackle the challenge of improving the robustness of multi-modal models against perturbations, such as uni-modal attacks and missing modalities. They provide valuable theoretical insight, emphasizing the importance of larger uni-modal representation margins and reliable integration for modalities in achieving higher robustness. They introduce a training procedure, Certifiable Robust Multi-modal Training (CRMT), which effectively addresses modality preference imbalances and enhances multi-modal model robustness. Experimental results validate the superiority of CRMT in comparison to existing methods, demonstrating its versatility and effectiveness. Overall, this paper contributes to the field by providing a theoretical foundation and a practical method for enhancing the robustness of multi-modal models.

### Strengths
Overall, the paper advances multi-modal robustness understanding and presents a practical solution in CRMT with strong empirical results, and the potential for broader applications in the ML/Multimodal community.
- The paper offers a fresh perspective on multi-modal robustness, emphasizing the importance of larger uni-modal representation margins and reliable integration within a joint multi-modal framework.
- The research is methodologically sound, with well-designed experiments and clear presentation.
- The authors effectively communicate complex concepts, enhancing accessibility.

### Weaknesses
The paper included some results on transformer as fusion models, particularlly the Multi-Modal Transformer-based framework with hierarchical attention on the VGGS dataset. However, all experriments, especailly the one with transformerr adopt training from scratch and did not consider any pre-training strategies, such as uni-modal pretraining, or multi-modal pretraining. It will be interesting to consider such methods as baselines and also to see how much CRMT can help to improve. 

Also, except for experimenting, it will be good if authors can discuss how does their method generalize to other fusion mechanisms, besides late fusion.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
