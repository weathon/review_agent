# Kronecker Mask and Interpretive Prompts are Language-Action Video Learners

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Contrastive language-image pretraining (CLIP) has significantly advanced image-based vision learning. A pressing topic subsequently arises: how can we effectively adapt CLIP to the video domain? Recent studies have focused on adjusting either the textual or visual branch of CLIP for action recognition. However, we argue that adaptations of both branches are crucial. In this paper, we propose a **C**ontrastive **L**anguage-**A**ction **V**ideo Learn**er** (**CLAVER**), designed to shift CLIP's focus from the alignment of static visual objects and concrete nouns to the alignment of dynamic action behaviors and abstract verbs. Specifically, we introduce a novel Kronecker mask attention for temporal modeling. Our tailored Kronecker mask offers three benefits 1) it expands the temporal receptive field for each token, 2) it serves as an effective spatiotemporal heterogeneity inductive bias, mitigating the issue of spatiotemporal homogenization, and 3) it can be seamlessly plugged into transformer-based models. Regarding the textual branch, we leverage large language models to generate diverse, sentence-level and semantically rich interpretive prompts of actions, which shift the model's focus towards the verb comprehension. Extensive experiments on various benchmarks and learning scenarios demonstrate the superiority and generality of our approach. The code will be available soon.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a Contrastive Language-Action Video Learner (CLAVER) to efficiently adapt both the visual and textual branches of the CLIP for improving video action recognition. CLAVER introduces the Kronecker mask for temporal modeling to capture the long-range and wide-range dependencies among frames. Moreover, CLAVER resorts to LLMs to generate interpretive prompts of actions to facilitate the alignment of action behaviors and verbs. Experimental results on multiple benchmarks, such as Kinetics-400 and Kinetics-600, show the superiority of the proposed method.

### Strengths
1 The proposed method incorporating the Kronecker mask and interpretive action prompts is demonstrated to be effective in supervised, few-shot, and zero-shot settings.

2 The visualization results sufficiently show that the proposed method captures better spatial-temporal attention as well as focuses more on verbs.

3 The proof of KMCTA can guarantee full rank is provided in detail in the Appendix. 

4 The paper is well-written with figures and tables nicely presented.

### Weaknesses
Overall, I think this paper introduces an effective and generalizable mechanism for improving video action recognition. The proposed method is novel and supported by extensive experimental results in both the main text and Supp. I only have some small concerns as follows:

1 Since the LLM tends to introduce some task-agnostic information in its output, will this noise impact the actual action recognition performance? I would like to see more analysis and examples here. Moreover, the proposed interpretive prompts look like a plug-and-play module. When it is combined with other existing models, will it contribute to performance improvement consistently? 

2 Can the authors give a complete analysis with examples to show when the proposed method may fail? This will help better understanding and deeper analysis of the proposed method.  

3 The font size in Figure 6 is too small for reading. The authors should modify it to give a better presentation. Moreover, Table 14 in Supp has the wrong format.

### Questions
See details in the weaknesses.

### Soundness
3

### Presentation
3

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
The paper introduces a novel framework called Contrastive Language-Action Video Learner (CLAVER), aimed at enhancing the adaptation of CLIP for video understanding, particularly in action recognition tasks. This research addresses a critical gap in existing methodologies, which often focus on modifying either the textual or visual components of CLIP in isolation. The authors argue that a dual adaptation of both branches is essential for effective video analysis. The key contributions are twofold: (1) Kronecker Mask Attention that significantly improves temporal modeling in video data (2) The authors utilize large language models to generate rich, sentence-level prompts that focus on action verbs, steering the model's learning towards a deeper understanding of dynamic behaviors rather than static objects. The extensive experiments across multiple benchmarks, showcasing the effectiveness and robustness of CLAVER compared to existing methods.

### Strengths
* The paper is well-written and easy to understand. The figure illustration and captions are informative.
* The proposed Kronecker mask temporal attention and Kronecker mask causal temporal attention schemes seem interesting. The authors have proved its effectiveness for temporal modeling
* Experimental results on four widely used benchmarks show that the proposed model could achieve superior performance with the existing contrastive baselines.

### Weaknesses
* The performance gain is marginal. The advantages on multiple benchmarks compared to the current state-of-the-art methods do not show a significant improvement, which somewhat undermines the effectiveness of this approach. It remains unclear whether the enhancements stem from the hyperparameter settings or other tricks.
* Can the authors provide details on how much additional computational cost and efficiency are incurred by adding the extra Kronecker mask temporal attention compared to the original CLIP method?

### Questions
Refer to weakness.

### Soundness
3

### Presentation
3

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
The author proposes CLAVER: a Contrastive Language-Action Video Learner, to shift CLIP’s focus for video action recognition tasks from the alignment of static visual objects and nouns to dynamic action behaviors and abstract verbs. They introduce CLAVER with generalizable Kronecker Mask Temporal Attention (KMTA) to expand the temporal receptive field for each token and serve as an effective spatiotemporal heterogeneity inductive bias, mitigating spatiotemporal homogenization. In addition, they utilize large language models to generate interpretive prompts of actions, which shift the model’s focus towards verb comprehension. They conduct extensive experiments on well-known action recognition datasets like Kinetics-400, Kinetics-600, HMDB-51, and UCF-101 demonstrating its competitive performance across models in full-shot, few-shot, and zero-shot settings.

### Strengths
•	Innovative Kronecker Mask Attention: The Kronecker mask temporal attention and Kronecker mask causal temporal are innovative and effective for temporal modeling. Compared with previous commonly used approaches such as joint attention, spatial attention, pipeline temporal attention, etc., it allows each patch at timestamp t to interact with all other patches and expands the temporal receptive field width of each token. It also alleviates the impact of spatiotemporal homogenization.

•	Leveraging large language models to create diverse, sentence-level prompts for action verbs is an effective approach. The author focuses on action decomposition, synonym conversion, which is an inspiring idea of “facilitating the alignment of action behaviors and verbs” by decomposing complex actions into basic and conveying the same core concept with varied expressions. 

•	This paper is well-written and provides a clear overview of CLAVER, with Kronecker Mask Temporal Attention and interpretive prompts. It conducts extensive experiments across several commonly used benchmarks with previous state of the art models as comparison and shows competitive performance, especially in few-shot and zero-shot settings, and provides comprehensive ablation studies to clearly claim the strength of CLAVER by parts.

### Weaknesses
•	Insufficient performance difference between KMTA and KMCTA. KMCTA aims to alleviate the low-rank bottleneck and the author claims that it is important to avoid the low-rank bottleneck to improve the representation power of transformer architecture by giving a formal proof. In the latter ablation studies, the author also successfully shows that KMCTA is profoundly affected by both PreTE and PostTE shuffling, which suggests that KMCTA possesses varying degrees of ability in mitigating spatiotemporal homogenization. However, based on the experiment results in Table. 1 and Table. 7, the difference between KMTA and KMCTA is marginal and we also lack experiments on zero-shot, few-shot settings for CLAVER with KMCTA, which I didn’t see how KMCTA solved the bottleneck according to the current experiments setting.

•	The pipeline of interpretive prompts. The author introduces an interesting pipeline of interpretive prompts by performing action decomposition, synonym conversion, and involving body parts. Then, they provide the prompt to LLaMA-3 for text completion. However, based on the illustration in Figure 4, and the textual description, I feel confused about how to make up these three steps from the left side of the figure to the right side of the figure into format prompts to LLaMA in order to perform text completion.

•	Qualitative analysis for word importance needs further proof. Based on the visualization for CLIP, X-CLIP in Figure. 5, while the author make a claim that CLIP tends to focus on nouns, whereas CLAVER prefers verbs, it’s hard for me to recognize the difference between CLIP and CLAVER’s visualization. As a result, the effectiveness of interpretive prompts for noun concept to verb concept transition might need further explanation.

### Questions
In addition to the questions raised in the weakness section, there are a few minor questions.

•	In Figure 4, the transaction from the left side to the right side might need further explanation. In Figure 5, the example might not fully explain the claim for word importance, please consider making another effective example. 

•	In Table 1, we can see the most powerful model for CLAVER is the CLAVER-L/14(KMT/KMCT). However, when conducting later experiments on Kinetic-600, and other few-shot, zero-shot experiments, the author uses CLAVER-B/16 (KMT) but not the strongest model in Table.1 even if it can follow the same setting. I’m wondering the reason why the author didn’t use their best model for all experiments.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on video action recognition tasks and presents CLAVER to improve CLIP by aligning video representations with verb understanding. The authors design the Kronecker mask attention for improved temporal modeling and interpretive prompts generated by large language models to enhance verb comprehension. The authors demonstrate that CLAVER shows competitive performance on benchmarks such as Kinetics-400, Kinetics-600, HMDB-51, and UCF-101.

### Strengths
1. The paper observes two challenges when applying CLIP to video action classification: How to perform effective temporal modeling and how to design suitable text descriptions for verb understanding. The authors then propose the Kronecker mask temporal attention (KMTA) and Kronecker mask causal temporal attention (KMCTA) to solve 1) and interpretive prompts to solve 2).
2. The paper conducts experiments on multiple video action classification datasets under supervised, few-shot, and zero-shot settings. The results show improved performance over baseline models and other state-of-the-art approaches.
3. The paper includes thorough ablation studies to isolate the contribution of each component, such as KMTA and interpretive prompts, providing evidence for their effectiveness.

### Weaknesses
1. My major concern about this work is its novelty. The proposed Kronecker Mask Attention seems to be a simple combination of pipeline temporal attention and joint attention by attending to all patches in other frames. I expect the authors to provide more insightful analysis and discussion about the proposed method and the differences from previous works.
2. The computational cost introduced by the Kronecker mask attention and the interpretive prompts could be further detailed, especially regarding training and inference time compared to other baselines.

### Questions
See weakness section

### Soundness
3

### Presentation
3

### Contribution
2
