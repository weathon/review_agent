# RL makes MLLMs see better than SFT

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
A dominant assumption in Multimodal Language Model (MLLM) research is that its performance is largely inherited from the LLM backbone, given its immense parameter scale and remarkable capabilities. 
This has created a void in the understanding of the vision encoder, which determines 'how MLLMs perceive images'.
The recent shift in MLLM training paradigms, from Supervised Finetuning (SFT) to Reinforcement Learning (RL), magnifies this oversight—namely, the significant lack of analysis on how such training reshapes the vision encoder as well as the MLLM.
To address this, we first investigate the impact of training strategies on MLLMs, where RL shows a clear advantage in strongly vision-related VQA benchmarks than SFT. 
Motivated by this, we conduct a critical yet under-explored analysis of the vision encoder of MLLMs through diverse and in-depth experiments, ranging from ImageNet classification and segmentation to gradient visualization. 
Our results demonstrate that MLLM's post-training strategy 'i.e, SFT or RL' not only leads to disctinct outcomes on MLLM downstream tasks, but also fundamentally reshapes MLLM's underlying visual representations. 
Specifically, our main finding is that 
RL produces stronger and more localized visual representations compared to SFT, boosting the ability of the vision encoder for MLLM.
We then reframe our findings into a simple recipe for building strong vision encoders for MLLMs, Preference-Instructed Vision OpTimization (PIVOT). 
When integrated into MLLMs, a PIVOT-trained vision encoder outperforms even larger and more heavily-trained counterparts, despite requiring less than 1\% of the computational cost of standard vision pretraining. This result opens an effective and efficient path for advancing the vision backbones of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work studies the impact of the RL technique DPO (Direct Preference Optimization) on the training of MLLM. The finding of the paper can be summarized on DPO being more effective than standard SFT when applied to a proper set of multimodal instruction tuning data containing both a positive and a negative response in text. The authors provide an extensive set of experimental analysis supporting their claims and propose a method called PIVOT to “warm up” visual encoders to be plugged into MLLM and boost their performance. The method boils down to pre-train them with a smaller LLM head using SFT followed by DPO. Only at that point the visual encoder is plugged in the bigger scale LLM to create the final MLLM. This provides a significant boost in performance, in the case where the visual encoder is kept frozen while training the MLLM (ala LLava), and a modest one when everything is trained end-to-end.

### Strengths
+ **Presentation:** I really appreciated the presentation in this paper. The work follows a very clear storyline and make an extremelly good case on why the proposed experiments make sense and on how to interpret the results. I found the reading very enjoyable and I would consider this a very good example on how to write a paper in an interesting way.

+ **Novelty:** To the best of my knowledge the effect of SFT vs RL for MLLM is still a quite under-explored area. This work does a nice job at taking a stab at it and starting to characterize the difference among the two. I particularly enjoyed the emphasis not only on the final performance, but also on the effect of the different strategies on the learned representations (e.g., the GRADCam experiments)

+ **Extensive experimental analysys:** The author run a significant number of experiments to explore their hypothesis and support their claim. Most experiments that came to my mind while reading the work were discussed in a subsequent part of the paper. After going through the main paper and supplementary I’m left with almost no doubt or question.

### Weaknesses
## Major

I haven’t found major weaknesses in this work.

## Minor

a. **Data advantage for DPO:** The comparison between SFT and DPO is not completely fair since the data used for the comparison are amenable to DPO training that per its own nature will use also negative answers in its optimization process, while SFT will not. This brings to two possible data advantage for DPO:
1. *Data scale:* De facto DPO is technically “seeing” more examples/information within one training run. From this perspective it is not surprising that it is more effective than SFT, because it can rely on the negative examples to pull the optimization away from that path, while SFT cannot. At the same time there isn’t a practical way that comes to my mind for SFT to leverage in any way the negative answers, therefore I don't have a better suggestion on how to compare the two.  Fig. 4 also shows that DPO is vastly more data efficient, hence this argument of “DPO” seeing more data is not the full picture.
2. *Data type:* The data used for Stage 2 are explicitly created to not have a clear ground truth but being more open, i.e., perfect for DPO suboptimal for SFT. SFT on these data will force the model to try to collapse its distribution to only one of the possible many options which are all good answers. DPO instead will only try to rank the good answer above the bad one in terms of likelihood. It would have been interesting to repeat the comparison between SFT and DPO on a more SFT friendly distribution of data (i.e., data with a clear answer). This would help to understand wheter the same findings hold also there or if the story of the paper should be adapted in “pick the best losses for the type of data you have”.

B. **PIVOT contributions/results being weak:** I found the PIVOT section of the paper significantly weaker than the rest. The core issue that I have is that in Tab. 1 the authors compare a visual encoder that has been “warmed up” to work good with a LLM and later kept frozen for the final part of the training to a visual encoder that has never been jointly trained with a LLM. It is not surprisingly that PIVOT encoders obtain significantly higher performance compared to their base model. It is still interesting that lines with +PIVOT are better than lines with +SFT, but this is just the storyline of the rest of the paper. This is even more evident in Tab. D appendix where the gain from PIVOT shrink significantly when going in a “Full train” regime where the visual encoder is allowed to adapt to the LLM during the training phase on Cambrian. All in all I found the PIVOT contribution significantly weaker than the rest and would have liked the paper even without it.

### Questions
1. In Fig. 4 the slope of SFT seems significantly more than DPO. Do you have experiment scaling the dataset size even further to verify wheter SFt eventually catch up with DPO? 

2. Can you comment on weakness A above? 


3. More of a suggestion, Fig. 11 in the main paper is very not informative/somehow misleading. I would suggest to move Fig. D from the appendix to the main paper as it does a way better job in explaining what is going on.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates how RL post-training compares to the standard STF one in MLLMs on vision-related tasks. Their findings show an advantage of using the RL paradigm for both the quality of visual representation and downstream performance on VQA. The authors propose a new training regime for MLLMs, called Preference-Instructed Vision OpTimization (PIVOT), which consists of a pretraining stage and a post-training stage using DPO on instruction-following samples and 20K preference pairs.

### Strengths
- Thorough analysis of the impact of using STF vs RL in the post-training stage showing significant differences- 
- Well written paper, easy to follow.

### Weaknesses
No experiments on other vision related tasks like image captioning, object hallucinations etc.

### Questions
- Significant benefits of using RL are demonstrated on VQA tasks. How about other vision related tasks like image captioning or object hallucinations? Did you perform any experiments to confirm that the performance transfers to these as well?
- I understand that the goal of this paper is to compare the effects of the STF vs RL in post-training on vision encoders. However, it would still be interesting to compare the performance of PIVOT with some of the established MLLM like BLIP2, LLaVa etc and those focusing on vision, like BRAVE: Broadening the Visual Encoding of Vision-Language Models (ECCV 2024), SPHINX: A Mixer of Weights, Visual Embeddings and Image Scales for Multi-modal Large Language Models (ECCV 2024) and similar (which are missing from the related work).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The research reveals that RL training (e.g., DPO) helps the visual encoder generate stronger and more precisely-localized visual features than SFT. This advantage results in better performance on both MLLM benchmarks (especially vision-heavy VQA) and classic vision tasks such as ImageNet classification and segmentation. Subsequently, the authors then consolidated these findings into PIVOT, a practical recipe, and validated its efficacy across a diverse range of vision encoders.

### Strengths
1. The work presents an under-explored area: how RL-based training affects the vision encoder in MLLMs, not just the language model. It  also provides systematic analyses of visual representations in RL-trained MLLMs for the first time.

2. Based on the analytical experiments, this work proposes PIVOT, a recipe that enhances the representational capacity of vision encoders through RL-based training with MLLM. Extensive experiments conducted on visual comprehension tasks consolidate its effectiveness.

### Weaknesses
1. The work only compares the performance of  visual encoders after post-training (SFT or RL). Could you provide an estimation of the visual encoder after Stage 1 (without any post-training) training on various visual comprehension tasks?

2. Most benchmarks are academic and may not fully reflect real-world multimodal challenges. Broader evaluation (e.g., on hallucination, robustness, or user-facing tasks) would be beneficial.


3. The study focuses exclusively on DPO as the RL method. While DPO is popular, other RL approaches (e.g., PPO, GRPO) are not explored, limiting the generalizability of the findings.

4. According to the experimental setup of PIVOT, a frozen visual encoder derived from post-training is integrated into a lightweight MLLM for visual understanding. In this context, it is hoped that the authors could provide an optimization approach for the lightweight MLLM through distillation and present corresponding performance evaluations. Specifically, this setting involves using the post-trained VLM(after SFT or RL) to generate corresponding data for LAION/CC/SBU-558K, mixing it with the original data, and then training a lightweight VLM. Only the projector layer would be optimized, and the final comprehension performance would be evaluated.

### Questions
Please see the problems listed in the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work aims to understand the effect of reinforcement learning versus supervised fine-tuning post-training on the visual encoders of large vision-language models. Accordingly, the work shows that, RL (via DPO) tends to improve vision-centric MLLM performance and yields stronger, more localized visual features than SFT. Following from this intuition, the work then proposes the upgrade the visual encoder quality by first training it with DPO in an MLLM post-training setting. Several experimental results support both the initial analyses and the usefulness of PIVOT.

### Strengths
Here are the primary strengths of the work:

- The motivation is very clear. The work isolates SFT vs. DPO for MLLM post-training and explicitly examines effects on both the MLLM and the vision encoder rather than only downstream scores.
- The work includes several encoder-centric analysis, complementing VQA results with ImageNet linear probes, segmentation probing, and qualitative localization, arguing that RL strengthens and localizes visual features.
- The work then proposes a simple extension of DPO to achieve stronger visual encoders, called "PIVOT", while showing reusing an RL-trained encoder in new MLLMs gives decent gains.

### Weaknesses
Here are the primary weaknesses of the work:

**W1: Limited Methodological Novelty:** The paper’s primary contribution is an analysis of SFT vs. DPO under a direct, unmodified application of DPO without any new objectives or reward. The proposed visual encoder-upgrading method, PIVOT, also follows straightly from this.

**W2: Limited to a Single RL Variant:** The field is shifting quickly and there are many newer RL variants utilized in MLLMs, e.g. GRPO, as also noted by the authors. However, the conclusions are demonstrated for DPO-style RL within a specific Stage-2 pipeline and it’s unclear how broadly they hold across other RL algorithms, reward definitions, or different post-training recipes.

### Questions
- Do the main findings (encoder strengthening and vision-centric gains) persist with other RL approaches (e.g., PPO/GRPO) or alternative pairwise objectives beyond DPO?

- Where does DPO help least (e.g., knowledge VQA)? Though the work is vision-centric, how does the end MLLM perform under text-only benchmarks, such as HellaSwag? Do you observe any depreciation?

- For the models leveraging a combination of SFT and DPO [A, B], where does your work stand? Would your conclusions hold for a practical pipeline with SFT→DPO (and DPO→SFT) under equal compute? There is a mention of this literature in the Appendix, stating how this work only focuses on the differences between SFT vs DPO, but I believe that a more detailed discussion is required here. 

I believe that the work could be improved further by addressing the aforementioned questions.

---
[A] Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., Chen, D., ... & Sun, M. (2025). Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 19985-19995).

[B] Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., ... & Darrell, T. (2023). Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525.

### Soundness
3

### Presentation
3

### Contribution
3
