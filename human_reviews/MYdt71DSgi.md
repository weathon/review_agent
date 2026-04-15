# Grounding Everything: Emerging Localization Properties in Vision-Language Transformers

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 5, 6, 5

## Abstract
Vision-language models have shown remarkable performance in various fields, ranging from zero-shot classification to captioning and prompt-based image generation. But so far, those models do not seem able to localize referential expressions and objects in images, with the result that they are only used as a post-process labeling step or that they need to be fine-tuned for this task. The following work, we show that vision-language (VL) models trained with image-level objectives hold object localization properties. We propose a Grounding Everything Model (GEM) that allows to leverage these properties without retraining or fine-tuning the pretrained model.  To this end, we extend the idea of v-v attention introduced by CLIPSurgery to a generalized self-self attention path and propose a set of regularizations that allows the model to better generalize across datasets and backbones. We further show how the concept of self-self attention corresponds to clustering, thus enforcing groups of tokens arising from the same object to be similar, while preserving the alignment with the language space. We evaluate the proposed GEM framework on three benchmark datasets and improve the performance in training-free open-vocabulary localization.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a post-hoc (no training) way to cluster features, and shows some applications to open-vocabulary segmentation.
The paper takes the value-value attention in CLIPSurgery and generalizes it to iterated value-value, key-key, query-query attention (or a combination thereof) to compute similarity. 
Experiments are on object localization, PascalVOC, PascalContext, and OpenImages-V7, where the method beats raw similarity between text and features. There is a nice comparison to k-means clustering, where results are a bit better than K-means (but quite similar nonetheless).

### Strengths
### Presentation
The paper is well-written.

### Experiments
I think the experiments are well done (though some comparisons are missing, see weaknesses). In particular, I like the insightful comparison to K-means.

Experimental results evaluated on multiple architectures and datasets. There are clear ablations of the different components and settings (temperature, iterations, self-self attention).

### Method
The method is simple, and reasonably principled

### Weaknesses
Overall I'm not seeing a use case for this method. K-means is a quicker-to-implement clustering/visualization method, and on the other end of the spectrum other techniques like DeepCut seem to offer better performance.

### Presentation
The title is "Grounding Everything Model". I think at best this is not a good fit for the paper, and at worst has the possibility to be a bit misleading. Firstly, this isn't a model, right? It is a post-hoc clustering technique and the authors evaluate it on several pretrained models. I suppose this is a form of grounding because it is a way to localize queries in an image, when the original architecture doesn't support that. But if you want to ground something, there are better models -- e.g. [Grouding-Dino](https://arxiv.org/abs/2303.05499)

### Experiments
1. The paper misses a lot of existing literature -- what about comparison to other papers that do clustering of pretrained features (e.g. [Deep Spectral Methods](https://github.com/lukemelas/deep-spectral-segmentation)? The only other clustering method compared to is K-means (and I guess CLIPSurgery, which I believe is just 1 iteration of V-V attention). 
2. The results are pretty weak -- not much better than K-means. Other feature-clustering approaches seem to have much better results -- e.g. [DeepCut](https://sampl-weizmann.github.io/DeepCut/). That paper also has extensive comparisons to existing lit that clusters deep features on VOC and other datasets.

### Questions
1. I wonder if this could be understood as an iterative approximation to some type of spectral clustering (computing neighbor laplacian, computing eigenvectors, the temperature setting in the alg washes out all eigenvectors with eigenvalues < some thresh, and then run k-means clustering on the eigenvectors). An exact analysis is escaping me here.
2. For the random gaussian clustering technique -- all vectors come from the same gaussian, right? And the resulting clustering from GEM is similar to that of K-means. That is a somewhat arbitrary clustering because the underlying process all comes from the same cluster. Why not generate data from different clusters, and show that GEM is better able to recover the underlying clusters?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a training-free method to address the image referential localization problem. It proposes GEM model and extends the v-v attention to self-self attention. The experimental results show that the proposed GEM framework improves the performance in training-free open-vocabulary localization.

### Strengths
1. This paper proposes a training-free method to address the open-vocabulary segmentation problem, and it improves the baseline model.
2. This paper shows that by merely computing the proposed heuristic similarity and without training, the pre-trained model already does well in open-vocabulary segmentation task.

### Weaknesses
1. This work claims the localization property is 'emerging', which is somehow overclaimed. The resulting model yields merely less than 50% mIoU in VOC, which is far from 'emerging' performance.
2. The so-called localization is usually used for a high-level concept. In this work, it should be replaced by the more precise term 'segmentation'.
3. The overall method is straightforward and tedious. It shows a way to compute a heuristic similarity with a high distinctiveness and does improve the performance. However, the procedures are too tedious and there is still a huge gap with the methods involved with training.
4. There is always a consensus that the final model usually needs two phases --- pre-training and finetuning. Why do we need a finetuning-free method for a segmentation task? A model pre-trained with a large quantity of data needs finetuning or alignment for a specific task unless it already has the emergent ability for multiple tasks. But today, we have seen that there is still a huge performance between a pre-trained model and a finetuned one, even in a single task. In this way, the finetuning-free property does not really matter in the context of a single task. The authors should elaborate more on this.

### Questions
The authors should give their explanations for each weakness above. 

Overall, I think this research is not really beneficial and promising for the development of this area, and this is not a paper that reaches the acceptance standard of ICLR.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper present Grounding Everything Model (GEM) in order use pre-trained Vision-Language (VL) models for object Localization without the need of re-training or fine-tuning the VL model. Building on CLIPSurgery (that incorporates value-value attention), the paper extends it to a generalized self-self attention path and propose a set of regularizations that allows the model to better generalize across datasets and backbones. Experimental results on three benchmarks (PascalVOC, Pascal Context, and OpenImages-V7) shows promising performance boost compared to existing training-free works.

### Strengths
The paper is well-written and easy to follow. Related literature is being reviewed in a comprehensive manner. The paper provides an in-depth analysis of the properties of self-self attention, along with its connection with k-means clustering, in various pretrained ViT transformer models.

### Weaknesses
My main point of criticism concerns the technical novelty of this work, since it builds on the recently proposed CLIPSurgery and proceeds by simply swapping value-value with self-self attention. 

Whilst this might appear as a marginal technical contribution, the authors provides good experimental results and a thorough analysis of v-v and self-self attention, with an interesting connection to k-means. It is not totally convincing, though, how this work is essentially different than CLIPSurgery. I would expect a more comprehensive comparison to CLIPSurgery, especially in methodological terms -- why the proposed method's perfomance cannot be achieved by CLIPSurgery by simply exploring the attention mechanism?

### Questions
In Table 2, k-means achieves the best results for k=3 (PascalVOC) and k=7 (Pascal Context):
 - What would be a possible explanation for this? 
 - How would a different clustering method affect the results (in other words., could a different clustering method achieve results than the proposed GEM's ones)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the Grounding Everything Model (GEM), leveraging the latent localization capabilities of VL models trained on web-scale datasets. In this paper, the authors present an extended version of the CLIPSurgery concept of v-v attention, called self-self attention, which can extract localization information from VL models. The GEM method effectively enables open vocabulary localization without additional training.

### Strengths
1) Overall, this paper is well-written, and the technical details are easy to follow. 

2) The main idea of avoiding additional training for open vocabulary localization is unquestionably important.

3) The main contribution of this paper is the self-self architecture, which is an extension of CLIPSurgery's concept of v-v attention.

### Weaknesses
**Technical Novelty.** The fact that v-v attention was suggested without considering that it can also be applied to keys and queries seems odd. I believe the CLIPSurgery paper may have been lacking in ablation since a properly executed ablation should reveal that v-v attention is not necessarily the most effective approach. According to this paper, they "...discovered that the major cause of this issue is the parameters (query and key) in the self-attention module" (see CLIP Surgery). Therefore, it seems strange that the query and key parameters are actually required now.

This brings me to the main concern I have. The proposed architecture is a slight extension of the v-v attention, and in particular, I cannot see how duplicate self-attention, in a similar manner to v-v attention, could be applied to other tasks or models. According to my understanding, this represents a very small ablation that the first paper missed, and is not something the community could make significant use of in the near future for another task or dataset.


**Motivation.** It is understandable that the authors are motivated to use pretrained vision and language models for localization. However, I am unsure why this is important since we already have SAM [1] (and other improved versions), which demonstrate that these supervised models are capable of almost completely solving this problem. The fact that SAM performs extremely well on this task raises the question of whether pretrained vision and language are still required, given its excellent performance.

[1] Segment Anything Model. ICCV 2023.


**Experiments.** For Table 2, did the authors use other methods than K-means? The K-means approach appears to be almost equivalent to the proposed approach, although the proposed approach has been modified to achieve the best results, while the K-means approach I assume has not been.

Additionally, this work has been applied to CLIP, but CLIP is already fairly updated while there are several other existing methods that are better, such as BLIPv2, LLaVa, and others. We cannot be certain whether the problem of object localization properties still exists in more advanced methods, so I would like to see whether the approach can generalize well to other models.

### Questions
I am concerned that this paper does not present a significant approach to one specific VL model, such as CLIP, for dealing with object localization in a training-free manner. My concerns have been listed above, and I would appreciate it if the authors could address them. I am also open to the authors' feedback and other reviewers' opinions.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
