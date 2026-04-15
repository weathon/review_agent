# Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Current large multimodal models (LMMs) face challenges in grounding, which requires the model to relate language components to visual entities. Contrary to the common practice that fine-tunes LMMs with additional grounding supervision, we find that the grounding ability can in fact emerge in LMMs trained without explicit grounding supervision. To reveal this emerging grounding, we introduce an "attend-and-segment" method which analyzes the attention within standard LMMs to provide a point prompt to a segmentation model (e.g., SAM) and perform pixel-level segmentation. Furthermore, to enhance the grounding ability, we propose DiffLMM, an LMM utilizing a diffusion-based visual encoder, as opposed to the standard CLIP visual encoder, and trained with the same weak supervision. Without being constrained by the biases and limited scale of grounding-specific supervision data, our approach enables strong visual grounding while preserving general conversation abilities. We achieve competitive performance on both grounding-specific and general visual question answering benchmarks, compared with grounding LMMs and generalist LMMs, respectively. Notably, we achieve a 44.2 grounding mask recall on grounded conversation generation, outperforming the extensively supervised model GLaMM.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper shows the possibility of grounding LMMs without grounding supervision. The paper presents a method called attend-and-segment that can be applied to existing LMMs to achieve pixel level grounding. The paper introduces DIFFLMM, which employs a visual encoder based on the diffusion model. The experiments are conducted on Grounded Conversation Generation (GCG) and VQA benchmarks.

### Strengths
1. The paper reads smooth and is mostly clear.

2. The motivation of using attention maps for pixel-level grounding is reasonable. 

3. The proposed DiffLMM achieves good experimental results compared to CLIP-based, and DINO-based methods for GCG, while maintaining general VQA performance.

### Weaknesses
1. From experiments Table4 and Figure2, it seems that the quality of the attention maps are not good enough for grounding. Please also report the result without SAM. And it seems the model highly relies on SAM, a foundation CV model, to produce good segmentation mask. So the sentence "the grounding ability can **emerge** in LMMs without grounding supervision. " is a little bit over-claimed. In addition, using SAM may also conflict with the claim "without explicit grounding supervision", as SAM is trained with a huge number of grounding related data. 

2. Lack of experiments on referring expression segmentation (RES) benchmark [a,b] and panoptic narrative grounding (PNG) benchmark [c]. RES benchmarks can better validate the effectiveness of the proposed method, as they require LMMs to perform both reasoning and segmentation. PNG benchmark validates the ability of LMMs to segment user-described objects and ground key phrases or words in a text sequence.

3. As pointed out in the paper, the paper is mostly related to  F-LMM (Wu et al., 2024a), however, no direct experimental comparisons are provided. 

4. Experimental results in Table1 shows that the AP and AR of DIFF-LMM are very low (only 5.6 AP and 26.0 AR), limiting its application in real scenarios.

[a] Kazemzadeh S, Ordonez V, Matten M, et al. Referitgame: Referring to objects in photographs of natural scenes[C]//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 787-798.
[b] Mao J, Huang J, Toshev A, et al. Generation and comprehension of unambiguous object descriptions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 11-20.
[c] González C, Ayobi N, Hernández I, et al. Panoptic narrative grounding[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 1364-1373.

### Questions
1. The method utilizes existing natural language processing tools (e.g., spaCy (Honnibal et al., 2020)) to parse the output sequence into noun phrases. However, in some cases only "noun phrases" may fail to capture the whole semantic meaning in some complex scenario, e.g. "the man in black holding an umbrella". How to handle this problem?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper exploits geometric and spatial cues within the word-image attentions of large multimodal models (LMMs), which is conducive to visual grounding in multimodal conversations. Then the authors introduce the attend-and-segment paradigm for grounded conversation. Specifically, the word-image attention maps are normalised and fed to the SAM model for mask prediction. In addition, the authors propose to apply diffusion U-Net as visual encoders for multimodal models, which helps enhance visual grounding performance. The proposed method is evaluated at the Grounded Conversation Generation (GCG) benchmark.

### Strengths
1. The paper is well-organised and easy to follow. The method and experimental settings are all clearly illustrated. 
2. The proposed method alleviates the need for pixel-level supervision for grounded conversation and preserves the excellent visual question-answering capability of modern LMMs.

### Weaknesses
1. The proposed attend-and-segment paradigm relies on the external tool SpaCy to parse noun phrases, which would simply enumerate all noun words and phrases from a sentence, regardless of whether the words correspond to visual objects in the images. Therefore, the parsing process can be quite noisy and might require prior human knowledge to filter out undesired nouns.

2. The proposed method is only evaluated on the GCG task, lacking experimental verification under tasks like referring segmentation which has been widely adopted in existing grounding LMMs.

3. The ad-hoc nature of using diffusion U-Net for enhanced grounding ability. Although the method circumvents additional grounding supervision, it requires re-training an LMM to incorporate the U-Net features on the million-scale pre-training and instruction-tuning datasets.

### Questions
1. Can the authors provide more details in obtaining object words / phrases by SpaCy? For example, how the nouns irrelevant to visual objects are filtered out?

2. Since the U-Net is already trained with cross-attention modules that interact with texts, would the word-image attention maps inside the U-Net be more capable of visual grounding than the attentions in the LLM part? If so, should we do visual grounding using u-net only instead of extracting LLM's attention maps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims to exploit the grounding ability of large multimodal models (LMMs). Unlike the traditional methods that finetunes LMMs using grounding supervision, this work introduces attend-and-segment to leverage attention maps from LMMs for pixel-level segmentation. Diffusion-based visual encoder is applied to capture both high-level and low-level visual features. The experimental results demonstrate the proposed DiffLMM achieves remarkable performance across different tasks, especially on the grounded conversation generation (GCG) task.

### Strengths
This work exploits the grounding ability of LMMs without explicit grounding supervision. Motivated by the attention maps from LMMs capturing the region of interest, attend-and-segment mechanism is proposed to find out the segmentation masks. The proposed DiffLMM achieves the impressive results on the grounded conversation generation task.

### Weaknesses
As the core motivation of this work, the efficacy of attention maps from LMMs should be validated. 1) The precision and recall of attention maps representing objects are expected to be reported, which indicate whether they find out the regions of interest and whether the irrelevant objects/background are detected respectively. 2) Employing a single point with the highest attention value may miss to hit multiple objects or multiple regions of interest. It is unclear how to deal with these situations and how good this design is for capturing the most relevant object or region. 3) The attention maps from different LMMs and different prompts are also worth exploration.

Applying the diffusion model as the visual encoder is claimed as another contribution of this paper. However, as shown in Table 5, the performance of DiffLMM using SD-1.5 + CLIP is even inferior to that of original LLaVA-1.5 using CLIP backbone without PE and IC, raising concerns regarding the effectiveness and necessity of diffusion-based visual encoder.

The applied SAM prompt (mask or point) is a critical technical implementation. This technical design should be specified in the method section, rather than only mentioned in the ablation study. Also, it is expected to be clarified if SAM prompts differ for different tasks.

The verification of scalability of DiffLMM is absent. To evaluate the scalability of the proposed method, experiments involving data scaling up are expected to be provided.

To extensively demonstrate both the grounding and conversation/reasoning abilities of DiffLMM, it would be better to also include the experimental results on tasks such as referring expression segmentation, panoptic narrative grounding and reasoning segmentation.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
