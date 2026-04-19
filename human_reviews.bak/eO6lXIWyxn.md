# ARTIST: Towards Disentangled Text Painter with Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Diffusion models have shown remarkable performance in generating a broad spectrum of visual content. However, their text rendering ability is still limited: they generate wrong characters or words that cannot blend well with the background image. To address this, we introduce a novel framework named ARTIST, which includes an additional textual diffusion model focusing on text structure learning. We first pretrain the textual diffusion model. Then we further fine-tune the visual model to learn how to inject textual structure information from the frozen textual model into the image. This disentangled architecture design and training strategy significantly enhance the text rendering ability of the diffusion models for text-rich image generation. Furthermore, we leverage pre-trained large-language models to infer the user's intention leading to better generation quality. Empirical results on the MARIO-Eval benchmark underscore the effectiveness of the proposed method, showing an improvement of up to 15\% in various metrics.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents ARTIST, a novel framework to address text rendering problems in diffusion models for image generation. They employ LLMs to identify users’ intentions and introduce dual stages of training to master text structure and visual quality. They achieve up to 15% improvement in various metrics on the MARIO-Eval benchmark, demonstrating the effectiveness of their approach.

### Strengths
This paper demonstrates a well-structured and clearly articulated research strategy. The experimental results presented in the paper showcase good empirical performance.

### Weaknesses
- The formatting of the main text has some issues, i.e., page 5 with the presentation of Fig. 2 and Eq. (2). Such formatting problems could hinder the overall readability and should be addressed.
- Visual results of ARTIST and the analysis of experiments are somewhat lacking. More qualitative results, similar to those presented in the 2nd column of Fig. 5, should be provided to better illustrate the differences between ARTIST and TextDiffuser. 
- The references need some revisions. For instance, the reference to "Classifier-Free Diffusion Guidance" should be updated to reflect its publication in the NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. Ensuring that all references are accurate and up-to-date.

### Questions
See the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new framework ARTIST that includes one additional text diffusion model to learn visual text structure, which helps disentangle the learning of text structure and text aesthetics. The experiments show improvement of OCR performance compared to previous methods. This paper also introduces large language model to resolve the issues of extracting target text.

### Strengths
1. The idea to disentangle the learning of text structure and text aesthetics is novel. The experiments also show the efficacy of the proposed method.
2. Utilizing LLM can help resolve the issues of target keyword extraction from prompts with no explicit mark for target text.

### Weaknesses
1. I am not convinced as to why we should use LLM to improve target text extraction since it is very easy for the user to specify the target text. 
2. The authors did not provide enough information about how LLM performs in generating a suitable layout.
2. The method part writing is not clear regarding the input to the model. How can a model decide which mask to generate the suitable word? For example, in Figure 2, how are "ARTIST" and "MODEL" specified for the two mask regions?

### Questions
1. How are the layouts generated using LLM? Are the layouts limited to normal-text layouts? Can LLM generate rotated or curved text layouts? Could authors provide detailed prompts for LLM?
2. Could the authors provide more information about the input for the visual examples, e.g. the layouts in Figures 3 and 4.
3. How does ARTIST deal with multiple-word generation? How does the model decide which mask to generate for each word?
4. Instead of adding another module to learn the text structure, how does the model perform when training the original model in a two-stage process?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
They present ARTIST, which first leverages an external LLM to produce the layout of the target text. The disentangled pipeline then produces the visual text and the figure, and fuses both as the final generated result. ARTIST demonstrates its creative and robust text rendering ability on the MARIO-Eval dataset.

### Strengths
- This paper is well-written and easy to follow.
- Text painting is a challenging task for current T2I models, and the idea of disentanglement between text and figure is well-motivated.
-  They provide lots of qualitative examples for the visual comparison.
- Their proposed ARTIST achieves notable improvements over the previous TextDiffuser, especially on the crucial OCR metric.

### Weaknesses
- It seems that the ARTIST framework is the combination of LLM-Layout and TextDiffusers, which both have been proposed before. Not sure if this achieves ICLR's novelty bar.
- There are so many metrics used for the evaluation (FID, CLIP-S, and OCR). Which one is the most appropriate to evaluate the overall performance? Or is there any way to combine all of them as a final metric?
- As a generative task, a human generation should be conducted to compare the performance in a human aspect.
- Since ARTIST relies on LLM to derive the layout, there should be a discussion about the quality of the generated layout. Is it visually appealing or in a reasonable position?
- Missing reference about LLM-Layout: [NeurIPS'23] Compositional Visual Planning and Generation with Large Language Models / [arXiv'23] Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models

### Questions
Please see the weaknesses

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies text rendering with images and proposes to use disentangled text and visual modules (both are fine-tuned from Stable Diffusion) along with LLM to identify which text and where to render. In the experimental results, the proposed approach outperforms the baseline, and also ablation study shows the effectiveness of the proposed components.

### Strengths
- The idea to use disentangled modules for text and image, along with LLM for generating guidance, looks novel.
- The proposed approach outperforms the baseline, TextDiffuser, in experimental results. 
- The presented ablation study shows the effectiveness of the proposed components.

### Weaknesses
- The experiment lacks an important qualitative evaluation, namely, human evaluation. This may limit to show effectiveness of the proposed approach.
- Some experimental results are not very convincing. e.g., FID is worse than Fine-tuned SD on MARIO-Eval benchmark. Also, in Fig. 5, it is not clear the proposed model is better than TextDiffuser.

### Questions
- When training visual module, do the targets also include text, or only images? 
- Does LLM always succeed? Is there any failure cases, and if so, why?
- What is "s" below as it is not clear. You need to define it first before using it.
page 4:  ... model to extract s which ...
page 4:  ... identify the essential s 

- You claim that "our computation requirement is still similar to the previous SOTA TextDiffuser", and can you provide actual model sizes such as the number of parameters of models?
- In Fig. 6, how can the proposed model without mask achieve the better CLIP and FID compared to the one with mask?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
