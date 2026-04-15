# HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing

- Decision: Accept (Poster)
- Scores: 3, 5, 6, 8

## Abstract
This study introduces HQ-Edit, a high-quality instruction-based image editing dataset with around 200,000 edits. Unlike prior approaches relying on attribute guidance or human feedback on building datasets, we devise a scalable data collection pipeline leveraging advanced foundation models, namely GPT-4V and DALL-E 3. To ensure its high quality, diverse examples are first collected online, expanded, and then used to create high-quality diptychs featuring input and output images with detailed text prompts, followed by precise alignment ensured through post-processing. In addition, we propose two evaluation metrics, Alignment and Coherence, to quantitatively assess the quality of image edit pairs using GPT-4V. HQ-Edits high-resolution images, rich in detail and accompanied by comprehensive editing prompts, substantially enhance the capabilities of existing image editing models. For example, an HQ-Edit finetuned InstructPix2Pix can attain state-of-the-art image editing performance, even surpassing those models fine-tuned with human-annotated data.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a new high resolution dataset on image editing named HQ-Edit. Compared to existing works, the proposed dataset contains high resolution and leveraged the most advanced vision-language model for creating human like instructions. Since all the instruction and image generated are using synthetic data, as a result, the proposed dataset is in a reasonable large scale for public usage. Simply leverage the dataset, the existing InstructPix2Pix can generate better results.

### Strengths
The key contribution in this paper is the proposed HQ-Edit dataset. Compared to existing widely used InstructPix2Pix dataset, HQ-Edit dataset is in a high resolution and better editing due to verification  process of a powerful vision-language models such as GPTV.  For example, after generating the dataset by Dalle 3.  GP4V will be used to rewrite the instructions to improve the instruction and editing alignment. Such process will greatly improve the dataset.  Besides, human evaluation has shown that the proposed dataset is better on editing image quality.

### Weaknesses
The key weakness of the proposed method lies on the limited quality improvement of dataset. 

First, InstructPix2Pix firstly proposed a pipeline to leverage prompt2prompt and gpt to generate the triplets. While the proposed "high quality" editing dataset is simply replaced the prompt2prompt method to Dalle3 and leveraged a more powerful gpt version to filter out the low quality instructions. Such improvement is incremental while not significantly improve the datasets.  The main reason is, similar to prompt2prompt, the HQ Edit dataset is still all synthetic pairs.  The proposed editing types are still similar to InstructPix2Pix.  Most importantly, when most of existing editing dataset such as Magicbrush, SeedX and UltraEdit have adopt real image as input,  the usage of synthetic image pairs has became less important. 

Second, the proposed dataset have shown multiple editing type, while leveraging Dalle3 is obviously not the optimal solutions.  SeedX has been demonstrated the remove and add case should leverage more accurate model such as image inpainting. Meanwhile, the quality of Dalle3 images have become less convincing to be recognize as "high resolution and high quality" since Flux and other generative super resolution can achieve much better higher resolution image editing.

### Questions
Since the proposed dataset contains multiple type of editing, it is important to show the results of different editing cases, especially on real images.   Moreover, it is also better to compare with some specialist models such as inpainting and object insertion model. 

In addition, I also notice that the edited results still can't keep the un-edited region unchanged. Many cases showing in the paper indicate that changing unrelated region will cause lots of identity shift on other regions. 

It also worths to show more global editing on more real images such as stylization (painting, sketch, water color, cartoon style) and image2image translation (day to night, winter to summer). EMU edit benchmark should be a good dataset to be evaluated. 

Reference 
[1]https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit
[2]https://huggingface.co/datasets/BleachNick/UltraEdit

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents HQ-Edit, a dataset designed for instruction-based image editing, which contains around 200,000 high-quality image edits. The authors have developed a scalable data collection pipeline that utilizes advanced foundation models, GPT-4V and DALL-E 3, to generate detailed text prompts paired with input and output images. This approach overcomes the limitations of previous methods that relied on attribute guidance or human feedback, which were constrained by scalability and diversity.

To quantitatively assess the quality of image-edit pairs, the authors introduce two novel metrics: Alignment and Coherence. Alignment evaluates the semantic consistency with edit prompts, while Coherence assesses the overall aesthetic quality of the edited image.

The author fine-tuned the InstructPix2Pix model using the HQ-EDIT dataset.Compared to existing text-based image editing models,their model performs best in Alignment and Coherence Score.

### Strengths
1. High-Quality Dataset: The paper introduces HQ-Edit, a dataset with approximately 200,000 high-quality image edits, which is a significant contribution to the field of instruction-based image editing.
2. Advanced Foundation Models: Leveraging state-of-the-art models like GPT-4V and DALL-E 3 ensures that the dataset benefits from the latest advancements in AI, leading to high-resolution and detailed images.
3. Broad Coverage of Editing Operations: HQ-Edit covers a wide range of editing tasks, from global operations like weather changes to local object edits, demonstrating its versatility.

### Weaknesses
1. Synthetic Data Limitations: Although the synthetic images are useful for training, the trained model may not perform well on real images.
2. Constrained Persuasiveness of Evaluation Metrics:The paper only conducted comparisons on the two evaluation metrics it proposed, Alignment and Coherence, without making comparisons on more widely used and popular metrics.

### Questions
1. In the appendix, the provided image scores suggest that GPT-4V seems to give higher coherence scores to synthetic images. The paper does not substantiate that higher coherence scores from GPT-4V are indicative of higher human ratings.This raises the question of whether GPT-4V is 、capable of assessing image quality accurately.
2. Both evaluation metrics have a scale ranging from 0 to 100. Even though there are some segmented scoring prompts, can GPT-4V truly differentiate between scores that are a few points apart?

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
This paper introduces HQ-Edit for instruction-based image editing task, which contains around 200,000 edits. By using GPT-4V and DALL-E 3, the author develops a scalable data collection pipeline, which consists of three steps (Expansion, Generation and Post-Processing). Besides, the author also designs two metrics by using GPT-4V, which own a higher correlation to human preference compared to the CLIP score. Based on HQ-Edit, the finetuned InstructPix2Pix can attain state-of-the-art image editing performance.

### Strengths
1. The proposed HQ-Edit can be a training data for instruction-based image editing task, which can promote the development of this area. 
2. The performance of finetuned InstructPix2Pix has proven the effectiveness of HQ-Edit.
3. The proposed evaluation metrics are superior to the CLIP score.

### Weaknesses
1. It seems that HQ-Edit only contains non-rigid pair data.
2. There are many types of operations for instruction-based image editing task (e.g., object addition, object removal, non-rigid operation, local transformation, global transformation). The figure 8 and figure 9 only show the transformation part. The author should show the results of all these operations to make a comprehensive comparison.
3. Regarding metrics, since there is already a large amount of Human Evaluation Scores, why not use a model to learn a reward model for scoring? Alternatively, we could fine-tune an MLLM for scoring. Simply using prompt engineering with GPT-4V may not be the best option.
4. There is some typo in line 512 – 516. It should be “enhances its performance on Coherence but hurts alignment”.
5. What is the architecture of InstructPix2Pix trained by the author? (Base or XL). If the XL version is finetuned with HQ-Edit, can the performance be further improved than Base version?

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a large-scale and high quality instruction-based image editing dataset, HQ-Edit, with 200K edits. This dataset contributes to address the challenges of low instruction alignment and quality in image editing by employing GPT-4, DALL-E3 and GPT-4V. This work presents a new and scalable data curation process involving with crafted prompt engineering and diptych generation. The authors also present baselines finetuned on HQ-Edit which shows state-of-the-art performance in instruction-based single image editing task.

### Strengths
1. Compared with previous work, the proposed dataset is high quality regarding to image resolution, content/edit-type diversity and prompt-image alignment.
2. Baseline method InstructPix2Pix finetuned on the proposed HQ-Edit achieves state-of-the-art performance.
3. The proposed data curation pipeline is scalable by leveraging pretrained generative model (e.g. DALL·E3) and visual-language model (e.g. GPT-4 / GPT-4V).
4. The paper is well-organized and easy to follow.

### Weaknesses
1. Compared with previous work, e.g. MagicBrush, the source images from HQ-Edit are generated by DALLE-3, which may introduce distribution bias between AIGC and photo realistic contents.
2. The necessity/importance analysis of using diptych generation is missing.
3. The proposed two metrics Alignment and Coherence are mainly used in the main evaluation. Given the validated limitation of CLIP directional similarity, other commonly used metrics are missing for quantitative evaluation, which may lead to unfair comparison.

### Questions
Referring to weakness 1, can you provide an analysis or design an experiment to show the potential generalization ability of the finetuned baseline model on photo-realistic data?

### Soundness
3

### Presentation
3

### Contribution
3
