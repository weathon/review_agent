# DeStyle2Style: Scalable Destylization-Driven Data Generation for Artistic Style Transfer

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
DeStyle2Style introduces a novel approach to artistic style transfer by reframing it as a data problem. Our key insight is destylization, reversing style transfer by removing stylistic elements from artworks to recover natural, style-reduced counterparts. This yields DeStyle-100K, a large-scale dataset that provides authentic supervision signals by aligning real artistic styles with their underlying content. To build DeStyle-100K, we develop DestyleNet, a text-guided destylization model that reconstructs style-reduced natural images, and DestyleCoT-Filter, a multi-stage evaluation model that employs Chain-of-Thought reasoning to automatically discard low-quality pairs while ensuring content fidelity and style accuracy. Furthermore, we introduce BCS-Bench, a benchmark with balanced stylistic diversity and content generality for systematic evaluation of style transfer methods. Our results demonstrate that scalable data generation via destylization offers a reliable supervision paradigm, effectively addressing the fundamental challenge of lacking ground-truth data in artistic style transfer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reframes artistic style transfer as a data generation problem. First, the authors develop DestyleNet—a text-guided destylization model that reconstructs natural images with reduced artistic styles. Second, they leverage GPT-4o (equipped with a Chain-of-Thought strategy) to build DestyleCoT-Filter, which enables the creation of DeStyle-100K, a new high-quality dataset. Third, using this dataset for training, they propose DeStyle2Style (based on FLUX-Dev), which achieves style transfer results comparable to SOTA approaches. Finally, they introduce BCS-Bench, a benchmark featuring balanced stylistic diversity and content generality, designed for the systematic evaluation of style transfer methods.

### Strengths
+ Using destylization offers a new perspective for creating paired data for artistic style transfer, enabling more authentic and higher-quality supervision signals.

+ The high-quality DeStyle-100K dataset and BCS-Bench benchmark (if publicly released) would provide valuable resources for the research community.

+ Both qualitative and quantitative comparisons validate the effectiveness of the constructed dataset and the DeStyle2Style model.

### Weaknesses
- DestyleNet is trained via full fine-tuning on a mere 60K paired samples—data synthesized using existing style transfer methods. As the authors highlight in L358, destylization is inherently a challenging task; this limited training data may undermine DestyleNet’s robustness. A more critical concern arises from its training paradigm: relying on outputs of existing style transfer methods means DestyleNet essentially learns the average of these methods’ inverse processes. This leads to a question: how can biases inherent to the base style transfer methods be avoided? For example, some methods prioritize color transfer over other stylistic elements, while others tend to distort content—would DestyleNet then only mimic such behaviors (e.g., merely removing colors or altering content) instead of achieving true, comprehensive destylization? Compounding these issues, the paper also lacks clarity on how to measure and evaluate DestyleNet’s effectiveness: no quantitative results or relevant discussions are provided to validate whether DestyleNet actually fulfills its intended destylization purpose.

- The primary novelty of this paper likely lies in its destylization data construction pipeline and the resulting dataset. In contrast, the architectures of DestyleNet and DeStyle2Style are largely inherited from existing mainstream frameworks such as OmniControl and Flux-Kontext.

### Questions
Please see weaknesses above.

### Soundness
2

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
This paper presents DeStyle-100K, a large-scale, high-quality dataset for supervised image style transfer built via a novel destylization pipeline. Using a DestyleNet model to reverse stylization and a DestyleCoT-Filter for quality control, it provides 100K aligned triplets <de-stylized image, reference image, style image>. Trained on this dataset, FLUX achieves superior results, reframing style transfer as a data-centric task.

### Strengths
1.The proposed dataset includes 100K high-quality triplet data points, which is highly beneficial to the style transfer community.

2.The complete construction of the pipeline is advantageous to the community.

3.The visual results of DestyleNet and the trained FLUX are good, content images are in the real-world domain and the stylized results are satisfactory.

4.Compared to the similar work OmniStyle[1], this paper have more diverse style images, help model to have better generalization ability.

[1]OmniStyle: Filtering High Quality Style Transfer Data at Scale, CVPR 2025.

### Weaknesses
1.The DeStyle model is trained on a dataset processed by current image-driven style transfer models. The upper bound of these methods also represents the limitation of the DeStyle-100K dataset. How to solve it?

2.As mentioned in [1], the de-stylized image and reference image have the consistent structure, is this will result in limited domain for image-driven style transfer? For example for Abstract style, if so, how to solve it?

3.In your triplets, the style image are constructed from style similarity based on CSD model. For my opinion, CSD model is not reliable because it is trained on WIKIART (most style are oil paintings). Picking from this metric might cause the different style between reference image and style image. 

4.Does the proposed DestyleNet perform better than the destyle model in USO[1] that build on a powerful customization framwork UNO? Your proposed data curation pipeline is similar to USO.

[1]USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning

If the authors solve all my concerns, I'd love to raise my score.

### Questions
1. Could you provide more visual results compared to SOTA style transfer methods? Only 3 samples in Figure 6.

2. Could you provide the comparison with SOTA methods AlignedGen[1].

[1]AlignedGen: Aligning Style Across Generated Images, NIPS 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a data-centric method for image stylization to address its ill-posed nature, where the ground-truth stylized image is often absent for a specific content reference image. While previous work often relies on synthetic stylized images as learning targets, the authors propose to synthesize the input content image from a human-created artwork to construct paired data with authentic training objectives. This is achieved by training an image de-stylization model, named DestyleNet, that removes stylistic elements from artworks to recover natural images. Leveraging this model, the authors curate the DeStyle-100K dataset consisting of 100K <destyle image, style reference, target image> paired data. To ensure high data quality, they also propose an automatic pipeline called DeStyleCoT-Filter, using MLLM to filter out data pair with either content or style mismatch. Finally, they introduce an image stylization benchmark BCS-Bench with balanced style and content diversity.

### Strengths
- The motivation behind the proposed method is clear. Due to the absence of paired data, current model-centric image stylization methods rely on synthetic data for end-to-end training, where the stylized target images are largely generated and curated through sophisticated data processing pipelines. The authors propose to, instead of synthesizing the learning target, synthesize the input content image. Such approach ensures high-quality in the learned data distribution, while the artifacts in the synthetic content images can effectively augment input data to improve the robustness of the model. The entire process is reasonable.
- The proposed image de-stylization method is interesting. For training the de-stylization model DestylNet, the authors construct an inversed problem and leverage existing image stylization methods with the condition and target switched. Such a design is intuitive and effective.
- The proposed data curation pipeline is well designed. Both the dataset and benchmark will benefit future work in this field.

### Weaknesses
- In the construction of the DeStyle-100K dataset, the authors collect 10K artworks from the internet. To improve data diversity the authors additionally generate 150K stylized images using FLUX which is extremely unproportionate to the curated artwork data. Even after post processing by DestyleCoT-Filter, the total amount of 100K images means the synthetic data consists of **at least 90\%** of the training dataset. This imbalance stands in stark contrast to the purpose of DeStyle-100K and diminishes the effectiveness of the proposed method from my perspective.
- In line 275-276 Section 3.4, the authors use MLLM to analyze the style discrepancy in the synthesized data pairs, where the style of an image is decomposed into attributes like color palette, texture, lighting and rendering effects. However, there lacks a detailed explanation for this step. How was the decomposition operated on the image? Is it also processed by MLLM like GPT?
- In quantitative comparisons, the authors use the textual description of the style reference as a proxy to condition text-guided models like Qwen-Image-Edit. Such setting is potentially adversarial for these methods and thus risks of unfair comparison. Combined with the imbalanced training data, the evidence for supporting the effectiveness of the proposed method is quite limited.
- Limitation is not discussed. From my point of view an obvious limitation will be the absence of text prompts in the synthesized data, and in the proposed DeStyle2Style model.
- There are several typos, especially in the citation style that seems unaligned with ICLR author guidelines. For instance: line 205, 211, 240, 247.

### Questions
- To improve the data diversity of the DeStyle-100K dataset. Wouldn’t it be simpler to employ the 60K triplets training data of DestyleNet, replacing the stylized target with the original style reference image and vice versa? In this way we could have both: 1) authentic learning target, since the target is now reference style image; 2) diverse input content images, including both the original 10K natural image curated in DeStyle-100K dataset, as well as digital art in the 60K triplets.
- For artworks like abstract art, Piet Mondrian for example. The content of these images can not be clearly defined, yet in other stylization pipelines they are eligible for serving as reference. I wonder how the proposed method generalizes to such data.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
DeStyle2Style introduces a novel data-centric approach by reversing style transfer through "destylization" to create a large-scale, aligned dataset called DeStyle-100K.

It develops two key models, DestyleNet for reconstructing content images and DestyleCoT-Filter for automatic quality control, alongside a new balanced benchmark, BCS-Bench, for evaluation.

This paradigm of generating supervised data via destylization effectively overcomes the fundamental challenge of lacking ground-truth data in artistic style transfer.

### Strengths
1.The method is very clear and the description is reasonable. 
2.The dataset pipeline is quite clear. 
3.The results achieved are quite good.

### Weaknesses
The authors did not clearly explain why the reverse-constructed data pipeline achieves better performance—whether it is due to the advantages of the data itself or the enhancement brought by the reverse data construction. The ablation experiments here are not clear. Besides, the authors did not provide SDXL-based experimental evidence to verify the reliability of the data pipeline. The number of parameters here may be a more critical factor affecting performance. Additional explanations are needed.
There are some printing errors in Table 2; please note this. The authors do not seem to have considered more improvements to the framework. Are VAE features important?

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
