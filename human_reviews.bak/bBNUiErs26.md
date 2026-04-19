# $\text{I}^2\text{AM}$: Interpreting Image-to-Image Latent Diffusion Models via Bi-Attribution Maps

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6, 5

## Abstract
Large-scale diffusion models have made significant advances in image generation, particularly through cross-attention mechanisms. While cross-attention has been well-studied in text-to-image tasks, their interpretability in image-to-image (I2I) diffusion models remains underexplored. This paper introduces Image-to-Image Attribution Maps $(\textbf{I}^2\textbf{AM})$, a method that enhances the interpretability of I2I models by visualizing bidirectional attribution maps, from the reference image to the generated image and vice versa. $\text{I}^2\text{AM}$ aggregates cross-attention scores across time steps, attention heads, and layers, offering insights into how critical features are transferred between images. We demonstrate the effectiveness of $\text{I}^2\text{AM}$ across object detection, inpainting, and super-resolution tasks. Our results demonstrate that $\text{I}^2\text{AM}$ successfully identifies key regions responsible for generating the output, even in complex scenes. Additionally, we introduce the Inpainting Mask Attention Consistency Score (IMACS) as a novel evaluation metric to assess the alignment between attribution maps and inpainting masks, which correlates strongly with existing performance metrics. Through extensive experiments, we show that $\text{I}^2\text{AM}$ enables model debugging and refinement, providing practical tools for improving I2I model's performance and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes a method called Image-to-Image Attention Mapping (I2AM) for improving the explainability of diffusion models. I2AM visualizes bidirectional attention mapping between reference and generated images to understand how the model processes image data. This method is particularly useful for image-to-image (I2I) tasks since it can capture complex relationships between reference and generated images. Furthermore, the authors introduce a new metric called Image Repair Attention Consistency Score (IMACS) to evaluate model performance and attention distribution consistency

### Strengths
1. The proposed method offers a novel perspective on interpreting LDMs by focusing on the relationship between reference and generated images. 

2. The authors also introduce a new metric, Image Repair Attention Consistency Score (IMACS), to evaluate model performance and attention distribution consistency, which adds value to the research.

3. The experimental results demonstrate the effectiveness of I2AM in identifying key feature regions for tasks such as object detection, image repair, and super-resolution.

4. The paper is written in a clear and concise manner.

### Weaknesses
1. The paper does not include an ablation study to demonstrate the effectiveness of the core components of the proposed method. 

2. The authors compare their method with several baseline models, but they do not mention any other attention-based methods or explain why they chose those specific models. Adding comparisons with other attention-based methods would help to understand the relative strengths and weaknesses of the proposed approach.

3. The authors do not discuss the limitations of their method.  Does the method work well for any diffusion models?

### Questions
See Weaknesses.  I will change my rating based on the responses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method to improve the interpretability of Image-to-Image (I2I) models by visualizing bidirectional attribution maps and proposing a new evaluation metric. The introduced Image-to-Image Attribution Maps ($I^2AM$) aggregate cross-attention scores across time steps, attention heads, and layers, providing insights into how key features transfer between images. Experimental results indicate that this method, enables model debugging and refinement, offering practical tools to enhance I2I model performance and interpretability.

### Strengths
- The paper introduces a well-designed method that improves interpretability in I2I models.

- The analysis of Image-to-Image Attribution Maps is interesting, demonstrating the utility of $I^2AM$ across tasks like object detection, inpainting, and super-resolution.

- The writing and presentation are generally clear and accessible, making it easier for readers to follow the proposed method.

### Weaknesses
- In the "Unsupervised manner & Unseen dataset" section of Table 1, a baseline comparison is missing. Applying [1]'s attribute map to identify key regions would provide a fair comparison and strengthen the analysis.

- Additional comparative experiments and visualizations would benefit the paper. For instance, more qualitative examples in tasks such as object detection, image inpainting, and super-resolution would enhance understanding of the method's effectiveness.

- The Specific-reference Attribution Map (SRAM) lacks formal definitions. It remains unclear how reference patch selection and attention aggregation are conducted, even with the explanations in Figure 4.

-  The choices for parameters δ and λ are not adequately justified through experiments. Additional ablation studies or experiments exploring the effect of varying δ and λ values would strengthen the argument for their selection, making the methodology more transparent and grounded.

[1] Tang, Raphael, et al. “What the DAAM: Interpreting Stable Diffusion Using Cross Attention.” Annual Meeting of the Association for Computational Linguistics (2022).

### Questions
Please refer to the Weaknesses for more details.

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
3

### Summary
The interpretability of cross-attention mechanisms in image-to-image (I2I) diffusion models remains underexplored, while cross-attention maps in text-to-image (T2I) models have been extensively studied. This paper advances interpretability in I2I models by visualizing attributions from two perspectives: from the reference image to the generated image and from the generated image back to the reference. The authors propose a novel method for interpreting latent diffusion models using cross-attention maps from these dual perspectives. This approach enables researchers to debug and enhance I2I models in various tasks, such as object detection, inpainting, and super-resolution. Additionally, they introduce an inpainting mask attention consistency score as an evaluation metric.

### Strengths
- This method visualizes bidirectional attribution maps across multiple axes, including diffusion time steps, attention heads, and layers, offering a comprehensive analysis. It also reveals insights into the coarse-to-fine process as layers deepen, enhancing interpretability. 

- Additionally, the paper introduces a specific-reference attribution map tailored for inpainting tasks, helping guide the model to focus on relevant object parts while disregarding irrelevant background information from the reference image.

- The authors conduct comprehensive experiments across various tasks to validate the improvements enabled by these visualization techniques.

### Weaknesses
- In ULAM, the column-wise averaging limits the ability to visualize specific region-to-region mappings. For instance, in virtual try-on applications, it’s challenging to verify whether a logo in the generated image aligns with the logo in the reference image. According to the authors' visualization in Figure 1, the reference image’s logo appears to contribute less to the generated image, making detailed correspondence difficult to interpret.

- Additionally, a question arises regarding the visualization approach: given a CLIP image embedding, how can each token be accurately mapped back to the original reference image? It remains unclear whether using a fixed CLIP image embedding alone is sufficient for effective visualization.

- Lastly, some formatting issues were noted, such as Figure 3 appearing before Figure 2.

### Questions
My main concern is that it remains unclear whether using a fixed CLIP image embedding alone is sufficient for effective visualization, as demonstrated in Figure 1. I would appreciate if the authors could provide further clarification on this point, as it may influence my overall rating.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper describes  a way to aggregate and visualize cross-attention similarity maps in image-to-image latent diffusion models. Those visualizations help to understand which regions of the generated image were influenced most by the reference image, and which parts of the reference image contributed most to the generated image. The paper also proposes a new metric for evaluating reference-based image inpainting through estimating how well the inpainting model captures details from the reference image. They perform experiments on tasks like image inpainting and image super-resolution in order to show the visualization capabilities of the proposed method.

### Strengths
The proposed method for cross-attention similarity map visualizations is clearly explained.

### Weaknesses
1. The paper’s contribution appears incremental, focusing on the adaptation of existing methods to a new context. It mostly describes a way to aggregate and visualize cross-attention similarity maps of a pre-trained image-to-image diffusion denoiser network by averaging across different dimensions. Similar approaches have already been employed by other papers for visualizing the cross-attention layers, as also discussed in the related work of this paper. So the main novelty of this work is the application of the same principles of cross-attention map visualizations to the image-to-image diffusion models. Therefore, the significance of this work to the field is quite limited.

2. The head-level attribution map introduced in Section 4.3 may not be fully meaningful, as it averages across layers. It’s unclear why a given head index should play a consistent role across all layers or how its visualization provides useful information. The authors could consider clarifying this assumption.

3. Inpainting Mask Attention Similarity Score (IMACS) in Section 4.4, Equation 8, appears to yield a matrix, but it’s unclear how this is transformed into a scalar score. Additional clarification on this process would improve understanding.

4. Figure 2 needs more descriptive labeling. Terms such as c_I^{(l)}/c_I​ should be clearly defined within the figure caption or the main text, as their meaning is not immediately obvious.

5. Lines 304-305 state, “the layer-level maps offer a better indicator of model performance across different layers and help uncover areas for potential improvement.” It’s not clear how a layer-level attribution map can serve as a performance indicator. Additional explanation or evidence supporting this claim would be helpful.

### Questions
1. The "right map" in Figure 1 is difficult to interpret—could the authors clarify what it represents? Additionally, there appears to be an artifact in the first row, second column of the right 3x3 grid; what caused this?

2. In the experiment described in Section 5.4, what loss functions were used to "densify attention distributions" and "improve semantic alignment"? A description of these would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a method called Image-to-Image Attribution Maps (I2AM) that enhances the interpretability of image-to-image (I2I) diffusion models by visualizing bidirectional attribution maps between the reference and generated images. I2AM aggregates cross-attention scores across time steps, attention heads, and layers, providing insights into feature transfers between images. It has been demonstrated to be effective in tasks like object detection, inpainting, and super-resolution, successfully identifying key regions for output generation even in complex scenes. Additionally, the paper introduces the Inpainting Mask Attention Consistency Score (IMACS), a new metric that assesses the alignment between attribution maps and inpainting masks, correlating strongly with existing performance metrics. Extensive experiments show that I2AM aids in model debugging and refinement, offering practical tools to improve the performance and interpretability of I2I models.

### Strengths
1. The paper introduces Image-to-Image Attribution Maps (I2AM), a novel method for enhancing the interpretability of image-to-image (I2I) diffusion models. I2AM visualizes bidirectional attribution maps, providing a clear view of feature transfers between reference and generated images.
2.Task Effectiveness: Demonstrates I2AM's effectiveness in object detection, inpainting, and super-resolution tasks.  Successfully identifies critical regions for output generation, even in complex scenes.
3.IMACS Metric: Introduces the Inpainting Mask Attention Consistency Score (IMACS), a new evaluation metric correlating with existing performance metrics.

### Weaknesses
1.The IMACS metric does not seem to be very relevant to the topic of this article, I2AM, which seems to be designed specifically for Inpainting and has no general use for other T2I tasks.
2.Some of the tasks selected in this article do not seem to be the most representative of I2I tasks, such as generative Inpating, image editing, image super-resolution, etc., so I have some concerns about the reliability and robustness of the conclusions in this article.

### Questions
1.The explanation in Figure 3 (b) is somewhat puzzling, why say "low-resolution signals result in lower involvement from early layers in controlling coarse features. " does not seem common sense, and why there is only one type of visualization in (a) and two in (b)?
2.Why did you choose PASD as the SR model? In fact, there are many influential generative SR models in the recent top conferences, like SUPIR, SeeSR. Will your approach come to the same conclusion on these models?

### Soundness
3

### Presentation
3

### Contribution
2
