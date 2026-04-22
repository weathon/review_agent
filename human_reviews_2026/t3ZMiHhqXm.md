# Person-Centric Annotations of LAION-400M: Auditing Bias and Its Transfer to Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Vision-language models trained on large-scale multimodal datasets show strong demographic biases, but the role of training data in producing these biases remains unclear. A major barrier has been the lack of demographic annotations in web-scale datasets such as LAION-400M. We address this gap by creating person-centric annotations for the full dataset, including over 276 million bounding boxes, perceived gender and race/ethnicity labels, and automatically generated captions. These annotations are produced through validated automatic labeling pipelines combining object detection, multimodal captioning, and finetuned classifiers. Using them, we uncover demographic imbalances and harmful associations, such as the disproportionate linking of men and individuals perceived as Black or Middle Eastern with crime-related and negative content. We also show that 60-70\% of gender bias in CLIP and Stable Diffusion can be linearly explained by direct co-occurrences in the data. Our resources establish the first large-scale empirical link between dataset composition and downstream model bias.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper annotates the images of LAION 400M by providing bounding boxes of person detections, inferred gender and race labels, as well as detailed captions for each detected person. The annotations reveal significant issues: males and people with Middle Eastern or Black appearances are more strongly correlated crime-related content or negative sentiment, a lot of which is shown to propagate into the models trained on the studied dataset. The authors also study the themes of captions concerning the racial and gender combinations, and uncover interesting patterns.

### Strengths
1. The paper provides gender and racial annotations for images in LAION-400M, which is a significant contribution given that LAION400M is web-scraped.
2. Instead of using off-the-shelf gender and race classifiers, authors particularly finetune models (e.g., SIGLIP for gender) to make them more aware of the domain, and to handle female, male, mixed and unclear cases (and the equivalent for race).
3. Analysis reveals more association of males and Middle-Eastern/Black races with crime and negative sentiments
4. Authors also attempt to tie dataset bias with model bias (CLIP and Stable Diffusion), and find a significant overlap.

### Weaknesses
1. The paper examines LAION400M, which was used to train earlier versions of OpenCLIP and Stable Diffusion, hardly used now. The biases are expected to grow with scale, as emphasized by previous papers [a], and a similar experiment on LAION 2B, DataComp, etc would have helped us analyse biases in more modern models. 
2. The identity-topic associations are dependent on the automatically generated captions from pretrained caption generators. Such models may carry their own biases, and it is hard to say if the generated captions are accurate. Similarly, topic analysis on the original captions may have revealed more patterns.
3. The authors study biases in downstream models via social categories. Crime-related and Sentiment-based analysis would have been valuable too. It often does not guarantee that presence of social category c in captions would ensure that the persons present in the corresponding images actually belong to category c, due to misalignment issues [b].
4. The authors mention that 60-70% of the biases in downstream models can be explained by those in the datasets. However, they do not discuss what leads to the rest of the biases, especially in Stable Diffusion.

[a] Birhane et al., 'Into the LAION’s Den: Investigating Hate in Multimodal Datasets', NeurIPS D&B 2023

[b] Udandarao et al., 'No “Zero-Shot” Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance', NeurIPS 2024

### Questions
1. What if the gender and race classifiers were not finetuned, and instead the ensemble VLMs were used to annotate the entire dataset? Do the authors avoid it for computational costs? What are their thoughts on analysis on some other dataset like CC-12M or DataComp - is a separate finetuning required for those cases too?
2. How do the authors verify that YOLOv11-l does not have gender/racial biases?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a large-scale study that adds person-centric annotations to the LAION-400M dataset. It automatically labels each detected person with perceived gender and race or ethnicity using a combination of object detection models and MLLMs. The paper then analyzes demographic distributions, harmful associations such as links to crime or negative sentiment, and the relationship between dataset bias and model bias in CLIP and Stable Diffusion. The main finding is that about 60–70% of model bias can be explained by co-occurrence patterns in the data.

### Strengths
- The paper tackles an important and timely topic.

- While prior work has provided protected-attribute annotations for smaller datasets such as COCO or GCC, this paper scales the effort to LAION-400M, a much larger and more representative dataset, aligning with recent trends in large-scale data research.

- The analysis is comprehensive and provides multiple insights. For example, the paper uncovers detailed demographic distributions, harmful associations, and correlations between data composition and model outputs.

- The study linking dataset bias and model bias is particularly interesting. The finding that a significant portion of model bias can be explained by dataset co-occurrences highlights the need for the community to address dataset-level bias more seriously.

### Weaknesses
- There is a potential risk that biases from the MLLMs used for demographic attribute annotation and caption generation propagate into the resulting annotations. For instance, if these models make more errors for certain genders or races, their biases may directly influence the final dataset. Although the paper validates agreement with human-labeled datasets, three concerns remain:

1. The agreement is relatively low for race annotations, and it is questionable to dismiss this easily. If certain racial groups have higher error rates, the final demographic distribution could diverge significantly from reality.

2. Relatedly, the paper only reports aggregate error rates but does not analyze error trends. It would be important to know whether errors are uniformly distributed or concentrated on specific groups, as this strongly affects the reliability of MLLM-based annotations.

3. No human study was conducted to verify the quality of the obtained annotations. While comparison with datasets such as FACET provides a proxy for human validation, a small-scale human study (even around 1K samples) on LAION-derived annotations would greatly strengthen confidence in their accuracy.

- In the dataset–model bias correlation experiment, the paper mentions that “the remaining 30–40% of bias stems from nonlinear or higher-order effects,” but does not provide any quantitative or qualitative analysis of these effects. Since this bias-transfer analysis is one of the central contributions, including at least some empirical investigation or hypothesis testing for these unexplained components would significantly reinforce the paper’s impact.

### Questions
**Overall assessment and suggestions**

This paper presents an interesting and valuable attempt to provide protected-attribute annotations and large-scale demographic analysis for a dataset of the scale of LAION-400M. The topic is timely and important, and the effort to enable systematic auditing of web-scale data is commendable.

However, I believe the paper does not sufficiently analyze (or mitigate) the potential biases introduced by the automatic annotation pipeline itself, especially those arising from the MLLMs used for labeling gender and race. Given that this work focuses on human-centric annotations, such limitations are critical and cannot be easily overlooked.

If the rebuttal provides a convincing analysis or additional validation addressing this issue, I would be happy to raise my score.

### Soundness
2

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
This work aims to examine demographic patterns in a large web dataset and how they reflect in downstream generative systems. The paper augments LAION-400M image-caption dataset by including person-level annotations, encompassing bounding boxes around individuals, with automatically inferred gender and ethnicity labels, and detailed captions for each detected person.

With this annotated dataset, the paper examines how different gender and ethnic groups are represented and how these identities intersect with themes like crime, sentiment, and broader contextual associations. They also explore how gender-related biases in the dataset correlate to biases in two models trained on LAION-400M, CLIP and Stable Diffusion.

### Strengths
* The study proposes an interesting methodology to examine demographic biases in pretraining datasets and their effects on generative models.
* The scope of this work is impressive, both in its scale and in its comprehensive examination of demographic patterns and biases within the dataset.
* The paper presents important findings on how sentiments and topics are associated with gender and ethnic identities, and to what extent gender bias in model generations correlates with biases in the training dataset.
* The additional annotations might facilitate future research, such as studying other forms of dataset-model interaction

### Weaknesses
* While the paper ambitiously combines large-scale demographic annotation with multiple layers of analysis, its broad scope makes the presentation overly dense. As a result, some key details and justifications are underdeveloped or omitted from the main paper, limiting clarity and depth in certain areas. 
* To generate gender and ethnicity labels, the paper fine-tunes a SigLIP classifier using a subset of the LAION-400M data labelled by three different MLLMs. The labelling process relies on consensus among these models, with training (and testing) data primarily drawn from images where all agreed. While this approach enhances label reliability, it may bias the classifier toward clear-cut examples and limit its robustness to the ambiguous or noisy cases.

Minor typos:
Line 52 mentions the word “intersectionalidentity”
Line 291, the caption says “compound score (orange)”, color is not exactly orange

### Questions
Some suggestions: 

* The paper uses YOLOv11-l for bounding box generation, relying on evaluations from datasets such as FACET and PHASE. Since LAION is a much noisier web-based dataset with the possibility of multiple people per image, assessing the detector’s accuracy on a subset of LAION would strengthen the work and ensure reliability in this context.

* The paper mentions a qualitative analysis of MLLMs over 4,939 bounding boxes to select one model for generating person-specific captions; however, the description of this process is somewhat vague. Providing more detail on this selection procedure and, if feasible, extending similar qualitative analysis to the gender and ethnicity detection pipelines would strengthen the reliability of the overall methodology.

I am willing to reconsider my assessment based on authors' response to the above. 

* (Minor): it would be good to hear authors' take on where they think the annotations could be used in the future

### Soundness
2

### Presentation
2

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
The paper generated person-centric annotations over LAION-400M, resulting in 270M odd detected person bounding boxes, 200M perceived gender and race/ethnicity labels (after filtering), and person-centric captions generated by MLLMs. The labels are used to quantify the level of demographic imbalances, thematic associations (via sparse autoencoders), and quantify how much of observed gender bias in CLIP and Stable Diffusion can be linearly explained by dataset co-occurrences.

The paper quantifies the extent to which bias in downstream models can be attributed to biases in the dataset. Futhermore, the dataset can be useful for future studies to understand dataset-model interactions in propagating or amplifying biases in large-scale datasets. The reliability of the findings can be improved by auditing potential issues of bias in the proposed process (MLLM ensemble labeling -> classifier training -> dataset labeling -> bias estimation).

### Strengths
- While the prevalence of bias in large models, and its attribution to bias in training datasets is well known, there are no large-scale annotated datasets with demographic labels. So the proposed dataset could be a valuable resource for more fine-grained studies seeking to understand and mitigate bias in models trained on large models.

- The workflow (YOLOv11 person detection -> MLLM ensemble labeling -> SigLIP finetuning -> full-dataset labeling -> analyses) for automated labeling is quite reasonable, well described and should be reproducible in principle.

- The attempt to quantitatively relate dataset co-occurrence statistics to measured model biases (CLIP, Stable Diffusion) has been lacking and this paper fills the gap.

### Weaknesses
There are several weaknesses in the proposed methodology, which reduces the reliability of the quantitative findings.

- Labeling relies on an MLLM ensemble consensus. However, these MLLMs may have inherent biases that would now propagate through the rest of the method. There is no analysis on the errors and biases of the MLLM ensemble. Similarly, bias in the pre-trained and fine-tuned SigLIP has not been analyzed.

- All the presented results in the paper are point estimates. How reliable are these estimates? Confidence intervals are missing. Similarly, how sensitive are the correlation estimates to hyperparameter choices in the full pipeline?

- Statements like “60–70% of gender bias in CLIP and Stable Diffusion can be linearly explained by direct co-occurrences in the data” are stronger than warranted. The result shows a strong *correlation* but not necessarily *causation*. These claims need to either be substantiated or rephrased.

### Questions
- In cases where the MLLM ensemble agree or disagreed, how do they relate to specific demographic groups?

- Bias analysis of the gender and race classifiers.

- To understand the reliability and robustness of the claims, confidence intervals and sensitivity to hyperparameters.

### Soundness
3

### Presentation
3

### Contribution
3
