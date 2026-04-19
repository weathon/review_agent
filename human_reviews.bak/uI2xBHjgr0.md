# Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
Multi-modal Large Language Models (MLLMs) have shown remarkable capabilities in many vision-language tasks. Nevertheless, most MLLMs still lack the Referential Comprehension (RC) ability to identify a specific object or area in images, limiting their application in fine-grained perception tasks. This paper proposes a novel method to enhance the RC capability for MLLMs.
Our model represents the referring object in the image using the coordinates of its bounding box and converts the coordinates into texts in a specific format. This allows the model to treat the coordinates as natural language.
Moreover, we construct the instruction tuning dataset with various designed RC tasks at a low cost by unleashing the potential of annotations in existing datasets.
To further boost the RC ability of the model, we propose a self-consistent bootstrapping method that extends dense object annotations of a dataset into high-quality referring-expression-bounding-box pairs.
The model is trained end-to-end with a parameter-efficient tuning framework that allows both modalities to benefit from multi-modal instruction tuning. This framework requires fewer trainable parameters and less training data.
Experimental results on conventional vision-language and RC tasks demonstrate the superior performance of our method. For instance, our model exhibits a 12.0\% absolute accuracy improvement over Instruct-BLIP on VSR and surpasses Kosmos-2 by 24.7% on RefCOCO_val under zero-shot settings. We also attain the top position on the leaderboard of MMBench. We will release the models, datasets, and codes for further research.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors argue that existing Multimodal Large Language Models (MLLMs) lack the referential comprehension (RC) ability, i.e., identifying a specific object or area in images. Thus, they propose to represent the referring object with the coordinates of its bounding box and convert the coordinates into texts in a specific format. By regarding the coordinates as natural language, they can use a small instruction tuning dataset to transfer the knowledge in pretrained models. Furthermore, they propose a self-consistent bootstrapping method to extend dense object annotations into high-quality query-box pairs. The model is trained with a parameter-efficient tuning framework. Results show that the whole pipeline can significant improve the referential comprehension ability of MLLMs.

### Strengths
+ The problem of improving the referential comprehension ability of MLLMs is a very important topic. This work puts more emphasis on this worth exploring direction.

### Weaknesses
+ The novelty of the whole pipeline is limited. Although many techniques are used, all techniques are well-studied and straightforward. For example: 1) using parameter efficient training to avoid overfitting; 2) building instruction dataset for instruction tuning; 3) transforming bounding boxes into coordinates into the text. The only "new" thing may be the self-consistent bootstrapping method. From my understanding, it looks more like a trick to filter data. Overall, I think the whole contribution is very limited.

### Questions
Based on the results in Table 2, the Pink model without * (without generated query-box pairs) shows very limited performance gains. It would be better to have more explanations to demonstrate the effectiveness of the proposed architecture. Otherwise, it feels like that the main performance gains come from the generated new dataset (i.e., Object365 with generated pairs).

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a method to enhance the referential comprehension (RC) ability to identify a specific object or area in images for multimodal large language models. The proposed method constructs the instruction tuning dataset with various designed RC tasks at a low cost by unleashing the potential of annotations in existing datasets. The proposed method achieves good performances on the public datasets.

### Strengths
This paper proposes a method to treat the coordinates as natural language by representing the referring object in the image using the coordinates of its bounding box and converting the coordinates into texts in a specific format. The model is trained end-to-end with a parameter-efficient tuning framework that allows both modalities to benefit from multi-modal instruction tuning.

### Weaknesses
The novelty  is limited. The proposed methods convert the coordinates into texts in a specific format. This idea is widely adopted in the multimodal large language models for  specific objects, e.g.  [GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest], [VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks].  The proposed methods just seem like the tricks of the object coordinates of the MLLMs

### Questions
1. Please highlight the contribution and novelty of the proposed methods.
2. Please add more details of the proposed adapters comparing the finetuning, LoRA and etc.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents Pink 1, a novel Multi-modal Large Language Model (MLLM) that enhances the Referential Comprehension (RC) capabilities of MLLMs. 

Pink 1 leverages an existing method where the referring object in an image is represented using the coordinates of its bounding box, which are then converted into text in a specific format. This allows the MLLM to treat the coordinates as natural language.

The authors also propose a unique method to construct an instruction tuning dataset with a diverse range of RC tasks using annotations from existing datasets. This method allows for low-cost dataset construction.

Further, a self-consistent bootstrapping method is introduced to extend dense object annotations into high-quality referring-expression-bounding-box pairs. Pink 1 is trained end-to-end with a parameter-efficient tuning framework, resulting in fewer trainable parameters and less training data.

The experimental results demonstrate the superior performance of Pink 1 on both conventional vision-language tasks and RC tasks, highlighting the potential of this approach.

### Strengths
1. Solid experimental results that validate the proposed model's superior performance on both conventional vision-language tasks and RC tasks. The authors also provided a detailed comparison with other models under the fine-tuning setting, demonstrating the effectiveness of their model.

2. While the method of representing the referring object in an image using the coordinates of its bounding box is not new, the authors introduced innovative methods such as a unique instruction tuning dataset construction and a self-consistent bootstrapping method.

3. The paper is well-written and organized, providing clear definitions and explanations of the proposed model and methodologies.

### Weaknesses
1. As acknowledged by the authors, the Pink utilizes the approach of converting bounding box coordinates into text to understand the location of objects within an image. This technique, while not novel, could potentially impose limitations on the model's ability to perceive fine-grained details, particularly in complex images with numerous or overlapping objects.

2. The novel instruction tuning dataset construction method and the self-consistent bootstrapping method proposed in this study are innovative. However, their effectiveness is largely dependent on the quality and diversity of the existing datasets used. There might be limitations when dealing with less common or more complex RC tasks not covered in the existing datasets.

3. Although the model performs well on the tested datasets, it's unclear how well it would generalize to other types of RC tasks or datasets that are more complex or have different characteristics.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
