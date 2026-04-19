# Searching for Parameter-Efficient Tuning Architecture for Text-to-image Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5

## Abstract
The large-scale text-to-image diffusion model, represented by Stable Diffusion, has achieved remarkable success in the field of image generation.  Transferring pretrained diffusion models to downstream domains with parameter-efficient tuning (PEFT)  methods such as Adapter and LoRa have become the most common paradigms. Despite their widespread usage, there has been limited research on systematically studying how the design of these components would impact the final tuning effectiveness.
In this paper, we investigate the automatic design of an optimal tuning architecture. Specifically, we employ a reinforcement learning-based neural network search method to facilitate the automatic design of the tuning architecture for PEFT of Stable Diffusion with few-shot training data. Our search space includes micro-structures similar to Adapter, LoRa, as well as their insertion positions. 
For effective searching and evaluation, we build a large-scale tuning dataset. Through our search, we successfully obtained a novel tuning architecture that reduces parameter count by 18\% compared to the widely adopted LoRa approach but still surpasses across various downstream tasks hugely.   We also conduct extensive analysis of the searched results, aiming to provide valuable insights to the community regarding parameter-efficient tuning for large-scale diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The author proposed a NAS framework specifically for the plug-in structures in the stable diffusion U-Net model. The search space is limited to whether to include the adapter and LoRa structure and a couple of hyperparameters in these two structures.  Existing RL-based optimization is adopted. A comparison of the optimized model (after searching) to models containing adapter and LoRa alone is performed, and superior results of the model obtained by the proposed method are reported. Several relatively small datasets are employed in the experiments for the few-shot setting, and only one face dataset is adopted for the fine-tuning setting. The paper is overall easy to follow, while several critical concerns are detailed below.

### Strengths
+ The manuscript is easy to follow
+ NAS on plug-in structure for diffusion U-Net is new
+ Superior results of the optimized model structure are reported in comparison to models with vanilla adapter and LoRa.

### Weaknesses
- The scope of the paper is small, where only adapter and LoRa are considered in the paper, and the model architecture is limited to diffusion U-Net. Is there any other plug-in structure that should be considered? And for the adapter and LoRa structure themself, only a couple of parameters are considered in the search space. How about other variable parts of the adapter and LoRa, e.g., the weights W? 
- The RL-based searching method is adopted. How about other search strategies?
- There are several datasets employed for the few-shot setting. Why only one dataset is considered for the fine-tuning setting?
- Other existing NAS methods should be included in the comparison study.

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

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
In this paper, the authors investigate the automatic design of an optimal tuning architecture. They employ a reinforcement learning based neural network search method to facilitate the automatic design of the tuning architecture for PEFT of Stable Diffusion with few-shot training data.

### Strengths
Through the proposed method, it was successfully obtained a novel tuning architecture that reduces parameter count by 18% compared to the widely adopted LoRa approach but still surpasses across various downstream tasks hugely. The authors conduct extensive analysis of the searched results.

### Weaknesses
1.	Insufficient innovation. The work in this article seems to be just fine-tuning on the original model, and the innovative work is not clear. It is recommended to re-elaborate in the abstract, introduction, and conclusion parts.
2.	Abstract writing is problematic. The abstract is recommended to be developed in the order of background, goals, methods, results, and conclusions. In another way, what is the background of the question? What work did the predecessors do? What's wrong with their job? What do you plan to achieve in this work? How did you go about achieving your goals? What are the main findings of the study? What is the conclusion?
3.	Absence of methodological details. While the article mentions a "novel tuning architecture," it fails to provide any specifics about the architecture, training process, or key innovations. Details such as the type of neural network used, training data preprocessing, and the mechanism for generating sparse labels are crucial to assessing the method's novelty and reliability.
4.	Some sentences are vague. For example, the claim of " there has been limited research on systematically studying how the design of these components would impact the final tuning effectiveness" lacks context – it's essential to specify how this comparison was made and against what reference.
5.	The methods part does not have enough mathematical formulas to support, and the innovation cannot be seen.
6.	Some charts are problematic. Such as fig.7 is too large, please reduce the image size so that one image takes up almost the entire page. And Fig.3 is confusing, it’s not clear which part is the work of this paper.
7.	The article does not mention whether the proposed deep learning method is fully reproducible. Lack of information about code availability, model architecture, hyperparameters, and data preprocessing steps could hinder the ability of other researchers to replicate the results.
8.	The article does not mention whether efforts were made to interpret or explain the model's decisions.
9.	Lack of limitations. The article does not discuss any limitations of the proposed method or the study itself. Addressing potential shortcomings, such as biases in data collection, limitations of the model architecture, or challenges in real-world deployment, demonstrates a comprehensive understanding of the research's scope.
10.	There is a problem of cluttering references. Please check the article thoroughly to eliminate all cluttered and uncited references. This should be achieved by describing each reference individually. This can be done by mentioning 1 or 2 phrases in each citation to show how it differs from the others and why it deserves a mention.

### Questions
1.	Insufficient innovation. The work in this article seems to be just fine-tuning on the original model, and the innovative work is not clear. It is recommended to re-elaborate in the abstract, introduction, and conclusion parts.
2.	Abstract writing is problematic. The abstract is recommended to be developed in the order of background, goals, methods, results, and conclusions. In another way, what is the background of the question? What work did the predecessors do? What's wrong with their job? What do you plan to achieve in this work? How did you go about achieving your goals? What are the main findings of the study? What is the conclusion?
3.	Absence of methodological details. While the article mentions a "novel tuning architecture," it fails to provide any specifics about the architecture, training process, or key innovations. Details such as the type of neural network used, training data preprocessing, and the mechanism for generating sparse labels are crucial to assessing the method's novelty and reliability.
4.	Some sentences are vague. For example, the claim of " there has been limited research on systematically studying how the design of these components would impact the final tuning effectiveness" lacks context – it's essential to specify how this comparison was made and against what reference.
5.	The methods part does not have enough mathematical formulas to support, and the innovation cannot be seen.
6.	Some charts are problematic. Such as fig.7 is too large, please reduce the image size so that one image takes up almost the entire page. And Fig.3 is confusing, it’s not clear which part is the work of this paper.
7.	The article does not mention whether the proposed deep learning method is fully reproducible. Lack of information about code availability, model architecture, hyperparameters, and data preprocessing steps could hinder the ability of other researchers to replicate the results.
8.	The article does not mention whether efforts were made to interpret or explain the model's decisions.
9.	Lack of limitations. The article does not discuss any limitations of the proposed method or the study itself. Addressing potential shortcomings, such as biases in data collection, limitations of the model architecture, or challenges in real-world deployment, demonstrates a comprehensive understanding of the research's scope.
10.	There is a problem of cluttering references. Please check the article thoroughly to eliminate all cluttered and uncited references. This should be achieved by describing each reference individually. This can be done by mentioning 1 or 2 phrases in each citation to show how it differs from the others and why it deserves a mention.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into the exploration of large-scale text-to-image diffusion models, emphasizing the achievements of Stable Diffusion in image generation. Its main goal is to investigate the influence of component design on the performance of parameter-efficient tuning (PEFT) methods, notably Adapter and LoRa. By harnessing reinforcement learning-based neural network search techniques, the study aims to automate the optimal tuning architecture's design for PEFT, taking into consideration structures similar to Adapter and LoRa.

### Strengths
1. Researching how to reduce the training and transfer costs of diffusion models is highly meaningful, especially for tasks with limited data.
2. The research has achieved a groundbreaking tuning architecture that reduces parameters by 18% compared to the popular LoRa approach, demonstrating superior performance across various tasks.
3. The method's versatility has been validated across a wide range of data domains.
4. The paper is well-written with clear logic, making it easy to understand.

### Weaknesses
1. Limited references. Several works [1-3], which aimed at reducing the costs of diffusion models, were not cited. Notably, the motivation and design approach of this study bear similarities to the paper [1].

   [1] Xiang C, Bao F, Li C, et al. A closer look at parameter-efficient tuning in diffusion models. arXiv preprint arXiv:2303.18181, 2023.

   [2] Kim B K, Song H K, Castells T, et al. On Architectural Compression of Text-to-Image Diffusion Models. ICCV Demo Track, 2023.

   [3] Go H, Lee Y, Kim J Y, et al. Towards practical plug-and-play diffusion models, CVPR 2023.

2. There's a limited comparison with other search methods. The authors assert that the proposed reinforcement learning approach is efficient, but additional experiments are needed to compare it with existing search methods to validate its efficiency.

### Questions
1. The impact of the search samples on model performance was not discussed.
2. It would be desirable to see experiments demonstrating the method's generalizability in more domains, such as the medical field.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a reinforcement learning based architecture search method for parameter efficient finetuning of text-to-image diffusion model using few-shot training data. They have experimented on dreambooth and finetuning tasks, and observed improved performance with lower parameter count.

### Strengths
1. The idea of parameter efficient finetuning with reinforcement learning is interesting. 
2. Experimental results are somewhat promising.

### Weaknesses
1. The paper used reinforcement learning like a blackbox. Proper motivation, justification and details of using which particular optimization methods are employed are missing. More details/citations are required. 
 2. LoRa, Adapter - these are parameter efficient finetuning methods. Adding reinforcement learning based search methods seems helping marginally w.r.t performance, training time. Also, why searching for parameters helps in image quality is not clear to me.
3. Comparison of reinforcement learning based search methods w.r.t grid search/ combinatorial search method would be required.
4. The rationale of using Eq.3 is not clear, why the authors choose to use power law method instead of any other combination?
5. The writing need to be improved. E.g., “Dreambooth” task is very weird, it should be called “personalized few-shot finetuning”. Overall, the motivation, method, experiments are not easy to follow.

### Questions
see weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
