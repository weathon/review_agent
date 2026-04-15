# Retrieval-Based Video Language Model for Efficient Long Video Question Answering

- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
The remarkable natural language understanding, reasoning, and generation capabilities of large language models (LLMs) have made them attractive for application to video question answering (VQA) tasks, utilizing video tokens as contextual input. However, employing LLMs for long video understanding presents significant challenges and remains under-explored. The extensive number of video tokens leads to considerable computational costs for LLMs while using aggregated tokens can result in loss of vision details. Moreover, the presence of abundant question-irrelevant tokens introduces noise to the VQA process.
To address these issues, we introduce a simple yet effective retrieval-based video language model (R-VLM) for efficient and interpretable long video QA. Specifically, given a question (query) and a long video, our model identifies and selects the most relevant $K$ video chunks and uses their associated visual tokens to serve as context for the LLM inference. This effectively reduces the number of video tokens, eliminates noise interference, and enhances system performance.
Our experimental results validate the effectiveness of our framework for comprehending long videos. Furthermore, based on the retrieved chunks, our model is interpretable that provides the justifications on where we get the answers.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenges of comprehending long videos using LLMs by introducing a retrieval-based video language model (R-VLM) for long-video QA. The model identifies and selects the most question-relevant video chunks and their associated visual tokens to serve as context for LLM inference, effectively reducing the number of video tokens, preserving informative information, eliminating noise interference, and enhancing system performance.

### Strengths
1. The paper is well-written and easy to follow.
2. Using the visual tokens of the most question-relevant K video chunks as context for the LLM inference can effectively reduce computational costs.

### Weaknesses
1. Since there are not any ground-truth locations of the question relevant chunks for supervision, how to make sure that the learned chunk selection is effective?
2. In this paper, the authors set the number of video chucks to 5. How does the model perform with fewer or more video chucks? Will the performance of the model increase with more retrieved video chucks?
3. When evaluating the model on four video datasets, the authors should clarify how much computational cost is saved through the retrieval-based chuck selection mechanism.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a Retrieval-based Video Language Model (R-VLM) for long video Question Answering (QA) tasks, aiming for efficient and interpretable long video QA. It computes the relevance between language token blocks and visual token blocks, selecting the top K visual token blocks as inputs for the Language Model (LLM) in VQA inference. This process matches and transmits the most relevant visual information to the LLM, reducing redundancy. Additionally, since the selected visual information is chosen based on its relevance to the question, this method also offers a degree of interpretability.

### Strengths
1. This method is relatively straightforward and easy to understand. By selecting the most relevant video segments, it reduces computational load, enhancing the efficiency and accuracy of the QA system.
   
2. Facing the growing content of long videos, this method provides a new perspective in the field of video understanding and retrieval.

### Weaknesses
1. Loss of detail: Although the method reduces distractions, key information might still be lost in the process of reducing visual tokens.
   
2. Lack of a dynamic adjustment mechanism: The model seems to lack the capability to dynamically adjust the number of selected video segments based on the content of the video, which could lead to information overload or oversimplification in some scenarios.
   
3. Resource consumption issues: Despite the reduction in the number of visual tokens, the overall resource consumption of the model (such as computational resources, memory, etc.) has not been fully discussed.
   
4. Regarding the selection of K, although the authors have explained it in the experimental section, the ablation of K still warrants exploration.

### Questions
1. How does the model handle the diversity and complexity of video content? Can it effectively process different types (such as documentaries, movies, news reports, etc.) and styles of videos?
   
2. When reducing visual tokens to improve efficiency, how is key information ensured not to be missed?
   
3. What is the model's generalizability? Whether it has good generalization capabilities for Out-Of-Distribution (OOD) data is worth discussing.
   
4. What about the real-time processing capability for long-duration videos? Is the model suitable for real-time video streams or scenarios requiring immediate responses?
   
5. In the ablation study section, the article does not conduct descriptive experiments on the hyperparameters used, such as $\lambda$ and K.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a new approach called retrieval-based video language model (R-VLM) for long range video understanding. In R-VLM, the video is divided into 'N' chunks of 4 seconds and a retrieval module selects the best 'K' chunks for a given question. The selected chunks are passed as input to the frozen LLM for question answering. The authors then evaluate R-VLM on 4 long range video QA datasets such as WildQA, QaEgo4D, lifeQA and Social-IQ 2.0 datasets. Experimental results demonstrate that R-VLM strongly outperforms previous approaches.

### Strengths
**1. Clarity:** The authors did an excellent job in the presentation and clarity of writing. The methodology, experiments and results section are presented in a systematic way.

**2. Strong results on multiple datasets:** The proposed approach demonstrated strong results on multiple long range QA datasets.

### Weaknesses
**1. Insufficient Contributions vis-à-vis (Maaz et al.):** The contributions of this work are limited and doesn't meet ICLR standards. There are a only couple of differences between this work and Video-ChatGPT in terms of using chunks vs global pooling. To reduce the dimensionality of individual chunks, the authors adopt the same spatial-temporal pooling used in (Maaz et al.).

**2. Unclear advantages of chunks:** The main contributions in this work is dividing video into small chunks of 4 seconds. However, the authors use a very small sampling rate (1fps) which is essentially 4 frames for 4 seconds. Its unclear how chunks with small sampling rate provide an advantage to uniform/random sampling of equal no. of frames over the entire video while keeping the entire pipeline unchanged. There are currently no comparisons included in the paper.

**3. References:** There are some issues with the reference style used. The references got mixed up with the text and are sometimes indistinguishable.

**4. Relation to brain/biology:** Although the authors connect the proposed approach to brain, I believe such a comparison might be unnecessary. There are currently no clear established relations to deep learning and brain while most of them being assumptions. 

**5. Lack of human studies in evaluation:** The authors use ChatGPT for evaluating the proposed model. ChatGPT is a LM trained on data which is not public and might have biases. Therefore, it is not an accurate evaluation method. The authors should add human evaluation by sampling 1000 questions/ground truth answers/model predictions and then calculating accuracy/average score for all the models.

### Questions
**1. More frames in a chunk vs less chunks:** Is there any advantage/dis-advantage of using more frames in a chunk and reducing the number of chunks?

**2. Results on QaEgo4D:** In Table-2, we observe that R-VLM significantly underperforms on QaEgo4D. What might be the reasons for this? Are there any significant difference between this dataset and others?

**3. Visualization results in Figure-2 (b):** The first three retrieved chunks and the last two are almost similar. However, the R-VLM is able to depict different terrains and vegetations. In fact, the uniform chunks have more diverse terrains than R-VLM.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a framework that retrieves the most relevant video chunks for a given question and projects their representations onto a pretrained Large Language Model (LLM): Vicuna. The LLM then uses these visual features in combination with question features to perform long video question-answering tasks. The experiments across four long video QA benchmarks show that the proposed framework outperforms two baseline models. Ablation studies reveal that the framework is more effective than methods solely based on temporal-spatial pooling, uniform sampling, or pretrained CLIP retrieval. Additionally, the importance of the soft matching loss is highlighted in the final ablation study.

### Strengths
+) The motivation for the proposed framework is clearly and effectively presented in the Introduction and Figure 1. The Introduction is well-crafted, making it easy to grasp the paper's concept.

+) The framework has been proven effective in capturing the key chunks relevant to the posed questions, outperforming the global temporal-spatial pooling baseline (Video-ChatGPT) and the Q-former baseline (Video-LLaMA) in four different long video QA benchmarks, despite Video-LLaMA being an off-the-shelf pretrained model not trained on the same data.

+) The ablation studies further confirm the effectiveness of the design decisions made in the framework. The learnable retrieval-based approach surpasses methods based on uniform sampling and pretrained CLIP representation retrieval.

+) The qualitative results show the framework's ability to select important video snippets to answer the given questions accurately.

### Weaknesses
-) Section 3.1 is poorly written and difficult to understand. It is unclear why two different spatial average pooling layers are used to obtain the global spatial average pool M tokens. It's also not clear why it is necessary to specify the N tokens as h/2 x w/2 from the first spatial average pooling. A more detailed explanation or an additional model figure could help clarify this process.

-) The paper lacks specific details, such as which variant of the pretrained CLIP model is used. Given that there are many variants that significantly outperform the original, this is an important detail. Also, the reason for using only 4 frames at 1 FPS per chunk is not explained; is this due to memory limitations?

-) Another average pooling is used for chunk token retrieval, but the rationale behind this is not clear. Previous works have shown the effectiveness of using a Q-former or a few transformer layers to summarize visual tokens. This design choice needs clarification. Furthermore, it is surprising that a simple linear layer is used to project the visual tokens into the LLM's inputs.

-) There is no ablation study to justify the choice of the hyperparameter K (=5). One could imagine that decreasing K might lead to a loss of information necessary to answer the question, while increasing K could provide more information but might also introduce redundancy, potentially confusing the LLM.

-) While qualitative results are provided, there is no discussion or analysis of the model's failure modes.

### Questions
o) Can the authors present more qualitative results, including both successful and unsuccessful examples?

o) Could the authors include the missing details and clarify the points of confusion regarding the use of many average pooling layers at different places and the simple linear projection of the visual tokens?

o) Could the authors conduct experiments with a Q-former for spatial-temporal visual feature aggregation as an alternative to average pooling?

o) Could the authors consider using a Q-former for projecting visual tokens instead of a simple linear layer?

o) Can the authors analyze the impact of different values for the hyperparameter K?

o) Could the authors finetune the vision-language branch proposed in Video-LLaMA using the same Video Instruction Data collected by Maaz et al. (2023) as was used to train the proposed framework for a fair comparison?

o) Would the authors consider revising Section 3.1 for better clarity?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
