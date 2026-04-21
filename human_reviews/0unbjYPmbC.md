# ChatSearch: a Dataset and a Generative Retrieval Model for General Conversational Image Retrieval

- Avg Score: 5.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 6

## Abstract
In this paper, we investigate the task of general conversational image retrieval on open-domain images.
The objective is to search for images based on interactive conversations between humans and computers. To advance this task, we curate a dataset called ChatSearch. This dataset includes a multimodal conversational context query for each target image, thereby requiring the retrieval system to infer the underlying retrieval intention from the multimodal dialogue conducted over multiple rounds. 
Simultaneously, we propose a generative retrieval model named ChatSearcher, which is trained end-to-end to accept and produce interleaved image-text inputs/outputs. ChatSearcher exhibits strong capability in reasoning with multimodal context and can leverage world knowledge to yield more sophisticated retrieval results. It demonstrates superior performance on the ChatSearch dataset and also achieves competitive results on other image retrieval tasks, such as zero-shot text-to-image retrieval and zero-shot composed image retrieval. With the availability of the ChatSearch dataset and the effectiveness of the ChatSearcher model, we anticipate that this work will inspire further research on interactive multimodal retrieval systems.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an automatically generated dataset for conversational image retrieval, in which a user may have multi-round interaction with a retriever in a natural language. The samples in the dataset is generated using existing models. The paper also presents a model for addressing the task. Some experimental results are presented on the dataset, which draws some insights on the data required for training.

### Strengths
(1) I think the task can be new and interesting. If I understand the task correctly, the model is required to take the conversation history into account for better performance.

### Weaknesses
(1) The paper does not provide sufficient details about the task definition. The paper will be better if it provides the input and output for each task (tChatSearch, iChatSearch, and mChatSearch). 

(2) Related to (1), these tasks seem to be interactive, in which a user can make different reactions to the same retrieval result (candidate images, etc.). However, as far as I understand, the dataset does not provide this type of interaction, and it only offers a single sequence of dialog without branches. This point seems important for interactive image retrieval but is not mentioned in the paper. I think this is not evaluated.

(3) I guess this is due to the page limitation, but the details on the dataset construction and the model is not fully provided to understand what is actually done. For example, the paper says some constraints are added to GPT-4 to imply some hints for image retrieval, but I cannot see how. 

(4) The insufficient details on the tasks and the methods make it hard for me to evaluate the paper.

### Questions
I would like clarification on (1)-(3) in the weakness section. I'm happy to increase my scores once some details are provided or I have made some misunderstandings.

UPDATE: I appreciate the authors' efforts to answer my questions. I increased my score.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a dataset referred to as ChatSearch, which is dialogue data in terms of image, such that the dialogue agents should retrieval corresponding image to respond the question in dialogue.
For the baseline, a system referred to as ChatSearcher is also proposed to reason multi modality between image and text.

### Strengths
[+] This paper addresses a situation when the dialogue is composed of an image, thus this paper proposes a dataset for image retrieval based on dialogue.

[+] New task about image retrieval in terms of dialogue

### Weaknesses
[-] The datasets from the works below seem to assume more general situations and to have more space for multimodal reasoning. What is the difference and contributions compared to this work?

1. PhotoChat: A Human-Human Dialogue Dataset with Photo Sharing Behavior for Joint Image-Text Modeling, ACL'21

2. TikTalk: A Video-Based Dialogue Dataset for Multi-Modal Chitchat in Real World ACM Mutimedia'23

3. DialogCC: Large-Scale Multi-Modal Dialogue Dataset Arxiv'23

[-] How can we handle it if there are no images that we want? I think our questions should be more diverse than the scope of the proposed dataset covers

[-] There are many systems for text-to-image retrieval. Does this dataset address the issues that previous text-to-image retrieval systems can not perform? I mean, we can also handle the problem of image retrieval from dialogue without involving the proposed datasets by integrating text-based dialogue systems and image retrieval systems based on many image corpus. It should be more convincing if there is evidential reason that the author has to collect dataset.

### Questions
See above. I am almost borderline, but there is no corresponding score for it. Therefore I want to decide my evaluations after rebuttal.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of conversational image search, where the goal is to interactively retrieve images according to the human's dialogue history. To resolve this problem, the authors propose a pipeline to automatically construct a dataset to include multimodal conversational context and target retrieval candidates. Then they trained a generative retrieval model called ChatSearcher to accepted interleaved image-text context and perform search.

### Strengths
- The paper is clearly written with well-defined motivation and real-world downstream use case
- The dataset construction pipeline is novel and completes the main story of the paper. The quality seems okay (since the author claimed that they have human validation)
- The system built, though far from solving the task, is approaching towards the correct direction

### Weaknesses
- It is kind of disappointing to base the validation and test set on a ancient dataset (MSCOCO), which has been mainly used for tuning model for almost a decade. Imaging that the COCO dataset would never cover any interesting visual entity after 2015. IMO, it would be better to consider base your test set on image collections with more diverse domains, such as the visual news dataset or CC3M dataset.

- The description on conversational instruction tuning is not sufficient for reader to understand how it actually work. Particularly, that instructPix2Pix is an image editing dataset. It would be nice to show a figure that explains how data from each domain look like and how they are processed to train model. 

- It also seems to me that there is a high chance where a user request can not be fulfilled within the candidate image sets. For example, when user are asking for a iPhone with foldable screen but there is no such a thing. What would the evaluation to handle such a case where it is unanswerable?  I couldn't find any discussion on it.

### Questions
- Why do authors emphasize your model is a generative retriever? In text retrieval (particularly entity retrieval), generative retriever usually refers to the models that generates the exact content of the document (via constrained decoding). Whereas in this case, the model is not really generating the image (but an embedding that approximates the image neighborhood). Giving the model terminology of generative retriever give the reader an impression that you are generating the image in pixel space, so that we can do exact visual matching. 

- Do we have any experiments ablating the effects of each component in the instruction-tuning datasets? It seems that Table 5 is such an ablation but where is instructpix2pix? 

- What is the retrieval image candidate set in each eval setting, are we only consider re-trieving against 5k (or 1k) images?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper the authors propose a new dataset ChatSearch which included multimodal conversations. The retrieval is performed by inferring details from multiple rounds of conversations. In addition, the authors also introduce a generative retrieval model ChatSearcher trained end-to-end and produces interleaved image-text inputs or outputs. Experimental results show that ChatSearcher shows superior performance on ChatSearch and comparable results on zero-shot image retrieval.

### Strengths
**Originality:** The authors introduce a dataset for conversational image retrieval which deals with multimodal form of dialogue. Most of the previous works focus on having image as static and textual dialogue. This limits the users ability to chat using images. The propose dataset overcomes the disadvantage. The proposed pipeline using LLMs for the dataset creation is also novel and requires less human effort. 

**Quality:** In addition to the dataset, the authors also propose a strong baseline model which learns from the conversational image dataset and performs better than CLIP on standard image retrieval datasets. The ablation studies are sound and well structured.

### Weaknesses
**Clarity:** 
-  I find the section 3.1 very difficult to follow. The words like "target image", "reference text" etc. appear multiple times and are confusing. The authors can introduce certain mathematical notations and provide sufficient examples for a better flow of the pipeline description. 

- Figure-2 contains a lot of sub-figures and details. The entire pipeline is not clearly understood from the figure and caption doesn't provide sufficient context. The figure can be improved by breaking down into individual sub-figures and the text can be made clearer.

### Questions
1. In the text dialogue construction, how does the authors ensure that GPT-4 doesn't generate unrelated image content or repetitive text dialogue.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
