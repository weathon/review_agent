# Video-STaR: Self-Training Enables Video Instruction Tuning with Any Supervision

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 6

## Abstract
The performance and reasoning capabilities of Large Multi-modal Models (LMMs) is dependent on the size and quality of their training datasets. However, collecting datasets that support chain-of-thought instruction tuning is highly challenging. Existing video instruction tuning datasets are often derived by prompting large language models with video captions to generate question-answer pairs, which makes them predominantly descriptive rather than reasoning-focused. 
Meanwhile, many labeled video datasets with diverse labels and supervision exist -- however, we find that their integration into LMMs is non-trivial. 
Herein, we present $\underline{\text{Video}}$ $\underline{\text{S}}\text{elf}$-$\underline{\text{T}}\text{raining}$ $\text{with}$ $\underline{\text{a}}\text{ugmented}$ $\underline{\text{R}}\text{easoning}$ (Video-STaR), the first self-training approach for video instruction tuning. 
Video-STaR allows the utilization of *any* labeled video dataset for video instruction tuning.
In Video-STaR, an LMM cycles between instruction generation and finetuning, which we show (I) improves general video understanding and (II) adapts LMMs to novel downstream tasks with existing supervision. 
During instruction generation, an LMM is prompted to propose an answer.  The answers are then filtered only to those that contain the original video labels, and the LMM is then re-trained on the generated dataset. 
By training exclusively on generated answers containing the correct video labels, Video-STaR leverages these existing labels as weak supervision for video instruction tuning.
Our results demonstrate that Video-STaR-augmented LMMs achieve notable improvements in (I) general Video QA, where TempCompass performance improved by 6.1%, *and* (II) downstream tasks, with a 9.9% increase in Kinetics700-QA accuracy and a 4.0% improvement in action quality assessment on FineDiving, while also exhibiting better interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel method to perform instruction tuning of Large Multi-modal Models (LMMs) on labeled video datasets, where the label format is arbitrary and different from that needed for instruction tuning. The main idea is to use the original LMM itself to generate question-answer pairs for instruction tuning, while using the original dataset labels as a "verifier" to filter out low-quality generations. The authors perform multiple cycles of dataset generation and model training to obtain the final model Video-STAR. Video-STAR shows improved performance on TempCompass and zero-shot video QA datasets.

### Strengths
* The idea of turning existing labeled video datasets into instruction tuning dataset and use them for self-training is elegant and effective.
* The paper is well written and is easy to follow
* I appreciate the ablation study

### Weaknesses
* The main weakness of the paper is the Video-LLaVA baselines, i.e. Video-LLaVA+ and Video-LLaVA-gemini, which are the main points of comparison in the paper. Video-LLaVA-gemini has been fine-tuned only on several thousand sample pairs, which is incomparable to hundreds of thousands of video samples used for Video-STAR. As for Video-LLaVA+, the number of training samples is comparable, yet there is no information about how those samples were obtained. The way the samples in Video-LLaVA+ are constructed from the original video labels would affect the final model performance a lot, so it is important to explain this in details.
* Additionally, one important baseline for dataset creation is missing. It looks like the SoTA method for creating instruction datasets from video is VideoInstruct [A]. I would expect the authors to compare to this method of dataset creation, both in answer generation and label rationalization settings, to understand how much the video stream actually helps.
* It is somewhat unclear from the paper if Label Rationalization can be used to generate the entire dataset, without using Answer Generation. Somewhere in the paper the authors suggest that it leads to hallucinations, yet in Tables 6 and 7 they show the improvement.  

This is a preliminary rating and I will revise my score once the weaknesses and questions are addressed.

[A] -  Maaz et al. "Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models"

### Questions
* In Tables 6 and 7, the last row "- Generation" means that neither generation nor rationalization are used, or only that generation is not used while rationalization is used? Either way, it is important to see both experiments to better understand the contribution of each method.
* When doing self-training in multiple cycles, every time the model is initialized from the same non-instruction-finetuned checkpoint. Why not reuse the model from the previous cycle of self-training as initialization? 
* How do the authors deal with numerical labels such as temporal localization, bounding box, performance score? Do they expect the LVM to predict those directly and compute the L1 distance in the verifier? If so, how accurate is the prediction of the model in this case? Is Answer Generation capable of generating good question-answer pairs or Label Rationalization is more helpful here?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of leveraging existing video datasets with diverse non-instruction labels to enhance video instruction tuning. The authors introduce Video-STaR, an automated loop for instruction generation, filtering, and self-tuning. Notably, they incorporate a Label Rationalization step within the loop to backtrace the rationale from video labels on challenging data. Experiments demonstrate that Video-STaR enables Large Multimodal Models to adapt to new tasks effectively.

### Strengths
1. This paper addresses a significant issue: how to utilize existing video dataset labels to improve instruction tuning. This paper presents the first self-training pipeline that could inspire future approaches in this area
2. The writing is clear. The authors also conduct extensive experiments to validate the proposed Video-STaR approach.

### Weaknesses
1. **Missing Experiments**: There is a lack of validation on the effectiveness of label filtering, including an ablation study on the verifier and clarification on why ground-truth (GT) labels should be included in the answers.
2. **Heuristic and Inflexible Label Verification**: The label verification process is heuristic, limiting flexibility and scalability. Since different videos contain diverse labels, the proposed approach requires custom parsers and verifiers for varying label formats. For example, timestamps may need distinct regex patterns due to format variations. Additionally, label matching, such as “chopping wood” vs. “smashing,” as seen in Table 2, appears to demand considerable human effort.
3. **Weak Baseline in Video-LLaVA Gemini**: Video-LLaVA Gemini is a relatively weak baseline. In Table 4, the authors used Gemini-pro to label only 2K samples for fine-tuning, a notably smaller dataset than those used for other baselines. The choice of 2K samples is not straightforward. The authors should attempt to label more videos with Gemini and include a curve showing the relationship between labeled data quantity and fine-tuning performance. How much labeled data is needed to achieve results comparable to Video-STaR’s 550K in both Table 3 and Table 4?
4. **Marginal Cross-Dataset Generalization (Table 4 & Table 7)**: Video-STaR’s improvement in cross-dataset generalization appears marginal. The authors attribute this to noise within the benchmark itself. If this is the case, comparisons and ablation studies across these four datasets may not convey valuable information. The authors could consider using alternative high-quality datasets to better evaluate Video-STaR’s true effectiveness in enhancing cross-dataset generalization.
5. **Typos**: Table index is incorrect. Line 260 Table 2.3 --> 2. Line 306 Table 3.2 --> 2

### Questions
1. Recent studies indicate that using data generated by large language models (LLMs) for self-training can ultimately have a catastrophic impact on model performance. How do the authors address this issue? An interesting experiment could involve alternately training two models—Video-LLaVA (Model A) and another model (Model B)—using the Video-STaR strategy. In this setup, results generated by Model A could fine-tune Model B, and subsequently, results from Model B could fine-tune Model A. Observing the outcomes could reveal any differences in performance and stability under this iterative training approach.

### Soundness
2

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
Large Multi-modal Models have achieved great success recently. However, when the problem comes to improve the reasoning capabilities of LMMs, the first and huge challenge is to get large scale and high quality datasets. Especially for video modalities, the problem is more hard by the difficulty and time-consuming. In this paper, the authors proposed Video-STaR, which is a method to generate video datasets with rationales based on LMMs. Besides, the authors releases a new video datasets named VSTAR-1M generated by the method.

### Strengths
* The topic of automatically generating video datasets with accurate rationales is interesting and important for future multi-modality researches. It is very difficult to use human labor to fine-grain annotate videos with various reasoning details. 
* The authors release a large scale video instruction tuning dataset with rich CoT reasoning which significant saves the time and computing resources for future works.

### Weaknesses
* Limited technical novelty. The idea of this paper is very similar to STaR and all the proposed modules including answer generation, label rationalization and label verification are processed only for text modality, with no special considerations for video modality. It means that it is easy to transfer from normal STaR. Besides, for the methods section, the authors description is not clear and it is better to write it with more details
* Lack of experimental description, theoretical analysis, or other deeply analysis of some assumptions. In the paper, the authors claim that Label Rationalizations have higher possibility to result in hallucination. However, the authors don't provide further analysis, nor do some experiments to prove it (even only on some partial samples and do significance testing). Rather, the authors propose a case to show the problem in Figure.3. However, in cycle 1 I think the problem lays in LMMs repeats the questions instead of hallucination, and in cylce 2, Answer Generation also get a non-perfect result "with an overall score of 64.68" as the ground truth is 65.6

### Questions
* The VSTAR-1M has about 1M videos mentioned in the paper but the experiments only tuned with 550k videos shown in Table 4. May I ask the authors if we apply the full VSTAR videos to tune LMMs, whether the results would be better?
* Since neither the paper nor the appendix describes the verify process carefully, I briefly skimmed through the code and found that there is a 'verifier_type' for each sample. Is the type manually labeled for different samples or is it automatically judged by some rules?
* In the paper it is mentioned that bounding boxes can be used as labeled data, so does this data enable LMMs to localize objects better than before? For example, to get better results when doing video segmentation or object tracing or simply image object detection?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel self-training approach, Video Self-Training with augmented Reasoning (Video-STaR), to improve the performance of Large Multi-modal Models (LMMs) in video instruction tuning. By leveraging any labeled video dataset, Video-STaR enhances video question answering and adapts LMMs to new downstream tasks. The method cycles between generating answers, verifying them using the video labels, and fine-tuning the model. Experimental results show significant improvements in video understanding tasks, such as a 6.1% increase in Video QA performance and enhanced action quality assessment accuracy.

### Strengths
- **Originality**: Video-STaR introduces an innovative self-training approach for video instruction tuning, enabling the use of any labeled video dataset for instruction tuning. This represents a novel solution to a critical problem in video QA and reasoning by removing the dependency on manual dataset collection.

- **Quality**: The paper is well-structured, with detailed experimental evaluations demonstrating significant improvements in video QA accuracy and adaptability to downstream tasks. It also addresses weaknesses in previous methods, such as reducing hallucinations in model answers.

- **Clarity**: The methodology, including the self-training cycles of generation, filtering, and tuning, is clearly articulated, with effective visual aids to illustrate the process.

- **Significance**: The paper contributes meaningfully to the field of video QA and LMM adaptation, showing improvements in both general video understanding and task-specific performance. It also creates a large and diverse dataset (VSTAR-1M), further enriching the resources available for multimodal research.

### Weaknesses
- **Computational Intensity**: The iterative nature of the self-training cycles, involving both answer generation and label rationalization, introduces significant computational overhead. This could limit the scalability and efficiency of the proposed method, especially in resource-constrained environments.

- **Hallucination in Rationalization**: The reliance on label rationalization, especially for difficult tasks, may increase the likelihood of hallucinations. This undermines the robustness of the system in generating accurate explanations and answers, particularly in complex video tasks like FineDiving.

- **Generalizability of Label Rationalization**: While the method assumes that all video labels require rationalization, certain straightforward labels may not benefit from this additional step. This leads to unnecessary computational load without proportional gains in performance.

- **Limited Dataset Variety**: The study largely focuses on specific datasets like FineDiving, STAR-benchmark, and Kinetics700. Expanding the evaluation to a broader set of tasks could help demonstrate the full generalizability and adaptability of the approach to more diverse real-world video datasets.

### Questions
1. **Clarification on Dataset Labeling**: Can the authors clarify how they handle datasets where video labels are ambiguous or not well-defined? It would be helpful to understand how such cases are managed during the label verification step to avoid generating incorrect instructions.

2. **Self-Training Cycles**: Could the authors provide more details on how they determine when the self-training process has plateaued? Is there a specific performance threshold or metric used to stop the cycles?

3. **Hallucination Mitigation**: The paper acknowledges hallucination as a potential issue, especially in label rationalization. Are there additional methods or heuristics that the authors could explore to reduce hallucinations during both answer generation and rationalization phases?

4. **Generalizability to Complex Tasks**: Video-STaR shows notable improvements in performance on diverse video tasks. Could the authors elaborate on how the method handles particularly complex reasoning tasks beyond action recognition and quality assessment? How might this approach generalize to more abstract tasks, such as subjective video analysis?

5. **Computational Overhead**: Given the iterative nature of the self-training process, what are the computational requirements and trade-offs for implementing Video-STaR at scale? Would this method be feasible for larger video datasets, and if so, what optimizations could be applied?

### Soundness
2

### Presentation
2

### Contribution
2
