# Beyond Fixed Resolution: Enhancing VLLMs with Adaptive Input Scaling

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 6

## Abstract
Real-world vision-language applications demand varying levels of perceptual granularity. However, most existing visual large language models (VLLMs), such as LLaVA, pre-assume a fixed resolution for downstream tasks, which leads to subpar performance. To address this problem, we first conduct a comprehensive and pioneering investigation into the resolution preferences of different vision-language tasks, revealing a correlation between resolution preferences with 1.image complexity, and 2.uncertainty variance of the VLLM at different image input resolutions. Building on this insight, we propose an empirical formula to determine the optimal resolution for a given vision-language task, accounting for these two factors as the zeroth-order and first-order terms in the Taylor expansion on a given image input. Second, based on rigorous experiments, we propose a novel parameter-efficient fine-tuning technique to extend the visual input resolution of pre-trained VLLMs to the identified optimal resolution. Extensive experiments on various vision-language tasks validate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a "two-stage" adaptive resolution pipeline. The authors introduce an empirical formula for choosing an optimal resolution. Experiments show that this method is effective on LLAVA.

### Strengths
The method lifts the performance of LLaVA. The proposed task-driven dynamic resolution is meaningful.

### Weaknesses
1. The pipeline is not end-to-end and looks unsightly. It seems like two VLMs cascaded.
2. The baseline, only LLaVA, is limited.
3. I feel it would be best for the authors to design the dynamic resolution of the task-driven mode in an end-to-end manner. The current approach is too heavy in image preprocessing

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Regarding the subpar performance of VLLM in downstream tasks when using fixed resolution for image processing, this work proposes a task-wise resolution selection method and adapts the model to optimal resolution on each task through post-training. By measuring image complexity and model uncertainty variance across resolutions and combining the two, an empirical formula for calculating optimal resolution based on the baseline resolution is obtained. In the resolution adaptation process, by adding LORA and training only certain key parameters, the performance of the proposed 7B model in downstream tasks surpasses all baselines and is comparable to the 13B model.

### Strengths
1. The overall logic is sound, and the experimental results demonstrate that the method proposed in the paper alleviates the shortcomings of fix resolution in downstream VQA tasks.
2. The article provides comprehensive experiments and analysis on the calculation and validation of optimal resolution.

### Weaknesses
1. The research findings of the article are relatively superficial, lacking deeper exploration. The relationship between the task and its optimal resolution in this paper seems more like overfitting to the benchmark itself. It is more important to focus on the sample-wise resolution selection or to derive the relationship between optimal resolution and more abstract task categories, rather than specific benchmarks.
2. There is a lack of comparison with other dynamic resolution methods, such as the dynamic number of tiles used in InternVL[1].
3. There are too few captions for the figures and tables. To understand the details of the figures and tables, one needs to refer back to a specific section.  

[1] Chen, Zhe, et al. "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
1. Intuitively, the selection of optimal resolution should not only be related to image complexity and uncertainty variance but also to the specific QA. For example, asking whether a photo contains a panda versus asking about the number of bamboo in a panda's arms would require different levels of image detail, which is more crucial than image complexity itself. The paper lacks sufficient analysis on this aspect.
2. According to Eq.3 in the paper, the task-wise optimal resolution is determined by C(T) and V(T), where C(T) and V(T) are the means of all samples in the task. What are the statistical distributions of these two values in each task? If there is significant variance, would it affect the significance of the mean itself?
3. The article does not provide specific information about the training data. It would be helpful to include more details about the experiments in the experimental section. Conversely, the case study section seems a bit lengthy.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper investigates the resolution preferences of different vision-language tasks and proposes an empirical formula to determine optimal resolution. It also presents a novel parameter-efficient fine-tuning technique to extend the visual input resolution of pre-trained models, which is validated by extensive experiments.

### Strengths
This paper explores the resolution preferences of different vision-language tasks and formulates an empirical formula to determine a relatively appropriate resolution. It also proposes a new parameter-efficient fine-tuning technique to enhance the visual input resolution of pre-trained models.

### Weaknesses
1. The author's starting point is good. However, the assumption that all existing VLLMs have a fixed resolution is invalid. Many existing VLLMs are of dynamic resolution, such as MiniCPM-V2 and Qwen2VL. The author seems to have directly ignored such methods and there is no discussion on them at all.

2. Regarding task selection, I don't seem to see that the author has selected tasks that are highly dependent on resolution for statistical evaluation, such as the DocVQA dataset. From my experience, this is a task scenario that is highly dependent on resolution. Statistical data in this task scenario can provide some inspiration.

### Questions
See the weakness.

### Soundness
2

### Presentation
1

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
This paper presents an investigation of the optimal image resolution for different vision-language tasks. It then propose a parameter-efficient fine-tuning technique to extend pretrained VLLMs to the target resolution.

### Strengths
This paper presents an interesting investigation in the choice of image resolution for different vision-language tasks with VLLMs. It reveals that image resolution would influence the performance for different downstream tasks. To solve the resolution variants problem, the authors futher propose a parameter-efficient fine-tuning techinque to tailor pretrained VLLMs to different image resolution. The experiments are extensive and well-organized.

### Weaknesses
Investigation on only LLaVA models might impair the generalization of this analysis.

### Questions
Does discrete image representation in VLLMs also suffer from the same image resolution problems?

### Soundness
3

### Presentation
3

### Contribution
3
