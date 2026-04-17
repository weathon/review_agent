# SupGRPO: Enhancing GRPO with Matching-based Online SFT for Text Spotting

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Text spotting requires both accurate text recognition and precise spatial localization. Current specialised spotters excel at predicting tight bounding boxes in natural scenes, but falter on complex or artistic text, whereas multimodal large language models (MLLMs) possess strong recognition capabilities yet remain weak at localisation. 
To equip the text spotter with general and powerful recognition capabilities and to maximize its localization ability, we explore two MLLM-based fine-tuning methods: Supervised Fine-Tuning (SFT) and reinforcement learning fine-tuning based on Group Relative Policy Optimisation (GRPO). An interesting finding is that SFT is less effective than GRPO at enhancing recognition, while GRPO is less effective than SFT at enhancing detection.
To compensate for each other's shortcomings, we introduce a joint training strategy, SupGRPO, which simultaneously optimizes the model using both SFT and GRPO. SupGRPO employs the specially designed reward functions and develops a matching‑based online SFT applied solely to coordinate tokens. It both mitigates the reward sparsity problem of GRPO and avoids the instance order dependency problem of SFT.
To evaluate particularly challenging cases, we curate ATS, a dataset for artistic text spotting. Experiments demonstrate that SupGRPO improves both text recognition and detection, validating the proposed approach. We will release ATS and our code upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SupGRPO for text spotting. The method performs joint training by jointly optimizing GRPO and matching-based online SFT, and designs the ATS dataset for artistic and complex scenes. Experiments show that the proposed method achieves excellent performance on multiple text spotting datasets.

### Strengths
1. This paper introduces GRPO into the text spotting task for the first time and designs four task-specific reward functions: format, text content, IoU precision, and IoU recall.
2. The authors constructed an artistic text dataset containing approximately 9K samples to expand the applicability of existing datasets in artistic and complex scenes.
3. Experimental results show that introducing SupGRPO significantly improves the model’s performance on the text spotting task.

### Weaknesses
1. SupGRPO is trained on a dataset that includes ATS, which mainly contains artistic text and differs from other datasets that focus on natural scenes. The compared models are not fine-tuned on this dataset. We suggest fine-tuning the baselines on the same dataset to verify the method’s effectiveness.
2. Although the authors selected strong general-purpose MLLMs (e.g., Qwen2.5-VL, InternVL 3.5) as baselines, the experiments still lack comparisons with OCR-optimized multimodal models (e.g., Ocean-OCR, TextMonkey). It is recommended to include such comparisons to more comprehensively verify the performance and competitiveness of SupGRPO in text spotting.
3. In the ablation study, the authors claim that “rewarding only text content degrades detection performance,” yet Table 7 shows that the detection score increases significantly from 26.3 to 61.1 (Baseline → Text). This inconsistency should be clarified by explaining the evaluation setup or experimental configuration.

### Questions
Please refer to weaknesses 1 and 2.

### Soundness
3

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
This paper proposes SupGRPO, a method for text spotting that combines both SFT and GRPO. In SupGRPO, the authors introduce a matching mechanism before SFT to reduce the instance order dependency problem. They also design several text-spotting–specific reward functions for GRPO. Furthermore, the authors construct an ATS dataset for training and evaluation. Experimental results demonstrate that their method outperforms other compared MLLMs.

### Strengths
1.This paper analyzes the drawbacks of the SFT and GRPO frameworks in the text spotting task. To address these issues, the authors propose a combined training method that achieves better performance.
2.Experimental results show that the proposed method outperforms other general MLLMs on both text recognition and detection tasks.
3.The authors create a new and challenging text spotting dataset, ATS, for evaluation.
4.The paper is well organized and clearly presented.

### Weaknesses
1.The authors show that their method achieves significant improvement on the new ATS dataset. However, their model has been trained on the ATS training set, which is unfair. The authors should provide results without training on ATS or fine-tune other models on the ATS for a fair comparison.
2.Only general MLLMs are compared. The authors should further include comparisons with MLLMs specifically designed for OCR tasks, such as GOT-OCR 2.0.
3. Why should SFT and GRPO be trained jointly in a single stage? A comparison with the commonly used two-stage training approach (SFT followed by GRPO) should be included as a baseline.
4.To better verify the effectiveness of their method, the authors should evaluate models with different scales and architectures.

### Questions
Answer the questions in Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work introduces SupGRPO, which GRPO with SFT to improve the text spotting capabilities of multimodal large language models. Additionally, the paper presents the ATS dataset comprising artistic text images. Experiments across multiple text spotting benchmarks validate the effectiveness of the proposed method.

### Strengths
- This paper is clearly written and easy to follow.
- This paper gives a successful try of GRPO on text spotting.
- This paper provides interesting observations in applying SFT and GRPO to text spotting.

### Weaknesses
- The contribution is somewhat limited. Authors apply SFT plus GRPO to text spotting and modify the rewards. They re-annotate erroneous OCR labels in TextSeg and WAS. However, the influence of the annotation error is not shown.
- Applying RL for a sole text spotting task is less meaningful.
- The effectiveness of introducing ATS during training for other benchmarks is not validated. 
- The detection performance lags far behind existing text spotting specialists.

### Questions
- Considering the text reward, why use the word-level F1-score which falls short on reflecting the error degree of recognition. Is ANLS or edit distance better?
- What about comparing SupGRPO to standard supervised learning with extensive data augmentation? Would this reduce the performance gap attributed to GRPO?
- The paper lacks ablation studies to show the influence of introducing ATS training data for other benchmarks.

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
4

### Summary
This paper aims to solve the problem of inaccurate text detection of MLLMs. First, they find that though SFT and GRPO can both improve the detection and recognition performance, SFT is superior to GRPO for detection but GRPO is superior to SFT for recognition. As a result, they combine GRPO and SFT to train MLLMs to better spotting text, and corresponding reward functions and a strategy that SFT only applies to matching coordinate tokens are designed. A dataset ATS is curated with existing datasets and extensive experiments are implemented.

### Strengths
1) Addressing the very low accuracy of text detection of MLLMs is valuable since in many scenarios both recognition and detection are important for real applications.
2) The proposed SupGRPO, which utilizes the specific Format Reward, Text Reward, IoU Precision Reward and IoU Recall Reward for policy optimization and simultaneously uses matching-based online SFT only for coordinate tokens, can boost both the detection and the recognition performance.
3) The detection performance of MLLMs can be improved with large margin by the proposed method, and the recognition performance can also be improved greatly. Especially, on ATS and CTW datasets, the recognition performance can be improved with large margin.

### Weaknesses
1) Though the detection accuracy is improved a lot, it is still can not be comparable with specialized text spotting models. Maybe the detection output of the specialized models is input to the MLLMs, the overall end-to-end recognition accuracy can be higher than that of MLLMs trained with SupGRPO, and the overall parameter amount and inference latency are comparable.
2) Tab. 1, Tab. 4-7, the recognition accuracy is not stated clearly. Is it the end-to-end recognition accuracy or the recognition accuracy of the cropped text region using the detection GT? If it is the end-to-end accuracy, the analysis for detection and recognition improvement is coupled together because end-to-end recognition accuracy includes detection’ performance implicitly. Please clarify it.
3) Eqa.6, do the predicted text strings and the ground truth text strings eliminate duplicate words? How to deal with duplicate words in the same image?
4) Fig.2 is not referred in the main body of the paper.
5) After trained with SupGRPO, does the performance of the MLLMs on other metrics except OCR degrade?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
