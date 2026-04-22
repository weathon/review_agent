# Unlabeled Data vs. Pre-trained Knowledge: Rethinking SSL in the Era of Large Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Semi-supervised learning (SSL) alleviates the cost of data labeling
process by exploiting unlabeled data and has achieved promising results. Meanwhile, with the development of large foundation models, exploiting pre-trained models becomes a promising way to address the label scarcity in the downstream tasks, such as various parameter-efficient fine-tuning techniques. This raises a natural yet critical question: When labeled data is limited, should we rely on unlabeled data or pre-trained models? To investigate this issue, we conduct a fair comparison between SSL methods and pre-trained models (e.g., CLIP) on representative image classification tasks under a controlled supervision budget. 
Experiments reveal that SSL has met its "Waterloo" in the era of large models, as pre-trained models show both high efficiency and strong performance on widely adopted SSL benchmarks. This underscores the urgent need for SSL researchers to explore new avenues, such as deeper integration between the SSL and pre-trained models. Furthermore, we investigate the potential of Multi-Modal Large Language Models (MLLMs) in image classification tasks. Results show that, despite their massive parameter scales, MLLMs still face significant performance limitations, highlighting that even a seemingly well-studied task remains highly challenging.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper compares unlabeled data and pre-trained models with a limited amount of labeled data. Various semi-supervised learning scenarios were considered, including semi-, open-set, open-world, and long-tailed semi-supervised learning. Based on extensive experimental results, valuable conclusions have been drawn, showing that pre-trained models can often perform better, though they still face many limitations and require further research.

### Strengths
- The problem studied, i.e., comparing unlabeled data and pre-trained models, is interesting and important to the literature.

- The experiments are comprehensive and cover various semi-supervised learning settings, including standard, open-set, open-world, and long-tailed semi-supervised learning. Therefore, the conclusions are convincing.

- The writing is excellent, making the paper easy to understand.

- This paper is an important position paper for the field of weakly supervised learning.

### Weaknesses
- There is a lack of related work discussing the use of zero-shot CLIP for traditional weakly supervised learning, a more general form of semi-supervised learning. Additionally, some work discusses semi-supervised learning with CLIP.

- The title would be better with "semi-supervised learning" instead of "SSL" since "SSL" has multiple meanings, such as self-supervised learning.

- In experiments on standard semi-supervised learning, SOTA SSL algorithms such as FixMatch seem to achieve better performance on CIFAR-10 and CIFAR-100. Therefore, concluding that pre-trained models are better for standard semi-supervised learning is incomplete.

[1] Delving into Weakly Supervised Learning with Pre-Trained Models, https://openreview.net/forum?id=RgWATMmWmz

### Questions
- For open-set and open-world SSL learning, is the method using pre-trained models the same as standard semi-supervised learning since it does not use unlabeled data?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a controlled comparison of classic semi-supervised learning (SSL) methods versus pre-trained vision–language models (VLMs) across four regimes: standard SSL, open-set, open-world, and long-tailed classification. Under matched supervision budgets, the study finds that VLMs (e.g., CLIP with prompt/adapter tuning) often match or surpass SSL in accuracy while being more compute-efficient.

### Strengths
This is a timely study in the era of large models and has the potential to deliver useful insight for the SSL community. The experiments are relatively comprehensive, covering main settings in SSL research. Takeaways are clearly stated.

### Weaknesses
The benchmark datasets—CIFAR-10/100, STL-10, and ImageNet—may overlap semantically with VLM pretraining data, potentially advantaging the VLMs in this comparison. By contrast, in domains where the pre-trained model has seen less data (e.g., medical imaging), it is unclear whether the conclusions would still hold. Experiments probing this low-overlap setting are currently missing.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates traditional self-supervised learning (SSL) methods and CLIP-based prompt-tuning approaches. The authors conduct experiments across various image classification settings and present several empirical findings.

### Strengths
The paper presents comprehensive experiments.

### Weaknesses
1. This paper focuses solely on experimental comparisons without proposing any new method. Therefore, the experimental setup is critical and should be described in detail in the main text.

2. The comparison settings are entirely unfair. For example, traditional SSL methods use ResNet backbones (e.g., ResNet-28-2), whereas prompt-tuning methods use ViT/B-16. Moreover, comparing SSL and few-shot learning is conceptually unsound. SSL with CLIP has already been explored, and under such unfair experimental conditions, the conclusions drawn from the experiments are not meaningful.

3. Since CLIP is trained on diverse web-scale data, its robustness to OOD (out-of-distribution) data is expected and not surprising.

4. Recent studies [1] have shown that MLLMs perform better in image classification tasks using no-thinking or direct-answering prompts. It would be more informative to report the performance of MLLMs under such settings, as this would better reflect their true classification capabilities.

5. CLIP models can also be used for SSL settings. For example, [2]. It is questionable why comparing few-shot CLIP-based prompting methods with ResNet-based SSL methods. 

6. Overall, the contribution of this paper appears limited. The authors merely conduct inference-based experiments and report results that are largely unsurprising. There are no technical or methodological innovations.

Reference:
[1] To Think or Not To Think: A Study of Thinking in Rule-Based Visual Reinforcement Fine-Tuning. NeurIPS 2025.

[2] Candidate pseudolabel learning: enhancing visionlanguage models by prompt tuning with unlabeled data. ICML 2024.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
