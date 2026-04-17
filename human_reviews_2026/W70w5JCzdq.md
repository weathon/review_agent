# DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Model merging has emerged as an efficient and flexible paradigm for multi-task learning, with numerous methods being proposed in recent years. 
However, these state-of-the-art techniques are typically evaluated on benchmark suites that are highly favorable to model merging, and their robustness in more realistic settings remains largely unexplored.
In this work, we first investigate the vulnerabilities of model-merging methods and pinpoint the source-model characteristics that critically underlie them.
Specifically, we identify two factors that are particularly harmful to the merging process: (1) disparities in task vector norms, and (2) the low confidence of the source models. To address this issue, we propose **DisTaC** (**Dis**tillation for **Ta**sk vector **C**onditioning), a novel method that pre-conditions these problematic task vectors before the merge. DisTaC leverages knowledge distillation to adjust a task vector's norm and increase source-model confidence while preserving its essential task-specific knowledge. Our extensive experiments demonstrate that by pre-conditioning task vectors with DisTaC, state-of-the-art merging techniques can successfully integrate models that exhibit these harmful traits, where they would otherwise fail, and achieve significant performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors identify two critical failure modes that are common in practical scenarios but overlooked in idealized benchmarks: (1) Task Vector Norm Disparity, where task vectors have vastly different magnitudes; and (2) Low-Confidence Source Models, where models produce high-entropy predictions, often as a result of calibration techniques like label smoothing.

The paper proposes a lightweight pre-conditioning method. DisTaC uses KD on unlabeled data to prepare models for merging. To correct norm disparities, it first rescales a task vector to a target norm and then uses KD to distill knowledge from the original, high-performing model, thereby recovering the performance lost during scaling. To correct for low confidence, it distills knowledge using a higher temperature for the student model than the teacher, which encourages the student to produce lower-entropy, more confident predictions.

The authors demonstrate empirically on eight vision tasks with ViT-B-32 and ViT-L-14 backbones that these failure modes significantly degrade the performance of state-of-the-art merging methods. They then show that applying DisTaC before merging consistently restores performance to the level of the idealized benchmark, with minimal computational overhead. Finally, the paper offers practical guidelines for model merging.

### Strengths
1. The paper's primary contribution is the identification and diagnosis of two practical failure modes in model merging. While the component techniques (knowledge distillation) are not new, their application to pre-condition task vectors to solve these specific robustness issues is novel.

2. The claims are substantiated by a thorough and well-designed set of experiments. The inclusion of theoretical motivation (Proposition 1 and the analysis in Appendix C) adds rigor and provides a deeper understanding of the observed phenomena. The ablation studies, such as the analysis of shrinking versus stretching vectors in Figure 3, are particularly convincing and support the practical guidelines offered.

3. By addressing the gap between idealized academic benchmarks, the paper provides a crucial step towards making model merging a more practical and reliable tool. DisTaC is a simple, effective, and computationally cheap solution that can be readily adopted by practitioners.

### Weaknesses
* The experiments are confined exclusively to CLIP-based models on vision tasks. The general applicability of DisTaC would be strengthened by demonstrating its effectiveness in other modalities, such as merging fine-tuned LLMs, or on MLLMs.

* The method's reliance on unlabeled data from the task distribution is a potential limitation. The paper would benefit from an analysis of the method's sensitivity to the quantity of this unlabeled data. For instance, how much data is needed for DisTaC to be effective, and how does performance degrade if the unlabeled data is from a slightly shifted distribution?

### Questions
1. Figure 3 and the discussion in Section 6.1 suggest that shrinking a task vector can sometimes lead to better performance than the original model. This is a very interesting observation. Could you elaborate on the potential reasons for this? Does this imply a regularization effect, suggesting that the initial fine-tuning may have overshot a more optimal, shorter task vector?

2. Regarding the influence of learning rate and training steps on the effectiveness of model merging, you may refer to the error analysis in Theorem 3.1 of [1], where the impact of the norm of the task vector was also observed. In [2], analysis of the properties of task vectors was conducted.

*[1] Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging. arXiv preprint arXiv:2505.19892*  
*[2] MAP: Low-compute Model Merging with Amortized Pareto Fronts via Quadratic Approximation. arXiv preprint arXiv:2406.07529*

### Soundness
3

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
2

### Summary
The paper investigates the robustness of model merging methods. The authors identify two critical failure modes in existing model-merging pipelines:

+ Task vector norm disparities — differences in magnitudes caused by heterogeneous fine-tuning hyperparameters (e.g., learning rates, steps).

+ Low pretrained-model confidence — often resulting from training techniques such as label smoothing, Mixup, or focal loss.

To mitigate these issues, the authors propose DisTaC, a lightweight rescaling task vectors to a target norm and increasing model confidence by training students with higher temperatures than teachers. Empirical evaluations on eight vision tasks demonstrate that DisTaC restores and often improves post-merge accuracy.

### Strengths
The experiments are extensive, including ablation studies, visual analyses (e.g., layer-wise norm shifts in Figs. 5–6), and additional tests with Mixup and focal loss to demonstrate generality.

The authors go beyond idealized benchmarks and pinpoint practical causes of model-merging failures, which is interesting and meaningful.

### Weaknesses
The experiments are restricted to vision-only tasks using CLIP ViT backbones. No tests are conducted on NLP, speech, or multimodal architectures.

While the paper argues that unlabeled data are “readily available,” this assumption may not hold in all domains, such as the security issue, or we can not find a comprehensive set for LLMs.

While DisTaC generally improves results, the paper lacks a sensitivity analysis of unlabeled data quality that would clarify when it might fail or overfit.

The authors assert that DisTaC is computationally lightweight (“500 steps, unlabeled data only”), but do not quantitatively compare its runtime or cost to baseline merges, nor provide results on large-scale LLMs such as Qwen.

### Questions
Why not provide the results of the Original version when combined with DisTaC in Table 1? It is helpful for us to understand that DisTaC may be harmful for performance in some cases.

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
4

### Summary
The paper investigates the robustness limits of state-of-the-art multi-task model merging methods, identifying two critical failure modes: disparities in task vector norms and low confidence in source models. To address these, the authors propose DisTaC—a pre-conditioning framework utilizing knowledge distillation to harmonize task vector norms and boost model confidence before merging. Extensive experiments on eight vision tasks with CLIP backbones demonstrate that DisTaC consistently restores or enhances merging performance under challenging, realistic scenarios where conventional merging fails.

### Strengths
1. The paper provides a sharp empirical and theoretical analysis of robustness shortcomings in existing model merging approaches, uncovering how norm disparity and low confidence can produce significant accuracy degradation after merging. 
2. DisTaC is a clear, well-motivated conditioning method, leveraging knowledge distillation with unlabeled data for practical pre-conditioning. By integrating both norm scaling and confidence sharpening, it directly addresses the two key failure modes.

### Weaknesses
1. Since the pre-conditioning step is applied independently to each source model, the total computational cost scales linearly with the number of models. For large-scale ensembles, this cumulative overhead can become prohibitive, undermining the method's primary efficiency advantage over full-scale retraining. The distillation cost also scales with model parameter count, becoming computationally prohibitive for large-scale foundation models (e.g., LLMs). 
2. The study’s empirical validation is confined to vision tasks, leaving its applicability to other modalities, particularly Large Language Models (LLMs), unverified. Given that task vector dynamics in LLMs may be substantially more complex, the method's effectiveness for merging models with diverse reasoning and generative capabilities is not guaranteed.

### Questions
1. How does DisTaC's independent conditioning of each task vector affect the final merge performance when tasks are fundamentally antagonistic or highly complementary?

### Soundness
2

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
This paper investigates the robustness of multi-task model merging methods under realistic, non-idealized conditions. The authors identify and empirically validate two key factors that degrade the performance of existing merging techniques: (1) disparities in the norms of task vectors, often arising from different fine-tuning hyperparameters, and (2) low prediction confidence of the source models, which can result from calibration techniques like label smoothing. To address these issues, the paper proposes DisTaC, a pre-conditioning step that uses knowledge distillation on unlabeled data. DisTaC adjusts task vector norms and increases model confidence before the merge, thereby making state-of-the-art merging algorithms more robust. Experiments on vision tasks show that DisTaC significantly improves performance in these challenging scenarios, often recovering it to the level of an ideal merge.

### Strengths
1. The paper addresses a novel and highly relevant problem. While most research focuses on improving performance on idealized benchmarks, this work astutely investigates the robustness of merging methods in more practical, pessimistic settings, which is a critical step for real-world applicability.

2. The problem formulation is exceptionally clear and well-motivated. The failure modes are demonstrated with convincing empirical evidence (Figure 1), making the motivation for the proposed solution immediately apparent and easy to follow.

3. The paper provides valuable and actionable insights for practitioners. The discussions around "shrinking vs. stretching" task vectors and the trade-off between source model calibration and merge performance offer practical guidelines that extend beyond the method itself.

### Weaknesses
1. The empirical evaluation is limited in scope. All experiments are conducted on CLIP/ViT models for vision classification tasks. Given the current prominence of model merging in the context of Large Language Models (LLMs), the absence of any experiments on language tasks is a significant limitation and leaves the generalizability of the findings in question.

2. The investigation of failure modes, while insightful, feels somewhat narrow. The paper focuses on norm disparity and low confidence but does not explore other plausible sources of incompatibility, such as differences in optimizer states, dataset quality/size, or the specific architecture of parameter-efficient fine-tuning (e.g., LoRA rank).

3. The method introduces a dependency on unlabeled data from the task distribution. While the authors argue this is a reasonable requirement, it makes the method less self-contained than truly data-free merging techniques and may be a practical constraint in some scenarios.

4. Proposition 1 formalizes a basic geometric intuition about vector addition but offers little deep insight into the functional consequences within a neural network. The choice of cosine similarity as the primary metric is also debatable, as its connection to actual task performance is not explicitly established.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
