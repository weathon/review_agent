# Bilateral Information-aware Test-time Adaptation for Vision-Language Models

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 4, 6, 6

## Abstract
Test-time adaptation (TTA) fine-tunes models using new data encountered during inference, which enables the vision-language models to handle test data with covariant shifts. Unlike training-time adaptation, TTA does not require a test-distributed validation set or consider the worst-case distribution within a given tolerance. However, previous methods primarily focused on adaption-objective design, while the data tend to be fully utilized or simply filtered through a fixed low-entropy selection criteria. In this paper, we analyze the weakness of previous selection criterion and find that only selecting fixed proportion of low-entropy samples fails to ensure optimal performance across various datasets and can lead the model to becoming over-confident in wrongly classified samples, showing unexpected overfitting to atypical features and compromising effective adaptation. To improve upon them, we propose \textit{Bilateral Information-aware Test-Time Adaptation} (BITTA), which simultaneously leverages two distinct parts of the test inputs during adaptation. Specifically, a dynamic proportion of low-entropy samples are used to learn the core representation under covariant shifts, while high-entropy samples are adopted to unlearn atypical features. This dual approach prevents the model from undesired memorization and ensures  extensive optimal performance. Comprehensive experiments validate the effectiveness in various datasets and model architectures. The code is publicly available at: https://github.com/tmlr-group/BITTA.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tackles a known failure mode in VLM TTA: selecting only low-entropy samples for entropy minimization (EM) can overfit "atypical features" and amplify overconfidence on misclassified cases. The paper argues that the high-entropy samples which the current model is unconfident can serve as a potential candidate for regularization. Thus, BITTA proposes a bilateral strategy: (i) learn with low-entropy samples using a standard TTA objective, and (ii) unlearn with a small set of high-entropy samples by maximizing their predictive entropy. The intent is to curb memorization of atypical features while preserving core representations. BITTA is used as plug-in to multiple TTA algorithms, yielding gains on multiple benchmarks.

### Strengths
- Clear diagnosis of a common TTA pitfall (confidence selection → overfitting to atypical cues).

- Simple, compatible design: bilateral learning + unlearning that can wrap around existing TTA learners.

- Dynamic selection ratio that reflects dataset/noise distribution rather than a fixed value.

### Weaknesses
- What the method assumes. The paper assumes two things:
(1) Low-entropy and high-entropy samples both carry spurious (atypical) cues; and
(2) those cues are stronger or more frequent in the high-entropy group.
This is why the method adds an “unlearning” branch on high-entropy samples.

- What is missing. The paper does not make this assumption clear; it does not define what counts as an “atypical feature,” and never measures how much atypicality exists in each entropy group (low / medium / high). 

- Why this matters. Without a clear definition and measurement regarding “atypical feature,” we cannot tell whether the gains of BITTA come from truly removing reliance on "atypical" features or from a generic regularization effect. That weakens the main motivation for the bilateral design and the authors' claims.

 - What evidence would resolve this. Provide a formal definition an operational test for “atypical features”, and report their level by entropy bin. Then show that the unlearning process for high-entropy samples reduces these atypical signals—especially for misclassified low-entropy cases.

### Questions
- Regarding "atypical feature" assumption, what are "atypical features"? Would it be possible to provide an operational definition of “atypical features”?

- Would it be possible to quantify their prevalence/strength of "atypical features" by entropy bins (low/medium/high) to substantiate the "atypical" feature assumption?

- Would it be possible to show that the unlearning process for high-entropy samples can reduce these atypical signals—especially for misclassified low-entropy cases?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes BITTA, a bilateral information-aware TTA framework for VLMs (e.g., CLIP): low-entropy samples are used for learning (entropy minimization) and high-entropy samples for unlearning (entropy maximization), with a heuristic to dynamically set the low-entropy selection ratio. Experiments on CIFAR-10/100-C and ImageNet-C show consistent but modest gains and some compatibility with existing TTA methods.

### Strengths
1. Clear identification of a failure mode of fixed low-entropy selection (overconfidence on errors).
2. Simple, general plug-in that works with multiple TTA baselines and backbones.
3. Sensible diagnostics (entropy dynamics, t-SNE) and reasonable ablations (batch size, steps, λ).
3. Low computational overhead; easy to implement.

### Weaknesses
1. Technical novelty is limited; the idea (minimize on confident, maximize on uncertain) is incremental, and the theory is high-level with strong assumptions.
2. ImageNet generalization is under-evaluated: key variants (e.g., ImageNet-R/A/V2/Sketch) are not systematically included in the main results.
3. Missing important baselines (e.g., DiffTPT [1], DMN-ZS [2]), weakening the empirical case.
4. Gains are often modest with occasional regressions; clearer analysis of when it helps/hurts is needed.

[1] Diverse data augmentation with diffusions for effective test-time prompt tuning
[2] Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The article addresses the problem of test-time adaptation of a vision-language model given a stream of test data. In particular, the model selects samples with low entropy, assuming that the model's confidence reflects certainty in its predictions, and optimizes the model to further minimize entropy. Deviating from previous works, the approach selects samples of high entropy as well. For the latter, it is assumed that they have atypical features that need to be unlearned, something achieved by maximizing their entropy. The tradeoff between the two objectives aims to prevent overfitting and overconfidence while still learning from the stream of data. Experiments across domains and corruptions show the effectiveness of the approach (BITTA), outperforming recent competitors.

### Strengths
1. The use of samples with high entropy/low confidence is sound and very interesting. It is interesting because it has not been explored by previous approaches, focusing mostly on learning from high-confidence samples. While it is unclear what type of atypical information is included in samples with high entropy, maximizing the entropy of the latter can still act as a regularizer for avoiding overconfidence, thus improving the results. 

2. The paper shows several analyses concerning multiple aspects of the approach, such as the impact of batch-size and update steps (Fig. 6, Fig. 8.a), hyperparameters (Fig. 8.b), selection module (Fig. 7), and changes in the entropy dynamic (Fig. 8.c). Moreover, experiments showcase the generality of the approach to other architectures/methods (e.g., Tab. 5 and Tab. 8). Overall, these analyses support the design choices and provide insights on the potential of the method, how the latter works, and the tradeoffs to keep in mind when applying it. 

3. The appendix contains several details regarding design choices (e.g., the threshold estimate module of Appendix E), and the supplementary material includes the code. All in all, these additions make the submission transparent and provide strong support for its reproducibility.

### Weaknesses
1. The setting is very similar to that of Episodic TTA (e.g., [a,b,c]), where the model is updated as the stream of target data becomes available. Some of these competitors are missing in the experimental results, and including them would make the comparisons more comprehensive. 

2. Related to the previous point, these methods test with a batch size of 1 (i.e., [a,c]), assuming no priors on the batch constitution. On the other hand, BITTA is very sensitive to the batch-size (e.g., results of Fig. 8.a and the adaptive threshold of 294-306). Open questions are whether the model would be i) effective for extremely low batch sizes and ii) robust to non i.i.d. batches (e.g., samples of the same class, as in [d,e]). 

3. While it is intuitive that maximizing the entropy of low confident examples can both prevent overfitting and reduce overconfidence, the fact that high entropy samples share atypical features with low entropy ones (65-70) is not intuitive and not clarified with the qualitative examples of Fig. 3.a and H.3. It would be helpful to either clarify the meaning of atypical features or provide evidence of these shared spurious factors, or down weigh the related statements. 

4. Lines 270-274 indicate that minimizing entropy on high-confidence samples leads to overconfidence and maximizing entropy on low-confidence ones reduces overfitting. This is also suggested by Fig. 2, Fig. 3.b, and Fig. 8.c at the level of entropy. To further analyze the phenomenon of overconfidence, it could be interesting to show the expected calibration error [f], an analysis reported by related TTA works exploring overconfidence and potential solutions (e.g., [g,h]). 

5. While maximizing entropy is a good unlearning strategy, it would have been interesting to explore other alternatives (e.g., [i]) to justify this design choice. Note that different unlearning choices could lead to different effects in terms of overfitting and overconfidence. Moreover, related works do not discuss how the paper relates to the machine unlearning literature and related approaches that used unlearning for downstream tasks (e.g., debiasing [j]).


**References** ([a,b,f] already in the manuscript):

[a] Karmanov, Adilbek, et al. "Efficient test-time adaptation of vision-language models." CVPR 2024.\
[b] Zhang, Ce, et al. "Dual prototype evolving for test-time generalization of vision-language models." NeurIPS 2024.\
[c] Zhou, Lihua, et al. "Bayesian test-time adaptation for vision-language models." CVPR 2025.\
[d] Gong, Taesik, et al. "Note: Robust continual test-time adaptation against temporal correlation." NeurIPS 2022.\
[e] Niu, Shuaicheng, et al. "Towards stable test-time adaptation in dynamic wild world." ICLR 2023.\
[f] Guo, Chuan, et al. "On calibration of modern neural networks." ICML, 2017.\
[g] Yoon, Hee Suk, et al. "C-TPT: Calibrated test-time prompt tuning for vision-language models via text-feature dispersion." ICLR 2024. \
[h] Farina, Matteo, et al. "Frustratingly easy test-time adaptation of vision-language models." NeurIPS 2024.\
[i] Fan, Chongyu, et al. "Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation." ICLR 2024.\
[j] Chen, Ruizhe, et al. "Fast model debias with machine unlearning." NeurIPS 2023.

### Questions
Following from the weaknesses above:
1. How does the method compare with other TTA approaches working online?
2. Is the method robust to the batch composition?
3. Is it possible to clarify the meaning of atypical features?
4. Is BITTA reducing the overconfidence of the model/improving its calibration?
5. How does BITTA relate to approaches for machine unlearning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes BITTA, a method for test-time adaptation (TTA) that simultaneously performs learning on low-entropy (confident) samples and unlearning on high-entropy (uncertain) samples. Empirical evaluation is performed on several corruption and domain shift benchmarks, showing small but consistent improvements over prior TTA baselines.

### Strengths
S1. **Conceptually simple but reasonable intuition.** The idea of balancing learn and unlearn at test time is intuitive, and connects nicely to existing observations that confident-only updates lead to confirmation bias. 

S2. **Dynamic adaptation ratio.** Instead of fixing the proportion of confident vs. uncertain samples, BITTA adjusts it adaptively based on batch entropy values. This is a lightweight heuristic that makes the algorithm more flexible. 

S3. **Well-written and structured.** The methodology section is easy to follow, and the figures illustrating the bilateral update flow are clear.

### Weaknesses
W1. **Marginal improvement magnitude.** Most gains over strong baselines are within +0.3--1.0 percent points, often within the variance range reported in prior TTA studies. It would be great if the authors report the performance of average and variance of each experiment with multiple times. 

W2. **Comparison of previous unlearning methods.** The paper states that high-entropy samples trigger *unlearning* to mitigate the overconfidence. I believe that there are a number of research in terms of unlearning works, so it is important to compare them with the proposed method that authors proposed in this paper. Sorry for not referring several unlearning methods to compare due to lack of expertise about unlearning domain. 

W3. **No computational analysis.** BITTA claims to be lightweight, but no runtime or memory cost comparison is reported against other baselines. 

W4. **Incremental novelty.** I agree that the authors address an important problem in the TTA domain. However, the proposed method is conceptually simple and lacks substantial novelty. If the approach had demonstrated a larger performance gap over prior TTA methods, its impact could have been justified despite the simplicity. Unfortunately, the observed improvements are rather marginal (as described in W1), which limits the overall significance of the contribution.

### Questions
BITTA is a clean and well-written paper that presents a modest yet reasonable enhancement to entropy-based test-time adaptation. The idea of learning from confident samples and unlearning uncertain ones is conceptually sound and practically implementable. However, the contribution remains incremental, and the performance gains are minor. I recommend **rejection** at this stage, but I believe the idea has potential. If the authors further refine their method and demonstrate a larger improvement over existing baselines, the work could be strong enough for acceptance in a future submission.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem in test-time adaptation of vision–language models whereby updating solely on low-entropy samples leads to inadvertent overfitting and sample-selection ratios that lack cross-domain robustness. The proposed method minimizes entropy for low-entropy samples while simultaneously strengthening image–text alignment and inter-class separability, and maximizes predictive entropy for high-entropy samples to suppress memorization of atypical features, thereby striking a balance between adaptation and robustness. The authors also provide theoretical support regarding the separability of hard samples and coverage guarantees for proportion prediction, and demonstrate consistent performance gains on CIFAR-10/100-C, ImageNet-C, and multiple cross-domain datasets, as well as in combination with methods such as TPT, CTPT, and BAT.

### Strengths
1. Proposes a bilateral mechanism and uses dynamic proportion estimation to mitigate overfitting and sensitivity to sample selection.
2. Provides an analysis of the separability of hard samples and coverage guarantees for proportion prediction, enhancing the method’s interpretability and robustness.
3. Achieves consistent gains on CIFAR-10/100-C, ImageNet-C, and multiple cross-domain datasets.

### Weaknesses
The paper needs additional experiments to further demonstrate the effectiveness of the method.

### Questions
1. How is the linear relationship between the dynamic low-entropy proportion and the number of classes fitted? Which data points are used, what is the goodness of fit, and are scatter plots with regression lines on CIFAR-10-C, CIFAR-100-C and ImageNet-C, together with a small-scale sensitivity check, included?
2. Why is the high-entropy proportion fixed at 0.1? Is a brief sweep at 0.05, 0.10, 0.15, 0.20 on CIFAR-10-C and ImageNet-C (severity 5), reporting both Top-1 and ECE?
3. In the component ablations, how much gain is attributed to low-entropy learning only, high-entropy unlearning only, and both together? Are the independent contributions of each component in the bilateral mechanism quantified?
4. Does unlearning improve model uncertainty and risk control? Are ECE, NLL, and rejection AUROC reported based on existing outputs and compared with baselines?
5. On which corruption types are the gains primarily concentrated? Are results broken down by distortion type provided, in addition to the main table, to clarify performance differences across corruptions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors argue that previous methods have focused on improving the objective functions and have ignored the dataset perspective of test time adaptation. The standard paradigm revolves around picking up the most confident samples and applying TTA methods on these. This relies on an assumption that all low entropy samples are correct predictions. However, this can lead to memorization of atypical features as shown in figure 2 of their motivation where model becomes confident about its wrong predictions. Authors propose an interesting work around to this, they utilize high entropy samples and maximize the entropy on these samples. This leads to unlearning of such atypical features leading to higher performance across the board. Lastly, to improve the training process, authors propose that the optimal percentage for low entropy samples vary for each dataset.

### Strengths
- Paper introduce a novel data centric perspective to the test time adaptation. 
- paper is well motivated. 
- presentation is well done. 
- Results are promising.

### Weaknesses
It is unclear which features are atypical. could it also be because of high confident incorrect predictions being part of the training mix? maybe some sort of attention map visualizations would be nice here as well. 
How does it compare to other regularization techniques such as weight decay.

### Questions
- Comparison with other regularization techniques. 
- visualization of attention maps over low confident misclassifications before TTA and high confident misclassifications after TTA.
- You could have high entropy correctly classified samples. Does something like this exist and if yes, then does it impact your method negatively?

### Soundness
3

### Presentation
4

### Contribution
3
