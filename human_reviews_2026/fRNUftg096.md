# Uncertainty quantification in clinical settings: A retinal fundus screening study and benchmarking

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 2, 8, 4

## Abstract
We offer the most extensive benchmark for uncertainty quantification (UQ) in retinal AI screening, providing practical guidance for clinical evaluators/regulators and highlighting the importance of risk–coverage–accuracy analysis. We methodically assess six well-known post-hoc UQ techniques in three main diseases: glaucoma (115K+ images), age-related macular degeneration (29K+ images), and diabetic retinopathy (105K+ images). Our benchmark comprises three Vision Transformer variations, standardized train/test/calibration splits, and evaluation on both public datasets and in-house clinical data from a local hospital. 
Results show that screening models can be miscalibrated and overconfident, and although UQ is helpful, its benefits are highly method- and disease-dependent. Our risk–coverage–accuracy analysis shows coverage drastically decreases as risk limits increase, and no single approach is consistently dependable in all contexts. 
While neither method consistently outperforms the others, Deep Ensembles and Test-Time Augmentation (TTA) are the two practical UQ approaches that most frequently enhance selective prediction and/or calibration. Conformal Prediction (CP) serves as a must-have safety rail, ensuring alignment between nominal and observed coverage. However, no method can reliably achieve the 2\% target-risk required for autonomous screening without sacrificing coverage. These findings highlight the need for more robust post-hoc UQ methods, both for in-distribution scenarios and under domain shifts (out-of-distribution), as well as improved mechanisms for capturing disagreements and implementing policy-aware thresholding in human-in-the-loop workflows. To facilitate progress in this field, we release our benchmark, which includes standardized data splits, trained model checkpoints, code, and an online demo for interactive exploration, thereby providing a reference for future UQ research in ophthalmic AI screening.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The article studies classification performance and uncertainty quantification in a retinal fundus screening. The study is based on two vision transformers that are pretrained on ImageNet or on a large dataset of unlabeled fundus images, and evaluates these models both on publicly available datasets (~10-100K images) as well as a separate clinical dataset collected at a local hospital (500 images). Different uncertainty quantification strategies are evaluated and compared to ground-truth test data. Key findings are i) there is a significant drop in classification performance from the public dataset to the private one, raising doubts about the applicability of current machine learning models in clinical practice, ii) uncertainty quantification approaches generally improve reliability, albeit only to a small degree and not consistent with uncertainty patterns reflected in clinicians, iii) split-conformal prediction is essential for callibration, as model confidences and uncertainty estimates tend to be severly overconfident.

### Strengths
This is a practical article that provides a summary of the status quo in uncertainty quantification and deep learning in a clinically relevant use-case and with actual data from real-world-deployment (lab to clinic). The article is well-written and the research is sound and carried out well. I very much enjoyed reading.

The fact that lab-to-clinic experiments are made makes the contributions significant and credible. The article has the potential to guide future research in uncertainty quantification and retinal fundus screening and has therefore significant scientific value. I recommend acceptance.

### Weaknesses
There are very few weak points that I could identify (see also questions). The article's contribution is not a new methodology but a scientifically sound evaluation of the status quo, which is largely missing in todays AI landscape, so the lack of new methodology should not be counted as weakness.

A weakness appears to be the focus on vision transformer architectures, which tend to be really large machine learning models that require substantial amounts of pretraining. It is unclear whether smaller model architectures would have performed better in this setting, as datasets are relatively small. However, the approaches are representative for the status quo in deep learning.

The real-world dataset for testing real-world-deployment appears to have a comparable small size (500 instances) from a machine learning perspective.

### Questions
Will all models and datasets be publicly released?

Can the authors further motivate the focus on vision transformers for their study?

The discrepency between model/UQ performance on publicly sourced data and newly collected data is striking to me. How much would a little bit of fine tuning on test-examples help with model performance? Can the authors pinpoint the reasons for this significant drop (e.g. preprocessing of images, different measurement device, etc.)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a study of a real benchmark that provides reality check on the promises of medical AI.
The authors used real-world clinical dataset to measure the crucial "lab-to-clinic" gap. Their findings are interesting; we are still a long way from safely automating these screening tasks. This is a good message for the community. Plus, they deserve credit for open-sourcing everything: their code, models, and data splits which sets a high bar for reproducibility and makes their work useful.
That said, my main concern is that they put all their eggs in one basket by only using Vision Transformer models. I'm left wondering if these same conclusions hold true for the classic CNNs that many people still use. It feels like a missed opportunity to make their findings more universal. They also completely sidestepped the practical cost of these methods. A "Deep Ensemble" sounds great, but it means training and running five models instead of one, which is a massive resource drain. For a hospital on a budget, that’s a potential deal-breaker. A bit of discussion on this trade-off would have made the paper more grounded.

### Strengths
On top of the points in the summary, these also are strengths in the paper
•	Comprehensive benchmark covering three diseases with rigorous evaluation metrics.
•	Strong commitment to reproducibility (releasing code, models, splits, demo).

### Weaknesses
Many issues with the paper in its current form:
•	Limited Novelty: The level of mathematical grounding of the ideas in the paper is somewhat below ICLR standards. The ICLR standards assume theoretical mathematical analysis of “why” these methods fails, in addition to the empirical evidence.
•	A main concern is that they put all their eggs in one basket by only using Vision Transformer models. I'm left wondering if these same conclusions hold true for the classic CNNs that many people still use.
•	Small clinical dataset (536 images).
•       Only two ViT architectures tested, both with frozen features.
•	AMD models perform poorly (low AUPRC).
•	Benchmarking study with no methodological novelty.
•	No justification for hyperparameters (T=50, N=5, K=20).
•	Identifies domain shift degradation but offers little insight into why or how to fix it

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a large-scale benchmark for uncertainty quantification in retinal AI screening across three diseases glaucoma, diabetic retinopathy, and age-related macular degeneration. They evaluate six post-hoc UQ techniques on two ViT backbones (ViT/DINOv2 and RETFound-Green). The authors assemble >114k glaucoma, >100k DR, and >28k AMD fundus images; define standardized train/test/calibration splits; and report discrimination, calibration, selective prediction, and conformal prediction (CP) results, including an external clinical test on a 536-image glaucoma set with 3-rater labels.

### Strengths
Clear, end-to-end benchmark framing. The work covers discrimination (AUROC/AUPRC), calibration (ECE/NLL/Brier), selective prediction (AURC, Risk@90% coverage, Coverage@5% risk), and CP validity, with sound descriptions of each metric. 
Risk–coverage analyses are thoughtfully interpreted; ensembles show the most consistent gains in glaucoma/AMD, while signals are weaker for DR. 
External clinical test & disagreement analysis. The hospital glaucoma set (n=536) reveals a realistic domain shift; the analysis of physician disagreement vs. UQ scores (with a significance test) is valuable.

### Weaknesses
Missing baselines and UQ methods post-hoc calibration baselines like temperature scaling and isotonic regression/Dirichlet calibration are not compared. Similarly, Laplace approximation, SWAG, and evidential deep learning

### Questions
Why was the calibration set drawn from the test sets rather than from held-out training/validation data?
it would be also useful if the authors provides reliability diagrams (before/after best calibrator) with ECE bins fixed across methods.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors provide a benchmark for understanding the predicted uncertainty in a highly relevant retinal screening task including the popular diabetic retinopathy detection. The authors just compare the 6 widely used methods for uncertaintiy quantification like the Monte-carlo dropout, test-time augmentations and use the already proposed uncertainty extraction mechanisms for these methods to calculate unertainty and orgainize it as a proper benchmark to accelerate the development of reliable models in this field. Mostly they use the existing labelled datasets for popular tasks like diabetic retinopathy but also include data from some hospital for the macular and galucoma task and evaluate them under setups like domain shift by testing on local clinical dataset. Based on this, the authors draw important conclusions for the effectiveness of these methods. The setups are the usual selective prediction and calibration analysis.

### Strengths
Overall the analysis can be quite useful for enhancing the research in this direction, since the authors have interesting insights like Glaucoma benefits the most from this uncertainty based selective prediction or deep ensembles emerge as a reliable approach and can decently decompose uncertainty into aleatoric and epistemic components. Also findings like the uncertainty estimates become unreliable under domain shift are intersting. Section 3.5 discussing disagreement analysis among the physicians is also an important point of discussion for this setup.

### Weaknesses
The authors have considered very simple methods and have not considered more recent methods for selective classification like the SelectiveNet or Self-adaptive training and have also missed some other works like Deep Gamblers since it is still not exactly clear which of these approaches will work the best for these tasks and whether they are a better alternative to these considered simple method. Also there was another recent paper [1] which proposed extensions to large language models to make them more relialble which also considered the Diabetic Retinopathy dataset and showed advantages for the selective classification setup. Secondly the authors have also missed out on popular calibration methods like focal loss [2]. Like the current benchmarks seem to consider somewhat standard/older methods and is not entirely convincing how much effectively (or not) the current best models can solve this problem. 

[1] Plex: Towards Reliability using Pretrained Large Model Extensions
[2] Calibrating Deep Neural Networks using Focal Loss

### Questions
Please see the weaknesses section. A major question is why only such limited baselines were considered which are not very recent and so it cannot directly indicate the state of model development for these tasks? Also are there any other possible tasks that can be included to make this study more comprehensive or maybe like does these tasks indicate something more about other relevant problems similar in nature?

### Soundness
3

### Presentation
3

### Contribution
3
