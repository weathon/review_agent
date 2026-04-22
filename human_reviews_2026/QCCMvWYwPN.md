# M$^3$Ret: A Mixed Multimodal Image Dataset and Benchmark for Personalized Multi-Retinal Disease Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In ophthalmic clinical practice, various imaging examinations, such as retinal fundus photography and OCT imaging, provide ophthalmologists with non-invasive methods to assess the condition of the retina and highlighting the importance of multimodal data. The imaging examinations are individually tailored according to each patient’s clinical condition, resulting in diverse modality combinations. However, existing multimodal ophthalmic imaging datasets only collected one combination of multimodal data for single disease detection. Correspondingly, previous multimodal models were designed to learn from a fixed combination of modalities, overlooking the personalized nature of clinical examinations and the variability in modality combinations. As a result, the models often fail to generalize well to real-world clinical applications. To bridge the gap, this paper proposes (1) $\mathbf{\mathsf{M^3Ret}}$, a $\textbf{M}$ixed $\textbf{M}$ultimodal ophthalmic imaging dataset for personalized $\textbf{M}$ulti-$\textbf{Ret}$inal disease detection, which consists of scanning laser ophthalmoscopy (SLO) images and optical coherence tomography (OCT) images and includes various modality combinations, and (2) $\mathbf{\mathsf{PersonNet}}$, a new baseline model for personalized multimodal multi-retinal disease detection, which can handles samples with various modality combinations during both training and inference phase, (3) benchmark results of our $\mathsf{PersonNet}$ and 13 existing multimodal learning methods, which demonstrate the superiority of the proposed $\mathsf{PersonNet}$ and highlight the significant room for improvement before clinical application can be achieved.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the gap between existing multimodal ophthalmic datasets/models and real-world clinical needs, where personalized examinations lead to diverse modality combinations and coexisting retinal diseases. Although this article still has some weaknesses in terms of the extensiveness and quality of the dataset, it has made significant progress compared to past work. In my view, this paper is marginally above the acceptance bar of ICLR.

### Strengths
1. Unlike existing datasets (e.g., FairVision, MMC-AMD) that only support single-disease detection or fixed modality combinations, M³Ret includes 7 mixed modality combinations (unimodal, bimodal, trimodal) and multi-label disease labels, fully reflecting personalized clinical examinations.

2. With 8,558 eye samples, it is one of the largest multimodal ophthalmic datasets. Its disease prevalence rates are consistent with real-world statistics (e.g., close to Teo et al.’s 4.07% ME and 6.17% DR), avoiding bias from overrepresented diseases (e.g., FairVision’s 48.7% glaucoma). 

3. Comprehensive experiments cover 13 baseline methods across complete and incomplete multimodal learning and explore subgroup analysis (age/gender fairness), and single-vs-multimodal comparisons, revealing limitations of existing SOTA and providing insights into model behavior.

### Weaknesses
1. M³Ret only includes three diseases (DR, ME, glaucoma), excluding other common retinal diseases (e.g., age-related macular degeneration (AMD)), reducing its applicability to broader clinical scenarios.

2. M³Ret is collected from a single hospital, lacking diversity in patient demographics (e.g., ethnicity, regional medical practices) and imaging devices (only Optos Panoramic 200 for UWF-SLO, CIRRUS HD-OCT 500 for OCT), limiting generalization to other clinical settings.

3. A portion of labels are derived from treatment records (not direct diagnoses) or marked as “unclear”, which may introduce noise into training.

### Questions
1. PersonNet’s fusion module uses modality combination-aware fully connected layers. How many such layers are there (one per combination?), and how do you avoid overfitting given the varying sample sizes across combinations (e.g., Tri-modal only has 118 samples)?

2. The paper suggests future work report computational metrics (e.g., FLOPS, FPS). Have you tested PersonNet’s performance on edge devices (e.g., mobile GPUs) to evaluate its deployment potential in primary care settings?

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
5

### Summary
The paper introduces M3Ret, a mixed-modality ophthalmic dataset with multiple modality combinations. 

It also proposes PersonNet, a baseline for personalized multi-disease detection with a memory-bank completion and fusion strategy. 

Tens of multimodal methods are benchmarked on this dataset.

### Strengths
+ The problem of multiple retinal disease and AI diagnosis is timely.

+ The dataset scale is attractive.

+ Overall the paper is easy-to-follow and clearly-presented.

### Weaknesses
- This paper claims “first” to support diverse combinations for eye disease, which is not true. Some prior works are listed as follows.

[1] Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset and Generalization-Reinforced Semi-Supervised Learning. [https://arxiv.org/abs/2308.13411]

[2] FairVision: Equitable Deep Learning for Eye Disease
Screening via Fair Identity Scaling [https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision]

- The center and vendor type in this work is not clearly detailed, and may be rather limited. Instead, cross-site testing and generalization to other vendors are needed, to enrich the diversity of the benchmark.

- For the evaluation metrics, it may be more rationale to consider precision, recall, F1-score and their class-wise metric.

- Besides, please do more analysis and show the baseline outcomes on the per-class per-disease performance.

- The technique novelties of the proposed method are limited. Specifically, memory-bank completion with class-wise prototypes is a straightforward instantiation of missing-modality completion.

- Since this paper considers the imcomplete modality settings, some stronger generative baselines should also be compared.

- The experiments and validation seem insufficient. For example, the sensitivity to prototype bank size/update. Besides, how does the class imbalance problem impact the performance?

### Questions
Please refer to the weakness section, and address these concerns point-by-point.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenges of incomplete modality combinations and multi-disease detection in ophthalmic multimodal image analysis by proposing the $M^3Ret$ dataset and the $PersonNet$ method. $M^3Ret$ comprises retinal images with diverse modality combinations and supports the detection of three retinal diseases. $PersonNet$ handles varying modality combinations through missing modality completion and personalized fusion strategies. The study conducts a benchmark evaluation of the proposed method alongside existing multimodal learning approaches on the $M^3Ret$ dataset and provides an analysis of the results.

### Strengths
1. The $M^3Ret$ dataset holds certain advantages in scale and encompasses diverse imaging modality combinations, which better reflects real-world clinical heterogeneity compared to existing datasets that only include complete modality pairs.

2. This paper identifies a limitation in current ophthalmic multimodal research, specifically, its reliance on fixed, complete modality combinations. And introduces the problem of personalized detection, which is more aligned with practical clinical scenarios.

### Weaknesses
1. All evaluations of the proposed PersonNet method were conducted solely on the $M^3Ret$ dataset, with no testing performed on external datasets. This limitation fails to demonstrate the method's generalization capability and broader effectiveness.

2. The core components of PersonNet, the class-wise prototype-based missing modality completion and the SKNet-inspired feature fusion—represent relatively straightforward and conventional approaches within the existing literature on incomplete multimodal learning. The paper does not sufficiently justify the significant innovation of the proposed method compared to existing techniques.

3. Critical ablation studies are reported only on the validation set, lacking final verification on the test set. This omission undermines the reliability of the conclusions drawn.

### Questions
As indicated in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces M$^3$Ret, a mixed multimodal ophthalmic dataset of 8,558 eyes / 5,235 patients covering UWF-SLO, macular OCT (128 B-scans), disc OCT (200 B-scans) with seven real-world modality combinations. It defines two benchmark suites (DA: multi-disease ME/DR/Glaucoma; DB: glaucoma) with stratified 6:2:2 patient splits, and provides PersonNet as a baseline supporting incomplete-modality inputs. The benchmark reports comprehensive performance metrics. Authors have release code and a data link and discuss ethics.

### Strengths
**S1.** This paper is well-organized and easy to follow.

**S2.** Seven observed modality combinations reflect routine workflows (uni-/bi-/tri-modal sampling), which is rare in this space.

**S3.** The paper reports age/sex distributions and disease prevalence close to epidemiology, supporting external validity. In addition, the provided dataset and code are both complete.

### Weaknesses
**W1.** Most labels come from EMR diagnoses; when those are missing, experts infer them from treatment records. Some cases are still marked ‘unclear,’ and there’s no inter-rater protocol, adjudication process, or label-noise audit.

**W2.** DA and DB are reorganizations of the same hospital cohort with stratified 6:2:2 splits, but it is not explicit whether splits are patient-disjoint within and across DA/DB, nor whether tri-modal cases can leak information between tasks.

**W3.** Benchmark scope is misses clinically critical metrics including ROC-AUC, calibration (ECE/Brier), or subgroup performance (e.g., age/sex strata) and analyses.

**W4.** Missing details on OCT resampling/padding and on whether duplicate scans are kept or discarded.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
