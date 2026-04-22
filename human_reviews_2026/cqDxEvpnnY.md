# Principles-Driven Machine Learning for UV Spectral Prediction

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Despite recent advances in machine learning, accurate UV spectral prediction remains a challenging task. UV spectra exhibit inherently broad absorption bands characterized by peak positions, band shapes, and curvature profiles that machine learning models still struggle to capture effectively.
Based on the ideas behind UV spectroscopic principles and analytical techniques, we present three methods that aims at identifying these three characteristics: Peak Position Awareness (PPA), Curriculum Learning for Interpolated Abstracted Spectra (CLIAS), and Spectrum Curvature Limitation (SCL).
Our performance evaluation shows that our methods can successfully capture genuine spectral characteristics, achieving consistent improvements over diverse models, especially when the training order of CLIAS and SCL is carefully considered in combination with PPA. We also show important evidence that our best models outperform the state-of-the-art UV-adVISor model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
I am concerned that the data volume used is relatively low for this area. In particular, with just over 3,000 examples, the training/testing partitioning strategy relies on random segmentation, without any criterion to prevent data leakage or other practices commonly used in the literature. I also find it notable that there is no comparison with other methods beyond a baseline, and that no standard partitions are used for method comparison. This means we do not really know how the method compares with its competitors, since the evaluation is performed on a split created by the authors themselves.

### Strengths
Using ML for molecular-level UV spectral prediction is an interesting topic addressed in this article. Specifically, of the three proposals, the results show that peak detection yields the best performance relative to a baseline. The other two variants produce only marginal differences compared with the first variant and the baseline.

### Weaknesses
- Only marginal improvements over baselines, with just one of the three proposals showing any gain.
- No benchmarks are used for the problem, which prevents comparison with competitive methods on standardised test data.

### Questions
Questions for the authors:

- Is it possible to compare performance on the same test data against other methods, avoiding splits created by the authors themselves?
- Can you explain qualitatively whether the observed MAE improvements translate into meaningful qualitative gains for the type of analysis conducted? 
- Could you show practical examples that make clear the improvement is relevant within the problem domain?

### Soundness
2

### Presentation
3

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
The paper proposes several enhancements to neural networks for the task of predicting absorption spectra of chemical compounds. The proposed improvements are:
 * predicting peak locations separately
 * curriculum learning focusing on peak locations first
 * regularizing the curvature of second derivative of spectra

These methods are independent of the network architecture, and experiments are performed with 4 different architectures.

### Strengths
* The paper works on a relevant problem where machine learning methods traditionally struggle.
* The results show a clear improvement in performance over the baseline model.

### Weaknesses
* The method is not sufficiently explained. In particular, it is not clear how PPA/PCM is integrated with the rest of the method. By itself this is just a classifier predicting peak locations. But how are these locations then used?
* Other unclear aspects:
  * "peak positions are computed by an algorithm to simply compare neighboring values for finding local maxima"
    Which algorithm?
    Do you mean that $j$ is a peak if $y_j > y_{j-1}$ and $y_j > y_{j+1}$?
  * "Semantically, yi,j is an absorption rate"
    Is this the absorption over some fixed path length and fixed concentration?
    Or are these values normalized per sample? Or is this normalized absorbance?
  * "manually selects a promising subset of $\{P_1, P_2 , ... , P_{k_a-1}\}$"
    What were the criteria for this selection?
    And does this means that you have to manually select these subsets for every training sample?
  * "P+C→P+S" what is that? It stands for "PPA with CLIAS and then PPA with SCL"
    But that doesn't explain what is actually going on. What does the resulting algorithm do?
  * Figure 3: "Histograms show the counts of the spectra, categorized by the absorption rates yielded for the 
  wavelengths in the captions."
    I don't understand what this figure is supposed to show.
  * It would help the structure of the paper if you say what kind of model is used. It only becomes clear that the method is generic over the model architecture in the experiment section.

### Questions
* "the difference in wavelengths between two adjacent indices in P are identical among all located between closest peak positions, or index 0 or N − 1."
   what does this mean?
* The curvature limitation is implemented as a squared error term, so a high second derivative is penalized, but so is a low one. So this means a specific second derivative $b_\text{cur}$ is preferred. Would a simple squared error term not work?
 * "with random seeds (42, 123 and 456)"
   without giving the code, reporting the random seeds is useless.
 * "since peak positions are not apriori knowledge on the target molecule,"
   -> "since peak positions are not apriori known on the target molecule,"
 * "we also include All trains a model by incorporating" ?
 * "binary vector $pv = (v_0, v_1,... , v_{N-1})$,"
   Use single letter variable names. In this case the vector should just be called $v$.
 * "S_i is a sequence yi,0 → yi,1 → · · · → yi,N −1 ,"
   Write a sequence as "$[y_{i,0}, y_{i,1}, ..., y_{i,N-1}]$"
 * "wavelength (w + j) nm", this means that the sequence is fixed to a rate of 1 sample per nm.
   Wouldn't it make sense to allow this to be flexible (i.e. (α + βj) nm)?
   As a convenient notation, I would recommend to define a vector $w$ of wavelengths, so $v_j$ is the absorption at wavelength $w_j$. This can simplify the presentation in later parts of the paper.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of UV spectral prediction, where machine learning models often fail to capture key physical characteristics such as peak positions, band shapes, and curvature. The authors propose three "principles-driven" methods to target these issues individually: Peak Position Awareness (PPA), an auxiliary classification task for peak locations; Curriculum Learning for CLIAS and SCL. Experiments are conducted on a dataset of 3,170 spectra, applying these methods to four baseline architectures (MLP, LSTM, Transformer, BiLSTM). The results indicate that a specific multi-stage training combination achieves consistent improvements over the baselines and reportedly outperforms the current SOTA UV-adVISor.

### Strengths
1. The wavelength-specific analysis (Section 5.3) is a strength. It moves beyond a single aggregate metric (mean MAE) to show that the improvements are concentrated in less frequent, but physically significant, absorption regions, which is a valuable finding.

2. The paper provides a direct comparison against the current SOTA (UV-adVISor) on its original datasets (Table 2) and demonstrates superior performance, which strengthens the paper's claims of practical utility.

### Weaknesses
1. The core technical contributions are weak. PPA is a standard application of auxiliary task learning, CLIAS is a standard application of curriculum learning, and SCL is a straightforward  application of a custom, physics-informed regularization loss. The paper's contribution is primarily an application and combination of existing techniques to a specific scientific domain, which appears to be below the novelty bar

2. All claims are based on a very small dataset of 3,170 spectra. This is insufficient to reliably evaluate the generalization of such a complex, multi-stage training procedure, especially for data-hungry models like the Transformer. The reported gains may be a result of overfitting this specific, small dataset rather than a genuine, generalizable improvement

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes three training/regularization ideas tailored to predicting UV–Vis spectra with few, broad features. Peak Position Awareness (PPA) adds an auxiliary classifier that predicts peak locations and feeds a peak one‑hot vector to the spectral regressor; CLIAS (curriculum learning on interpolated, abstracted spectra) trains on progressively denser wavelength grids; and SCL penalizes excessive second‑derivative curvature to enforce physically realistic line broadening

Using SMILES‑token encoders with MLP/LSTM/Transformer/BiLSTM decoders, the authors train on a merged experimental dataset of 3,170 spectra (230–400 nm, 1 nm resolution) and report consistent 15–23% MAE reductions over baselines, with a two‑stage schedule (P+C → P+S) outperforming “all‑at‑once” combinations

Wavelength‑specific analysis indicates gains concentrate on infrequent but important absorption regimes (e.g., longer wavelengths or low‑frequency bins), and qualitative examples show smoother, more realistic predictions after applying PPA/CLIAS/SCL

### Strengths
This is the reason why experts battle the bitter lesson - throwing in all kinds of inductive bias into models. And it helps (with 3K points) PPA reflects chemists’ workflow (identify peaks first), SCL encodes qualitative broadening constraints, and CLIAS teaches band envelopes before fine detail; this produces architecture‑agnostic gains across MLP/LSTM/Transformer/BiLSTM. 

Clean ablation of the three ideas; results are reported with multiple seeds and both MAE/RMSE. The wavelength‑wise distribution and per‑bin MAE plots are informative, showing that improvements arrive where data are sparse and naive losses would otherwise under‑weight errors

### Weaknesses
"While training neural networks (NNs) is one way to predict UV spectra" i mean sure. this paper is also doing that. Maybe make a more concrete point ? 

'An essential challenge related to UV spectroscopy is also raised.' by whom ? 

Why only UV ? HIstorically the equipment that measures these also measures (more easily!) visible light. Why not do UV-Vis like everyone else?  The only use use i can think of this is maybe detectors for liquid chromatography, since most organic molecules don't have too much going on in the visible? The paper does not do a good job at justifying why UV only. 

More broadly- why not apply this to other clases of spectra with broad shoulders in the sciences ? (saxs?)  

Why not use a larger dataset? There must exist bigger training datsets for UV. UV-VIs papers train on like 100K 

All models rely on SMILES tokenization; message‑passing GNNs (with aromaticity/bond order), 3D‑aware or equivariant models, and pretrained molecular encoders are known to help spectral tasks. Even the paper notes these could be “combined,” but they are not evaluated. Including an MPNN baseline on graphs (or 3D à la McNaughton et al. 2023, which the paper cites) would test whether PPA/CLIAS/SCL still provide additive value beyond stronger representations

PPA’s ground‑truth peaks are obtained via local‑maxima detection with manual thresholds (height, distance, width), which injects arbitrary transformations into the task. The curriculum uses hand‑selected grids (e.g., {43, 86, 171}) and linear interpolation for missing wavelengths . This risks schedule‑tuning to a dataset. A more principled scheme (e.g., learn the curriculum, use spectral‑domain smoothing priors, or parametric envelope targets) would improve robustness and portability.  

While the paper acknowledges that mean MAE under‑weights rare absorptions, the primary model selection still uses global MAE/RMSE. 

The fact that UV-adivsor results are taken at face value and cannot be replicated using the same splits and metrics would be better. 

The poor performance of transformer underperformance is poorly explained

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
