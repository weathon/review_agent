# Rethinking Convergence in Deep Learning: The Predictive-Corrective Paradigm for Anatomy-Informed Brain MRI Segmentation

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 6, 6, 2, 4

## Abstract
In medical image segmentation, although end-to-end deep learning has achieved substantial progress, obtaining accurate results typically requires many training iterations and large-scale annotated datasets, which limits efficiency and practicality in data-scarce clinical scenarios. To address this issue, we propose a Predictive–Corrective (PC) paradigm that decouples segmentation into a fast, anatomy-informed prediction stage followed by a focused refinement stage. Based on this paradigm, we develop PCMambaNet, which comprises two cooperative modules: a Predictive Prior Module (PPM) that generates a coarse anatomical approximation at low computational cost and injects symmetry priors via inter-hemispheric similarity and thresholding to highlight diagnostically relevant asymmetric regions, and a Corrective Residual Network (CRN) that models the residual error, concentrating capacity on refining challenging regions and delineating pathological boundaries.
Experiments on multiple high-resolution brain MRI benchmarks show that PCMambaNet attains competitive accuracy with relatively few training epochs and exhibits clear advantages in data-limited settings. Extended experiments further indicate that the proposed PC paradigm remains applicable to organs without strong left–right symmetry. Overall, this work demonstrates that explicitly incorporating anatomy-informed priors and decoupling prediction from refinement is an effective way to improve both training efficiency and data efficiency in medical image segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a "predictive-corrective" approach to medical image segmentation that uses anatomical inductive biases to reduce the effective capacity of a model, splitting inference into a "predictive prior module" (PPM) and then a "corrective residual net" (CRN), making it converge faster, require less training data, and perform better, tested on brain MRI segmentation tasks.

### Strengths
1. The paper is written and presented well, and organized in an understandable, digestible fashion. It is well-contextualized in the field and related work.
2. The approach is intuitive, and it makes sense that certain medical imaging tasks could benefit from inductive bias assumptions based on key domain knowledge that makes things easier to learn, and with less data. Why use a sledgehammer when a regular hammer suffices?
3. The experimental design is solid: a strong variety of evaluation metrics, strong baseline models, etc.
4. Results on the evaluated datasets are strong: improvement over other models using many epochs is not huge but it is consistent, and those models are hard to beat (Table 1). Single epoch performance is impressive, although single epoch performance of other models (Fig 4) aren't that far behind.
5. Ablation studies (table 2) are thorough and make me convinced of the use of the different components.

### Weaknesses
**Major Weaknesses:**
1. I feel that maybe taking this inductive bias assumption to make things a bit easier to train may only work so well for fairly simple individual dataset/modality scenarios with not as much variety in the data, and/or simple segmentation tasks. In the authors words: "Our core insight is that a complex modeling task can be effectively decoupled into two simpler, more manageable sub-tasks" it's unclear how well this intuition extends into other datasets, modalities, and tasks, as only brain MRI (which has unusually high spatial symmetry compared to many medical image analysis tasks) is evaluated, and the segmentation task is a fairly easy one at that. Overall, I am left wondering if this approach would generalize well to other scenarios, yet only this single modality and segmentation task is evaluated. The paper would therefore very much benefit from evaluating on another task in biomedical image segmentation, or at least a more challenging task in brain MRI, such as tumor segmentation, as the proposed simple model's superiority could be due to the evaluated task being fairly easy.
    1. Another concern in regards to the generality of the approach: take a more challenging task also in brain MRI, of tumor segmentation (e.g. as in the BraTS challenge). Is the mask (eq 8) designed to filter out abnormalities prone to false positives in such scenarios, such that this approach would no longer work? Are there types of noticeable assymetries in brain MRI that are healthy?
2. This was also touched on in my previous bullet, but overall, the generality of the proposed method as a useful tool for medical image segmentation would be much more substantiated if another modality/segmentation task was tested. 
3. The approach is quite computationally expensive (see Table 7) yet the limitations section discussing this isn't in the main text. As shown in Tab. 7,  the model is quite compute intensive, especially compared to U-Net which has solid performance, yet is orders of magnitude smaller and faster! The tradeoff of greatly increased computation for relatively small performance gains (Table 1) is not too convincing to me, although to be fair, in some medical imaging scenarios, training set size is the bottleneck, not compute resources. This is a key limitation and should be made clear in the main text, such as having dedicated sentences covering limitations in the conclusion section.  

**Minor Weaknesses:**
1. The proposed method's single epoch performance is impressive, although single epoch performance of other models (Fig 4) aren't that far behind.
2. Sec 3.3/Table 3, data efficiency analysis: Seems promising but needs to be compared to some baseline models for the same training amount percentages to demonstrate that the model is more efficiency (which does seem to be implied by Fig. 4).
3. Small formatting things: use \citep{} for citations to make them parenthetical. Write quotations with ``x'' in LaTeX. The text in certain plots, e.g. Figure 1, is too small to read without significantly zooming in. Typo : "The numben of train" in Table 3.

**Overall Justification of my rating and thoughts for revisions:**
The approach is technically solid and well-motivated for this domain. However, the limited scope of only being evaluated for a fairly easy segmentation task in brain MRI, and the possibility of the inductive biases used being constrained to only this evaluated scenario, makes the generality and impact of the work unclear. Also, the significant computation time (Table 7) needs to be addressed, as it is not mentioned in the main text, yet is a clear limitation. The approach has promise, and I think the most important way to improve the work and demonstrate its impact at the level of ICLR (as opposed to a venue like MICCAI, which it is more ready for in its current state) is via evaluation on some other medical image segmentation dataset/modality/task where these types of spatial inductive biases will be useful, or at the very least, on a harder task in brain MRI such as tumor segmentation (See e.g. the BraTS challenge).

### Questions
1. Why is a mamba backbone used? could the contextual modulation factor multiplied with the hidden state (eq 6) be formulated for other model types/backbones? Since its just computed using the "symmetry checking" mask, I think this could be integrated into different models' feature maps/representations (e.g. CNNs, transformers) etc. I'm curious how for CNNs especially this could help further make the model simpler and faster, which could help with the high computational cost (Table 7).
2. How does this perform under domain shift/how does the simplicity of the inductive bias affect overfitting to the training domain, and generalization? For example, training on your first dataset and testing on the second dataset, or vice-versa.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper identifies a key limitation of the "brute-force" end-to-end paradigm: slow convergence and a heavy reliance on large datasets, which is particularly problematic in medical imaging. The authors propose a new Predictive-Corrective (PC) paradigm to accelerate learning by decoupling the modeling task. The paper instantiates this paradigm in a model named PCMambaNet for brain MRI segmentation. This model consists of two components: a Predictive Prior Module (PPM) that leverages anatomical symmetry to focus computational efforts, and a Corrective Residual Network (CRN) that refines segmentation by modeling residual errors. Experiments demonstrate PCMambaNet achieves state-of-the-art accuracy within just 1–5 epochs, substantially outperforming traditional end-to-end methods.

### Strengths
1. The "Predictive-Corrective" paradigm is an insightful conceptual contribution. The idea of decoupling the task into a coarse, prior-driven prediction and a focused residual correction is an elegant and powerful alternative.

2. Good experimental results. The experimental evaluations are thorough, demonstrating clear advantage in convergence speed and segmentation accuracy compared to established baselines.

### Weaknesses
1. Computational efficiency is compromised due to the added complexity of the predictive-corrective structure. Although superior in speed of convergence, the model will likely be considerably slower in inference than simpler baselines

2. The predictive module relies heavily on predefined anatomical symmetry, limiting generalizability to medical tasks lacking clear structural symmetry or well-defined anatomical priors.


3. Experimental validation is primarily limited to brain MRI segmentation. More extensive validation across diverse medical imaging tasks and modalities would strengthen the broader applicability and robustness claims.

### Questions
1. How does PCMambaNet handle inaccuracies from the predictive prior module, especially when encountering cases with significant anatomical anomalies?

2. Could the authors elaborate on potential strategies to reduce computational overhead and enhance inference efficiency without sacrificing the predictive-corrective paradigm’s key advantages?

3. Has PCMambaNet been tested on tasks beyond brain MRI segmentation, and if so, how does performance compare to traditional end-to-end models in those contexts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript proposes a Predictive-Corrective (PC) paradigm, which decouples modeling into two stages to accelerate convergence. The proposed PCMambaNet integrates a Predictive Prior Module (PPM) that leverages brain symmetry to generate coarse "focus maps" of the regions, and a Corrective Residual Network (CRN) that refines these regions for precise segmentation. By using the anatomical priors, the method achieves state-of-the-art brain MRI segmentation within 1-5 epochs.

### Strengths
- Ablation studies isolating PPM and CRN contributions.
- Quantitative comparisons against multiple baselines (e.g., UNet, SwinUNETR, nnUNet).
- Analysis of convergence dynamics under limited data.

### Weaknesses
Major:
- The Predictive Prior Module assumes that the brain exhibits ideal bilateral symmetry. This assumption may not hold under patient-specific rotations, misalignments, or post-surgical deformations. In practice, even small registration errors or head tilts could distort the left-right difference map and might mislead the Corrective Residual Network. It would be valuable to discuss robustness under affine transformations or to evaluate performance after introducing controlled perturbations. 
- Experiments are limited to brain MRI segmentation tasks. The narrow scope may limit the paper's interest and impact for the broader ICLR audience.


Minor:
- Table 1 does reports results as mean +- or confidence interval. No statistical testing has been performed.
- It will be great if you label the models and ground truth columns in Figure 3.

### Questions
- The current title suggests a general rethinking of convergence. However, when reading it, the content does not match the title's expectations. The actual contributions are focused on a predictive-corrective architecture for anatomy-informed brain MRI segmentation that converges pretty fast to an accurate segmentation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a Predictive-Corrective (PC) paradigm aimed at accelerating training and improving data efficiency. The authors decouple the learning process into two stages: a lightweight predictive module that provides an initial coarse estimation using domain priors (the bilateral symmetry of the human brain), and a corrective residual network that refines the output by focusing on residual errors. The framework is instantiated in the context of high-resolution brain MRI segmentation, where the bilateral symmetry of the brain is used as anatomical prior knowledge. Experimental results show that the proposed method achieves state-of-the-art segmentation accuracy while converging within only a few epochs, demonstrating both efficiency and robustness.

### Strengths
- The paper tackles an important and realistic challenge in medical imaging, the limited amount of labelled data compared to natural image domains.

- The proposed method shows that satisfactory segmentation performance can be achieved with only a few training epochs, which is promising for brain MRI applications.

- The approach achieves state-of-the-art (SOTA) performance, demonstrating its effectiveness.

### Weaknesses
- Writing and presentation issues:

(1) Figures 1 and 2 are too small, and Figure 2, as the main framework illustration, is not clearly presented. The inclusion of training epochs in the framework diagram is confusing; such information belongs to the experimental section rather than the model design. The conceptual logic of the framework should be emphasised instead. (2) Figure 3 lacks clear explanations (e.g., what each column represents). (3)There are numerous typos and formatting errors throughout the paper. For example, in Table 3 “numben” should be “number,” and punctuation spacing errors like “Figure 1 ``baseline.By''” occur frequently. These mistakes give the impression of a lack of careful proofreading.

- The introduction suggests a general solution for accelerating and data-efficient learning in medical imaging, but the experiments focus only on brain MRI segmentation, which limits the generality of the claimed contributions.

- The method mainly leverages a simple prior (brain symmetry) for segmentation, which is not plug-and-play or easily transferable to other tasks.

- Although the method converges quickly at the beginning, the final convergence takes roughly the same number of epochs as previous methods, which weakens the claim of significantly accelerated training.

- The ablation study only includes Mamba as the baseline. The authors should clarify why this was chosen and include comparisons with other relevant baselines, not only the current SOTA.

### Questions
Besides the weaknesses mentioned above, I am also curious about the hyperparameter $\theta$: how sensitive is the model to this parameter, and what value was ultimately used?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces PCMambaNet, a segmentation network built on the Predictive-Corrective (PC) paradigm, which decouples prediction and refinement to accelerate learning. The Predictive Prior Module (PPM) leverages anatomical knowledge and bilateral symmetry to generate a coarse focus map of diagnostically relevant asymmetric regions, while the Corrective Residual Network (CRN) refines these regions to produce precise segmentations. Experiments on high-resolution brain MRI show that PCMambaNet achieves state-of-the-art accuracy within 1–5 epochs in the presence of sufficient data, while still outperforming baselines when datasets are small.

### Strengths
-**Novel and creative paradigm for efficient segmentation** -The paper introduces the PC paradigm, which decouples coarse prediction (PPM) from fine refinement (CRN). This separation of the two blocks is well-motivated and offers a creative architectural approach that enables faster and more data-efficient learning.

-**Integration of anatomical prior as an inductive bias** -The model leverages the brain’s bilateral symmetry as an inductive bias, demonstrating how existing anatomical knowledge can improve segmentation performance in medical imaging, a domain where labeled data is often limited.

-**Demonstrated efficiency and data advantage** - Table 1 shows that PCMambaNet achieves faster training and strong performance when sufficient data is available, while also delivering improved results on smaller datasets.

### Weaknesses
1. **On clarity and definition of main claims**-  I find the claims in this paper unclear. The paper emphasizes faster convergence, but this is not evident from Figure 1. Visually, Figure 4(a) suggests that some methods take longer to reach their best DICE value, yet no quantitative metric for convergence is provided. How is convergence defined? For example, if we define it as the number of epochs required to reach a 5% margin of error and remain there, how many epochs does each method take? The claim would be more accurate if phrased as “better performance faster.” However, unlike the abstract statement that PCMambaNet “achieves state-of-the-art accuracy while converging within only 1–5 epochs,” Table 1 shows that for the small dataset, for better performance 200 epochs are required. I suggest the authors drop the convergence claim and instead explicitly frame these two objectives—(1) faster training and improved performance with enough data, and (2) improved performance with small datasets—as clear contributions in the introduction.

2. **On writing clarity and textual issues**- Some sentences in the manuscript appear unfinished, unpolished, or inconsistent, which reduces readability and can confuse the reader. Please see the Minor Comments section below for examples. 

3. **On generality of claims versus experiments**-The method relies on a symmetry-based prior, but it is not clear for which organs or imaging domains this prior is appropriate. All experiments are conducted on brain MRI datasets, which suggests that PCMambaNet may be specialized for the brain, and its applicability to other organs or medical imaging tasks remains uncertain.

**Minor Comments**

-Plots in figure 1 are hard to read; the fonts are small.
 
-Figure 3’s qualitative results are unclear: it is difficult to tell which segmentation corresponds to which method, and the column-dataset correspondence is ambiguous. The caption refers to arrows, but rectangles are shown in the images. Improving figure readability and clarifying labels would make the qualitative evaluation more interpretable.

-There is no explicit mention of the segmentation task early in the paper. Readers should know the exact task sooner, as it helps understand the method better.

-Table 3 does not include baselines, requiring the reader to cross-reference Table 1 multiple times to compare results. Including baseline methods directly in Table 3 would improve clarity and make comparisons more immediate.

-There is this seemingly unfinished sentence in section 3.2 "Role of the CRN." Did this intend to be the heading of a new paragraph? It currently reads like an incomplete sentence and may confuse readers.

-The sentence “First, relying s egraded accuracy and boundary quality, underscoring the need for a refinement stage” in Section 3.2 is unclear and appears incomplete. It should be rewritten for clarity.

-I had to go to the appendix to figure out which dataset was considered easy versus hard and how many samples were in each. This is core information that helps readers interpret Table 1 and should be in the main text.

 -It would have been more informative if the “data efficiency” experiments compared performance at each subset percentage against a baseline trained with the same number of samples. While it is true that for most metrics the results using only 10% of the data outperform baselines trained on the entire dataset, a direct comparison at each subset level is needed to fully understand how well the method performs under limited data conditions.

### Questions
I would appreciate it if you could clarify a few points:

**Q1**. Could you clarify what type of pathology, if any, is present in each dataset? It is unclear whether the MRBrainS13 brains include any pathological cases, which raises the question of how well the model is expected to perform on datasets without pathology. More broadly, can this method be applied to segment other organs with pathology? If so, why were only brain datasets used in your experiments?

**Q2.** In Figure 3 (left column), I don’t see much difference in the qualitative results across methods. Am I missing something? which one is the ground truth?

**Q3.** In Section 3.2 (Ablation Study), you claim that “On the small-scale dataset, removing the PPM or replacing it with a random mask causes a significant drop in performance.” I don’t see this for the case of removing the PPM. The improvement appears quite marginal—for example, for WM, the DICE score changes from 0.7468 without PPM to 0.7517 with PPM. Am I looking at the wrong row?

**Q4.** Also in Section 3.2, regarding “replacing our high-capacity CRN with a standard convolutional block results in a noticeable performance decline,” what I see from Table 2 is that for the small dataset the improvement is not significant. Interestingly, you see more improvement for the larger dataset. Why do you think that is? Also I am curios, how many parameters the "CRN" component add to the model compared to a simple convolutional block?

**Q5.** Could you provide a brief description of the dataset preprocessing steps? According to Section A4, MRBrainS13 contains only 20 subjects—how many training and test samples does this correspond to? Additionally, what image resolution is used as input to the network?

**Q6.** Regarding the Dice score reported in the text (“our model reaches a Dice score of 93.11%”): from the numbers in Table 3, this seems to correspond to GM. However, when comparing with U-Net, the reported Dice value is 93.32%, which from Table 1 appears to be the DICE value for WM. Could you clarify which class each reported number corresponds to?

**Q7.** In Section 3.4, I’m not entirely sure what the main argument is. Is it that the PPM helps improve foreground segmentation?You mention that the “feature maps are extracted from different layers.” Could you specify which layer indices are used, and map each feature map to its corresponding network layer (e.g., “feature map N corresponds to output of layer L3”)?

I am happy to reconsider my score if these concerns, discussed in both the weaknesses and questions sections, are adequately addressed.

### Soundness
3

### Presentation
2

### Contribution
3
