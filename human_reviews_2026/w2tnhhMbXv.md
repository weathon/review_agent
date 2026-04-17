# cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning

- Decision: Accept (Oral)
- Scores: 8, 2, 6, 8, 6

## Abstract
Computer-Aided Design (CAD) plays a central role in engineering and manufacturing, making it possible to create precise and editable 3D models. Using a variety of sensor or user-provided data as inputs for CAD reconstruction can democratize access to design applications. However, most existing methods focus on a single input modality: point clouds, images, or texts, which limits their generalizability and robustness, while few multimodal approaches struggle to deliver competitive quality. Leveraging advances in vision-language models (VLM), we propose $\texttt{cadrille}$, a multimodal CAD reconstruction model that takes inputs of three modalities and outputs executable Python code for CAD reconstruction. Inspired by large language model (LLM) training paradigm, we adopt a two-stage pipeline: supervised fine-tuning (SFT) on large-scale procedurally generated data, followed by reinforcement learning (RL) fine-tuning using online feedback, obtained programatically. In the DeepCAD benchmark, our SFT model outperforms existing single-modal approaches in all three input modalities simultaneously. More importantly, after RL fine-tuning, $\texttt{cadrille}$ sets new state-of-the-art in as many as 10 benchmarks across three modalities and four datasets, including a real-world one.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a multi-modal CAD reconstruction model that takes point clouds, images and text as input modalities and outputs executable python code in the CADQuery language. 

A pre-trained Qwen2-VL-2B is first fine tuned on synthetic data and then reinforcement learning (RL) fine tuning employed, using datasets of CAD models create by humans.  This final RL fine-tuning stage does not require a CAD command sequence and could even be performed using mesh data.  This rather unique capability has the potential to greatly expand the amount of data available for training.  

The method is evaluated on four datasets, including one containing realistic point clouds.  The model is shown to be state of the art when compared with 10 benchmark methods.

### Strengths
This work sets new state-of-the-art for the generation of CAD models using the "CAD command sequence" approach.

The large number of benchmark methods and datasets used for evaluation is impressive.

The ability to train using CAD data from step files or even just 3D meshes is a big benefit.

The good performance on the realistic CC3D is particularly encouraging

It's also interesting and exciting  that  hard example mining was found to give faster convergence during RL fine-tuning

### Weaknesses
Figures 1 and 2 are not very inspiring.  While they include all the important details, they somehow don't make the visual impact which should be possible given the quality of the results.  Perhaps just adding more images to them would help.  

The quantitative comparison between DPO and Dr. CPPO were welcome, but leave me wondering about the decision to use Dr. CPPO over vanilla Dr GRPO or CPPO.  Clearly online RL was beneficial here, but it's hard to tell if the Dr. CPPO algorithm has any benefit over more traditional methods.  I guess that vanilla GRPO wouldn't be possible because of the need to keep the reference model in vram for the KL loss.

### Questions
Dr. CPPO
The expensive part of a CAD RL workflow is usually sampling from the model, rebuilding the CAD and computing the rewards.  Was there any evidence that dropping all but the N samples with the most extreme advantages helped?   More details would be very interesting.

It's a big advantage of this method that the CAD command sequence is not required for the RL fine tuning phase.   This would open up the possibility of training on the large ABC dataset.  I would be very interested to hear about results for this kind of experiment.  Are there problems with unsupported features in the CADQuery language  like edge blends?

Abstract:
"...in the DeepCAD benchmark...
I suggest "... on the DeepCAD benchmark"

Line 211
"...the invalidity ratio (IR) is to 10%, "
Should this be “is up to 10%”?

### Soundness
4

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
This paper introduces a multimodal CAD reconstruction model that unifies point clouds, multi-view images, and textual descriptions as input modalities to generate executable Python code for parametric CAD models.

Building upon a multi-modal architecture, cadrille adopts a two-stage training pipeline: supervised fine-tuning (SFT) on large-scale procedurally generated CAD data followed by reinforcement learning (RL) fine-tuning using handcrafted data.

Evaluated on four datasets (DeepCAD, Fusion360, CC3D, and Omni-CAD), cadrille achieves state-of-the-art results in 10 benchmarks spanning all three input modalities.

### Strengths
1. Writing and presentation of this paper are highly complete and easy to follow.

2. Comparison and ablation of this paper are thorough and comprehensive.

3. Datasets involved in experiments are various.

### Weaknesses
**The reviewer has several serious concerns on some statements in this paper the reliability of evaluation and experiments:**

1. Statement at Lines 52-53:

> "However, the first multimodal methods in that vein (Xu et al., 2024b; Wang et al., 2025b) are dramatically inferior to single-modal approaches, so the full potential of VLMs for CAD reconstruction is yet to be unleashed."

And statement at Line 139:

> "Recent CAD-GPT (Wang et al., 2025b) predicts a CAD model given a single image and textual description, while CAD-MLLM (Xu et al., 2024b) pioneers three-modal CAD reconstruc- tion, yet both these methods fall behind single-modal state-of-the-art results (Rukhovich et al., 2024; Khan et al., 2024b; Chen et al., 2025) by a large margin (up to two orders of magnitude!)."

Might be improper. The authors claim that single-modal state-of-the-art methods' performance is "up to two orders of magnitude!". However, since CAD-GPT and CAD-MLLM have not yet been publicly released as far as I know, it is unclear how this quantitative comparison was obtained. **Without transparent and reproducible evaluation settings aligned across all methods, such a strong claim (especially “up to two orders of magnitude!”) might be misleading.**

Moreover, geometric evaluations—such as those based on Chamfer Distance—are typically preceded by a scaling operation that normalizes shapes into a common coordinate space (e.g., $[−1, 1]^3$ or $[-100, 100]^3$). Differences in normalization schemes, evaluation implementations, or testing environments can significantly affect the absolute magnitude of reported errors.

For example, comparing error values across differently normalized spaces (e.g., $[−100, 100]^3$ vs. $[−1, 1]^3$) is not meaningful without proper alignment or clarification. **Authors are generally expected to maintain consistent scaling in the normalized space across all compared methods to ensure a fair and meaningful evaluation.** (But the reviewer doesn't find the elaboration on the scale of this normalized coordinate space used in this paper's experiment.)

2. Building on the previous concern, the reviewer only observes that in Line 322:

> "CD values are multiplied by $10^3$"

**The authors have not yet specified the scale of the normalized coordinate space used for Chamfer Distance (CD) evaluations in the main text.**

Line 809 says that “we report mean CD since it is the only CD metric reported by CAD-MLLM”.

CAD-MLLM reports its normalized space is $[-0.5, 0.5]^3$. In Table 4, is the scale of normalized space identical to $[-0.5, 0.5]^3$ for all tested methods? (including DeepCAD, CAD-MLLM, Point2CAD, cadrille, any other baselines or methods)

The reviewer is therefore concerned that the comparisons of CD values across methods may not be valid **unless all approaches were evaluated in the same normalized space**. The reviewer thinks it's also essential to explain why LLM-based method cadrille outperforms Point2CAD with one orders of magnitude, since Point2CAD has a setting to fitting/reconstruct input point cloud with paramatric geometry primitive. (Since the proposed method is reported to be better than Point2CAD, how is comparison with sota-method [1] within that setting?)

Consistent normalization is essential for meaningful quantitative comparison, and this detail should be explicitly clarified.

Although the experiments are comprehensive, **the reviewer has serious concerns about the validity of the reported results due to the issues outlined above**—particularly the lack of clarity regarding normalization protocols and the unverifiable performance claims against unreleased methods.

References:

[1] Split-and-Fit: Learning B-Reps via Structure-Aware Voronoi Partitioning, Liu et al. (https://arxiv.org/abs/2406.05261)

### Questions
1. Between lines 393 and 395, the results appear identical across all columns except for the IR of CAD-Recode when conditioned on the point cloud. Is this result typed correctly?

(The reviewer gives a score of 2 because of **serious concerns about the correctness and reliability of the paper’s experiments**. If these issues are clearly explained and properly addressed, the reviewer is open to raising the score. The reviewer acknowledges the authors’ efforts in presenting comprehensive experiments and analysis. However, for the conclusions to be meaningful and useful to the research community, the evaluation must be methodologically sound and correct. Once the reviewers' concerns on experiments are resolved, the reviewer may also have additional questions about the paper’s core contribution.)

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes cadrille, a multimodal framework for CAD reconstruction that unifies three input modalities (point clouds, multi-view images, and text descriptions) into a single generative model that outputs executable Python CAD code. cadrille combines SFT strategy on large-scale procedural CAD data with  RL fine-tuning on real handcrafted datasets using programmatic rewards. The proposed hybrid training pipeline significantly improves model validity and generalization across domains. Experiments on such as DeepCAD, Fusion360 dataset demonstrate consistent SOTA results across ten benchmarks.

### Strengths
1. The paper evaluates the proposed method on four different datasets (DeepCAD, Fusion360, CC3D, and Omni-CAD) and covers three input modalities. The reported performance improvements are significant and consistent, supporting the effectiveness of the proposed method.

2. The SFT (on procedural data) + RL (on handcrafted data) approach is a highly effective and novel solution to the well-known domain gap problem. Integrating LLM-style RL fine-tuning into multimodal CAD reconstruction is a novel and impactful idea.

3. Handling point clouds, images, and text within a single framework is technically elegant and practically valuable.

4. The paper includes excellent ablation analyses. It proves that its RL adaptation strategy is superior to a naive SFT on a mixed dataset

### Weaknesses
1. The model inherits high computational overhead from large VLMs, but efficiency metrics are not reported.

2. The SFT stage relies on the CAD-Recode dataset, which the authors note lacks certain operations (e.g., symmetric extrusions) found in DeepCAD. It is unclear if the RL stage can learn to generate new CAD operations it has never seen during SFT.

3. CAD-MLLM is supposed to  be compared and discussed in the experiment section, as this would more effectively demonstrate the superiority of the proposed model. Currently, its brief mention in the related work section is insufficient and appears too weak to prove the advanced performance of proposed model.

### Questions
1. Table 1 does not show the results of cadrille using only Rp. When cadrille is trained using only Rp, is it completely equivalent to CAD-Recode?

2. Regarding the results in the fourth row of Table 2, how is Rpi+Dpi mixed? Does it sample from both datasets with the same probability in each iteration, or just sequentially fine-tuned using Rpi first and then Dpi?

3. How sensitive is the RL fine-tuning performance to the weighting between the IoU and invalidity penalties in the reward function?

4. Table 1 does not show the results of cadrille using only Rp. When cadrille is trained using only Rp, is it completely equivalent to CAD-Recode?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Cadrille, a multimodal CAD reconstruction system that unifies three input modalities—point clouds, multi-view images, and text—to produce executable Python CAD programs. The model builds upon a vision-language backbone and adopts a two-stage training paradigm: (1) SFT on large-scale synthetic CAD data, followed by (2) RL fine-tuning using programmatic IoU- and validity-based feedback. Cadrille achieves state-of-the-art results on multiple benchmarks varied by datasets and experimental setups.

### Strengths
Cadrille is, to my knowledge, the first CAD framework that jointly handles point clouds, images, and text inputs in a single model while maintaining competitive performance with single-modal baselines.

The quantitative results are comprehensive and informative. The authors evaluated a wide range of existing methods to demonstrate the model’s advantages across them, including experiments on multiple datasets and under various setups.

The qualitative results further highlight the model’s contributions, showing clear improvements in generation quality. The selected examples are non-trivial and effectively illustrate the strengths of the proposed approach. Along with the quantitative results, this work presents strong empirical coverage on its effectiveness.

The paper carefully details the multimodal data generation pipeline, reward definition, and training setup, contributing valuable reproducibility and practical insight to the community.

### Weaknesses
While the authors stated that their two-stage training strategy is inspired by LLM pretraining, the core SFT + RL paradigm is nearly identical to previous works such as CADFusion [2] and CAD-Coder [1]. The main difference lies in applying this strategy across modalities rather than within a single one. This distinction should be discussed more explicitly; the claim of “first to use RL fine-tuning for multimodal CAD” is true, but the conceptual framework remains evolutionary rather than fundamentally new.

[1] CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation. ArXiv.

[2] Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models. ICML 2025.

### Questions
This seems to be primarily a naming issue: the term “multimodal CAD reconstruction” is somewhat confusing, as the tasks defined in this work do not involve reconstructing original CAD shapes. In my view, “multimodal CAD generation” would be a more accurate description.

Could the authors clarify the backbone model size and report ablations on model scaling to ensure fairness versus the baselines?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents cadrille, a multimodal CAD reconstruction framework that unifies three input modalities (point clouds, multi-view images, and textual descriptions) to generate executable Python code for parametric CAD models. Built upon the Qwen2-VL vision-language model, the approach employs a two-stage training paradigm: (1) supervised fine-tuning (SFT) on large-scale procedurally generated data (from CAD-Recode), and (2) reinforcement learning (RL) fine-tuning on handcrafted data without requiring CAD sequence annotations. The method achieves state-of-the-art results across 10 benchmarks, with improvements in reducing invalidity ratios.

The key contributions are: (1) a multimodal CAD reconstruction method achieving SOTA performance, (2) a novel training strategy that separates large-scale synthetic data for SFT and smaller real-world data for annotation-free RL fine-tuning, and (3) comprehensive evaluation demonstrating robustness across synthetic and real-world scenarios.

### Strengths
1. SOTA: This work introduces a unified multimodal model for CAD reconstruction can match or exceed specialized single-modal approaches. 
2. Training paradigm: The separation of procedurally generated data (SFT) and handcrafted data (RL) elegantly addresses the volume-quality tradeoff in CAD datasets. The key insight that RL can work without CAD sequence annotations (only requiring meshes) substantially lowers the barrier for future work.

### Weaknesses
1. Insufficient architectural innovation: The core contribution is training methodology rather than model design.

2. Critical phenomena lack adequate explanation: Cross-modal transfer (RL on images improves point clouds): The paper acknowledges this as "surprising" but provides no mechanistic explanation or controlled experiments. Possible hypotheses are not tested.

3. RL fine-tuning design choices lack justification.

### Questions
1. Cross-modal transfer mechanism: Can you explain why RL on images improves point cloud reconstruction? Does RL on point clouds also improve image reconstruction?

2. Dataset mixing failure (Table 3, row 4): The negative result when mixing Rpi+Dpi is attributed to "operation inconsistency", with the authors noting that "DeepCAD models are constructed using commands like extruded cuts and symmetric extrusions, which are not present in the generation procedure of the CAD-Recode dataset." However, this explanation raises questions: extruded cuts and extrusions are fundamental CAD operations that should theoretically be present in both datasets.

3. RL on point clouds: Why was RL fine-tuning not performed on point clouds? Why only images?

4. Hard example mining: R_th = 7.5 appears hand-tuned. How sensitive is performance to this threshold?

5. Can the author release the code for their paper？

### Soundness
3

### Presentation
2

### Contribution
3
