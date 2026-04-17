# Controllable Preference Alignment for Ambiguous Medical Image Segmentation via Text and Dice Guidance

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
In medical imaging, different experts often provide different but equally valid segmentations, making ambiguity an inherent challenge. A good model should therefore capture this variability by producing a distribution of plausible masks, rather than a single deterministic output. Diffusion models are well-suited for this task because of their ability to generate diverse samples, but standard training does not guarantee clinically meaningful segmentations. Prior work in ambiguous segmentation, such as diffusion-based approaches, lacks semantic control. This work introduces a multi-modal framework that makes diffusion-based segmentation both controllable and clinically aligned. The model is conditioned on input images and descriptive text from clinical metadata, and Direct Preference Optimization (DPO) is adapted by using Dice-based signals from multi-rater annotations instead of subjective human feedback. Three preference strategies are explored, with a consensus-based Mean Dice signal proving most effective. With DDIM sampling, inference is accelerated by a factor of three, making the approach practical for clinical use. Experiments on LIDC-IDRI demonstrate state-of-the-art segmentation quality while preserving diversity, and introduce a controllable preference knob that enables practitioners to directly adjust the balance between per-sample accuracy and distributional variability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper transfers preference alignment (DPO) to the setting of ambiguous medical image segmentation, uses multi-annotator Dice to automatically construct preference pairs, and couples it with a text-conditioned reference diffusion model (AmbiSeg).

### Strengths
(1) This paper presents a diffusion-based architecture that incorporates textual descriptions as conditioning inputs, enabling the model to generate clinically guided and ambiguity-aware segmentations.

(2) This paper adapts preference optimization (DPO) to ambiguous medical segmentation, exploring consensus-based, expert-specific, and stochastic preference strategies.

(3) Extensive experimental results show the effectiveness of the propsoed framework.

### Weaknesses
(1) Quantitative comparisons do not explicitly mark the best/second-best results (e.g., bold/underline/shaded cells), reducing table informativeness and comparability.

(2) The manuscript repeatedly emphasizes a controllable preference knob, but the Methods section does not define this knob’s specific variable, semantics, or scope (which parameter family it belongs to, its tunable range, and how it affects the objective).

(3) In the two-stage framework, Stage-2 (TDG-DiffDPO) directly adopts an existing Diffusion-DPO objective; no new training form is introduced. Overall, it reads as “porting DPO to segmentation with textual conditioning and constructing preferences via Dice scores,” rather than proposing a new preference-optimization principle or diffusion-alignment framework.

(4) All experiments are conducted on the single LIDC-IDRI dataset; the lack of cross-center/cross-modality or external validation makes it difficult to substantiate the claimed stability and generality of “controllable alignment” under distribution shift.

### Questions
(1) Figure 1 is not cited in the main text, which harms narrative completeness and retrievability.

(2) Notation is inconsistent—for example, in Sec.3.2, the image is first denoted by b and later by x; the notation should be unified throughout to avoid ambiguity.

(3) Effects of the parameter $\lambda$ should be investigated.

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
3

### Summary
This work proposes a two-stage framework for ambiguous medical image segmentation that generates a distribution of plausible masks using diffusion models. In the first stage, AmbiSeg is trained as a text-conditioned diffusion model that leverages image–text pairs to produce multiple diverse segmentation hypotheses reflecting inter-rater variability. In the second stage, TDG-DiffDPO fine-tunes AmbiSeg using Dice-guided Direct Preference Optimization, aligning the generated mask distribution toward clinically preferred segmentations based on multi-expert annotations. Experiments on the LIDC-IDRI lung CT dataset demonstrate that the proposed model achieves higher per-sample Dice and improved clinical alignment compared to prior probabilistic and diffusion-based baselines such as Probabilistic U-Net, PHi-Seg, and CIMD, while maintaining strong diversity and 3× faster inference with DDIM sampling.

### Strengths
1. The authors extend Diffusion-DPO to medical segmentation by using Dice similarity as the evaluation metric for preference alignment and explore three implementation strategies for constructing preference pairs. This adaptation enables automatic, clinically relevant supervision without human preference labels.
2. The idea of conditioning on text is promising, as shown both quantitatively (Table 5) and qualitatively (Section A.7).

### Weaknesses
1. Encoding short, structured metadata phrases with Bio-ClinicalBERT appears unnecessarily complex; simpler embeddings could achieve comparable results with lower cost. The inclusion of annotation IDs as conditioning inputs is not well justified and may risk introducing sample-specific bias rather than genuine semantic conditioning.
2. Despite improved ambiguity modeling and controllability, the model’s Dice scores remain lower than competing deterministic or diffusion-based baselines, suggesting a trade-off where preference alignment improves diversity but reduces segmentation precision.
3. All experiments are conducted solely on the LIDC-IDRI lung CT dataset.

### Questions
1. Could the authors evaluate the necessity of Bio-ClinicalBERT by testing a simpler text encoder or adding an ablation to show its impact on performance?
2. Can the authors include experiments on additional datasets to demonstrate generalizability, for example by following Probabilistic U-Net’s setup to simulate ambiguity on datasets like Cityscapes?
3. Could the authors provide a deeper analysis on why the Mean-Dice preference strategy performs best, and under what conditions the other strategies fail?
4. (Optional) It would be interesting to see a quantitative evaluation of text adherence—for instance, training a classifier on metadata and testing whether generated masks align with the corresponding text descriptions.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TDG-DiffDPO to enable segmentation via text and dice guidance. Experiments are performed on LIDC-IDRI datasets.

### Strengths
1. The paper is well-motivated by the key challenges in text-guided medical image segmentation.
2. The paper presents both quantitative and qualitative results to demonstrate the outperformance of TDG-DiffDPO.

### Weaknesses
1. The architecture of TDG-DiffDPO looks very similar to previous work, for example, latent diffusion model, and DiffSeg.
2. The experiment only performs on one testbed (LIDC-IDRI). The generalizability remains a concern. 
3. Several significant segmentation baselines are missing, for example, nnunet.
4. What is the segmentation performance of the other baseline?
5. Using the DDIM for faster sampling is hard to be considered as a novelty.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
