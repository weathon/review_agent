# BrainMIND: Interpret Fine-grained Spatial Mapping of Brain Activity to Multi-semantic Concepts

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 0, 0

## Abstract
Understanding how population-coding in the human visual cortex shape high-level semantic representations remains a significant challenge. Prior work has either focused on region-level text decoding or relied on simple linear models to probe single-semantic decoding at the voxel level. Consequently, systematic exploration of semantic diversity remains limited at both the region level and the fine-grained voxel level. To address this gap, we introduce BrainMIND, a data-driven framework for analyzing multi-concept semantic selectivity in the visual cortex. We use a conditional variational autoencoder (CVAE) whose latent space is constrained by brain data and spatial locations of voxels. The CVAE decodes the structured latent space into CLIP-aligned semantic embeddings, which then condition a fine-tuned large language model to generate interpretable captions. We validate BrainMIND on widely recognized cortical regions, demonstrating interpretable region-level and voxel-level semantic selectivity. We reveal that individual voxels exhibit mixed selectivity across multiple semantic dimensions, and filling a key gap in voxel-wise neural decoding. Our results demonstrate that BrainMIND provides an interpretable bridge from brain regions to their constituent voxels, enabling controlled, fine-grained exploration of semantic organization in the higher visual cortex.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes BrainMIND, a conditional variational autoencoder (CVAE) with a dynamic mixture-of-Gaussians prior, conditioned on both fMRI voxel positions and brain responses, to decode multi-semantic representations from the Natural Scenes Dataset (NSD). The decoded latent features are mapped into CLIP space and then converted to natural language via a fine-tuned large language model (LLM). The authors claim this approach reveals mixed selectivity at both the region and voxel levels and provides interpretable multi-semantic mappings across the visual cortex.

### Strengths
-The work addresses a meaningful and underexplored problem — interpretable, voxel-level, multi-semantic decoding rather than simple reconstruction or region-level prediction.

-Authors connect findings to established cortical selectivity patterns (FFA, PPA, VWFA) and report consistency with known priors, suggesting neuroscientific validity.

-The paper promises open code and weights, and the ethics section is solid and transparent.

### Weaknesses
generally the text is poorly written, with some mistakes and lot of AI generated content. This is not a problem per-se but please double check the whole text and especially the citations. Many of them are kind of wrong with maybe the right author but incorrect titles or the other way around.

Furthermore, the paper oscillates between semantic reconstruction and interpretability. It’s unclear whether BrainMIND is intended as a decoding model (predicting content from brain data) or as a representational analysis tool. If it is intended as a decoding model lot of comparison is missing with existent literature. There is only one comparison with BrainSCUBA, with limited improvements.

The paper completely lack evaluation, do we need the router? What is the impact?

Also I'm not fully convinced by the math presented in the paper. For example, the gating is not-differentiable, but there are other potential flaws.

### Questions
Honestly I think the paper should be improved in several directions:

- Improved text for clarity and readability
- Improved explanation of what is the research question here, and why (and how) the method proposed is solving the question
- Clear math
- Fair comparison with prior work (properly cited)

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes BrainMIND, a voxel-level fMRI decoding framework for interpretable brain-to-text reconstruction. Voxel-wise neural decoding is an important problem for understanding the relationship between human brain representations and latent embeddings of intelligent models. However, the manuscript contains several serious issues that undermine confidence in the work.

### Strengths
Exploring fine-grained mapping between brain activity patterns and AI model representations is a meaningful direction in bridging neuroscience and AI.

### Weaknesses
* The bibliography contains multiple citation errors. For example, several references list non-existent or placeholder author names such as [1–2], and some cited works cannot be traced to public sources or published papers (e.g., [2]). 

  [1] Aoxiao Luo, John D Smith, and Jane Doe. Brain diffusion for visual exploration: Cortical discovery using large scale generative models. arXiv preprint arXiv:2306.03089, 2023.

  [2] Yujia Wang, John D Smith, and Jane Doe. Incorporating clip into brain decoding: Zero-shot learning for fmri analysis. NeuroImage, 250:118956, 2022.

* Another major concern to me is the lack of comparison to existing work on the topic of fMRI-to-caption decoding. While the paper compares against BrainSCUBA, the brain-to-text decoding field already contains numerous advanced methods such as [1-3].

  [1] Neuro-Vision to Language: Enhancing Brain Recording-based Visual Reconstruction and Language Interaction. NeurIPS 2024.

  [2] Exploring the Visual Feature Space for Multimodal Neural Decoding. ICCV 2025.

  [3] Bridging the Gap between Brain and Machine in Interpreting Visual Semantics: Towards Self-adaptive Brain-to-Text Decoding, ICCV2025.

* The current methods lack crucial details for understanding.

### Questions
Implementation details are missing, such as the description and configuration of comparison methods, the choice of LLM, raising questions about reproducibility.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes BrainMIND, a position‑conditioned CVAE with a dynamic mixture prior that decodes semantics embeddings and uses LLMs to produce natural‑language semantics at both ROI and voxel levels. I found that  the submission raises serious Ethical & Reproducibility Concerns. There are multiple placeholder‑style or internally inconsistent references. These issues impede a fair scientific assessment at this time.

### Strengths
n.a.

### Weaknesses
n.a.

### Questions
n.a.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a way to generate words from voxel-wise selectivity. The task is not new, and this paper contains numerous issues with AI generated citations.

### Strengths
Investigating the selectivity in higher visual cortex is an interesting problem.

### Weaknesses
I have serious serious concerns regarding this paper.

There are numerous citations which are totally made up and reference non-existent papers:
* Yujia Wang, John D Smith, and Jane Doe. Incorporating clip into brain decoding: Zero-shot learning for fmri analysis. NeuroImage, 250:118956, 2022.
* Aoxiao Luo, John D Smith, and Jane Doe. Brain diffusion for visual exploration: Cortical discovery using large scale generative models. arXiv preprint arXiv:2306.03089, 2023.
* Enrico Ferrante, John D Smith, and Jane Doe. Brain captioning: Decoding human brain activity into images and text. arXiv preprint arXiv:2305.11560, 2023.
* Justin Giles, Andrew Luo, and Leyla Isik. Clip-decoding: A generalist brain decoder for reconstructing arbitrary image-caption pairs. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.


Second, by the commonly accepted definitions of encoding and decoding by Thomas Naselaris, this work does not perform decoding, yet says `Our framework achieves multi-semantic and position-aware decoding at both the coarse ROI scale and the fine-grained voxel level`. This work is **investigating selectivity** not decoding. This sentence shows that the authors are seemingly unaware of the question they are investigating. 

Third, the figures are very strange. There are no cortical flat-maps or inflated maps, instead the authors seemingly plot out all figures using voxel positions. Which may change depending on the reference frame the MRI was transformed into. Note that this may vary from subject to subject. 

Fourth, the authors at no point in the paper describe how the words for each voxel are generated.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1
