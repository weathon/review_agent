# A Joint Diffusion Model with Pre-Trained Priors for RNA Sequence-Structure Co-Design

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
RNA molecules underlie regulation, catalysis, and therapeutics in biological systems, yet de novo RNA design remains difficult with the tight and highly non-linear sequence-structure coupling. 
The RNA sequence-structure co-design problem generates nucleotide sequences and 3D conformations jointly, which is challenging due to RNA’s conformational flexibility, non-canonical base pairing, and the scarcity of 3D data. 
We introduce a joint generative framework that embeds RoseTTAFold2NA as the denoiser into a dual diffusion model, injecting rich cross-molecular priors while enabling sample-efficient learning from limited RNA data. Our method couples a discrete diffusion process for sequences with an $SE(3)$-equivariant diffusion for rigid-frame translations and rotations over all-atom coordinates. The architecture supports flexible conditioning,
and is further enhanced at inference via lightweight RL techniques that optimize task-aligned rewards. 
Across de novo RNA design as well as complex and protein-conditioned design tasks, our approach yields high self-consistency and confidence scores, improving over recent diffusion/flow baselines trained from scratch. Results demonstrate that leveraging pre-trained structural priors within a joint diffusion framework is a powerful paradigm for RNA design under data scarcity, enabling high-fidelity generation of standalone RNAs and functional RNA-protein interfaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel joint RNA sequence and structure denoising model that integrates the RoseTTAFold2NA structural prediction network as the denoiser within a generative diffusion framework. This approach is designed to predict RNA sequences under various conditioning settings, including protein-guided design. The central motivation is to leverage the strong inductive bias of a pretrained structural model (RoseTTAFold2NA) to overcome the significant data scarcity, particularly for RNA sequences with paired structural annotations.

### Strengths
1. The innovative use of the pretrained RoseTTAFold2NA model as the core denoiser provides a powerful structural inductive bias. This design choice is highly effective in mitigating the challenges posed by the limited availability of high-quality, structure-annotated RNA data, thereby enhancing the model's generative capacity.

2. The method extends beyond existing structural diffusion models (like RFdiffusion) by integrating modern conditional guidance to explicitly incorporate protein sequence and/or structure information.

### Weaknesses
1. The primary evaluation relies heavily on self-consistency metrics (scTM, scRMSD, sc-lDDT), which measure the agreement between the generated structure and the structure predicted from the generated sequence. While valuable for checking folding consistency, these internal metrics do not constitute external validation against true ground-truth structures or de novo design goals.

2. The work employs highly complex, high-dimensional inputs (full protein sequence and backbone coordinates) as conditioning signals injected directly into the main backbone. Unlike simpler conditioning approaches, the efficacy of using such complex conditioning without dedicated analysis is questionable. A deeper justification or ablation is needed to validate that this complex, integrated injection mechanism is superior to simpler or more modular conditioning strategies.

### Questions
The content in this section is identical to the points raised in Weaknesses. I would be very willing to increase my score if the authors successfully address these concerns.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a joint diffusion model for RNA sequence–structure co-design, embedding the pre-trained RoseTTAFold2NA as a denoiser within dual diffusion processes: a discrete diffusion for RNA sequences and an SE(3)-equivariant diffusion for 3D structures. The method enables simultaneous generation of sequences and structures and supports conditional tasks such as RNA–protein complex or protein-conditioned RNA design. Experiments show solid improvements over previous RNA-specific diffusion or flow models trained from scratch, demonstrating that leveraging pre-trained structural priors can improve sample efficiency and accuracy in data-scarce RNA settings.

### Strengths
The work is technically sound and well-executed, with clear formulation and comprehensive evaluation. Integrating pre-trained structural knowledge from RoseTTAFold2NA into a joint diffusion framework is a reasonable and effective strategy for improving RNA generative design. The paper is well-written, the methodology is consistent with recent trends in biomolecular generation, and the results are convincing and reproducible. Overall, the paper completes a meaningful and well-defined task without major flaws.

### Weaknesses
The novelty of the approach is limited. The overall framework closely parallels prior work in protein design, particularly “Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design”, which introduced a similar idea of coupling discrete and continuous generative processes for sequence–structure modeling. The current paper largely adapts this concept to the RNA setting, without introducing substantial methodological innovation or theoretical insight beyond that extension. While the adaptation is meaningful, the contribution is incremental rather than conceptually new.

### Questions
None

### Soundness
3

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
3

### Summary
The authors propose a new method for RNA structure generation that utilises RF2NA as a pretrained structure encoder and train a generative model on top of this, similar to how RFDiffusion was trained for proteins.

### Strengths
[S1]  The authors use a pretrained encoder for their diffusion process and demonstrate via ablations that this is a strong contributor to their performance.

[S2] The authors do not investigate only RNA generation in isolation, but also test RNA-Protein complex design, a more practically relevant task.

### Weaknesses
[W1] Missing baselines: the authors describe RNA-FrameFlow and gRNAde as a RNA backbone structure generation and RNA inverse folding model, but do not compare to it empirically. Given that in that paper the authors show that their method outperforms MMDiff (the main baseline in this paper), the authors should compare to it.

[W2] While the authors perform an ablation study with respect to their pre-trained prior, their framework contains a lot more components they claim are important, for example RL-enhanced diffusion inference, various auxiliary losses and their codiffusion objective. More systematic ablations here would strengthen the paper, for example 1) what happens without these auxiliary losses, or 2) what happens if one trains their model with just backbone design and then adds gRNAde on top instead of codiffusion?

### Questions
[Q1] Recent all atom structure prediction models unify proteins rna etc into a consistent framework instead of having specialised structure predictions methods like RF2NA. Do you think something similar is realistic/desirable in design?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a joint generative framework for RNA sequence and structure co-designs by using pretrained RoseTTAFold2NA as a denoiser for a dual-track diffusion model. They combine a discrete diffusion process for RNA sequence generation with a SE(3)-equivalent diffusion process for structure design, while leveraging the shared information embeded in the pretrained molecule folding model. They also introduced a RL techinquie to further optimize the generation during inference time. 

In the empirical experiments, they showed their pretrained joint method consistently outperforms other joint baselines such as MMDiff, RNAFlow across three tasks settings, including single RNA co-design, conditional RNA designs.

### Strengths
1. The paper is well structured, and the motivation for leveraging pretrained cross-molecular priors is clearly articulated.

2. Integrating RoseTTAFold2NA into both discrete and continuous diffusion streams for RNA design is a clever and impactful idea that enables joint modeling of RNA sequence and structure, particularly given the scarcity of RNA data.

3. The method shows significant performance improvements over baselines across different metrics in single RNA design and significant gains in complex design.

4. The proposed framework is flexible and versatile; it naturally extends to conditional generation tasks and supports inference-time RL guidance.

### Weaknesses
1. Comparisons are primarily against a small set of joint models (MMDiff, RNAFlow, Random Generation). It would further strengthen the paper to compare against separately trained structure generators + inverse folding tools (gRNAde, RNAFrameFlow-style pipelines) or sequence generators + folding tools.

2. RoseTTAFold2NA is pretrained on large biomolecular datasets, including PDB structures that may overlap with evaluation sets; the paper should more clearly address dataset leakage or steps taken to avoid it

3. While the integration of RF2NA into diffusion is novel for RNA, the general paradigm mirrors prior work in protein design (RFdiffusion), slightly reducing novelty.

### Questions
Overall solid paper. Despite minor concerns about data leakage and missing comparisons with separate-stage baselines, the empirical gains and clear methodology justify acceptance.

### Soundness
3

### Presentation
3

### Contribution
3
