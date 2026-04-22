# Perturbation Guided Drug Molecule Design via Latent Rectified Flow

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Phenotypic drug discovery generates rich multi-modal biological data, yet translating complex cellular responses into molecular design remains a computational bottleneck. Existing generative methods operate on single modalities (transcriptomic or morphological alone) and condition on post-treatment measurements without leveraging paired control-treatment dynamics. We present **Pert2Mol**, the first framework for multi-modal phenotype-to-structure generation that integrates transcriptomic and morphological features from paired control-treatment experiments. Pert2Mol employs separate ResNet and cross-attention encoders for microscopy images and gene expression profiles, with bidirectional cross-attention between control and treatment states to capture perturbation dynamics rather than simple differential measurements. These multi-modal embeddings condition a rectified flow transformer that learns velocity fields along straight-line trajectories from noise to molecular structures, enabling deterministic generation with superior efficiency over diffusion models. We introduce Student-Teacher Self-Representation (SERE) learning where an exponential moving average teacher supervises student representations across network depths, stabilizing training in high-dimensional multi-modal spaces. Unlike previous approaches that require preprocessed differential expression vectors, Pert2Mol learns perturbation effects directly from raw paired experimental data. Experiments on large-scale datasets demonstrate the first successful multi-modal framework for phenotype-driven molecular generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this work, the authors study the problem of generating chemical structures that achieve a target biological effect, conditioned on paired data from control and treatment samples. The model, Pert2Mol, integrates both transcriptomic and imaging data, which are first encoded separately and then concatenated to produce the final conditioning vector. The generative model is based on rectified flow transformers. The authors introduce a student-teacher self-representation scheme to improve training stability and sampling. The model is evaluated on a multi-modal dataset of chemically perturbed cell populations, and the generated molecules are assessed using metrics for chemical validity, drug-likeness, and target similarity.

### Strengths
- The paper proposes to integrate transcriptomes and imaging data for de-novo molecule design, offering a potentially more comprehensive phenotype-to-structure mapping.
- The paper is well-written and easy to understand.

### Weaknesses
- The authors claim that "no existing method tackles the task of perturbation-guided drug molecule design". This is inaccurate. This is a very active field and many methods have been proposed to solve the problem of molecule generation conditioned on a desired gene expression effects [1-5].
- Given the above claim, the authors only compare against a single diffusion baseline. A direct comparison and benchmarking against the models listed is necessary to properly evaluate Pert2Mol's claimed advantages.

[1] https://www.nature.com/articles/s41467-019-13807-w  
[2] https://pubs.acs.org/doi/10.1021/acs.jcim.2c01301  
[3] https://academic.oup.com/bib/article/25/6/bbae525/7845937  
[4] https://academic.oup.com/bioinformatics/article/40/5/btae189/7649318  
[5] https://www.nature.com/articles/s41587-021-00946-z

### Questions
- How was sampling for the molecules presented in Figure 3 done?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposed focuses on integrating transcriptomic and morphological data for modeling control–treatment perturbation dynamics in drug design. The work applies a rectified flow framework to this multi-modal setting, combining an image encoder and an RNA encoder to learn joint representations from microscopy images and transcriptomic profiles. The motivation appears to be improving generative modeling performance by leveraging complementary information across modalities.

While the overall direction of multi-modal integration is interesting and potentially useful for biological discovery, the paper lacks sufficient novelty. 
The approach essentially extends an existing rectified flow framework to a multi-modal context without introducing substantial methodological innovation. 
The contribution is therefore more of an application than a conceptual advancement.

In terms of evaluation, the experiments are quite limited. The authors compare their method only with a generic diffusion model, without specifying the exact implementation or baseline details. 
This makes it difficult to assess the validity or significance of the reported improvements. 
Moreover, the experimental section would benefit from comparisons against other established multi-modal generative models or perturbation prediction methods. 
Ablation results suggest some value in the multi-modal setup, but they are not convincing enough to demonstrate a clear advantage over existing techniques.

### Strengths
multi-modal algo are an interesting methods

### Weaknesses
not novel, only appliying rectified flow in multi-modal settings.

limited evaluation, only compared with a diffusion model, not specifying which one.

### Questions
could you compare with other multi-modal approaches?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a conditional generative model for molecules, specifically designed for applications in phenotypic drug discovery. The task's complexity arises from the multimodal conditioning signal, which includes microscopy images and gene expression data from both pre- and post-treatment states. Given that generative modeling of molecules is inherently challenging, the authors perform this task in the continuous latent space of an autoencoder. They adopt flow matching as the generative modeling paradigm. The use of a transformer-based approximate vector field facilitates the incorporation of conditioning information via two mechanisms: cross-attention and adaptive normalization. To ensure stable training, the authors employ a self-supervised loss between transformer layers, applied alongside the primary flow matching objective. The framework's effectiveness is demonstrated on several datasets. Beyond generation, the authors also leverage the learned data representations to perform drug repurposing and retrieval tasks.

### Strengths
(As my expertise lies in machine learning rather than the application domain, my evaluation will focus on the methodological aspects of the work.)

- This work presents an elegant integration of standard state-of-the-art (SOTA) methods to address the target task.
- The framework is described in great detail (with the exception of the aspects noted in the Weaknesses and Questions sections), providing sufficient material to support the re-implementation of the method.
- The proposed "Student-Teacher Self-representation (SERE)" is a potentially interesting contribution for stabilizing training. However, as noted below, this component requires significantly more justification and analysis to validate its effectiveness.
- A further strength is the use of the learned condition representations for downstream tasks, such as drug repurposing and retrieval, as demonstrated in the "Experiments" section.

### Weaknesses
- The "Methods" section is largely dedicated to a detailed listing of neural network architectures, implementation choices, and hyperparameters. While this detail is valuable for reproducibility, it is difficult to evaluate this descriptive catalogue as a primary scientific contribution.
- The paper's contribution could be substantially strengthened by including more rigorous empirical analysis. For example:
   - A comprehensive ablation study on key hyperparameters.
   - An analysis of the contribution of individual model components (e.g., evaluating performance using only cross-attention for conditioning versus the full model).
- The SERE method, noted as a potential strength, is a significant weakness in its current form. It suffers from a superficial description, a lack of rigorous analysis, and is not supported by an ablation study. (This is detailed further in the "Questions" section).
- Overall, the paper provides excessive implementation detail on standard components while remaining vague on its more novel aspects (such as SERE) and the precise mechanisms of its core components (such as the molecular representation).

### Questions
1. Molecular Representation: Could the authors describe this component in more detail? The text seems to present conflicting or incomplete information. The first reference points to "RoBERTa" which suggests a masked modeling objective. However, the second reference describes a contrastive learning method. The word "contrastive" appears in Figure 2 but is absent from the main text (outside of the "Related Work" section). Furthermore, the paper states, "Tokenization uses learned molecular motifs," but provides no details on how these motifs are learned. Please clarify the exact architecture and training objective of the molecular encoder.
2. Student-Teacher Self-representation (SERE): Is this method a novel contribution of this paper? No references are provided in its description. Additionally, the description is contradictory. The text first implies that the student and teacher are different layers within the same transformer vector field. However, it later mentions "higher-" and "lower-noise layers," which suggests a comparison across different denoising timesteps. Please clarify: (a) if SERE is novel, and (b) the precise mechanism of the student-teacher relationship (i.e., which layers or timesteps are being compared).
3. Figure 2: the diagram shows the "SMILES Encoder" as an input to the ReT. However, this connection and its purpose are not described in the text. How is the output of the SMILES Encoder integrated into the model during this stage?

### Soundness
3

### Presentation
2

### Contribution
1
