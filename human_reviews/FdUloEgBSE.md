# Text-guided Diffusion Model for 3D Molecule Generation

- Avg Score: 3.75
- Decision: Reject
- Scores: 1, 6, 3, 5

## Abstract
The *de novo* generation of molecules with desired properties is a critical task in fields like biology, chemistry, and drug discovery.
Recent advancements in diffusion models, particularly equivariant diffusion models, have shown promise in generating 3D molecular structures.
However, these models largely work under value guidance, typically conditioning on a single property value, which might limit their ability to address complex real-world requirements.
To address this, we propose the text guidance instead, and introduce TEDMol, a new *Text-guided Diffusion Model for 3D Molecule Generation*.
It aims to integrate the capabilities of language models with diffusion models, thereby providing a deeper level of language understanding in 3D molecule generation.
Specifically, TEDMol utilizes textual conditions to guide the reverse process, enabling the adept and flexible generation of 3D molecules.
Our experimental results on various tasks demonstrate that TEDMol not only enhances the stability and diversity of the generated molecules, but also excels in capturing and utilizing information derived from textual descriptions.
Our approach forms a flexible and efficient text-guided molecular diffusion framework, providing a powerful tool for generating 3D molecular structures in response to complex, textual conditions.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
A diffusion model for 3d molecule generation conditioned on text descriptions is proposed. It uses a combination of BERT and GIN to encode a text description to a “reference geometry”. Then, an EDM diffusion model is conditioned on this using ILVR.

### Strengths
To the best of my knowledge this is the first work that proposes text-guided diffusion for 3D molecule design in the domain of conditional diffusion-based generative modeling for 3D molecule generation.

### Weaknesses
- A similar idea has been proposed and published already, called SILVR (https://pubs.acs.org/doi/epdf/10.1021/acs.jcim.3c00667). In contrast to SILVR, the authors propose to create a reference geometry first using an ML-based mapping from textual description to geometry. Hence the generative model is not guided directly by text, but by some reference geometry the text-geometry mapping is outputting.
- It is unclear how single and multiple (two) QM properties are "encoded" in Sec. 4.1 and 4.2. Important information on model architecture is missing. For example, the “multi-modal conversion module” translating the text to a “reference geometry” is not clearly described. Nevertheless, the authors highlight their results in bold claiming strong and significant improvements in the main text.
- QM9 contains only very small molecules. A better benchmark set would be something like GEOM-Drugs. Results are missing error bars. In particular, the results in Table 2 are only marginally better and probably not significant.
- The results on PubChem have only been evaluated on selected examples. The reader is left guessing whether the text aligns with the depicted molecules; no additional metrics are given to prove correlation / correctness. 
- Ablation studies are missing: Is the improved performance in Table 1 compared to EDM due to text description, or because ILVR is the better conditioning approach?
- The evaluation in Table 1 is only done with a property network instead of actual ground truth calculated values. Beyond that they do not seem to be of relaxed structures and the quality of the generated geometries is not checked. Therefore, the classifier might be out of training domain and/or the structure-dependent property might be quite different in equilibrium compared to the predicted 3d structure.
- The text contains many spelling mistakes and unscientific language, the captions of plots and tables are often insufficient, citations are partly wrong.

### Questions
- Why is it important to generated 3d molecules? It seems for the text-to-molecule use case generation of molecular graphs would be preferable.
- Why should conditioning on text work better than on numerical values, particularly for single property conditioning? What additional information does the text description contain here?
- Example in Fig 1: What is a “small HOMO-LUMO gap”? The text here is imprecise, so how can this condition actually be properly evaluated?
- How is the reference geometry c_p obtained? What is the precise architecture of the “multi-modal conversion module”? How is it trained?
- The text-to-reference geometry encoding cannot be a deterministic mapping, as multiple geometries could correspond to the same text description. Why should this not restrict the conditioned diffusion model? Why is the “reference geometry” in Fig 2 not a connected molecule?
- If c_p is a 3d structure, how is it matched with the atoms generated by the diffusion model? Has it necessarily the same number of atoms?
- Has the same property model been used for all models in Table 1?
- What are the concrete text descriptions used for single- and multi-property guidance?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a molecular generation using a text-conditioned diffusion model.  It is particularly effective when specifying multiple physical conditions simultaneously, such as specific heat and dipole moments.  The texts are obtained from PubChem or converted from physical properties using templates.  The text conditions generate reference structures through a generation model, and the other generation model mixes the reference structures with the structures during the reverse diffusion process in the same way as ILVR for image generation.   They empirically demonstrate conditional generation performance surpassing EDM and EEGSDE in the QM9 dataset.  Examples of generating molecules based on language instructions, such as polycyclic compounds, are also provided.

### Strengths
* The results of multiple conditions in Section 4.2 convince the readers about the benefit of the text conditioning.
* The examples in Section 4.3 show the flexible conditioning ability, which previous work cannot.
* Even the single conditioning is empirically better or competitive to the baseline methods.

### Weaknesses
* Since the text dataset used in Section 4 is not publicly available, it is hard to reproduce their results in the subsequent research.
* Although the direct use of C_p is not recommended in the main text, the empirical evaluation of it is not available.  Since the proposed method is complex, the readers would want to see more supporting evidence of the current design choice.

### Questions
* Is the reference of EEGSDE to (Igashov et al., 2022) correct?  Do you mean Equivariant Energy-Guided SDE for Inverse Molecular Design, https://arxiv.org/abs/2209.15408?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a diffusion model for generating 3D molecules conditioned with text descriptions. This model uses textual prompts to guide the reverse process and accurately generates 3D molecular structures that match the condition while maintaining chemical validity. Empirical results demonstrate the effectiveness of the proposed model compared with baselines.

### Strengths
- This is a novel application of diffusion models on text-guided 3D molecule design. Text can indeed naturally combine multiple conditions to control the generation of molecules that humans want, so this task makes sense.
- The paper is well organized.

### Weaknesses
- This paper is not the first molecule translation task, but the author does not compare with the former baseline [1] which also includes single-objective and multi-objective molecule generation.
  - [1] Liu, Shengchao, et al. "Multi-modal molecule structure-text model for text-based retrieval and editing." *arXiv preprint arXiv:2212.10789* (2022).

- The equivariant diffusion model, iterative latent variable refinement, and multi-modal conversion module are all from existing works , making the technical contribution limited.
- The dataset is constructed manually with a set of textual templates, and the authors do not give details on how to build it. This may lack diversity and be far away from real-world challenges.

- The experiment in Tab. 3 is conditioned on two quantum properties. However, the results only report the MAE of a single property, lacking the proportion of molecules that meet both conditions.
- Authors believe that text can naturally capture information in multiple conditions, but the experiment is only about the combination of the two quantum properties. The conditions related to the structure are only shown in the case study, which is insufficient.
- I am confused about how to use two encoders and a decoder (a GIN molecular graph encoder and a language encoder-decoder extended from BERT) to obtain the condition vector.

### Questions
- Can you provide more details about the dataset you collected and analyze if it lacks diversity?
-  What are the pre-trained data and training objectives of the multi-modal module? Why is the geometry information contained in the condition vector?
- Why not compare it with the previous baseline and report the metrics about whether your generated molecules meet multiple conditions at the same time?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a text-conditional generative model of molecules in 3D space by pairing a diffusion model with a text encoder used to guide the denoising process. The authors then perform a series of experiments to validate the proposed approach, showing how it can generate molecules conditioned on one or several properties, as well as handle more free-form textual descriptions.

### Strengths
(S1): This work explores an important topic of molecule generation. While 2D-based generative models have long been adopted in the pharma industry, models operating in 3D directly are a newer frontier, which many practitioners are excited about, and so developing such models is worthwhile. 

(S2): The high-level design seems sensible, and it makes use of relatively modern DL components. The text conditioning idea is interesting from an ML point of view (even if I'm not sure about its practicality – see (W2)). 

(S3): The aspects of the method that the authors choose to talk about in depth are explained quite clearly (however, many other aspects seem to be not talked about – see (W1)).

### Weaknesses
(W1): Many aspects of this work are not clear to me. 

- (a) Many existing diffusion-based models for generating molecules (or point clouds more generally) have a caveat around number of atoms (points), which has to be fixed beforehand. Is it also the case here? When sampling from the model, do you sample the number of atoms separately? Is that conditioned on the text? 

- (b) How does $\Gamma$ work? As I understand, it is a model mapping from text to a molecular conformation, i.e. the output is a set. How is this set decoded? Are there any issues in pretraining this component related to the fact that the atoms do not have a predefined ordering? 

- (c) What are the different training stages involved here? I understand $\Gamma$ is trained beforehand and frozen; is the unconditional EDM also pretrained (without involving text) and frozen? Are the parameters of the linear transformation $\theta$ then trained separately, in a third stage? I can't fully match this with Figure 2 which suggests two stages, not three. 

- (d) Why is the form of conditioning (linear combination of reference geometry and the current latent) so constrained? It seems the text encoder has very limited ability to "send information" to the diffusion process, as it cannot "send" arbitrary activations and rather something that is constrained to be an actual geometry. Is this design informed by having very little paired data (i.e. samples having both molecular conformation and text description at the same time)? 

(W2): While the text conditioning is theoretically nice and flexible, I fear it may not be fully practical in a real drug design setting. In those scenarios, the generation would typically be also conditioned on a property that is specific to a given drug discovery target; for those properties there would normally be very few samples available [1], and they wouldn't be known upfront, so it's not clear how such properties would fit in the text-conditioned model (one may have to retrain the model for each project?). On the other hand, this issue is not specific to this work, and it is perhaps why simple 2D-based models (utilizing autoencoders or genetic algorithms) were so far more widely adopted in pharma than 3D-based ones, as the former are easier to condition on arbitrary properties not known beforehand. That being said, including 3D in the generative process is generally an important direction, and those classes of models are less mature than the 2D-based ones, so a smaller practicality is perhaps fine at this stage. At the very least, I would hope for some more discussion of this limitation in the "Limitations" section i.e. of the fact that the properties the model can condition on have to be known beforehand when training the text encoder, and this assumption may not fully reflect the real-world usecase. 

 

=== Other comments === 

(O1): Table 2 presents some results, bolding the best numbers, however, the differences between most of these results are tiny, and they don't look statistically significant. So, first, some significance test would be good here, and second, could the authors discuss why the differences are so small? It seems the novelty/stability percentages are influenced much more strongly by the property being conditioned on than the generative model itself, which I found somewhat surprising and counter-intuitive. 

(O2): Given that the related works discussion mentions [2], it would also make sense to refer to more modern extensions of that work [3,4]. 

 

=== Nitpicks === 

Below I list nitpicks (e.g. typos, grammar errors), which did not have a significant impact on my review score, but it would be good to fix those to improve the paper further. 

- "By “unconditional”, some studies (…) craft atom coordinates and types without external constraints.  By “conditional”, (…)" - the use of the word "by" seems off to me 

- "searching suitable molecules in the drug design" -> I would drop "the" in this context 

- "$\mathcal{N}(O,I_n)$" - should the $O$ be $0$? 

- "rests on the assumption that, with the model distribution $p(G) = p(x, h)$ remains invariant" - I would either say "if (…) remains invariant" or "with (…) remaining invariant" 

- "Experiment" (title of Section 4) - maybe "Experiments", as there are several 

- missing space before the citation of Igashov et al 

- "novelty is enhanced instead in comparison to the baseline" - the use of the word "instead" seems slightly off to me 

- "Mvaluation Metrics" (title of Appendix B.1) 

 

=== References === 

[1] "FS-Mol: A Few-Shot Learning Dataset of Molecules" 

[2] "Junction Tree Variational Autoencoder for Molecular Graph Generation" 

[3] "Learning to Extend Molecular Scaffolds with Structural Motifs" 

[4] "Hierarchical Generation of Molecular Graphs using Structural Motifs"

### Questions
See the "Weaknesses" section above (particularly (W1)) for specific questions.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
