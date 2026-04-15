# P(all-atom) Is Unlocking New Path For Protein Design

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 3, 5

## Abstract
We introduce Pallatom, an innovative protein generation model capable of producing protein structures with all-atom coordinates. Pallatom directly learns and models the joint distribution $P(\textit{structure}, \textit{seq})$ by focusing on $P(\textit{all-atom})$, effectively addressing the interdependence between sequence and structure in protein generation. To achieve this, we propose a novel network architecture specifically designed for all-atom protein generation. Our model employs a dual-track framework that tokenizes proteins into token-level and atomic-level representations, integrating them through a multi-layer decoding process with "traversing" representations and recycling mechanism. We also introduce the $\texttt{atom14}$ representation method, which unifies the description of unknown side-chain coordinates, ensuring high fidelity between the generated all-atom conformation and its physical structure. Experimental results demonstrate that Pallatom excels in key metrics of protein design, including designability, diversity, and novelty, showing significant improvements across the board. Our model not only enhances the accuracy of protein generation but also exhibits excellent training efficiency, paving the way for future applications in larger and more complex systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Pallatom, a generative model for creating all-atom protein structures, aiming to directly model the joint distribution of protein structure and sequence. Pallatom utilizes a novel dual-track architecture that combines atomic- and token-level representations, which are refined iteratively through a multi-layer decoding process. The model introduces the atom14 representation, allowing accurate depiction of side-chain configurations while addressing unknown side-chain coordinates. Pallatom demonstrates high performance in metrics such as designability, diversity, and novelty, positioning it as a promising tool for de novo protein design and efficient generation of complex protein structures.

### Strengths
Quality
The paper is methodologically rigorous, presenting a well-designed and thoroughly tested model. Pallatom’s dual-track framework is detailed with an iterative decoding and recycling mechanism that leverages both local and global attention, maintaining consistency between atomic- and token-level representations. The model demonstrates strong results across several key metrics for protein generation—designability, diversity, and novelty—indicating a high-quality, comprehensive evaluation. The detailed ablation studies, use of both PDB and AlphaFold databases for training, and comparisons with several state-of-the-art models highlight the thoroughness of the experimental design. The adoption of diffusion models for sampling noise levels further enhances the robustness and adaptability of the model.

Clarity
The paper is largely clear, particularly in how it describes the architecture and methods of Pallatom. The introduction effectively frames the motivation behind the study, distinguishing Pallatom from previous stepwise or co-design approaches, and introduces the challenges of protein generation concisely. Figures, such as those depicting Pallatom’s architecture, provide useful visual aids, enhancing the reader's understanding of the dual-track representation and multi-layer decoding. However, some sections, such as the mathematical descriptions of the diffusion model and recycling mechanism, could benefit from additional explanation for readers less familiar with advanced protein modeling or generative techniques.

Significance
Pallatom's potential impact on de novo protein design and drug discovery is substantial, especially given the model’s ability to produce highly accurate all-atom structures without requiring iterative backbone and sequence optimization steps. Its emphasis on producing designable and diverse structures makes it a practical tool for real-world applications, where structural accuracy and diversity are key to designing proteins for specific functions. Moreover, the model’s scalability to larger protein lengths without sacrificing accuracy is particularly significant for the study of large, complex protein systems and could pave the way for future expansion to include small molecules or non-standard amino acids.

### Weaknesses
1. Limited Interpretability of the Model
While Pallatom achieves strong performance, the model’s interpretability remains limited, which can be a barrier for users seeking insights into how predictions are generated. This is particularly relevant in protein design, where understanding which features drive structure generation is essential for real-world applications.
2. Scalability Constraints on Complex Systems
Pallatom’s training and testing focus largely on shorter protein sequences (up to length 128), and although it exhibits scalability for slightly longer sequences, its practical applicability to more complex protein systems, such as large complexes or multi-chain proteins, remains unclear.
3. Atom14 Representation Constraints
Although the atom14 representation improves handling of unknown side-chain conformations, it is a simplified model that may not capture all physical details, especially for less common or modified amino acids. This could limit Pallatom’s effectiveness in fine-grained applications like enzyme engineering or ligand binding, where side-chain interactions are critical.

### Questions
1. The authors state that the atom14 representation is used to avoid potential conflicts arising from the simultaneous design of sequence and structure, but they do not elaborate on the specific nature of these conflicts.  The authors should explain why padding the initial protein with L residues as x = {xl}Ll=1 → x0, considering it as a 3D point cloud distribution P (x0) ∈ RL×14×3, is necessary to address these conflicts. 
2. The authors mention that the dual-track framework is inspired by AlphaFold3, but they do not fully explain the rationale behind their specific modifications to the framework for protein generation. The authors should explain how the traversal atomic-level representations help to prevent the repeated accumulation of redundant structural information.  The authors could also provide a more detailed explanation of how the dual-track recycling mechanism works and why it is effective.  For example, the authors could provide more detail on the benefits of using a minimal decoding unit and supervised training on intermediate coordinates.
3. Discuss the limitations of the Pallatom model in terms of its ability to generate proteins with diverse secondary structures. The results in Figure 2B and 2C show that Pallatom, like other comparative methods, has difficulty generating proteins with predominantly β-sheet structures. The authors should discuss the reasons for this limitation and potential strategies for addressing it in future work.
4.  **Explore the potential of Pallatom for designing larger and more complex protein systems, such as antibody complexes and self-assembling materials.** The authors mention this as a direction for future work, but they could expand on the specific challenges and opportunities involved in extending the model to these more complex systems.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposed to design protein by directly generating all atom coordinates. Experimental results show improvements on newly proposed metrics describing designablility, diversity and novelty.

### Strengths
Improved evaluation scores on proposed metrics.

### Weaknesses
While the concept shows promise, the network architecture is actually not novel. The core of the model appears to be heavily based on AlphaFold3, with adaptations made to handle cases without input sequences and MSAs. In several statements, the authors may have overstated the novelty of their approach (see questions).

The paper lacks sufficient evaluations to become useful. The main benchmark focuses on the model's ability in designing monomers with 60-120 residues, while other important scenarios such as motif-scaffolding, loop design etc (see details in RFdiffusion) are ignored. In fact, in the industry and research of protein design, de novo design is a worst setting with regard to real applicability such as drug discovery and enzyme engineering.

On the benchmark of PMPNN 1 (which is the most common setting), Pallatom only out-performs baseline methods on newly proposed metrics. This makes the comparison less convincing, especially given that the model is not specifically designed to improve diversity and novelty.

### Questions
1. P2 066 "we propose a new amino acid coordinates representation, atom14" The atom14 notation is used no later than AF2 and is not new at all.

2. P4 167-170 "We find that if residual connections ... repeatedly broadcast and inappropriately accumulated" This so-called challenge is not well-described. Readers cannot find any evidence in supporting the statement, which seems to be describing a bug in the author's residual network modeling.

3. All algorithms displayed in the appendix are almost directly copied from AF3's supplementary. The authors should either reduce the copied contents or justify why such massive copying is necessary.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This study proposed a new architecture called Pallatom to generate protein structures. When the side chain coordinates are unknown, they are generated by a strategy named Atom14. The generative performance is evaluated and compared with several existing methods in terms of designability, diversity, and confidence.

### Strengths
1. The paper provides a comprehensive description of the implementation of the proposed method.
2. The proteins generated by the method exhibit notably high levels of diversity and novelty, indicating considerably strong performance in these areas.

### Weaknesses
1. The rationale behind the study lacks clear explanation and supporting evidence, such as empirical justification for the importance of conducting all-atom protein generation.

2. The `atom14` representation seems rather trivial, as it mainly consists of initializing 14 atom entities per AA and padding extra atoms to the C_alpha. This method does not appear to represent a substantial novelty.

3. The methodology section is poorly articulated, making it difficult to assess the significance of the proposed modules or workflows in relation to existing frameworks.

4. The figures are challenging to decipher due to the small font size used.

5.  The evaluation criteria are not well-justified. For instance, why it is important for a model to design novel structures?

6. The overall presentation has a large room for improvement, with many typos to be fixed.

### Questions
1. In the statement "(A) series of protein generation models based on SE(3)" could you specify why it is referred to as recent when the references are from 2020 and 2021? Are there more recent studies that support this claim?

2. Could you provide evidence or references to clarify the challenges faced by "The P(structure | seq) · P(seq) strategy faces challenges when sampling in the high-dimensional (space)"? Also, could you explain why explicit side-chain interactions are considered essential in the context of the "P(backbone) · P(seq | backbone) strategy"?

3. Where does the statement "The ultimate goal of protein generation is to directly obtain a sequence along with its corresponding structure" originate from? Could you provide references and clear explanations to support this claim?

4. How was the number 14 determined for the `atom14` representation? How does this approach handle amino acids with more than 14 atoms, such as Tryptophan?

5. Could you clarify why the existing designability evaluation metric is not considered suitable for evaluating all-atom generation results, given that RMSD is calculated on atoms rather than C_alpha?

6. Why are there references to previous studies that used designability, but none for diversity and novelty? Could you provide references that discuss these aspects?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose Pallatom, a model for all-atom protein generation. The authors design atom14 to represent the unknown amino acid side-chain coordinates.

### Strengths
1. The authors define three kinds of metrics, DES, SIV and NOV, and show Pallatom has excellent performance on these metrics.
2. Pallatom considers the information from side-chains.

### Weaknesses
1. There is no formal definition for the defined metrics, and the authors only claim that these metrics are evaluated by TM-score, which makes it difficult to follow the authors' idea. I'm not sure if DIV can be an ideal metric, since the final target is to generate protein structures that are similar to natural proteins. High diversity indeed indicates the design ability of Pallatom, but the authors should also investigate whether high diversity aligns with natural proteins.

2. The paper is difficult to follow. When the authors refer the appendix in the main body, please tell the readers where we can find exactly in the appendix. Please do not just let us find the content in appendix by ourselves, given that you have 12 pages in appendix.

3. The application seems limited, because only the unconditional experiments are conducted.

### Questions
I am really surprised to see that in the training data, the authors only take approximately 7k and 27k structures from PDB and AFDB, respectively. Are these structures sufficient for training this model? Additionally, I find that the authors set the maximum training length as 128. What is the motivation behind this decision?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents Pallatom, a generative model for protein structures based on all-atom coordinates. Precisely, the standard all-atom representation is extended to obtain the $\texttt{atom14}$ representation, which pads the original 6 all-atom features with 8 more virtual atoms. At its core, Pallatom is a diffusion generative model which is trained to denoise the $\texttt{atom14}$ representations, adapting parts of the AlphaFold3 model from the structure prediction to the generative setting. Experiments in _de novo_ protein design show that Pallatom is capable of generating coherent and diverse low-to-medium-sized proteins.

**Judgment**

In my opinion, this paper is a tremendous effort in solving the de novo protein generation task with denoising diffusion. However, I could not fully appreciate the author's contribution because the way it is presented is rather convoluted and because of some unfortunate choices (such as the rather confusing notation borrowed from another paper). I am sure this has to do with the fact that the authors had to limit the writing to 10 pages. Perhaps ICLR is not the right venue for this kind of papers, because of the page limits; probably a journal article would do more justice to such a technically involved work. I am giving a score of 5 for the moment being, I'll be happy to revise after hearing from my fellow reviewers and the author's replies.

### Strengths
- the paper tackles a very difficult task, with a potentially disrupting impact in the comp-bio area.
- the overall design appears very thoughtful. Adapting AlphaFold3 designs to the generative process is a real challenge, which judging by the experimental results has been mastered.
- the $\texttt{atom14}$ representation is innovative and seems effective, although its importance has not been verified by ablations.
- experimental results are excellent, although they are measured on custom metrics.

### Weaknesses
**Disclaimer**

I couldn't properly understand the methodological contribution of this paper (details below). This might bias my overall judgment.


**Weaknesses**

The main problem with this paper is the way it is presented. While I had no problems in understanding papers on the same subject (e.g. Protpardelle), here I had a very difficult time reading. Some reasons why:

1) the paper takes too many concepts from granted (e.g., the all-atom representation is not explained, examples of sequence/structure conflicts that motivate the $\texttt{atom14}$ representation are lacking, and so on). Perhaps it would be better to add a brief explanation of the all-atom representation in the introduction, and providing 1-2 specific examples of sequence/structure conflicts in Section 2.1 to motivate the $\texttt{atom14}$ representation.

2) it uses the unfortunate notation of Karras et al. (2022), which makes understanding the contribution more difficult (e.g. $\sigma(t)$ vs. $\sigma_t$, $F_{\theta}$ undefined). Perhaps you could add a notation table to help the reader. Also please make sure that all the symbols and acronyms used in the paper are defined before or immediately after their first use.

3) it is not well-structured, or at least, it uses section names that are confusing (e.g. section 2 is about "preliminaries", yet some contributions, e.g. $\texttt{atom14}$ are described there). Maybe it would be better to reorganize the sections so that your contribution stands out from work that has already been done, although I understand this could be a very disruptive change.

Other major problems I have found:
- $\texttt{atom14}$ is not ablated, it is not clear whether its addition is really needed to improve performances. It would be better to include a specific ablation on Pallatom with and without the $\texttt{atom14}$ representation, using the metrics you defined for all-atom models.
- comparing backbone-only models with their own metrics to all-atom models with their own metrics is not really meaningful in my opinion. However, I'm not sure there is a metric that could be used in this case; perhaps you can just compare the ratio of novel and unique protein structures produced by both classes of methods?

### Questions
- why representations are initialized with the standard conformation of alanine and not other amino acids (e.g. methionine)?
- Pallatom is evaluated with custom metrics. Can you provide the scores obtained by Pallatom in the metrics used by competitors (e.g. Protpardelle)?

### Soundness
3

### Presentation
1

### Contribution
3
