# IgGM: A Generative Model for Functional Antibody and Nanobody Design

- Decision: Accept (Poster)
- Scores: 6, 5, 5, 8, 5

## Abstract
Immunoglobulins are crucial proteins produced by the immune system to identify and bind to foreign substances, playing an essential role in shielding organisms from infections and diseases. Designing specific antibodies opens new pathways for disease treatment. With the rise of deep learning, AI-driven drug design has become possible, leading to several methods for antibody design. However, many of these approaches require additional conditions that differ from real-world scenarios, making it challenging to incorporate them into existing antibody design processes. Here, we introduce IgGM, a generative model for the de novo design of immunoglobulins with functional specificity. IgGM simultaneously generates antibody sequences and structures for a given antigen, consisting of three core components: a pre-trained language model for extracting sequence features, a feature learning module for identifying pertinent features, and a prediction module that outputs designed antibody sequences and the predicted complete antibody-antigen complex structure. IgGM effectively predicts structures and designs novel antibodies and nanobodies. This makes it highly applicable in a wide range of practical situations related to antibody and nanobody design. Code is available at: https://github.com/TencentAI4S/IgGM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces IgGM, a generative model for designing antibody/antigen structures and sequences for a given antigen. IgGM uses features from a pre-trained language model and a diffusion framework to generate antibodies and nanobodies. The model shows great performance in both structure prediction and antibody design.

### Strengths
1. The authors made a comprehensive comparision between IgGM and other baseline methods cross multiple metrics, demonstrating impressive results.
2. This paper is well-structued, with carefully prepared figures that enhance the reader's understanding.

### Weaknesses
Major comments:

1. One major limitation is that the generation of antibodies depends on their framework regions, which predetermine the length of CDRs and may not always be available when handling a new antigen. Since the optimal CDR lengths are not known a priori, the authors should demonstrate the capability of IgGM in designing antibodies with varying CDR lengths.

2. However, in line 147, the authors state that their method "...enables the design of the whole antibody structure even without experimental structures." This claim requires clarification, as the results presented only demonstrate the ability to design CDRs.

3. I suggest that the authors should compare their method with other antibody deisgn methods, such as:
 - Co-design models: AbX [1] (very similar framework with IgGM)
 - Structure-only models: RFdiffusion for antibody design [2]
 - Sequence-only models: AbLang [3] and IgLM [4]

Minor comments:

1. In figure  6, "Aligin" should be "Align"

References

[1] Zhu T, Ren M, Zhang H. Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints[C]//Forty-first International Conference on Machine Learning.

[2] Bennett N R, Watson J L, Ragotte R J, et al. Atomically accurate de novo design of single-domain antibodies[J]. bioRxiv, 2024.

[3] Olsen T H, Moal I H, Deane C M. AbLang: an antibody language model for completing antibody sequences[J]. Bioinformatics Advances, 2022, 2(1): vbac046.

[4] Shuai R W, Ruffolo J A, Gray J J. IgLM: Infilling language modeling for antibody sequence design[J]. Cell Systems, 2023, 14(11): 979-989. e4.

### Questions
1. How is epitope information encoded in the model, and to what extent can the model follow the guidance of a specific epitope?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces IgGM, a diffusion model for simultaneously generating the structure and sequence of antibody CDR regions, as well as the structure of antibody FR regions. By employing a two-stage training approach (folding -> design), IgGM is capable of conducting antibody design and docking tasks, and achieves commendable performance.

### Strengths
1. Compared to most existing antibody design works, this study undertakes the design of the antibody CDR region without providing the antibody framework, consistent with dyMEAN. This setting is closer to real-world requirements and poses a greater challenge. 

2. The two-stage training approach brings a significant performance improvement. 

3. Additionally, targeted designs focusing on inter-chain interactions were conducted.

### Weaknesses
1. It seems that the work does not fundamentally differ from DiffAb in terms of diffusion algorithm, and it bears some resemblance to ESMFold in model design. 

2. Having only five samples for each example is somewhat limited. 

3. The main paper does not provide an explanation of the diffusion algorithm, nor does it detail the representation of proteins and the prior distribution associated with each modality.

### Questions
1. In line 278, the probabilities of 4:2:2:2 are assigned to the model to design CDR H3, CDR H, and all CDR. So, do you mean CDR-H3 only for 4/10, all CDR-Hs for 2/10, and all CDRs for 2/10 ?

2. I am confused as to why you ultimately trained a consistency model. I comprehend that the key advantage of a consistency model lies in accelerating the generation process, enabling completion in fewer steps. However, in a scientific problem such as antibody design, is acceleration truly significant? Related to this, you have employed 10 different sampling steps in the appendix, but aside from the success rate (SR), the other four metrics seem to decline with the increase in the number of steps. Even SR shows a decline after surpassing 10 steps. These experimental results differ from what we might anticipate; I would appreciate a detailed study and explanation. Additionally, how does the original model perform without using the consistency model?

3. Some physical metrics (like energy) are needed for a more refined evaluation. 

4. The model name (DiffIg) in Figure 4 has not been updated. 

5. The experiment in Section 4.2 seems to be the most important one in this paper, but I am somewhat unclear about the experimental setup here.
  
    5.1. 'Structure prediction⇒docking⇒CDR generation⇒side-chain packing' is not the process of dyMEAN. On the contrary, this is the 'Existing Works Pipeline' shown in Figure 1 to highlight the end-to-end approach of dyMEAN.
  
    5.2. In the stage of structure prediction, how is the sequence of the CDR part defined? Are random sequences used, or are special symbols like [MASK] employed?
  
    5.3. The AAR of dyMEAN on CDR-H3 is much lower than reported in the literature. Although the training and test data used in this paper differ from the original dyMEAN article, such a significant disparity needs to be explained (43.65%->29.4%).
  
    5.4. In both Section 4.1 and 4.2, there is a version of IgGM that uses AF3-predicted structures for initialization, making the prior distribution inconsistent with N(0, I). How does IgGM predict the structure in this scenario? Was an additional IgGM trained with AF3 predicted structures as prior? If it's the latter, how were the noise addition and denoising processes conducted?"

6. Finally, for any researcher working on AI-driven antibody design, aligning in silico evaluation with wet-lab experimental results is an essential consideration. Unfortunately, I did not find any discussion related to this in the main paper. If IgGM were to be eventually validated experimentally, do you believe the advantages of IgGM over other methods would still persist? Also, what improvements could be made?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces IgGM, a generative model designed for creating functional antibody and nanobody structures. It integrates sequence and structural generation using a multi-level network approach comprising a pre-trained language model, feature encoder, and prediction module. Experimental results demonstrate its applicability in antibody and nanobody design tasks.

### Strengths
1. IgGM successfully establishes an antibody co-design pipeline that integrates several key elements previously deemed essential by the research community.
2. IgGM shows promising docking success rates over AF3, suggesting that it effectively captures essential antibody-antigen interaction patterns.

### Weaknesses
1. Although the IgGM framework demonstrates its effectiveness on various antibody design tasks, it relies heavily on components and algorithms established in prior work, limiting the originality and impact of its contributions to the field.
2. The study lacks direct comparisons with RFDiffusion while they perform similar tasks
3. The authors didn't report the variance/robustness of their performance.

### Questions
1.  It would be valuable to explore why AlphaFold 3 (AF3) shows stronger performance in structure prediction than in docking-related metrics, perhaps due to limitations in capturing finer antigen-binding dynamics.
2. Why did the authors opt not to use RAbD as the test set, as has been customary in prior studies?

### Soundness
3

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
4

### Summary
The article discusses the challenges in practical applications where obtaining the structures of antigens and antibodies (including the framework region) is often unfeasible. To address this, it introduces an end-to-end algorithm called IgGM, which simultaneously predicts the sequences and structures of the CDR regions, performs docking based on the epitope, and predicts the structure of the antigen-antibody complex. Additionally, the article employs a two-stage training method to enhance the model's performance in predicting both structures and sequences.

### Strengths
The innovation of the article lies in further extending the scope of co-design. The model can not only design amino acids in the CDR region, but also complete antigen docking, which is unprecedented. IgGM not only has excellent performance in predicting antibody structure, especially in the CDR region, but also outperforms existing algorithms in docking. IgGM not only shows SOTA performance on conventional antibodies, but also performs well on nanobodies, further demonstrating the application value of IgGM.

### Weaknesses
1. The core architecture of the algorithm is very similar to AF2, and the accuracy of structural prediction seems to be attributed to AF2. However, the innovation in this part is not strong enough.
2. The article did not elaborate on the introduction and explanation of the Inter chain Feature Embedding Module and Structure Encoder.
3. In line 343, the authors define the success rate as DockQ > 0.23. The authors should either provide a reference to justify the selection of this threshold or elaborate on the rationale behind it.
4. From the overall text, it appears that the length of the antibody CDR regions is also specified; however, the authors do not clarify this in the problem formulation.

### Questions
1. It was observed in Table 1 that HDock does not have an asterisk (*) symbol. Does this mean that Hdock did not use the information of the epidemic, resulting in lower indicators such as DockQ?
2. ESM-PPI is a model retrained based on Sabdab data. Will the training data of ESM-PPI appear in the validation set of IgGM? Has this part been considered?
3. As shown in Table 1, IgGM (AF3) uses the structure predicted by AF3 as the initial state. However, the authors do not clearly specify whether the sequence input to AF3 is the ground truth sequence. If it is, this would indirectly leak the answer, leading to inflated metrics for both the subsequent structures and sequences. A more rigorous approach would be to use 'initial state sequence + AF3 (random initial state sequence)' instead of 'random initial state sequence + AF3 (ground truth sequence)'

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present IgGM, a novel generative model designed for creating antibodies and nanobodies.

### Strengths
The authors curated a high-quality dataset that includes recent data from 2023, ensuring the model is trained on up-to-date antibody and nanobody structures. 

By employing diffusion models—a powerful approach in generative modeling—the authors introduce a novel solution to antibody and nanobody design. This choice is especially suited to capturing complex structural and functional patterns in biological data. The study includes ablation studies and comparisons with multiple alternative models.

### Weaknesses
The manuscript could benefit from a clearer and more detailed explanation of the features and methods. Improved clarity in this section would enhance readers' understanding of the model’s construction and functionality.

While the authors have effectively applied diffusion models, the novelty is somewhat limited since diffusion models have already been established in various generative tasks.

### Questions
The authors did not explicitly mention the threshold used for redundancy reduction in the summary provided. This information is typically included in the methods section, where they would specify the sequence identity cutoff or other criteria used to filter out redundant sequences.

While the authors have shared the test data, any plans for sharing the training data have not been clearly outlined.

### Soundness
2

### Presentation
2

### Contribution
2
