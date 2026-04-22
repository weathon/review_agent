# FractalFold: Towards Fractal Structure Modeling for Hierarchical Inverse Protein Folding

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Inverse protein folding aims to design amino acid sequences that fold into desired backbone structures, representing a long-standing challenge in computational protein design. While recent deep learning approaches have achieved significant progress, existing methods predominantly treat protein structures as flattened sequences, overlooking their inherent hierarchical and fractal organization. To address this limitation, we propose FractalFold, a novel transformer-based model that performs structure-informed inverse folding by recursively invoking multi-level atomic fractal transformers. FractalFold employs a coarse-to-fine sequence refinement paradigm that mirrors the intrinsic hierarchical nature of protein structures. To generalize our approach to quasi-fractal proteins with variable-length structural segments, we introduce the Hierarchical Fractal Segmentation Module (HFSM), which leverages attention patterns from pre-trained protein language models to recursively partition protein structures into tree-organized patches. Extensive experiments on the CATH benchmarks demonstrate that FractalFold achieves state-of-the-art performance in sequence recovery rate and perplexity while generating sequences with enhanced foldability, establishing a new paradigm for structure-informed protein design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the inverse protein folding problem by proposing FractalFold, a novel transformer-based model. The authors argue that existing methods overlook the inherent hierarchical and fractal organization of protein structures, treating them as flattened sequences. FractalFold's architecture is designed to align with this fractal geometry, employing a coarse-to-fine sequence refinement paradigm. A key component is the Hierarchical Fractal Segmentation Module (HFSM), which uses attention patterns from pre-trained protein language models to recursively partition the structure into tree-organized patches. This enables a multi-scale, one-shot decoding process that aims to reduce error accumulation and computational complexity. The authors report state-of-the-art performance on CATH benchmarks for sequence recovery, perplexity, and refoldability.

### Strengths
The central idea of explicitly modeling the hierarchical and fractal nature of protein structures is novel and provides a strong, physically-motivated inductive bias. This is a clear departure from methods that treat structures as simple graphs or flat sequences.

The ablation studies in Table 3 effectively validate the contributions of the key design choices: the pre-trained GVP encoder, the one-shot decoding strategy (vs. autoregressive), and the HFSM module .

### Weaknesses
The latest baselines in the paper were published in 2023. There has been many new inverse folding methods in the 2 years, which were reported to have better performance than FractalFold. I noticed some of them are discussed in the related work section in the appendix, could the authors explain why they are not compared? Besides, some SOTA inverse folding methods are neither mentioned in the related work section nor compared, which also needs further explanation.

Event though baselines in 2024 and 2025 are not compared, FractalFold’s improvement seems insignificant, with respect to both structural and sequence recovery.

Considering FractalFold uses a hierarchical design, a possible advantage would be better understanding of proteins’ functional properties and developability like thermostability, if the mechanism really learns some useful features. I suggest the authors to explore them to comprehensively evaluate the potential benefit of the architecture.

The paper claims the architecture "captures essential sequence-structure relationships" and "opens promising avenues" for tasks requiring "multi-scale understanding", but does not provide direct evidence of this beyond folding.

There is a clear contradiction in the definition of the HFSM's link probability, $a_i$. In Figure 2, the diagram specifies $a_i = A_{i, j+1} + A_{i+1, j}$. However, the core methodology in Section 3.2 and Algorithm 1 relies on a different definition, $a_i = \sqrt{A_{i,i+1} \times A_{i+1,i}}$ (Equation 4). This inconsistency creates confusion about the exact mechanism used for segmentation.

The case study presented in Figure 3 and its caption contains conflicting information. The caption identifies the protein as "Irtu.A (114 residues)" with an average pLDDT of 92.2 , but the figure itself is clearly labeled "4owz.A" with "Length: 134" and different metrics (Average pLDDT 96.2, pTM Score 0.880). This same "4owz.A" figure is duplicated as Figure 5. This discrepancy makes the case study impossible to interpret correctly.

The HFSM (Algorithm 1) requires the segment counts for each fractal scale, $\{L_k\}_{k=1}^K$, as an input parameter. The paper does not explain how these crucial hyperparameters are selected. The algorithm only determines where to place breakpoints based on $L_k$, not the value of $L_k$ itself, which seems fundamental to the fractal decomposition.

The paper claims the HFSM generalizes FractalFold to "quasi-fractal proteins". This term is not clearly defined. There is no empirical validation (e.g., analysis across different CATH topologies) to demonstrate that the model's performance is particularly strong or improved on proteins that are less 'classically' fractal, which would be needed to substantiate this generalization claim.

### Questions
In table 2, could the the metrics of wild-type sequences be reported as a baseline, considering the inaccuracy of the folding models?

Could the authors clarify the confusions mentioned in the weaknesses section?

### Soundness
1

### Presentation
1

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
This paper studies the important inverse folding problem in protein design. The authors propose FractalFold, a novel transformer-based model that performs structure-informed inverse folding by recursively invoking multi-level atomic fractal transformers. FractalFold achieves state-of-the-art performance in several benchmarks.

### Strengths
- This paper is well-written and easy to follow.
- FractalFold first introduced the concept of fractals in the problem of inverse folding.
- The authors conduct ablation studies to analyze the design choice.

### Weaknesses
- Without parentheses, the citation format is unclear.
- Several recent related works are missing. For example, LM-Design [1] and Bridge-IF [2]. 
- HFSM uses pretrained ESM2 to extract features. However, in Table 1, all baselines do not use any pre-trained knowledge. The performance comparison may be unfair, and can not verify the effectiveness of the fractal decoding framework.
- ProteinMPNN is widely used in de novo protein design when combined with RfDiffusion and AF3. It is interesting to evaluate whether the proposed method can be used in such an important case [3].

[1] Structure-informed Language Models Are Protein Designers

[2] Learning Inverse Protein Folding with Markov Bridges

[3] A Holistic Evaluation of Protein Foundation Models

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces FractalFold, a transformer-based framework that models protein inverse folding by aligning architectural design with the fractal and hierarchical nature of protein structures.

### Strengths
- The idea of introducing a fractal inductive bias for protein inverse folding is original and aligns well with the hierarchical organization of real proteins. The recursive, coarse-to-fine architecture provides a biologically meaningful modeling perspective rarely seen in this area.

### Weaknesses
- Baseline selection is relatively old (only up to 2023). The authors should compare with more recent methods such as SPDesign (https://academic.oup.com/bib/article/25/3/bbae146/7642672), SurfPro (https://arxiv.org/abs/2405.06693), or BC-Design (https://www.biorxiv.org/content/10.1101/2024.10.28.620755v2). They also mentioned many recent works for inverse folding in their related work sections, but they are not compared with.
- The architecture is biologically-inspired, but the experimental results are purely quantitative. The authors should provide some biological insights into the model's predictions. For instance, an example of how this FractalFold model works well on a specific protein structure, while baseline methods fail, is not provided. This could potentially be used to show the superiority of the proposed method. Other questions are listed in the questions section.

### Questions
- Does fractal hierarchy generalize to multi-chain or quaternary complexes (e.g., antibodies, enzymes)?
- How does the model behave on intrinsically disordered or low-structure regions where fractal segmentation may be ill-defined?
- Figure 3 is inconsistent with its caption.

### Soundness
2

### Presentation
2

### Contribution
2
