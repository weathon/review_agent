# GeoMFormer: A General Architecture for Geometric Molecular Representation Learning

- Decision: Reject
- Scores: 6, 8, 6, 5

## Abstract
Molecular modeling, a central topic in quantum mechanics, aims to accurately calculate the properties and simulate the behaviors of molecular systems. The molecular model is governed by physical laws, which impose geometric constraints such as invariance and equivariance to coordinate rotation and translation. While numerous deep learning approaches have been developed to learn molecular representations under these constraints, most of them are built upon heuristic and costly modules. We argue that there is a strong need for a general and flexible framework for learning both invariant and equivariant features. In this work, we introduce a novel Transformer-based molecular model called GeoMFormer to achieve this goal. Using the standard Transformer modules, two separate streams are developed to maintain and learn invariant and equivariant representations. Carefully designed cross-attention modules bridge the two streams, allowing information fusion and enhancing geometric modeling in each stream. As a general and flexible architecture, we show that many previous architectures can be viewed as special instantiations of GeoMFormer. Extensive experiments are conducted to demonstrate the power of GeoMFormer. All empirical results show that GeoMFormer achieves strong performance on both invariant and equivariant tasks of different types and scales. Code and models will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a Transformer for molecular property prediction. The framework roots from separating invariant/equivariant representations and introduces cross-attention to fuse the two representations effectively. The framework provides a recipe for scaling up the number of parameters easily. Experiments on the multiple benchmarks (OC20, PCQM4Mv2, Molecule3D, MD17, etc) demonstrate SoTA or near SoTA performance.

### Strengths
1. The work stems from designing architectures that separate in-/equi-variant representations which provides flexibility of scaling up the # parameters through introducing the cross-attention between the two representations. 
2. Proposed GeoMFormer demonstrates SoTA or at lest competitive performance on multiple standard benchmarks when compared with strong baselines. 
3. The paper is well written and easy to follow.

### Weaknesses
1. The model achieves strong performance in many benchmarks but also contains more # parameters than most baseline models.

### Questions
1. Following the weakness, training GeMFormer of different sizes can be useful to elaborate how much performance gain stems from scaling up # parameters. Since the proposed method designs an equivariant Transformer that can be better scaled, such experiments can further demonstrate the advantage of GeMFormer. 
2. Is cutoff of interatomic distances applied when implementing attention between atoms in GeMFormer? If yes, how is the cutoff determined?
3. The work has included benchmarks on multiple standard molecular property prediction tasks. How does the model perform on QM9. Though it's a bit older compared with some benchmarks, it's still a widely used. It would be helpful to see where GeMFormer sits on QM9 as well. 
4. In appendix D.7, the authors present ablation studies on N-body simulation task. However, as the work focuses mostly on molecular property predictions. It would be better to include some ablations on molecular benchmarks to further validation the design choices.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new Transformer-based architecture, GeoMFormer, for molecular modeling that considers both invariant and equivariant properties in the tasks. Specifically, GeoMFormer comes with two streams: one captures invariance, and the other captures equivariance. A cross-attention mechanism is proposed, which allows one stream to extract information from another one. Extensive experiments have been conducted, and GeoMFormer shows significant improvements over baselines across all six tasks considered. Ablation studies clearly demonstrate the effectiveness of each module in the model.

### Strengths
1. This paper is well-written and well-motivated.
2. The proposed architecture has a clear and consistent design principle, resulting in an elegant and effective model that performs even better than those utilizing complex features.
3. The design of cross-attention is intuitive but effective in preserving more structural information. 
4. Extensive experiments have clearly demonstrated the superiority of GeoMFormer over many competitive baselines.

### Weaknesses
1. I'm generally satisfied with the extensive experiments, but it would be better to benchmark GeoMFormer on QM9 due to its popularity.
2. For experiments on PCQM4MV2, it seems only GeoMFormer and Uni-Mol+ are directly comparable, as both of them essentially utilize more supervision information.

### Questions
No.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a novel molecular model called GeoMFormer for molecular modeling, which is based on the Transformer architecture. GeoMFormer incorporates two distinct streams for preserving invariant and equivariant representations and employs cross-attention modules to facilitate information exchange between them. The proposed GeoMFormer is a general and adaptable framework, with other existing architectures being special cases. Extensive experiments demonstrate GeoMFormer's strong performance on invariant and equivariant tasks across various types and scales. The potential of our GeoMFormer can be further explored in a broad range of applications in molecular modeling.

### Strengths
GeoMFormer divides the process of learning invariant and equivariant representations into distinct streams, which are interconnected through self-attention and cross-attention modules. Through these cross-attention modules, the invariant stream is enriched with structural information from the equivariant stream, while the equivariant stream benefits from non-linear transformations originating from the invariant stream. This enables the comprehensive modeling of interatomic interactions, both within individual feature spaces and across them, in a unified and simultaneous fashion.


The proposed architecture is general. Many established methods can be seen as specific instances within our framework. For instance, PaiNN (Schütt et al., 2021) and TorchMD-NET (Thölke & De Fabritiis, 2022) can be configured as distinct realizations by adhering to the design principles of GeoMFormer and selecting appropriate configurations for essential building components.



The experimental results are thorough, including invariant and equivariant tasks, covering multiple mainstream tasks in the molecule modeling. The proposed GeoMFormer was compared with a bunch of state-of-the-art methods and outperform them in various setups.



The paper is well-written and easy to follow.

### Weaknesses
No.

### Questions
I look forward to more discussion of application of GeoMFormer in many other areas outside molecule modeling.



In the experiment, I wonder why different invariant tasks use different neural architectures as baseline methods, e.g., Section 5.2 and 5.3. The same thing happens for equivariant tasks. 


I wonder if the author could report the results of multiple runs, e.g., standard deviation to enhance the empirical results.


In N-BODY SIMULATION in Section 5.4, what are the input feature and groundtruth? input is 3D position, output is velocity? Please add more details. Thanks!

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new  Transformer-based molecular model, GeoMFormer based on standard Transformer modules. The architecture has an invariant representation brach and an equivariant representation brach. Cross-attentions are used to fuse the two kinds of representations in the architecture. Extensive experiments are conducted on many datasets. The model outperforms baselines.


------- After rebuttal -----

I would like to raise my score from "reject" to "marginally below the acceptance threshold" beacause of the new ablation experiment on MD17.

In my view, this paper still does not meet the standards required for ICLR because its motivation is not clear. The central argument of this paper is "effectively learning both invariant representations and equivariant representations simultaneously". As a study focused on information fusion, it is pivotal to analyze the distinct characteristics of these two types of information. Such an analysis is vital to demonstrate the importance of their fusion. Unfortunately, this paper presents a superficial exploration of these differences, which undermines its overall quanlity.

In addition, some experiments do not well support the core contribution of this paper. For instance, in the ablation studies in Table 8, the model without any cross-attn can still achieve competitive performance. The cross-attn module is the main technological contribution of this paper.

### Strengths
S1: The proposed model is validated on extensive experiments and tasks;

S2. The performance of the model is commendable;

S3: The paper is well written.

### Weaknesses
W1: The intuition and rationale of modeling is not clear. There is a lack of analysis of the difference between an invariant representation and an equivariant representation in terms of the information they contain.

W2: The novelty in this paper is limited. Simultaneously, using an invariant representation and an equivariant representation, such as VisNet, is not new. Cross-attention is well-known in multimodal deep learning.

W3: The experimental results cannot support the advantage of fusing two kinds of representation. The ablation studies in Table 8 are interesting. The model without any cross-attn can still achieve competitive performance. The cross-attn module, the main contribution of this paper, may contribute little to performance in experiments.

### Questions
Q1: Can you give empirical evidence that invariant and equivariant representations have different useful information for molecular learning?

Q2: In Table 8, the model without cross-attn can still achieve competitive performance. From this point, is it very important to fuse two kinds of representation on such a powerful Transfermor model?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
