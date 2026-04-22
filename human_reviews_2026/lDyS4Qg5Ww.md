# RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
The inverse design of RNA three-dimensional (3D) structures is crucial for engineering functional RNAs in synthetic biology and therapeutics. While recent deep learning approaches have advanced this field, they are typically optimized and evaluated using native sequence recovery, which is a limited surrogate for structural fidelity, since different sequences can fold into similar 3D structures and high recovery does not necessarily indicate correct folding. To address this limitation, we propose RIDER, an RNA Inverse DEsign framework with Reinforcement learning that directly optimizes for 3D structural similarity. First, we develop and pre-train a GNN-based generative diffusion model conditioned on the target 3D structure, achieving a $9\\%$ improvement in native sequence recovery over state-of-the-art methods. Then, we fine-tune the model with an improved policy gradient algorithm using four task-specific reward functions based on 3D self-consistency metrics. Experimental results show that RIDER improves structural similarity by over $100\\%$ across all metrics and discovers designs that are distinct from native sequences.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose to extend recent RNA inverse folding methods such as RiboDiffusion and gRNAde to generate non-native RNA sequences with near-native structural similarities. Overall, the method's performance is compelling, pointing to the importance in future work of using reinforcement learning to steer pretrained generative RNA models to more relevant regions of biochemical design space.

### Strengths
1. The authors cleverly combine the best architectural aspects of previous RNA inverse folding methods such as gRNAde and RiboDiffusion.
2. Structure-focused and reinforcement learning-based finetuning of RNA inverse folding methods does seem to contribute a substantial improvement to their ability to design structurally-native RNA sequences.
3. The authors follow best practices in RNA inverse folding evaluation by adopting gRNAde's datasets and benchmarking protocols.
4. The authors' ablation experiment showing that using AlphaFold 3 instead of RhoFold in their method does not impact performance much is encouraging to see.

### Weaknesses
1. Although the authors made their code available for review, it could be cleaned up and documented more carefully. For instance, including a descriptive README.md file with it would go a long way to helping users navigate the codebase.
2. The authors claim they are the first to point out the limitations of using native sequence recovery as a primary metric for RNA inverse folding. However, this has already been well discussed within the context of an adjacent topic (i.e., protein inverse folding).
3. Lines 119-122 seem a bit indirect in saying that RNA structure prediction is important, yet not the most important, but still important for RNA design. Although most structural metrics related to RNA design are ultimately dependent on the accuracy of RNA structure prediction algorithms, the authors' descriptions here are vague and unclear concerning which directions are important to pursue to improve RNA design in future work.

### Questions
1. Have the authors investigated how well their fine-tuned (DM-RL) checkpoints perform for multi-state RNA design, as investigated in previous works such as gRNAde? This would interesting to see (i.e., whether reinforcement learning can capture such multi-state RNA conformational information implicitly).

### Soundness
3

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
5

### Summary
The paper proposes a two-stage framework for RNA inverse folding. First, a graph neural diffusion model (DM-GNN) is pretrained to reconstruct native sequences from 3D RNA structures. Then, the model is fine-tuned via reinforcement learning (DM-RL) to directly optimize structural fidelity metrics (GDT-TS, TM-score, RMSD) predicted by a structure oracle. This approach substantially improves 3D self-consistency compared to prior RNA design methods, while maintaining sequence diversity and foldability.

### Strengths
The paper is technically sound and well executed. The writing is clear, structured, and easy to follow. The empirical results are strong, demonstrating consistent and substantial improvements over prior state-of-the-art RNA inverse-folding approaches. The overall methodology—a diffusion model pretrained on sequence recovery and fine-tuned via reinforcement learning for 3D structural fidelity—is reasonable and well motivated. The success of this framework aligns with prior evidence in the protein inverse-folding literature, where diffusion combined with RL optimization has been shown to yield similar performance gains.

### Weaknesses
While the paper is well executed, it lacks genuine methodological novelty. The proposed framework, diffusion-based sequence generation followed by reinforcement learning fine-tuning for structural fidelity, has already been explored extensively in the context of protein inverse folding. The current work largely ports that established recipe to RNA, with the main differences being the input domain (RNA rather than protein), the smaller token vocabulary (4 bases vs. 20 amino acids), and the use of RNA-specific structure predictors.

The use of RL to steer diffusion models is also not novel; similar formulations have been reported in both general diffusion literature and multiple protein design papers. As such, the contribution lies primarily in application rather than new ML methodology. Moreover, the paper underplays or omits discussion of this related body of work, giving the impression that concepts such as “native sequence recovery not reflecting structural fidelity” are new insights, whereas they are well established in the protein design literature.

In its current form, the paper presents a well-engineered application of known ideas to a new molecular domain, which would be more appropriate for a domain-specific or biological modeling venue rather than ICLR, which prioritizes algorithmic or conceptual advances in machine learning.

### Questions
See weaknesses.

### Soundness
3

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
5

### Summary
The paper tackles RNA sequence design conditioned on a target 3D structure. The main contribution is a RL methodology that optimizes for structural similarity metrics (as opposed to previous supervised learning methods). Results show that the RL training leads to significant improvements over the baseline gRNAde model.

### Strengths
- The problem chosen is significant, as RNA inverse design methods are used practically and validated in wet labs recently. The paper identifies an important research gap and tackles the problem of RL-driven structural similarity optimization in RNA design well. To the best of my knowledge, this is one of the first works to tackle this important problem.

- The paper is generally well written overall. The appendix/supplementary could be better organized.

- I found the experiments and results convincing and supportive of the main claims of the paper around RL optimization. The result of the supervised trained model also outperforming gRNAde on the challenging single-state Das split is also a significant gain.

### Weaknesses
My main issue with this paper is how its trying to optimizer via RL directly for structural metrics using RNA 3D structure prediction that is not itself very reliable. This way, the model may optimizer for structure as predicted by the folding model, but the folding model may be very poor (as recent CASP contests for RNA show...) -- this means that the model may not really be doing what its meant to do. 

I realise that this is not the current aim of the paper. I would still have liked to see discussions on this limitation within the paper, as it can limit practical applicability.

Other than this, I don't see any major issues with the methodology/from a technical/ML point of view.

### Questions
- A major question I had based on the manuscript: Can you show a case study or explain what may happens in the following situation: I wish to design a particular backbone 3D structure. I have its native sequence. The 3D structure predictor I happen to use simply cannot fold the native sequence into the target 3D structure (its a poor predictor). So what will happen if I use the proposed RL finetuning method? Won't the model sort of 'hack' the 3D structure predictor I am using and end up designing something that folds well according to the structure predictor, but may not do so in nature?

- Additionally, and relatedly to the weakness I identified (+ my Q above), can the RL methodology be extended to other oracles? Can you show experiments with, say, secondary structure ensemble properties such as Mean Free Energy, or some other reasonable properties of interest for RNA design? I think that could strengthen the paper and also show the generality of RL finetuning a strong base supervised model.

Other suggestions:

- On contribution claim: "We identify the limitations of native sequence recovery..." - I don't think this should be presented as a claim. This is well known in the community.

### Soundness
3

### Presentation
3

### Contribution
3
