# A Resolution-Agnostic Geometric Transformer for Chromosome Modeling Using Inertial Frame

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 4

## Abstract
Chromosomes are the carriers of genetic information. Further understanding their 3D structure can help reveal gene-regulatory mechanisms and cellular functions. However, high-resolution 3D structures are often missing due to the high cost and inherent noise of experimental screening. A standard pipeline for reconstructing the chromosome 3D structure first applies the single-cell Hi-C high-throughput screening method to measure pairwise interactions between DNA fragments at different resolutions; then it adopts computational methods to reconstruct the 3D structures from these contacts. These include traditional numerical methods and deep learning models, which struggle with limited model expressiveness and poor generalization across resolutions. To handle this issue, we propose InertialGenome, a novel transformer-based framework for robust and resolution-agnostic chromosome reconstruction. InertialGenome first adopts the inertial frame for the pose canonicalization. Then, based on such an invariant pose, it proposes a Transformer with geometry-aware positional encoding, leveraging Nyström estimation. To verify the effectiveness of InertialGenome, we conduct experiments on two single-cell 3D reconstruction datasets with four resolutions, reaching superior performance over all four computational baselines. Additionally, we observe that the 3D structure reconstructed by InertialGenome is more in line with the results of real experimental results on two functional verification tasks. Finally, we leverage InertialGenome for cross-resolution transfer learning, yielding up to a 5\% improvement from low to high resolution. The source code is available at https://github.com/yize1203/InertialGenome.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces InertialGenome, a novel Transformer-based framework for reconstructing 3D chromosome structures from Hi-C data. The core contributions are a two-stage process: first, it canonicalizes the initial 3D coordinates using an inertial frame to achieve pose-invariance, removing translational and rotational ambiguity. Second, it employs a geometry-aware Transformer, enhanced with 3D rotary position embeddings (RoPE) and Nyström-based structural features, to refine these coordinates. The proposed method demonstrates state-of-the-art performance on two single-cell datasets across four resolutions, showing particular strength in generalization and cross-resolution transfer learning.

### Strengths
The primary contribution is the elegant and effective use of inertial frame canonicalization as a preprocessing step for 3D chromosome structure refinement. This method directly addresses the fundamental problem of pose variance in geometric deep learning, providing a principled way to standardize inputs. This standardization appears to be the key driver behind the model's impressive resolution-agnostic capabilities and strong generalization performance.

The paper's novelty lies not in a single component, but in the intelligent synthesis of several ideas. While inertial frames and Transformers are not new in themselves, their application and combination in this domain are novel and well-motivated. The theoretical analysis of the inertial frame's stability (Section 3.5) using the Davis-Kahan theorem is a significant strength.

### Weaknesses
The most critical weakness is the omission of highly relevant and recent related work in both the discussion and the experimental comparison. This significantly impacts the claims of novelty and the comprehensiveness of the evaluation.

1. Missing Baseline: A key missing baseline is CHROMFORMER (NIPS'2022), which also uses a Transformer-based architecture for this exact problem. Without a direct comparison, it is difficult to ascertain whether the performance gains of InertialGenome stem from the novel inertial frame canonicalization or simply from the power of the Transformer backbone, which CHROMFORMER already established.

2. Unaddressed Novelty: The paper's geometry-aware positional encoding (3D-RoPE) is presented as a novel adaptation. However, the recent work "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING (ICML'2025)" appears to propose a very similar 3D extension of RoPE. The lack of citation and discussion of this work obscures the precise novelty of the authors' formulation.

### Questions
1. Comparison with CHROMFORMER: Could the authors please comment on the CHROMFORMER (NIPS'2022) model? Given its use of a Transformer for the same task, it appears to be a crucial baseline. Can you provide a comparison, or a compelling argument for its exclusion, to better situate your performance results?

2. Novelty of 3D-RoPE: The recently proposed STRING (ICML'2025) introduced a similar 3D rotary position encoding. Could you please clarify the novelty of your 3D-RoPE implementation in relation to this prior work?

3. Ablation Study for Loss Parameter α: Your hybrid loss function in Equation (12) is controlled by the parameter α. Could you please provide an ablation study showing how performance (e.g., dSCC and dRMSE) varies with different values of α? This is important for understanding the interplay between the two loss components and for reproducibility.

4. In Section 3.4, you motivate the Value-Weighted MSE by stating that Hi-C data has higher reliability for smaller distances. However, another common approach for handling data with varying reliability or potential outliers is to use the Mean Absolute Error (MAE), which is inherently less sensitive to large errors than MSE. Could you please elaborate on the reasoning for choosing a weighted MSE scheme over a simpler, more robust loss function like MAE?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This model tackles the problem of inferring genomic 3D coordinates from a distance matrix derived from Hi-C data.  The approach involves training a transformer model to infer coordinates that, essentially, maximize the agreement between the observed distance matrix and the inferred one.  The novel aspects include using something called the Nystrom method to infer candidate sets of points from an RBF kernel matrix, and using RoPE embeddings in 3D.

### Strengths
The overall approach contains several novel ideas, as outlined above.

The paper contains fairly extensive empirical results, largely following the evaluation protocols proposed in previous work in this area.

### Weaknesses
Overall, I found the empirical validation of the methods pretty unsatisfying.  In general, it's difficult to benchmark 3D genome reconstruction methods, because it is hard to find an orthogonal source of gold standard structures.   One common approach is to aggregate the single-cell reconstructions and compare the result to a bulk Hi-C dataset from the same type of cells.  Another approach is it to take similar approach using imaging data (e.g., chromatin tracing).

The list of methods to infer 3D chromatin contacts (line 44) is radically incomplete.  You should focus on methods that infer genome-wide contacts, which would leave off 3C, 4C, and 5C, but would include GAM, SPRITE, ChIA-PET, and many others.

Similarly, the list of methods for 3D reconstruction (lines 72-77) is incomplete.  The list later on the same page is better.  You should re-organize the text so that all the related work is discussed at once.

The critique of related work is pretty vague.  You just say that all of these methods "rely on simplistic modeling of contact matrices as the sole input, lacking deeper structural interpretation, and their model expression ability may be constrained and limited."  It's not clear what you mean by "lacking deeper structural interpretation" nor what the "model expression" refers to.  I'd rather see specific critiques of particular methods, or at the very least a more precise critique.

line 110: Not every method uses these two steps.  You should make clear that the problem does not have to be solved in this fashion.  And even if it is broken down into two steps, there is not agreement in the field about the proper transfer function to translate from counts to distances.

Minor points:

The abstract should make it clear earlier (around line 15)  you are talking about single-cell Hi-C data.  You should also be careful to describe it as such in the intro to Section 4.  If you just say "Hi-C," that typically means bulk Hi-C.

line 107: The number of bins in the genome depends on what reference genome you use, not what cell line you are looking at.

line 116: What does "canonicalized" mean here?

line 216: Give the Williams & Seeger citation when you first mention the Nystrom method.

line 218: I thought the phrase "and the effectiveness of this approach has been confirmed by research" was oddly vague.  What kind of research?

A brief description of the Nystrom method (as in lines 218-221, but shorter) should appear in the Introduction.

### Questions
Why did you only use two single-cell Hi-C datasets?  There are many more available.

How did you decide which methods to compare your method against?  Did you run them yourself, or just compare to published results?

I don't actually understand the sentences (lines 81-83) that describe how an "inertial frame" is used.  Some definitions of terms would be helpful.  What do you mean by "inertial frame" here, and "position canonicalization"?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces InertialGenome, a novel Transformer-based framework for 3D chromosome reconstruction that is designed to be robust to different resolutions. The author proposes an inertial frame canonicalization and geometry-aware positional encoding scheme that combines 3D RoPE and Nyström approximation. The method demonstrates strong empirical results, outperforming baselines.

### Strengths
1. The use of inertial frame canonicalization to achieve pose invariance is a reasonable and well-justified preprocessing step.

2. The model consistently outperforms existing baselines in reconstruction metrics, TAD consistency and cross-resolution generalization.

### Weaknesses
1. In Sec 3.2, the textual description (L180-183) appears to contradict the mathematical formulation in Eq 5, where 'Selective' and 'Separate' modes seem functionally identical.

2. In Eq 2, the query/key vectors are intended to be 6D vector. Can we understand six as a pair, just as we considered two as a pair in RoPE? This lack of clear explanation hinders understanding.

3. It is not explicitly stated, but is the 'vocab_size' simply the total number of bins for a given chromosome (248,947 at 1Kb resolution)? This confirmation would be helpful for understanding.

4. The paper claims RoPE-3D has limitations in 'capturing pairwise distance relationships' (L214), but this claim is not sufficiently substantiated or demonstrated. The paper should provide an ablation study to demonstrate the contribution of this Nyström approximation .

5. In Sec 3.2, $u_i$, $A$, and $O$ are used in the main text but are only clearly defined several lines later, which hinder understanding.

6. The paper relies exclusively on distance-based metrics (dSCC, dRMSE). For a task focused on 3D structure reconstruction, TM-score or RMSD are essential for evaluating the global structural similarity of the predicted conformations.

7. The paper claims the method is 'resolution-agnostic' and attributes this to the inertial-frame alignment and RoPE. This explanation is high-level and lacks a deep analysis. A more thorough explanation is needed to precisely understand why this specific combination of components grants robustness to resolution changes, which is a key claim of the paper.

### Questions
1. On Line 219, the phrase "relationships between anchor points" is used twice in the same sentence. Is this a typographical error?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposed InertialGenome, a novel transformer-based framework for robust and resolution-agnostic chromosome reconstruction.  With this new design, it greatly improved the accuracy compraed to previous methods.

### Strengths
1. The background and problem setting is clearly addressed.
2. The network architecture is well designed and improved over previous graph-neural-network based approaches.
3. The performance improvement is impressive.

### Weaknesses
1. I think the problem setting has a siginificant limitation: there is no other ways to validate the predicted coordinates. In your setting, you compared the projected contact map from your 3D coordinates and the original contact map. However, single-cell Hi-C is very sparse and it can not fully capture the real 3D coordinates. Therefore, the reconstructed structure can be wrong even it has high agreement with the contact map.  Instead, I suggested to include chromatin tracing data to validate your methods, which you can refer to Higashi, which adopted some popular dataset.

2. The ablation study of the design is very limited, which I think need substantial improvement.
2.1 How to balance the structural stability loss and MSE loss? Are both losses needed?
2.2 What is the performances without inertial frame canonicalization?
2.3 What is ROPE's contribution to the performances?
2.4 What is the contribution of Nystr¨ om Approximation for Structure Tokenization?

3. The cross-resolution transfer benchmark is not so useful. We should expect we have different models for different resolutions, since they have very different biological focuses.

INERTIAL FRAME CANONICALIZATION

### Questions
1. Problem setting for real chromatin tracing data.
2. Ablation study to understand the key contribution of the designs.
3. The biological validation is not so meaningful, since you should expect to have similar TAD/compartment observations if you can make distance map close to the original distance/contact map, but that does not mean

### Soundness
1

### Presentation
2

### Contribution
2
