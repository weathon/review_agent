# Follow the MEP: Scalable Neural Representations for Minimum-Energy Path Discovery in Molecular Systems

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Characterizing conformational transitions in physical systems remains a fundamental challenge, as traditional sampling methods struggle with the high-dimensional nature of molecular systems and high-energy barriers between stable states. These rare events often represent the most biologically significant processes, yet may require months of continuous simulation to observe. One way to understand the function and mechanics of such systems is through the minimum energy path (MEP), which represents the most probable transition pathway between stable states in the high-friction, low-temperature limit. We present a method that reformulates MEP discovery as a fast and scalable neural optimization problem. By representing paths as implicit neural representations and training with differentiable molecular force fields, our method discovers transition pathways without expensive sampling. Our approach scales to large biomolecular systems through a simple loss function derived from the path's likelihood via the Onsager-Machlup action and a scalable new architecture, AdaPath. We demonstrate this approach using four proteins, including an explicitly hydrated BPTI system with over 3,500 atoms. Our method identifies a MEP that captures the same conformational change observed in a millisecond-scale molecular dynamics (MD) simulation, obtaining this pathway in minutes on a standard GPU, rather than the weeks required on a specialized cluster.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents AdaPath, which targets efficient discovery of minimum-energy transition paths for large biomolecular systems. Built upon the Onsager-Machlup action functional, AdaPath adopts a simple energy-based loss, which reformulated the MEP discovery as a continuous neural optimization problem. Meanwhile, the model is established with learnable embedding sharing across atoms and a conditioning network for the progress coordinate, which is tailored for scalability. Experiments on the 9-residue peptide AIB9 and the 58-residue protein BPTI with explicit water demonstrate the scalability of AdaPath to explore transition paths for large biomolecular systems with thousands of atoms.

### Strengths
- This paper presents a concise and effective derivation and simplification of the OM action potential, requiring only two endpoints and a given potential function for training. This greatly simplifies the overall training procedure, thereby enhancing the practical utility of AdaPath.
- The model architecture is well designed: it addresses the challenge of continuous paths through interpolation, and the use of shared atom embeddings enhances the model’s scalability.
- AdaPath is able to reasonably sample transition paths in both the AIB9 and BPTI systems, which differ greatly in scale. Moreover, it successfully reproduces intermediate states in the large BPTI system, comprising thousands of atoms with explicit water molecules, which is quite impressive.

### Weaknesses
- It is noted that AdaPath embeds only the progress coordinate and atom types, while lacking molecular topology and structural information in its inputs. This implies that the model must be trained separately for each system, making out-of-distribution generalization virtually impossible.

### Questions
1. AdaPath uses only atomic embeddings and the progress coordinate as inputs, without incorporating any topological or structural information. This design likely limits its out-of-distribution generalization capability, requiring retraining for each unseen system. Does this pose a challenge to the model’s practical applicability? I would appreciate clarification from the authors on this point.
2. Appendix G.1 shows that AdaPath undergoes a two-stage training process on the BPTI system, with different hyperparameters used at each stage. This raises some concerns regarding the complexity and robustness of the training procedure. If the model were to be applied to other molecular systems, would it require task-specific fine-tuning of the training strategy, or is the current training procedure generally applicable? I would like the authors to provide empirical evidence supporting the robustness of the training process across different systems.
3. Although the experiments on the BPTI system are quite impressive, evaluating the model on only two cases (i.e., AIB9 and BPTI) is not sufficient to demonstrate its overall feasibility. I recommend that the authors include additional examples of transition paths for biologically relevant processes. For instance, using fast-folding proteins [1] as target systems to study transitions between unfolded and folded states.
4. Section 2.2 reads more like a summary of related work, and placing it within the Method section may lead to confusion. It is recommended that the authors present this part as a separate section.

**References**

> [1] Lindorff-Larsen, K., Piana, S., Dror, R. O., & Shaw, D. E. (2011). How fast-folding proteins fold. Science, 334(6055), 517-520.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reformulates Minimum Energy Path (MEP) discovery as  a neural optimization problem. The transition pathways are parametrized with Implicit Neural Representations (INR) that are trained using a differentiable force-field and the proposed loss function. The loss function is derived from the transition path log probability and is essentially the batch-averaged potential energy along the transition pathway. A new INR architecture called AdaPath is proposed. The key feature of AdaPath is incorporating the progress coordinate $s$ via adaptive layer normalization and gating. Experiments are performed on two proteins: AIB9 and BPTI with explicit water molecules. For both proteins, the recovered transition pathways seem to be closely aligned with the results of long MD simulations.

### Strengths
The authors propose a scalable and efficient approach for MEP discovery. The experimental results demonstrate that trained INRs for a 58-residue protein with explicit water closely resemble states from long MD simulation performed in [1]. The training of INR requires $2 \times 10^{5}$ force field evaluations compared to $4 \times 10^{11}$ force field evaluations in MD simulation.

[1] Shaw, D. E., Maragakis, P., Lindorff-Larsen, K., Piana, S., Dror, R. O., Eastwood, M. P., ... & Wriggers, W. (2010). Atomic-level characterization of the structural dynamics of proteins. Science, 330(6002), 341-346.

### Weaknesses
- The loss function derived by authors does not seem to be well motivated or enforce continuity of the resulting transition pathway. The function $L = \frac{1}{B}\sum_{i=1}^B U(\phi_{\theta}(s_j))$ can be alternatively derived by assuming the independence of $x_{i+1}$ from $x_i$ (see Equations 3, 4): 
$$
log P(x_0, \dots, x_N) = \{ \text{assume } x_0, \dots, x_N \text{ are independent}, x_i \sim \text{Boltzman} 
\} = \sum_{i=1}^N log P(x_i) \propto \sum_{i=1}^N U(x_i).
$$
 When the transition pathway is reparametrized with $\phi_{\theta}$, we get the exact loss from AdaPath paper. However, the assumption of independence obviously does not hold for states in MEP as those must come form a continuous trajectory in the conformation space.
- The authors state a new AdaPath architecture and the new loss function as main contributions of the paper. Upon closer inspection, the derived loss seems to be a simplification of the loss functions introduced in [1]. However, the paper lacks experiments that demonstrate the benefits of the proposed loss function compared to loss function from [1]. Moreover, it would be great to see how the MLP architecture performs on AIB9 and BPTI with loss from [1].
- The proposed method’s efficiency is compared to the efficiency of classical methods only in terms of the number of force field evaluations. This does not take into consideration the computational resources required for the training of the neural network itself. Thus, it would be logical to compare those methods in terms of GPU-hours or wall-time.
- In Section D, the maximum energy for NEB is lower than the average energy, which seems to be impossible. This raises questions about the validity of baseline method results.

[1] Ramakrishnan, K., Schaaf, L. L., Lin, C., Wang, G., & Torr, P. (2025). Implicit Neural Representations for Chemical Reaction Paths. arXiv preprint arXiv:2502.15843.

### Questions
- The authors state that the validity of the proposed loss function approximation depends critically on maintaining sufficiently small displacement vectors. How is this ensured in practice?
- By removing the higher order gradient terms from the task defined be the loss function becomes ill-posed as it allows trivial solutions without continuity (see Section 2.1). To enforce continuity the authors have to carefully tune the optimizer and use gradient clipping. Both the Muon optimizer and the gradient clipping help maintain lower Lipshitz constant. I wonder if the method will become more robust to those hyperparameters if the loss from [1] is used?

[1] Ramakrishnan, K., Schaaf, L. L., Lin, C., Wang, G., & Torr, P. (2025). Implicit Neural Representations for Chemical Reaction Paths. arXiv preprint arXiv:2502.15843.

### Soundness
2

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
This paper proposes AdaPath, a fast and scalable architecture for finding minimum energy paths. The framework is optimized with the loss function from the path’s likelihood with the Onsager-Machlup action, and validated for two proteins.

### Strengths
1. Experiments on large biomolecules (Significance)

While prior works mainly have tackled the problem on smaller biomolecules with lengths of 10~20 residues, the paper additionally tackles BPTI with 58 residues.

### Weaknesses
1. Lack of comparison with ML methods

The paper lacks detailed comparison with prior works. Though authors have cited some prior works, I missed points on why some baselines are missing. They should denote clearly if the prior works fail to scale to larger systems, or impossible to compare for a reason

- A similar work solve transition path sampling target with Onsager_machlup functional [1]. The authors should states the difference between them.

- Though I understand that prior works mostly generate ensembles of transition paths rather than a minimum energy path, the authors should still denote the results [2-5].

- While the Doobs Seq2Seq [4] and Doobs’ Lagrangian [5] is different, the authors include only one of them across table 1 and table 2, for a different system. Both should be stated.

2. Experiments - physical validation on transition states

While the computed MEP from AdaPath does not get off the low energy region dramatically, additional validation on the generated path would strengthen the paper a lot. For example, measuring the dynamics content, energy along the MEP would improve validity on the MEP. If a committor function analysis is possible (there might not be one for every molecular system, I understand even if this is missing), doing them on intermediate states would be great.

3. Presentation (clarity)

I think the paper could improve presentation in some aspects. The most important results, quantitative results in table 1 to table 3 are all in the appendix. Making some space in the main paper an putting them would be good. section 2.2 on related works could summarized briefly to discuss the difference of AdaPath to prior works, and details to the appendix.

4. Lack of evidence for the claim fast

The authors claim a fast and scalable neural optimization problem. However, I do not find any results showing that the proposed method is faster than prior works. Additionally, what is the exact time component the authors claim for fast?

[1] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional, ICML 2025

[2] Stochastic Optimal Control for Collective Variable Free Sampling of Molecular Transition Paths, NeurIPS 2023

[3] Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers, ICLR 2024

[4] Scaling Deep Learning Solutions for Transition Path Sampling, preprint

[5] Doob’s Lagrangian: A Sample-Efficient Variational Approach to Transition Path Sampling, NeurIPS 2024

Minor

- Seems like section 4, 5 should be section 3.2, 3.3 regarding that 3.1 is for a system type?
- line 241 “shared transformer-like MLP blocks with ada”, this means that a single MLP is used for every coordinates?

### Questions
1. Scalability claim

One of the main contribution the authors are claiming is scalability. Is this scalability only for the system size? If so, there are no problems, but if something different, I do not see any results to support it

2. Architecture details

Could the authors explain more about the architecture? I understood what the task is and what the input & output of the model, but how is the AdaLN blocks incorporated? The original adaLN blocks were used diffusion transformers [1], replacing the standard norm layer with scale and shift parameters. Why is the up and down MLP needed?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this work, the authors present a neural representation-based approach to find the Minimum-Energy-Path (MEP) between two molecular conformations, an important problem in computational chemistry. For this purpose, the authors derive a loss function based on a relaxation of the Onsager–Machlup action functional. In short, the derived loss function reduces to a sum over the potential energy along the path at discretized intervals.

To learn the neural representation, the authors present their key architecture, “AdaPath,” whose primary contribution is that it uses shared atom embeddings. The authors claim that this shared embedding is one of the main reasons for the success of their introduced method. The output of AdaPath is an offset to the linear interpolation connecting the endpoints. In the section introducing AdaPath, the authors discuss an important possible failure point of the method in the form of path stretching and discuss how this should be resolved by the implicit smoothness of neural architectures.

After an in-depth discussion of other approaches to MEP, the authors present their experimental evaluation of their method using the AIB9 and BPTI systems. Compared to the general literature on MEP, these systems are of considerable size. The presented results show that the found MEPs closely align with the FEP landscape in the case of AIB9 and follow the same intermediate states as found in other large-scale studies for BPTI.

### Strengths
With their specific focus on MEP discovery, the authors target an interesting and very important problem in computational biochemistry. While recently there has been significant focus on problems such as equilibrium and transition path sampling for molecular systems, MEP is still somewhat overlooked despite its importance.

The main strengths of the paper are in the scalability of the presented method, as exemplified by the experiments conducted. As stated in the paper, the size of the systems considered is significantly larger than is normally considered for MEP approaches or even related problems such as Transition Path Sampling.

For this reason alone, the paper already makes a good case for publication at ICLR. However, as highlighted below, there remain a few issues and questions that need to be clarified and/or resolved before I would feel fully comfortable voting for acceptance.

### Weaknesses
As discussed above, while I believe the paper has significant contribution and novelty, it currently suffers from two major issues that make me reluctant to vote for acceptance. First, the paper is not sufficiently structured and is hard to follow at times; and second, while the experimental results are impressive, they currently do not sufficiently validate that the main failure mode of path stretching is fully resolved by the smoothness of neural architectures.

Regarding the first point, as it stands, the paper would significantly benefit from more structure and formalism. While it currently contains a long derivation of the proposed loss function in Section 2.1, the lack of clear subheadings and paragraphs makes it hard to follow the flow. Adding some clear definition and theorem environments to this section would also help. The same holds for the section on AdaPath, to highlight when different components/limitations are discussed. Moving the formal overview of established approaches before the derivation of the loss function would also help, as would a short formal definition of the MEP problem at this early stage of the paper. Lastly, the introduction would benefit a lot from some additional clear subheadings, as it is currently very long.

Regarding the path-stretching failure mode, as stated, I am currently insufficiently convinced that this is fully resolved by the implicit smoothness of neural architectures. While the presented experimental results are impressive in terms of scaling, they make it hard to study such core failure modes. To resolve these issues, I believe moving the experimental results based on Alanine Dipeptide to the main body of the paper would greatly help, if accompanied by a further in-depth study of the found transition path. For example, it would be useful to see how the velocity along the path for AdaPath compares to that of traditional methods. If the presented method does indeed not suffer from the path-stretching failure mode, the maximum velocities should be roughly the same.

I believe both of these points, the structure of the paper and the experimental results, are necessary for me to be able to increase my score.

In addition to this, I have a few additional comments below, but I do not deem these as hard requirements for acceptance and/or think they should be relatively easy to address:
- The last sentence of the abstract has a very large claim that is, in my opinion, not sufficiently justified.
- Lines 35–38 are overly simplified and generally do not read well.
- The switch to a discussion about ML-based MEP methods in line 104 is abrupt and generally hard to follow.
- Figure 1 is placed a few pages earlier than it is first referenced. While I understand the placement of the figure here, it would be better to at least have a short reference to the figure on the same page. Aside from this, when referenced, the writing only says “1,” not “Fig. 1” or “Figure 1.”
- It should be made clearer in the experimental section that the results of different spline MEPs, as visualized in Figure 2, are due to the training of different models and not due to variation within the model from different seeds.
- It is unclear if all training runs result in successfully trained models; it would be good to get this clarified in the paper.
- When discussing the computational efficiency of the method (lines 417–422), the training time of each model should be included.

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
2
