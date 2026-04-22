# MiAD: Mirage Atom Diffusion for De Novo Crystal Generation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
In recent years, diffusion-based models have demonstrated exceptional performance in searching for simultaneously stable, unique, and novel (S.U.N.) crystalline materials. However, most of these models don't have the ability to change the number of atoms in the crystal during the generation process, which limits the variability of model sampling trajectories. In this paper, we demonstrate the severity of this restriction and introduce a simple yet powerful technique, mirage infusion, which enables diffusion models to change the state of the atoms that make up the crystal from existent to non-existent (mirage) and vice versa. We show that this technique improves model quality by up to $\times2.5$ compared to the same model without this modification. The resulting model, Mirage Atom Diffusion (MiAD), is an equivariant joint diffusion model for de novo crystal generation that is capable of altering the number of atoms during the generation process. MiAD achieves an $8.2\%$ S.U.N. rate on the MP-20 dataset, which substantially exceeds existing state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes adding mirage (non-existent) atoms to crystal systems to enable diffusion models to modify the number of atoms during inference, thus removing the restriction to pre-specify the number of atoms for crystal generation. The method, MiAD, generates significantly more stable, unique, and novel (S.U.N.) crystals than the baselines.

### Strengths
- The paper is well-structured and written.
- The proposed idea is simple with minor modifications to existing diffusion model training pipelines. 
- The method notably achieves higher S.U.N. metrics for crystal generation than the compared baselines.

### Weaknesses
I primarily have concerns about:
- lack of some appropriate experiments or discussions about design choices
- lack of conceptual discussions about the MiAD method

These are supported by the questions below. I am willing to improve the score if these are adequately addressed during the discussion period.

### Questions
- What is the appropriate method to select $N_m$? For MP-20, S.U.N. performance degrades (and then somewhat increases?) when $m$ exceeds 25 (if finer choices are used, performance could be better at other values as well, and since the method is quite sensitive to this hyperparameter, this is a concern). Is there any explanation for this? Is there a general rule of thumb about how much $m$ should be, if the maximum number of atoms in the training dataset is $N_d$?
- Can you provide the results of MiAD on datasets with a larger number of atoms, such as MPTS-52? This would show if mirage infusion and reduction also improve performance in such crystals.
- Conceptually, with the addition of mirage atoms, the distribution shifts from the original training distribution (and potentially changes the symmetry within the crystal). Does the model have to learn two symmetries simultaneously? Is there a study on the distribution of space groups before and after mirage infusion, along with the generated crystals before and after reduction?
- It would be great to add more discussion on the benefits of your method compared to the first subgroup of methods (under models that change the number of atoms) to highlight the importance of the proposed method in developing models that can change the number of atoms during inference. Furthermore, the result tables lack these baselines, which are more relevant for comparison with MiAD.
- The mirage infusion and reduction are implemented only with DiffCSP. Results with additional models that require no special modification (e.g., as mentioned, FlowMM, CrystalFlow, and MatterGen) will demonstrate the method's adaptability and whether performance improvements are agnostic to the underlying model.
- What is the quantitative estimate of the increase in training and inference budget (time, compute, memory) with the mirage atoms (since it is mentioned that the model incurs higher computational costs)?

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors address a key limitation in crystal material generation, where most existing diffusion-based models are constrained to a fixed number of atoms during the generation process. To overcome this, they propose MiAD (Mirage Atom Diffusion), a novel generative framework that extends DiffCSP by allowing the number of atoms to vary dynamically as the crystal structure evolves. The central idea, called “mirage infusion”, introduces a special placeholder atom type (type 0), referred to as a mirage atom, which can either appear or disappear during generation. This mechanism effectively enables the model to add or remove atoms adaptively, thereby expanding the diversity and flexibility of generated structures. Experimental results on the MP-20 dataset show that MiAD achieves a S.U.N. rate of 8.2%, outperforming leading baselines such as ADiT, WyFormer, and MatterGen, and demonstrating substantial improvements in both generative quality and material discovery potential.

### Strengths
- The paper is very well written. The limitation of fixed atom counts in current crystal diffusion models is well identified and practically significant for de novo materials discovery.
- The idea of mirage infusion technique is conceptually simple yet effective, implemented by augmenting the atom-type diffusion process with an additional “mirage” type and masking loss terms appropriately. It Gives the flexibility to the models to vary the number of atoms in a crystal during the generation process.
- Results are compared across several leading baselines (DiffCSP, FlowMM, ADiT, WyFormer, etc.) with DFT-based and MLIP-based stability evaluations. Also, the computational and structural validity results are produced in the appendix.

### Weaknesses
- The paper lacks methodological novelty. MiAD’s architecture remains largely identical to DiffCSP, with the only change being the addition of mirage atoms. While this tweak is clever, it is incremental rather than fundamentally new.
- The paper does not convincingly justify why varying atom numbers is scientifically important beyond improving diversity. For instance, how does this help in discovering more stable or experimentally realizable materials? 
- The paper could benefit from qualitative visualizations showing how mirage atoms evolve during diffusion, and examples of successful vs failed generations.
- The authors acknowledge higher computational overhead due to an increased average atom count, but there is no quantitative analysis or efficiency comparison. 
- Scalability on larger datasets (e.g., MPTS-52) remains unclear. It is not evident how computationally efficient the proposed approach is when incorporating these additional “mirage” atoms at larger scales. A detailed analysis of the computational overhead and efficiency on such datasets would strengthen the work.
- For the S.U.N. comparisons, several important baseline models—such as UniMat[1], TGDMat[2], CrysBFN[3], and Crystal-Text-LLM[4]—are not included. Additionally, key recent baselines like DiffCSP++ and SymmCD are missing from Table 1, while FlowLLM and WyFormer are absent from Table 2. The omission of these models leads to incomplete and inconsistent comparisons, making it difficult to accurately assess the relative performance and claimed improvements of the proposed approach.

[1] Yang, Sherry, et al. "Scalable diffusion for materials generation." arXiv preprint arXiv:2311.09235 (2023).

[2] Das, Kishalay, et al. "Periodic materials generation using text-guided joint diffusion model." ICLR 2025.

[3] Wu, Hanlin, et al. "A periodic Bayesian flow for material generation." ICLR 2025.

[4] Gruver, Nate, et al. "Fine-tuned language models generate stable inorganic materials as text." 2024.

### Questions
- Since MiAD’s architecture closely follows DiffCSP, apart from diffusion models have you tested it on other frameworks like LLMs or Flow-based models? Do that show similar performance improvement?
- The paper mentions that varying the number of atoms enhances generative diversity, but could the authors elaborate on its scientific relevance? Specifically, how does this ability contribute to discovering more stable or experimentally realizable materials?
- Can the authors provide qualitative visualizations or case studies showing how mirage atoms evolve during diffusion? For instance, examples of successful and failed generations would help illustrate the behavior and impact of the proposed mechanism.
- Could they provide quantitative comparisons (e.g., training time, memory usage) to clarify the extent of this overhead relative to DiffCSP?
- How scalable is MiAD to larger and more complex datasets such as MPTS-52?
- When incorporating mirage atoms at larger scales, how computationally efficient is the model, and are any optimization strategies employed to maintain tractable training and inference?

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
The paper focuses on de novo generation of crystalline materials and proposes to use mirage atoms (also called fake or virtual atoms in the literature), i.e. atoms with no type, to increase the model flexibility by allowing the number of atoms to vary during generation. The authors demonstrate that this simple technique leads to better S.U.N rate compared to other de novo generation models.

### Strengths
The paper is well written and the idea is simple yet powerful, resulting in a significant improvement of S.U.N rate in the DNG task by keeping the same backbone architecture as some of the baseline model. Additionally, the authors provide ablation studies for all key design choices, with detailed results reported in the appendix. The evaluation of the generated crystalline material samples is conducted both by DFT calculation and machine-learning based interatomic potentials.

### Weaknesses
I have the feeling that the contribution may be somewhat limited, even though it appears to provide a benefit in terms of the S.U.N. rate. In addition to [1] there is another concurrent work [2] (also out in August)  that uses virtual or fake nodes for molecules to allow for variable-sized output. In this paper, the idea is applied to materials, although it is not material-specific and could, in principle, be extended to any graph generation problem. As noted in the paper, the authors mask the mirage atom from the loss computation for the fractional coordinates, which they found to beneficial for material generation. It would have been interesting to see whether the same effect occurs in molecular generation. Additionally, the computation of metrics on the generated samples could be explained in greater detail (see Question)

**References**

[1] "Multi-domain Distribution Learning for De Novo Drug Design", Schneuing et al, 2025

[2] "FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation", Ian Dunn and David Koes, 2025

### Questions
- Why not test the way you used fake nodes for crystals on molecules also? 
- I am somewhat unsure whether the comparison presented is truly apples-to-apples. All the baseline models sample according to the empirical distribution of the dataset (maximum 20 atoms), while your approach samples up to 25 atoms due to the mirage atoms approach. A potentially a more fair comparison might involve using the empirical distribution generated by your model for the fixed-size baselines (Figure 4 you are presenting in Appendix). Related to the metrics computation, did you exclude generated samples that contain more than 20 atoms in the case of MP-20? Are these generated crystals automatically counted as novel in the evaluation? Since the fixed-size models are limited to the empirical distribution of MP-20, they cannot generate crystals with more than 20 atoms and therefore cannot outperform the proposed method.
- In the Appendix, Table 5 mentions that you attempt to position the mirage atoms at the geometric center of the real atoms. How is this computed, given that the atoms lives in a periodic space? Which geometric mean have you considered?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a material representation that allows the diffusion process to vary the number of atoms during the sampling process.

### Strengths
Strengths:
 - The paper is well written and easy to understand. Related works have been discussed properly.
 - The method is generalizable and can be applied to most pre-existing works without much hassle.
 - Results show the improvement in SUN metrics, although the unique and novel metrics are not better than baselines like FlowMM, DiffCSP and WyFormer.

### Weaknesses
Weaknesses:
 - This work only introduces a representation for a material that allows the number of atoms to be flexible. No new model architecture or algorithmic variation has been proposed, which limits the novelty of this work.
 - Datasets with bigger and more complicated structures should have been used to highlight the advantages of this approach. Please also include the results of MPTS-52 dataset in the table.
 - As noted in the ablation studies, the method is sensitive to hyperparameters. like the loss scaling and maximum number of atoms in the augmented atom, which means that for different datasets, these parameters will need to be tuned every time.
 - D3PM introduced several types of transition matrices in their work - uniform, gaussian, mask state. The authors have used the uniform transition matrix for mirage diffusion, but have they tried any other types of transitions? How would the model perform if the mirage atoms are treated as masked states in D3PM?
 - There are some missing citations for the works mentioned in the appendix like CrysBFN and TGDMat.
 - Typos and mistakes - line 292 ("stucture" should be "structure"), line 342 ("does not necessary mean" should be "does not necessarily mean")

### Questions
Text Guided Diffusion Models need to be discussed.  Please comment on that.

### Soundness
2

### Presentation
3

### Contribution
2
