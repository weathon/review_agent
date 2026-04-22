# Latent Diffusion-based 3D Molecular Recovery from Vibrational Spectra

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Infrared (IR) spectroscopy, a type of vibrational spectroscopy, is widely used for molecular structure determination and provides critical structural information for chemists. 
However, existing approaches for recovering molecular structures from IR spectra typically rely on one-dimensional SMILES strings or two-dimensional molecular graphs, which fail to capture the intricate relationship between spectral features and three-dimensional molecular geometry.
Recent advances in diffusion models have greatly enhanced the ability to generate molecular structures in 3D space. 
Yet, no existing model has explored the distribution of 3D molecular geometries corresponding to a single IR spectrum.
In this work, we introduce *IR-GeoDiff*, a latent diffusion model that integrates IR spectral information into both node and edge representations of molecular structures. We evaluate IR-GeoDiff from both spectral and structural perspectives, demonstrating its ability to recover the molecular distribution corresponding to a given IR spectrum. 
Furthermore, an attention-based analysis reveals that the model’s interpretation of IR spectra aligns with quantum mechanical principles of molecular vibrations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors aim to reconstruct three-dimensional molecular geometries from infrared (IR) spectra. Unlike prior work that maps spectra to 1D SMILES or 2D graphs, IR-GeoDiff seeks to recover full 3D atomic coordinates. Their method conditions a geometric latent diffusion model on spectral information using a classifier. The model integrates IR features into both node (atomic) and edge (bond) representations through cross-attention with spectral patches and functional group features. The paper evaluates the approach using the Spectral Information Similarity and chemical graph similarity on the QM9S dataset. The authors also find that spectral peaks receiving the highest attention frequently correspond to functional groups.

### Strengths
- To the best of my knowledge, this is the first paper that attempts to deduce 3D structure from IR
- Significant performance gains over baseline 3D generative diffusion models (EDM, GEOLDM)

### Weaknesses
- Method is essentially a recombination of known techniques
- Assumes that the atom types and the atom count are known beforehand
- Metrics seem problematic

### Questions
- How reasonable is it to assume that only a single 3D structure generated an IR spectra? In an experiment there is surely a distribution over structures in a sample, is it not?
- You mention that number of cases where molecules exhibit high graph similarity but low SIS, and vice versa. At the same time, only ~200 test samples were used for spectral metrics due to computational cost of SIS. What would be better metrics?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work appears to be primarily a benchmark study. Building upon previous molecular generation approaches that failed to capture spectral features, the authors aim to address the problem of modeling the distribution of 3D molecular geometries corresponding to a single IR spectrum, under the assumption that atom types and counts are fixed. In the implementation, they construct a corresponding dataset based on QM9 and develop a framework that integrates techniques such as classifier guidance, latent diffusion models (LDM), and GeoLDM to achieve this goal.

### Strengths
1. The task is novel and introduces a fresh perspective to the field.
2. The *Background* and *Related Works* sections are well-organized and provide a thorough summary, which is appropriate for a benchmark-focused study.
3. The overall framework maintains good SE(3)-equivariance properties throughout the model design.
4. The integration of spectral features is reasonable and well-supported by the ablation studies.

### Weaknesses
1. The paper makes two main claims regarding the task: (1) it aims to model the distribution of 3D molecular geometries corresponding to a single IR spectrum, as stated in the *Abstract* and *Introduction*; and (2) it seeks to learn a probabilistic model $\theta$ that captures the conditional distribution of molecular geometries given an IR spectrum, i.e., $p_{\theta}(G|S)$, as described in the *Preliminaries* section. However, I question the accuracy of this task formulation, since in the actual implementation the authors assume that the atom types $h$ and the atom count $N$ of each molecule are known, and focus solely on modeling the conditional distribution over atomic coordinates $x$. Therefore, the model effectively learns $p_{\theta}(x|S,h)$ rather than $p_{\theta}(G|S)$, and the paper should have explicitly clarified at the outset what the true modeling objective of the task is.  
2. The benchmark dataset is undersized: it relies solely on QM9, which includes molecules with at most nine heavy atoms. Contemporary 3D molecular tasks typically involve much larger molecules—e.g., those in GEOM-Drugs or PCQM4Mv2 average tens to hundreds of heavy atoms—so larger, multiscale benchmarks would be more representative and comprehensive.  
3. The overall framework is primarily built upon several well-established methods, including classifier guidance, latent diffusion models (LDM), and GEOLDM. Although this design is adequate for a benchmark-oriented study, it offers limited novelty beyond existing approaches.  
4. The baselines used in this work are relatively outdated and limited in number. In the field of molecular generation, many newer and more powerful methods beyond EDM and GeoLDM have been proposed, which could be adapted for comparison. At the very least, some of these recent approaches should have been considered to provide a more comprehensive evaluation.  
5. The results and implementation of EDM and GeoLDM appear questionable and inconsistent. In your setting, both the atom types and atom counts are provided as part of the input, whereas for EDM and GeoLDM, only the number of atoms is fixed and the atom types are omitted—a significant loss of information. This discrepancy raises concerns about the fairness of the comparison. Please refer to **Question 2** for further discussion.  
6. There are no formulations provided for any of the evaluation metrics in the manuscript.

### Questions
1. The manuscript states that *“in the subsequent diffusion training stage, the spectral classifier is frozen to ensure a stable and consistent conditioning signal, while the autoencoder remains learnable.”* Why does the autoencoder remain learnable during the diffusion training stage? Is there any ablation study or experimental evidence supporting this design choice?
2. The manuscript claims that *“denoising networks jointly model position- and feature-level noise, which precludes specifying atom types as inputs for EDM and GeoLDM.”* Why is this the case? The paper provides neither a theoretical formulation nor an architectural illustration to justify this restriction.
3. In the definition of *molecular accuracy*, it is stated that *“if at least one sampled molecule exactly matches the reference structure.”* How is “matches” defined in this context? Does it refer to exact coordinate alignment, atom-type correspondence, or another structural similarity measure?
4. During evaluation, is there any metric that measures spatial deviation—such as the root mean square deviation (RMSD)—between the generated positions $ x^{\prime}$ and the reference positions $ x $?

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
4

### Summary
This paper proposes IR-GeoDiff, a latent diffusion model designed to reconstruct three-dimensional molecular geometries directly from one-dimensional infrared spectra. Existing methods, which typically generate one-dimensional SMILES strings or two-dimensional molecular graphs, fail to capture the intrinsic relationship between infrared spectra and molecular geometry. IR-GeoDiff addresses this limitation through two key innovations: (1) It employs a multi-head cross-attention mechanism to integrate infrared spectral features into the node (atom) and edge (bond) features of the molecular geometry; (2) It utilizes a Transformer-based spectral classifier to simultaneously extract spectral and functional group features, which serve as conditional inputs for the generation process.
The model was evaluated based on both spectral and structural similarity metrics to assess the consistency between the generated geometries and their corresponding infrared spectra. Results demonstrate that IR-GeoDiff outperforms baseline models such as EDM and GEOLDM on the QM9S dataset. Furthermore, attention visualization analysis confirms that the model's interpretation of the spectra aligns with the principles of molecular vibration derived from quantum chemical calculations.

### Strengths
1. The work is innovative in its application of a latent diffusion model to the challenging task of recovering molecular geometry from infrared spectra. The model architecture is novel, particularly in its use of a multi-head cross-attention mechanism for feature fusion and a Transformer-based classifier for joint feature extraction.
2. The model design is sound and the experimental evaluation is comprehensive, encompassing both spectral and structural dimensions. The inclusion of attention visualization provides valuable validation for the model's interpretability.
3. The paper is well-structured, with detailed method descriptions. The inclusion of diagrams and formulas aids in comprehension.
4. This research addresses a significant gap by enabling the direct recovery of 3D structures from infrared spectra. It holds considerable potential for applications in computational chemistry and drug design.

### Weaknesses
1. The experimental setup has limitations. The chosen baseline models, EDM and GEOLDM, were proposed approximately three years ago. Given the rapid pace of development in this field, more recent and potentially superior models should be included for comparison.The speed of diffusion models is too slow, and in recent years, many new models have been developed. These new models should be compared.
2. The study relies on a single dataset, QM9S, for validation. This dataset is limited to small molecules (with a maximum of nine heavy atoms), which restricts the complexity and diversity of the tested structures and may not adequately demonstrate the model's generalizability.
3. The conclusion is underdeveloped. It primarily reiterates the main contributions and findings without a critical discussion of the study's limitations or a clear outlook on future research directions.
4. The related work section lacks depth. It emphasizes the primary motivation for developing IR-GeoDiff but does not sufficiently elaborate on the core challenges or the specific innovations designed to overcome them, such as the multi-head cross-attention mechanism for fusing spectral and geometric features.
5. There are inconsistencies between the manuscript text and the mathematical notation. For instance, the variable   in Equation 6 does not match the   in line 224 of the text.

### Questions
1. The "Experiments" section should be strengthened by benchmarking IR-GeoDiff against more recent state-of-the-art models to compellingly demonstrate its superior performance.The speed of diffusion models is too slow, and in recent years, many new models have been developed. These new models should be compared.
2. Additional experiments on datasets featuring more complex and diverse molecular structures are recommended to thoroughly validate the model's generalization capability.
3. The "Conclusion" should be expanded to explicitly outline the study's limitations and to propose specific, actionable directions for future work.
4. The "Related Work" section should be revised to provide a clearer exposition of the core challenges in this research area and to delineate how IR-GeoDiff's design, particularly its use of cross-attention, addresses these challenges.
5. A thorough review of the entire manuscript is necessary to ensure consistency between all variables in the formulas and the main text.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes IR-GeoDiff, a latent diffusion model designed to recover 3D molecular geometries from infrared (IR) vibrational spectra. The model integrates spectral features into both node and edge representations within an equivariant latent diffusion framework. Experiments on the QM9S dataset demonstrate that IR-GeoDiff can reconstruct 3D structures with given IR spectra, achieving promising results on authors' designed evaluation metrics.

### Strengths
1. The idea of reconstructing 3D molecular geometries from vibrational spectra is scientifically interesting and potentially impactful for computational chemistry and molecular spectroscopy.
2. The paper is overall well written and logically organized. It's easy to follow and recognize the paper's contributions.

### Weaknesses
1. The motivation for adopting a latent diffusion model is not sufficiently convincing. Latent diffusion models rely heavily on strong VAE encoders/decoders to build meaningful latent representations, but such powerful autoencoders are not yet well established for molecular 3D geometry. In contrast, existing non-latent 3D molecular diffusion models already achieve strong and stable results directly in coordinate space. The paper should clearly justify why the latent-space formulation is preferable in this domain.
2. The experiments only compare with EDM and GEOLDM. More recent and stronger baselines, including non-latent 3D molecular diffusion models, should be included for a fair evaluation.
3. The claimed contribution focuses on 3D molecular generation, but the primary structural metric is the Tanimoto similarity between Morgan fingerprints. This metric does not capture 3D conformational or geometric correctness. Evaluations for quality of generated 3D structures should be reported.
4. The method assumes that the atom types and atom count are known a priori. Although the authors justify this as reflecting certain experimental workflows, this assumption greatly limits practical usability. In real scenarios, these properties may be unknown or partially uncertain, making the proposed method inapplicable beyond constrained settings.

### Questions
1. Is the autoencoder module trained by the authors, or reused from prior molecular latent diffusion work (e.g., GEOLDM)? Please clarify its training procedure, data, and reconstruction performance.
2. When computing *molecular accuracy*, how is a correct molecule defined? Is it based on exact molecular graph structure, fingerprint, or SMILES? 
3. Since the paper reports top-$n$ accuracy, what value of $n$ is used, and how sensitive are results to this choice?

### Soundness
2

### Presentation
2

### Contribution
2
