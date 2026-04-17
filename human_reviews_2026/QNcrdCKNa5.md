# TopoScorer: a light, interpretable predictor for protein-protein binding affinity

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Protein-protein binding affinity underlies complex stability, selectivity, and therapeutic action, yet experimental measurement is low-throughput. Although a number of deep learning models are now end-to-end differentiable, they generally lack interpretable attributions, whereas traditional topology-based affinity predictors rely on non-differentiable persistent diagrams or barcodes. We present TopoScorer, a lightweight, interpretable, end-to-end–trainable affinity scorer that can act as a loss or reward to steer generative and discriminative protein models; across protein and mutation affinity benchmarks, it delivers performance comparable to state-of-the-art methods and, when integrated into a modern antibody-design workflow, improves affinity-related metrics of generated candidates. The core component of TopoScorer is Specter(Spectral Topology Encoder), a topology-driven, multi-channel, multi-scale differentiable feature extractor for protein–protein interfaces that converts full-atom coordinates into topo-spectral representations via Persistent Topological Hyperdigraph Laplacians (PTHLs) and differentiable spectral descriptors, preserving physicochemical-role–aware cues alongside 3D topological structure to yield compact, interpretable features suitable for learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TopoScorer, a model designed to predict the binding affinity of protein-protein interactions. The method is evaluated using standard data, and it also includes a case study demonstrating how an antibody generative model can be guided to produce designs with higher affinity. While the method is presented as being lightweight, interpretable, and end-to-end trainable, each of these aspects currently exhibits significant weaknesses.

### Strengths
- The featurization and model architecture are described with great detail.
- The concept of fine-tuning a generative antibody model to enhance affinity is both practically interesting and innovative.

### Weaknesses
**Major**

- Although the featurization and model architecture are well-explained, there is a lack of clarity regarding the training details, loss function, and data. Section 3.3, "BINDING AFFINITY PREDICTION MODEL," only covers the "Model Architecture" paragraph, leaving the training objective and data unclear.
- The proposed method appears to underperform existing approaches. For scoring mutations, Table 1 shows only a slight improvement in Spearman correlation over RDE-Net, while its Pearson correlation is substantially lower. A similar trend is observed for affinity prediction in Table 1 when compared against DSMBind, a fully unsupervised method.
- The evaluation seems to be affected by data leakage, and further analysis is needed. The affinity prediction section (Section 4.1) does not specify how the complexes were split (e.g., sequence or structure similarity). The data for antibody design is split by release time (Section 4.2), which is known to cause data leakage when used as the sole criterion [1].
- The method's key novelties, being "lightweight, interpretable, and end-to-end trainable", have considerable issues:
  - The claim that the model is "lightweight" is not supported. Running time is not analyzed. Section 4.1 only states that TopoScore has  approximately four times fewer parameters than RDE-Network, but it remains unconfirmed whether this translates into substantially lower memory or time consumption.
  - The "interpretability" of the method is only demonstrated through a single example (Figure 2b). It is unclear whether similar interpretability could be achieved by simply calculating the number of bonds rather than relying on topological features (please refer to Question 1).
  - While the Abstract states that "existing deep learning based approaches lack interpretability and a differentiable path from affinity back to the interface," multiple prior methods are indeed end-to-end differentiable, including for example [2], and DDGPred, which is mentioned in Related Work as end-to-end differentiable (line 94).

**Minor**
- The paper does not cite or discuss TopNetTree, a previous method that also uses topological features of protein-protein interfaces to predict affinity [3].
- Some sentences are missing references:
  - [Line 39]. The sentence "Deep learning has become the dominant paradigm for protein–protein binding affinity prediction, delivering state-of-the-art accuracy and throughput" lacks a reference.
  - [Line 45]. The "High training cost" of prior methods is not supported by any reference.
  - [Line 106]. "Reviews" are mentioned, but only one is cited.

References

- [1] Bushuiev et al, 2024, “Revealing data leakage in protein interaction benchmarks”, https://arxiv.org/abs/2404.10457
- [2] Shuai et al, 2025, “Sidechain conditioning and modeling for full-atom protein sequence design with FAMPNN”, https://www.biorxiv.org/content/10.1101/2025.02.13.637498v1.full.pdf
- [3] Want et al, 2020, “A topology-based network tree for the prediction of protein–protein binding affinity changes following mutation”, https://www.nature.com/articles/s42256-020-0149-6

### Questions
1. What would be the outcome if Figures 2e, f, and g were replaced with simple bar plots showing the counts of respective bonds? For instance, instead of Figure 2e, if there were simply three numbers representing carboxylate O - carboxylate O bonds for each of the three structures, would the same trend still be observable?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose TopoScorer a binding affinity prediction with a focus on interpretability. They designed Specter which is a differentiable feature extractor for encoding protein protein interface encoding for both structural and chemical information.  They showcase an differentiable deep learning based model that can steer a generative antibody design model.

The model architecture is formed based on hyper graph induced cross protein distances to encode the heavy atoms into physicochemical role aware classes within the binding interface. They also use soft filtration to persistent topological hyperdigraph laplacians and then summarize the topology with a six tuple of differentiable spectral statistics of eigen values. They differentiate the eigen values with a fallback schedule to handle ill conditioned classes. Multi channel encoding is used to encode the the interface as a multi channel graph from role aware atom types. They map Atom37 names to 11 chemical role aware classes to differentiate between backbone donors/acceptors, aromatic carbon, sulfur atoms etc. They employ transformer with multi head self attention to mix information across channels, scales and interface regions. The different heads are expected to understand hydrophobic, polar, long range scales etc.

### Strengths
For the sequence and structure co design model the fine tuning helps as shown in table 2. Addition of the proposed method helps steer optimization towards more plausible interface specially at the H3 region. It is great that the authors do a lot of ablation on the single channel vs multi channel to explain the interpretability.

### Weaknesses
From the results (table 1) in the affinity prediction task top scorer outperforms other baselines on spearman. For multiple mutations it is the same pattern. It is hard to evaluate and compare the model performances if the pearson and spearman results are not consistent although authors claim that ranking based metrics are better for the task. 

In addition the authors do not compare their model to other models notably GearBind which is also an all atom based graph model. For predicting affinity it is important to compare to the surface and structure based models such as AtomSurf as well. 
On PDBBind data (Figure 2h) the correlation is 0.298. It is hard to say how good is the score when the proposed method is the only one and the other correlation values (Figure 2i) shows the ablation of the TopoScorer method. 

Interpretability analysis is an important area of research and often most methods do not focus on it but on the benchmark tasks itself the model seems to underperform compared to the other methods (Table 1 PearsonR shows DSMBind is best, DDGPred is best on single mutation task as shown in PearsonR).

### Questions
Have the authors considered other methods such as SurfPro https://arxiv.org/pdf/2405.06693 which is a surface based models on the protein design tasks and compared the results? It will make the approach more robust if other SOTA surface aware methods are compared.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TopoScorer, a lightweight, differentiable scorer for protein–protein binding affinity. It builds Specter, a topo-spectral feature extractor that computes soft, differentiable spectral statistics of PTHL spectra across radii, and pairs this with a small ttransformer to predict affinity from multi-channel, physicochemical role–aware interface graphs. On PPB-Affinity and SKEMPI-2.0, the method achieves strong Spearman correlations, and when used as a frozen reward it improves an antibody co-design model on DockQ in a held-out set.

### Strengths
- Clean, coherent idea in a multi-scale, multi-channel topo-spectral features with a compact Transformer. Also it should be easy to plug into design loops.
- Differentiable topology done carefully and in a specialized knowgledge domain.Soft zero-counts, log-sum-exp min/max, Huberized std. Also, eigenvalue-only backprop with stability guarantees.
- Competitive ranking performance: best or near-best Spearman on PPB-Affinity and SKEMPI subsets. Clear reporting against physics and learned baselines.
- As a frozen reward, improves DockQ and SR in IgGM finetuning on a post-2013 SAbDab split.

### Weaknesses
- Generalization controls: time-based PPB-Affinity test is good, but no explicit homology/interface-similarity controls are reported (e.g., sequence-identity thresholds at contacting residues, SCOPe/CATH or interface clustering). This weakens cold-family claims.
- Robustness to structural noise: mutants come from FoldX; there’s no stress-test for AF vs crystal, side-chain repacking choices, or coordinate jitter/protonation.
- Channel/radius attributions are plausible, but there’s no deletion/counterfactual test showing predicted changes track

### Questions
- How do you prevent train–test leakage at the interface level (e.g., sequence identity on contacting residues, SCOPe/CATH families, interface RMSD clustering)?
- How sensitive are results to structure sources (AF vs crystal), side-chain packers, coordinate jitter, and protonation?
- In IgGM finetuning, how is the TopoScorer reward normalized across targets/sizes to avoid scale bias? Did you test reward-weight sweeps or label-shuffle controls?
- What radii set do you use by default, and how often do eigenvalue fallback/stability tricks trigger in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes a parameter-efficient protein-protein interaction (PPI) affinity prediction model, TopoScorer, based on a topological features of PPI complex structures. Specifically, the structure is represented by a multi-channel graph with role-aware atom types and physiochemical properties. The core component is a feature extractor named Specter, which employs a differentiable distance filter to obtain the multi-scale Hodge Laplacians which are then used to compute spectral features from the eigenvalues. The binding affinity predictor is then trained on the spectral features. Experiments on affinity and mutation impact prediction shows comparable performance with state-of-the-art methods.

Moreover, the predictor is used in the fine-tuning of a language model for antibody design, achieving improved performance than the larger RDE-Network. Interpretation analysis show the importance of interface connectivity features and side chain structures.

### Strengths
- The formulation and rationale of the proposed methods are clearly and soundly presented. The use of spectral features offer a rather novel insight into the

- The application of the predictor as fine-tuning for generative models proves good generalizability and meaningful representations.

- The biological relevance of the learned model patterns is demonstrated through interpretability analysis.

### Weaknesses
- The predictive performance is not significantly improved compared to the baselines, 

- Because of this, the major advantage of the proposed model may lie in its parameter efficiency and adaptability as guidance methods. However, this part needs some additional demonstration. See Questions.

### Questions
The work overall looks promising, but some important application aspects need to be addressed and I'm willing to raise the scoring with sufficient evidences:

- How does the predictive performance compare to other PTHL-based models such as topoformer?

- A more detailed comparison of the parameter and time efficiency between models would be appreciated.

-  Besides comparing with a totally different generative framework, how does classifier guidance with TopoScorer compare to other prediction or scoring models (eg compared in Table 1) as classifier guidance on the same generative model, in terms of performance, parameter size and running time?

- What is the side chain packing method used in the fine-tuning? Also, as side chain packing algorithms can take long to run, how does affect the running time of the fine-tuning loop?

- 4.3 and Fig 2: some additional justifications of the analysis are needed: how does the pattern of $\lambda_{sum}$ indicate the physiochemical properties of the amino acids and atoms *a priori*?

### Soundness
3

### Presentation
4

### Contribution
3
