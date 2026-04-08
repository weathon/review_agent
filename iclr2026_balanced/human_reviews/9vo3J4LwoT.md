## Human Reviewer 1

### Summary
SubDyve is a virtual screening framework that is label efficient and builds a similarity graph from substructures that are able to discriminate classes. It applies local and dynamic seed refinement based on FDR and performs network propagation to rank candidates. A joint GNN is used to calibrate the propagation. The method tackles two major gaps in NP-based screening by replacing generic fingerprints with mined subgraph patterns. These patterns capture activity relevant signals, while limiting graph topological bias through the usage of LFDR. The proposal also employs ensembles for sake of stabilizing results.

### Strengths
A major contribution of SubDyve is the integration of two complementary ideas: a target-aware sub-graph fingerprint (built from class-discriminating patterns mined from scarce labels) and an LFDR-guided refinement loop that calibrates uncertainty, so that the model is able  to control mFDR and to counteract graph-induced topological bias. Such a design comes from low-label virtual screening and achieves empirical gains considering baselines. The proposal emphasizes that recognizing early, when screening value are more valuable. Further,  the ablations indicate that no component alone achieves the combined performance, demonstrating the benefit of the proposal.

### Weaknesses
A key limitation of SubDyve is regarding its scalability, since each target requires building a separate subgraph fingerprint network. In this case, supervised subgraph mining and graph construction becomes increasingly expensive as the number of targets or compounds grows. The iterative LFDR refinement adds further computational cost at large scale. The method also depends on the availability of homologous protein sequences to obtain annotated compounds for seed generation, which constrains its use to well-characterized targets. These factors limit scalability in large or multi-target screening, and LFDR calibration remains less reliable in intermediate decision regions. The paper should consider these limitations and discuss how they may be effectively addressed.

### Questions
It’s not clear how the setting qualifies as zero shot given that seeds are curated from homologous proteins with high sequence identity. Please clarify.

I believe that adding additional ablation studies would strengthen the work. Evaluating the method with and without the pretrained ChemBERTa features and comparing single versus hybrid ranking would clarify the contribution of each component and better support the
claimed synergy. Please comment.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
Virtual screening (VS) is a key step in drug discovery that aims to identify potentially active compounds from large molecular libraries. In real-world scenarios, only a few active compounds are known, making it a typical low-label setting. Traditional methods often rely on generic molecular fingerprints or deep embeddings, ignoring substructure features critical for activity. Moreover, they usually treat molecules as independent samples, failing to leverage structural relationships among molecules. This paper proposes SubDyve, a subgraph-driven network propagation framework for robust virtual screening under low-label conditions, with effective control of false positives.

### Strengths
1. Introduce supervised subgraph mining into VS graph construction, capturing activity-relevant structural patterns and outperforming generic fingerprints.
2. Proposes an LFDR-based iterative seed updating strategy that expands the seed set while theoretically controlling the FDR upper bound, effectively reducing false positives.
3. With a few dozen seed molecules, SubDyve achieves nice early enrichment performance on both DUD-E and PU datasets, outperforming existing deep learning and fingerprint-based methods.

### Weaknesses
1. Subgraph mining and graph construction must be repeated per target, making cross-target scalability a potential bottleneck for large-scale multi-target screening.
2. The subgraph mining stage relies heavily on the representativeness of the initial seed set. If the seed structures are biased, the graph construction may lack generalizability.
3. All experiments are single-target independent models, with no evaluation on multi-target joint training or transfer learning scenarios.
4. Current baselines are mostly sequence or graph embedding models, with no direct comparison to recent 3D structure-aware models.

### Questions
Refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper addresses the problem of virtual screening (VS) in drug discovery under low-label regimes, where very few active ligands are known. The authors argue that existing network propagation (NP) methods suffer from reliance on generic molecular fingerprints and topological bias, leading to high false positive (FP) rates. The paper proposes SubDyve, a framework featuring two main components: 1) Constructing a task-specific similarity network by mining "class-discriminative subgraphs"; 2) Introducing a "dynamic seed refinement" mechanism that uses the local false discovery rate (LFDR) to iteratively add high-confidence (low FP probability) unlabeled molecules as new seeds, aiming to control FPs. Experiments on the DUD-E benchmark (zero-shot) and a large-scale ZINC dataset for CDK7 (PU learning) show that SubDyve outperforms baselines on early enrichment metrics.

### Strengths
1. The paper focuses on the low-label/few-shot VS problem, which is highly relevant and challenging in real-world drug discovery.
2. The primary highlight is the introduction of LFDR for a principled control of the dynamic seed set expansion to combat topological bias. This is more sophisticated than using fixed similarity or GNN score thresholds.
3. Evaluating the method on two challenging scenarios (zero-shot and large-scale PU) is a clear strength.
4. The results demonstrate a clear improvement in early enrichment metrics compared to the baseline methods.

### Weaknesses
1. The most significant concern is the method's scalability and preprocessing overhead. As acknowledged in the limitations (Section H), the method requires re-running the computationally intensive subgraph mining (SSM) and graph construction process for *every single* protein target. This could be a major practical barrier in a real-world drug discovery pipeline that needs to screen hundreds or thousands of targets.
2. The success of the entire pipeline (especially the SSM step) is highly dependent on the quality and quantity of the initial seed set $\mathcal{S}_{train}$. If the initial set is extremely small (e.g., < 10-20 molecules) or not representative, the subgraph mining step will likely fail or produce misleading patterns, causing the entire downstream pipeline (GNN training and LFDR refinement) to perform poorly. The 50 seeds used as the minimum in the experiment (Sec 4.2.2) is still a somewhat optimistic setting for a "low-label" scenario.

### Questions
1. Can the authors provide an end-to-end average time (e.g., on the DUD-E targets) required to build the subgraph network (including subgraph mining and similarity computation)? How does this compare to the preprocessing time of baselines (like RDKit+NP)?
2. How sensitive is the method to the size of  $S_{train} $? If $S_{train} $ contained only 10 or 5 molecules, would the SSM step still be effective?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces SubDyve, a subgraph-aware network propagation framework for label-efficient virtual screening in drug discovery. It combines subgraph fingerprint network construction, dynamic seed refinement using local false discovery rate (LFDR) estimation and a GNN-based model for stratified refinement. They evaluate on on DUD-E (10 targets) and a CDK7 target with a 10M-compound ZINC dataset, showing large performance gains over fingerprint-based and deep learning baselines.

### Strengths
1. Paper is well-written and structured. Also, URL containing code and experiments is provided for reproducibility.

2. The presented framework seems novel and efficient. For instance, the introduction of class-discriminative subgraph fingerprints captures fine-grained structural patterns that general-purpose fingerprints miss.

3. Comprehensive benchmarks demonstrate significant improvements across several evaluation metrics, such as outperforming large foundation models while using less data and compute.

4. They also provide beyond-performance analysis through case studies (Figure 2 and 3), where they show that subgraph-level information correlates with biologically meaningful substructures.

### Weaknesses
1. While substructure interpretability is discussed, there is little experimental validation linking retrieved subgraphs to real binding mechanisms.

2. The complexity of the presented pipeline (subgraph mining, LFDR estimation and GNN optimization) could make results to be very sensitive to changes in seeds, thresholds and hyperparameters. For instance, the effect of LFDR thresholds could be explored more systematically to assess stability and sensitivity. This can be specifically important for interpretability. 

3. Results are primarily shown on small-molecule benchmarks; applicability to structure-based or protein–ligand joint settings is unclear.

### Questions
1. How sensitive is SubDyve to the LFDR threshold and the number of iterations in seed refinement?

2. How are the mined subgraph patterns evaluated or visualized in terms of chemical or pharmacophoric relevance?

3. Could you check if the provided URL for the code is working? I can't access it.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2