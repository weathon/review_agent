# SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Ensuring synthesizability in generative small molecule design remains a major challenge. While recent developments in synthesizable molecule generation have demonstrated promising results, these efforts have been largely confined to 2D molecular graph representations, limiting the ability to perform geometry-based conditional generation. In this work, we present SYNCOGEN (Synthesizable Co-Generation), a single framework that combines simultaneous masked graph diffusion and flow matching for synthesizable 3D molecule generation. SYNCOGEN samples from the joint distribution of molecular building blocks, chemical reactions, and atomic coordinates. To train the model, we curated SYNSPACE, a dataset containing over 600K synthesis-aware building block graphs and 3.3M conformers. SYNCOGEN achieves state-of-the-art performance in unconditional small molecule graph and conformer generation, and the model delivers competitive performance in zero-shot molecular linker design and pharmacophore conditioning for protein ligand generation in drug discovery. Overall, this multimodal formulation represents a foundation for future applications enabled by non-autoregressive molecular generation, including analog expansion, lead optimization, and direct structure conditioning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SynCoGen, a framework that jointly generates synthesizable reaction graphs and 3D molecular coordinates using masked discrete diffusion and flow matching. The paper also introduces SynSpace, a new dataset with 600k synthesis-aware molecules. Experiments show strong results in unconditional 3D generation, fragment linking, and pharmacophore-conditioned tasks.

### Strengths
1. Novel idea of coupling synthesizability with 3D coordinate generation.
2. The dataset is well-curated and likely useful for future work.
3. Results are convincing and demonstrate clear improvements over baselines.

### Weaknesses
1. There are many node/edge modeling constraints, heavy notation, and complex noise processes that make the method difficult to follow and reproduce. It’s also unclear which components contribute most to the final performance — the paper did not provide sufficient ablation studies or analysis isolating the impact of individual design choices (e.g., compatibility masking, visibility-aware noise, joint time coupling). As a result, the practical necessity of the full system remains uncertain.
2. The reaction vocabulary is fixed and limits diversity. The model relies on a predefined set of 93 building blocks and 19 reaction templates, where each reaction involves at most one leaving group per reagent. While this makes synthesis simulation simpler and chemically reliable, it also restricts the model’s capacity to explore more diverse or complex chemical spaces.

### Questions
1. How scalable is SynCoGen to larger or more diverse reaction vocabularies? Would the framework still function if reactions involved multiple leaving groups or more complex coupling patterns?
2. The paper claims unified discrete–continuous time coupling. However, it seems that diffusion or flow updates on edge types may not take effect until the corresponding nodes are decoded. Did the authors experiment with decoupled or asynchronous schedules across modalities, and if so, how did that affect performance?

I would consider raising my scores if authors could give some resonable explanations to the questions.

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
The authors introduce.  SYNCOGEN, a multimodal generative framework that jointly models 3D molecular geometry and synthetic accessibility through reaction templates. The key idea is to integrate masked graph diffusion and flow matching into a unified generative process that simultaneously samples both the reaction graph over molecular building blocks and the corresponding atomic coordinates. This coupling enables SYNCOGEN to generate 3D molecules that are not only conformationally realistic but also chemically synthesizable via known reaction templates. To support this framework, the authors curate SYNSPACE, a large-scale dataset of over 600K synthesizable molecules and 3.3M associated low-energy conformations, represented as building-block reaction graphs. The model supports conditional generation tasks such as linker design and pharmacophore-guided generation, showcasing its utility in practical drug discovery.

### Strengths
The paper tackles one of the central challenges in molecular generative models i.e. providing accessible synthetic routes for generated candidates. Additionally the proposed method is able to generate favourable conformers.

### Weaknesses
- The writing of the paper is convoluted makes it difficult to follow and appreciate the methodology. It would be beneficial if the authors explain their method more clearly. For example, there are several components in the proposed workflow, a better workflow diagram explaining end-to-end working of SYNCOGEN will be helpful to the readers. 
- for a complicated method as yours, with so many components, a more detailed workflow will he helpful. The message in current fig1 is unclear
- To better understand the method, code could have helped. However the url provided (https://anonymous.4open.science/r/SynCoGen-13F7) does not have the code. Please provide reproducible code. 
- Another major issue I have that authors should benchmark their work against Shen et al (2025). It is the closest baseline to their work however haven't been benchmark against. I understand Shen et al, is conditional SBDD generation but their attempt to generate conformers while simultaneously providing reaction pathway is most similar to this work. In theory, the authors unconditional generation should be able to outperform Shen et al. Please include this experiment in your paper.

### Questions
- Fig 1 caption, what does non-linear node in this context mean?
- why do authors disallow macrocycles?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SYNCOGEN, a generative framework for designing synthesizable small molecules directly in 3D Cartesian space. The core idea is to jointly model molecular building blocks and chemical reactions alongside the continuous 3D atomic coordinates. The method combines a masked graph diffusion model for the reaction graph with a flow matching model for the coordinates, trained on a newly curated dataset called SYNSPACE. The authors demonstrate SYNCOGEN's capabilities through experiments in unconditional generation, where it achieves state-of-the-art performance in both 3D geometry and synthesizability metrics, as well as in conditional tasks like fragment linking and pharmacophore-conditioned generation, where it shows competitive results.

### Strengths
1. The primary contribution is the novel formulation of jointly generating a synthesizability-aware reaction graph and the corresponding 3D molecular conformation. This elegantly bridges two previously disconnected lines of research: 2D synthesizable generation and 3D all-atom generation. If successful, this approach could significantly accelerate drug discovery pipelines by producing spatially-aware designs that are also practically achievable in the lab. 
2. The technical execution is of high quality. The choice to combine masked discrete diffusion for the building block graph and equivariant flow matching for the 3D coordinates is a sophisticated and well-suited generative paradigm for this multimodal task. Furthermore, the curation of the SYNSPACE dataset, which explicitly links reaction graphs with low-energy 3D conformers, is a valuable resource contribution that could enable future research in this direction. 
3. The paper is well-written, clearly structured, and generally easy to follow. The methodology is explained with sufficient detail, and Figure 1 provides an excellent conceptual overview of the generative process. The experimental setup and evaluation metrics are well-defined.

### Weaknesses
1. While the creation of SYNSPACE is a commendable effort, its presentation as a key contribution is not fully supported by an analysis within the paper. The manuscript lacks a characterization of the dataset's properties. To understand the model's behavior and potential biases, it is crucial to see distributions of key molecular properties (e.g., molecular weight, logP, scaffold diversity) and how they compare to standard benchmarks like ZINC or GEOM-Drugs. It is difficult to assess the complexity and chemical scope of the space the model has learned.
2. The de novo generation experiments (Table 1) compare SYNCOGEN only against all-atom 3D generative models. While this is a necessary comparison for evaluating 3D conformer quality, it omits a critical class of baselines: state-of-the-art synthesizable generators that operate on fragments in 2D or 3D.  Does the added complexity of generating 3D coordinates come at the cost of performance on the core task of synthesizable graph generation? This comparison is needed to fully contextualize the paper's contribution.
3. A central motivation of the paper is the advantage of generating 3D coordinates alongside the synthesis plan. However, the experiments do not provide a clear, causal link demonstrating this advantage. The work successfully shows that it can be done, but not why it is better than alternative pipelines. For synthesizability alone, 2D methods suffice. 
4. The evaluation for pharmacophore-conditioned generation is performed on only three target proteins. This small sample size is insufficient to establish statistical significance or to claim general applicability. The results are promising but can be treated as anecdotal case studies rather than a robust validation of the method's capability.
5. The model generates molecules conditioned on a pharmacophore without any information about the target protein pocket. It is therefore surprising that it produces molecules with docking scores superior to the native ligand and other baselines. A deeper analysis is needed to convince the reader that this is not a coincidental finding.

### Questions
1. Could you please add an analysis of the SYNSPACE dataset's chemical property distributions to the appendix? This would help in understanding the chemical space your model operates in.
2. Could you comment on the feasibility of adding an experiment that explicitly highlights the synergy of your joint approach? For example, a task involving conditioning on a 3D property where a 2D-first pipeline would likely fail. 
3. Could you comment on the generalizability of your findings, given the small number of targets?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a multimodal generative framework that jointly samples a reaction graph with masked graph diffusion and 3D atomic coordinates with flow matching under a unified time schedule. To enable training, the authors curate SYNSPACE, comprising synthesis-aware building-block graphs with associated low-energy conformations. Empirically, SYNCOGEN achieves strong validity and retrosynthetic solve rates while producing low-energy, PoseBusters-plausible conformers. It further demonstrates fragment linking and pharmacophore-conditioned generation with competitive docking scores and markedly higher retrosynthesis success than baselines.

### Strengths
- Well-motivated co-generation: Jointly models reaction graphs and 3D coordinates, explicitly aligning “makeability” with structural plausibility in one framework.
- Diversified application scenarios: Demonstrates fragment linking and pharmacophore-guided design, suggesting practical utility beyond unconditional generation.
- Technical contribution: Clear integration of chemistry-aware constraints and templated reaction modeling that improve validity and retrosynthesis success. SYNSPACE is substantial, pairing reaction-level structure with optimized conformers, and includes pharmacophore annotations to support structure-informed tasks.

### Weaknesses
- Fixed reaction vocabulary: Reliance on fixed building blocks and templates restricts chemical diversity and may limit generalization performance.
- Baseline comparability: Many 3D baselines reported are not synthesis-aware. Stronger comparisons with synthesis-aware methods would be more convincing.
- Hard edge constraints can preclude valid chemotypes such as macrocycles.

### Questions
1. SYNCOGEN shows lower diversity and novelty. Is this mainly due to using fixed 93 building blocks and 19 reaction templates? How were these number chosen?
2. Can the model generate molecules with more building blocks than seen in training? How does performance changes for larger N?
3. For a fairer baseline, it would be more convincing to compare with other synthesis-aware models, such as CGFlow.
4. In SAMPLEEDGES, the authors draw parents stochastically. Did the authors test deterministic sampling for example argmax? How does this affect diversity and validity?

### Soundness
3

### Presentation
2

### Contribution
3
