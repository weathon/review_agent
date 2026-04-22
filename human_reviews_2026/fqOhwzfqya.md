# GraphIP–Bench: A Benchmark for Extraction Attacks and Defenses in Graph Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Graph machine learning models support applications in recommendation, finance and biomedicine, yet their parameters and training data are proprietary assets that face threats such as model extraction, inversion and membership inference. Prior work proposes many attacks and defenses, but comparisons remain unreliable because experiments use inconsistent datasets, threat models and evaluation metrics. We introduce \emph{GraphIP-Bench}, a unified benchmark and library that provides standardized datasets, reference implementations of diverse attack and defense families and a common evaluation protocol. The benchmark specifies clear metrics for extraction fidelity, task utility and computational cost, which together measure the protection–utility trade-off under a prescribed query-access threat model. Empirical analysis across citation, coauthor and commercial graphs shows that several watermarking methods preserve utility and enable ownership verification with different cost profiles, while data-free extraction lags behind data-driven extraction even at large budgets. To the best of our knowledge, this is the first benchmark that standardizes the rigorous evaluation of model-extraction attacks and defenses for graph neural networks. Our implementation is publicly available at: \href{https://anonymous.4open.science/r/GraphIPBench-7F7F}{https://anonymous.4open.science/r/GraphIPBench-7F7F}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
GraphIP-Bench is the first unified benchmark for evaluating model extraction attacks and defenses in GNNs, addressing the lack of standardized experimental protocols in prior research. It integrates diverse extraction attacks, defense families, and multiple public datasets with standardized query budgets, threat models, and metrics (fidelity, utility, and computational cost). The empirical results comprehensively compare the strengths and weaknesses of various models. The benchmark provides a reproducible infrastructure and a clear characterization of trade-offs to guide practical deployment.

### Strengths
1.Fixes public data splits, query sets, and budgets, enabling fair and standardized comparison across different methods.

2.Covers diverse attack/defense paradigms, data availability settings, and evaluation dimensions (security, utility, and efficiency).

3.Releases reference implementations, fixed seeds, and unified hardware/software configurations. It also reports computational cost and protection–utility trade-offs, facilitating real-world adoption.

### Weaknesses
1.The benchmark focuses mainly on ownership tracing and information-limiting defenses; it could be expanded to include other defense types, such as differential privacy.

2.As the authors mentioned the use of molecular property prediction in the main text, it would be beneficial to include more realistic and practically relevant tasks—the current datasets are largely academic citation networks, which are publicly available and thus do not realistically require extraction protection.

3.It might be valuable to include heterogeneous graphs, as this could reveal how structural diversity affects extraction and defense performance.

### Questions
See weakness.

### Soundness
3

### Presentation
3

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
This paper provides a systematic benchmark of model extraction attacks and defenses for graph neural networks (GNNs). It evaluates various methods across seven public datasets and releases corresponding source code. The overall experimental trends are consistent with prior studies, making it a relatively eproducible benchmark work.

### Strengths
1. Focuses on black-box scenarios, covering seven datasets, multiple attack variants, and five watermark-based defense methods, offering broad coverage.
2. The evaluation metrics are well-designed, balancing model fidelity, task performance, and computational efficiency.
3. The paper is clearly written and well-organized, with strong reproducibility.

### Weaknesses
1. The benchmark only considers black-box settings; for completeness, gray-box and white-box threat models should also be incorporated.
2. Lacks comparison with existing benchmarks in terms of scope and coverage.
3. Provides little discussion on the novelty and core ideas of the included methods, focusing mainly on empirical results.
4. The evaluation is limited to node classification, while link prediction and other tasks are not explored.

### Questions
1. Can the implemented methods be extended to link prediction or other graph tasks? It is recommended to include such experiments.
2. Future versions should consider introducing white-box and gray-box threat models to achieve a more comprehensive threat landscape.
3. Current experiments are conducted on small-scale datasets; it is advisable to include large-scale graphs (with millions or billions of nodes) to test scalability and practical applicability.

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
This paper introduces GraphIP-Bench, a unified benchmark for systematically evaluating model extraction attacks and defenses in graph neural networks. The benchmark standardizes datasets, query budgets, threat models, and evaluation metrics to enable reproducible and fair comparison across methods. The authors implement nine representative extraction attacks and six defense families under a single black-box protocol, measuring performance in terms of security, utility, and computational efficiency. By releasing a reproducible and extensible platform, this work provides a foundation for evaluating model extraction attacks and offers valuable insights for both research and industrial deployment.

### Strengths
1. The paper defines a rigorous experimental protocol with fixed data splits, query budgets, and endpoint assumptions, ensuring fair and reproducible comparison across methods.

2. The benchmark covers nine extraction and six defense families across seven public datasets, providing a broad and representative empirical basis.

3. The paper offers insights into the relative effectiveness and cost of different methods, highlighting sample efficiency, data-free limitations, and defense practicality with visualizations and quantitative metrics.

### Weaknesses
1. The benchmark does not include several representative and recently published model extraction works such as GNNStealing[1], STEALGNN[2], which limits the completeness of the comparative evaluation.

2. In the data-free setting, the benchmark relies on randomly generated graphs for querying and extraction. This simplification reduces the credibility of the conclusions, as the tested scenario does not accurately represent the state-of-the-art approaches.

3. While the paper thoroughly reports quantitative results across multiple datasets and methods, it primarily focuses on summarizing empirical outcomes without providing deeper analysis or interpretation of the underlying mechanisms that drive the observed trends. 

[1]Model Stealing Attacks Against Inductive Graph Neural Networks, S&P, 2022.
[2]Unveiling the Secrets without Data: Can Graph Neural Networks Be Exploited through Data-Free Model Extraction Attacks?, Usenix, 2024.

### Questions
1. Could the authors evaluate the effectiveness of each defense strategy by directly pairing it with every attack method? 

2. In the current benchmark, both the target and surrogate models adopt the same GNN architecture. How would the performance and conclusions change if different architectures were used, for example, when the target model is a GCN while the surrogate employs a GAT?

### Soundness
2

### Presentation
2

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
The paper proposes GraphIP–Bench, a benchmark and library for evaluating model extraction attacks and ownership defenses (watermarking, fingerprinting, integrity verification) on GNNs. It claims to provide a unified protocol with standardized datasets, query budgets, and metrics for fidelity, utility, and efficiency. Experiments span 7 public node-classification datasets and a mixture of “data-driven” and “data-free” attacks, alongside four watermarking methods and one integrity-checking method. The authors argue this is the first rigorous benchmark for GNN IP protection.

### Strengths
- Addresses an important and timely problem: intellectual property protection for GNNs under realistic MLaaS settings.
- Provides a unified evaluation protocol that attempts to standardize splits, query budgets, and metrics.
- Includes both attack and defense families, covering data-driven, data-free extraction, and watermarking/fingerprinting methods.
- Reports protection–utility trade-offs and computational cost, which are relevant for practical deployment.

### Weaknesses
- Novelty is overstated: Prior works (e.g., GNNFingers, GrOVe) already provide standardized pipelines or ownership verification frameworks.
- Threat-model inconsistencies: Evidence of regime leakage in “data-free” settings (non–data-free methods achieving high fidelity).
- Endpoint assumptions unclear: No systematic evaluation of label-only vs. probability endpoints, despite their importance in extraction literature.

### Questions
Given the strengths, I have the following concerns:

1. Model extraction on GNNs has been thoroughly studied (taxonomy, realizations, empirical baselines) and public implementations exist. Reducing the novelty of assembling them under a single umbrella without definitive new methodology or insight is concerning. For instance, ownership verification has been addressed by GNNFingers[1] and GrOVe[2]. Can the authors should clearly articulate what is new beyond co-locating attacks and defenses. Also, [3] already standardized model extraction attacks along 7 threat models (and 7 corresponding attacks), along which of these 7 threat models did the authors consider? Or is their threat model agnostic to the exisiting threat models as enumerated in [3]? Moreover, why is [2] not included in the analysis?
 

2. The endpoint assumptions remain vague in practice. The protocol claims to fix whether the API exposes labels vs. probabilities; however, the main text doesn’t clearly specify which endpoints were used per setting, despite reporting data‑free variants based on logits vs. labels. The absence of a clean ablation across label‑only vs. probability endpoints (a central axis in the extraction literature) reduces interpretability.

3. The authors reported a defense training times of 1–6 seconds for watermarking on large graphs (e.g., CoauthorPhysics) but no clarification on the number of training steps or early‑stopping criteria that would justify such short runs. Could the authors clarify this?

4. Are the experiments conducted in the transductive or inductive setting?

5. The paper claims to enforce four regimes (features+structure, features-only, structure-only, data-free). However, in Table 7 (Cora) and related plots in Figure 3, non–data-free algorithms such as AdvMEA and CEGA achieve very high accuracy and fidelity in the data-free block (e.g., AdvMEA approx. 68% accuracy/F1; CEGA approx. 71–79% fidelity across budgets). Is this an anomaly? I see the same in other datasets in the appendix tables. Under a strict data-free protocol, these methods should not access real graph features or adjacency, making such performance questionable. Can the authors explain how they guaranteed isolation of regimes. Specifically, that data-free attacks had no access to real graph features or adjacency? Were synthetic queries generated without reusing real node IDs or structural statistics?

I am willing to change my score if my concerns are addressed.

**Minor**
- It is rather confusing which method the authors are actually referring to. Could the authors include the corresponding citations as inline in front of the methods rather than list them all at ones? For instance, "The benchmark includes four watermarking methods RandomWM, BackdoorWM,
SurviveWM, and ImperceptibleWM (Zhao et al., 2021; Xu et al., 2023; Wang et al., 2023;
Zhang et al., 2024),". Rather the authors should cite like so: "RandomWM (Zhao et al., 2021)", etc.

- Dataset count inconsistencies, unresolved references (“Table ??”). For instance, in some cases, the authors mentioned that they utilized 5 datasets whereas they utilized 7.


[1] You et al. GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks

[2] Waheed et al. Grove: Ownership verification of graph neural networks using embeddings.

[3] Wu et al. Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realisation

### Soundness
3

### Presentation
3

### Contribution
2
