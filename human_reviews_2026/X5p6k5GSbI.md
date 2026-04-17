# APEX: Approximate-but-exhaustive search for ultra-large combinatorial synthesis libraries

- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Make-on-demand combinatorial synthesis libraries (CSLs) like Enamine REAL have significantly enabled drug discovery efforts. However, their large size presents a challenge for virtual screening, where the goal is to identify the top compounds in a library according to a computational objective (e.g., optimizing docking score) subject to computational constraints under a limited computational budget. For current library sizes---numbering in the tens of billions of compounds---and scoring functions of interest, a routine virtual screening campaign may be limited to scoring fewer than 0.1\% of the available compounds, leaving potentially many high scoring compounds undiscovered. Furthermore, as constraints (and sometimes objectives) change during the course of a virtual screening campaign, existing virtual screening algorithms typically offer little room for amortization. We propose the approximate-but-exhaustive search protocol for CSLs, or APEX. APEX utilizes a neural network surrogate that exploits the structure of CSLs in the prediction of objectives and constraints to make full enumeration on a consumer GPU possible in under a minute, allowing for exact retrieval of approximate top-$k$ sets. To demonstrate APEX's capabilities, we develop a benchmark CSL comprised of more than 10 million compounds, all of which have been annotated with their docking scores on five medically relevant targets along with physicohemical properties measured with RDKit such that, for any objective and set of constraints, the ground truth top-$k$ compounds can be identified and compared against the retrievals from any virtual screening algorithm. We show APEX's consistently strong performance both in retrieval accuracy and runtime compared to alternative methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents APEX, which aims to find drug candidates from combinatorial synthesis libraries. The proposed method trains a surrogate neural network model to predict docking scores. In turn, by utilizing the hierarchical structure of CSL, the embedding of each synthon is calculated only once, and these can be combined to reconstruct the embedding of the entire compound. In this way, APEX can do an approximate-but-exhaustive search on the user's request, achieving good retrieval accuracy and runtime.

### Strengths
The proposed method enabled exhaustive search, which is based on approximated info, but can provide good retrieval accuracy with reasonable runtime. APEX seems to be technically sound, has clear presentation, and showed reasonable perf according to increased compound and k value.

### Weaknesses
1. I believe there are several existing works that presented a surrogate model; however, these are not discussed in the paper. Also, it'd be great if authors could provide a comparison to these works.
2. Can you please share how much GPU memory is consumed for training and actual inference?
3. How does associative embedding work? Can the model consider reaction in different order? e.g., when doing A+B the reaction result can be A-B or B-A, where - is bonding.
4. Is there any recent work that provides better result than Thompson Sampling?

### Questions
Please see Weaknesses section.

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
This paper presents APEX (Approximate-but-Exhaustive Search), which is a neural network-based framework for virtual screening of ultra-large combinatorial synthesis libraries (CSLs). The core challenge addressed is that modern CSLs contain tens of billions of make-on-demand compounds, but computational constraints typically limit virtual screening to evaluating less than 0.1% of available compounds, potentially missing many high-scoring candidates. APEX addresses this limitation through a three-step approach: (1) training a multi-task surrogate model that predicts molecular properties from molecular representations using a neural network encoder; (2) training a reaction factorizer that exploits the combinatorial structure of CSLs to reconstruct the surrogate's embeddings from reaction and R-group assignment pairs, thereby enabling efficient amortization across the entire library; and (3) performing approximate-but-exhaustive search by computing factorized surrogate predictions for all compounds in the CSL and retrieving the top-k compounds that maximize a user-specified objective subject to constraints. The key innovation is the APEX factorization, which decomposes surrogate predictions into synthon associative contributions that can be precomputed and cached, reducing the computational complexity.

### Strengths
1. I appreciate that the authors explicitly address real-world virtual screening constraints that are often overlooked in academic papers. The paper recognizes that modern CSLs contain tens of billions of compounds, yet computational budgets typically allow evaluation of less than 0.1% of the library, which is a genuine bottleneck in industrial drug discovery settings. The emphasis on handling multiple constraints is also valuable, as constraint satisfaction is critical in real-world drug discovery applications.

2. The runtime performance is remarkable—screening over 10 billion compounds in approximately 30 seconds on a single Tesla T4 GPU (Table 1).

3. The factorization approach that decomposes molecular embeddings into synthon associative contributions is well-explained.

### Weaknesses
1. The text size in Figure 1 and Figure 2 is extremely small and nearly illegible at normal viewing resolution. I had to zoom to 500% magnification to read the text labels, annotations, and diagram components. The authors should significantly increase font sizes and improve the overall figure format.

2. The paper makes claims about screening 10 billion compounds using a surrogate model trained on only 1 million, yet provides insufficient evidence to support the reliability of this generalization. The fundamental assumption is that a surrogate model trained on 1M compounds can accurately predict properties across the full 10B chemical space, yet there is no theoretical or empirical justification for this distribution shift. In real-world deployment, the model would likely encounter compounds with unseen scaffolds and functional groups, where the reliability of the surrogate remains untested and uncertain.

3. APEX introduces a two-stage approximation: first, the surrogate model approximates the true objective, and then the factorizer approximates the surrogate. However, the cumulative error arising from these cascaded approximations is neither quantified nor theoretically bounded. It remains unclear how the authors ensure that error accumulation does not occur within their proposed framework.

4. The authors inject isotropic Gaussian noise into embeddings during training and claim this makes predictions robust. However, this is merely a statistical regularization technique that assumes embedding perturbations follow a simple normal distribution. This does not reflect the actual uncertainty structure in chemical space, where small structural changes can cause large, non-random shifts in molecular properties. The noise injection is mathematically convenient but chemically unmotivated. If the authors believe this approach is chemically grounded or relevant, please provide a detailed justification and explanation. 

5. The paper compares APEX against only a single baseline method (i.e., Thompson sampling). The introduction mentions multiple recent virtual screening algorithms (V-SYNTHES  and NGT) but provides no experimental comparison with any of them. If the authors consider Thompson sampling to be the only valid baseline, they should explicitly justify this choice and clarify why other established approaches were excluded from evaluation.

### Questions
1. The paper implements custom PyTorch CUDA C++ extension modules for top-k retrieval with chain-of-batches strategy and GPU-compatible AIR top-k algorithm. Does this custom hardware-level implementation mean that users would need to modify or fine-tune the CUDA code for different GPU architectures, library sizes, or constraint specifications? How portable is this implementation across different hardware (e.g., AMD GPUs, Apple Silicon, newer NVIDIA architectures)?

2. What is the trade-off between embedding dimension, surrogate accuracy, factorization fidelity, and memory consumption? Should users with larger synthon libraries or more complex objectives use higher d? Is there a principled way to determine the minimum d needed for a given retrieval accuracy target?

3. The paper assumes CSLs follow a specific hierarchical structure with reactions, R-groups, and synthons. How does APEX handle: (1) libraries with variable numbers of components per reaction (e.g., some reactions use 2 synthons, others use 4 or 5); (2) multi-step synthesis routes rather than single-step combinatorial reactions; (3) libraries with reaction templates that have positional dependencies (e.g., where synthon choice for R-group 1 constrains valid choices for R-group 2)?

4. All evaluations in the paper use predicted docking scores from GPU-accelerated AutoDock Vina as ground truth. However, docking scores are themselves approximations of true binding affinity, and there is often poor correlation between computational docking and experimental binding measurements. Have any of the APEX-retrieved compounds been validated experimentally?

5. As commercial CSLs continue to grow toward 100 billion or even trillion-scale libraries, how does APEX's performance scale? Does retrieval accuracy decrease as the library grows and the chemical space becomes more diverse? 

6. APEX performs a single-shot retrieval based on the trained surrogate and factorizer. In practice, virtual screening is often iterative: initial hits are validated, and the model is refined based on feedback. Does APEX support active learning workflows, or is APEX intended only as a one-time screening tool, requiring full retraining if the user wants to incorporate new data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript introduces APEX (Approximate-but-Exhaustive), a search protocol for ultra-large combinatorial synthesis libraries (CSLs) that aims to make “full-library” virtual screening feasible by combining (i) a neural surrogate trained on an enumerated subset and (ii) a factorization mechanism that decomposes surrogate embeddings into synthon-level contributions. This enables the model to evaluate all compounds implied by the CSL (up to 10B+ products) on a single GPU and then perform top-k retrieval under user-specified objectives and constraints.

### Strengths
- Problem importance: The paper tackles a very real bottleneck in structure-based and library-based discovery: current make-on-demand CSLs are now so large (10⁹–10¹⁰) that even “smart” virtual screening methods end up seeing <1% of the space, leaving high-scoring chemistry unexplored. A method that makes declarative, low-latency queries over the entire library is valuable to both method developers and practitioners. The paper also recognizes that constraints (Lipinski, Veber, fragment-like rules, etc.) are not an afterthought but central to realistic screening campaigns.

- Interesting and novel factoring mechanism: The key technical idea of learning a factorizer that reconstructs the surrogate embedding from (reaction, R-group, synthon) choices, is genuinely interesting. By pushing the complexity into a hierarchical, library-aware model (synthon → R-group → reaction → associative contributions), APEX can precompute a small number of neural evaluations and then reuse them combinatorially when traversing the CSL. This is a nice instantiation of “amortization over a structured chemical space,” and the linear projection to task heads (eqs. (13)–(16)) makes the top-k scan GPU-friendly. It is a nontrivial step beyond simply “batch the surrogate on GPU.”

### Weaknesses
- Dependence on synthon + reaction paths for every library compound: APEX only works because every product in the library is addressable as “reaction + R-group + synthon” and the factorizer has been trained on that exact CSL structure. This is fine for Enamine-style, synthon-organized libraries, but it also means the method is not directly applicable to an arbitrary compound library (e.g., a corporate merged screening collection, ChEMBL-like flat sets, or ad-hoc AI-generated enumerations) unless that library can first be expressed in this hierarchical CSL form. This limitation should be made explicit in the paper: APEX solves the CSL case, not the general VS case. Right now the narrative leans toward “fast virtual screening on massive libraries,” but the technical requirement is actually “fast screening on massive combinatorial libraries with known factorization.” Please clarify scope and portability, and discuss what would be needed to support (i) multiple CSLs with different factorization schemes and/or (ii) partially specified synthons.

- Benchmarked properties are quite basic: The experimental section mostly shows APEX recovering top docking hits and satisfying standard medicinal chemistry filters (Lipinski, Veber, Astex Rule of 3, Pfizer 3/75). These are useful sanity checks but they are all relatively low-level or RDKit-level properties. For a method whose selling point is “once your surrogate is trained, you can query anything,” it would be much more convincing to see: docking-like tasks with tighter resolution (e.g., distinguishing top 0.01% vs. top 0.1%); multi-target or composite objectives; a direct comparison on docking-score enrichment against baselines.

- Right now, the tasks mostly show that the factorization is not breaking the surrogate, but they don’t fully stress-test how well APEX preserves ordering on a hard, high-variance objective like docking. The paper would be much stronger with more analysis specifically on docking-score tasks, including per-target recall@k curves and order-preservation (Kendall-τ / Spearman-ρ) between surrogate, factorized surrogate, and ground truth.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes APEX, a surrogate-based  search framework for ultra-large combinatorial synthesis libraries. The workflow trains a surrogate model to predict an objective (docking), then trains a reaction factorizer that reconstructs the surrogate embedding from reaction, R-group, and synthon components. The model then performs GPU-accelerated top-k retrieval over the entire enumerated CSL using pre-computed, factorized contributions. The authors evaluate APEX on docking tasks for five targets and claim quickly retrieval for 1 M–12 M-sized datasets, extending to "potential" 10 B-scale enumeration.

### Strengths
1. Evaluation of the method is reasonable:  Includes five realistic molecular targets and measures runtime and recall, showing consistent acceleration over brute-force screening.

### Weaknesses
1. Exceptionally limited related work. The paper omits extensive prior work in active learning and surrogate-based virtual screening, such as
    - MEMES: Machine Learning Framework for Enhanced Molecular Screening [1]
    - Accelerating High-Throughput Virtual Screening through Molecular Pool-Based Active Learning [2]
    - Generative AI for Navigating Synthesizable Chemical Space [3] 

These works already report more sophisticated active learning and synthesizability-aware loops, which APEX neither cites nor compares against. Before publication, the paper requires a thorough literature review of enumeration and active learning in molecular design to contextualize its contribution. 

2. Novelty is weak. Using a learned surrogate for pre-screening enumerated chemical spaces is well-established. APEX mainly implements an efficient factorization of embeddings rather than a new search paradigm.
3. The “open CSL” is created by without synthesizabilty consdierations, so generated compounds are not necessarily synthesizable, unlike Gao et al.’s synthesizability approaches [3]
4. Baseline selection is poor. All comparisons are against non-ML heuristics (random search, Thompson sampling) rather than established machine learning-based or active learning approaches

### Questions
1. Definition of “approximate-but-exhaustive”: Please formalize what is meant by “exhaustive search” when using a factorized surrogate. Is there any theoretical guarantee that top-k ranking under the surrogate approximates the true top-k under the physical objective?
2. Baselines: Why are active learning and ML-based selection strategies excluded from comparison? How would APEX perform relative to these?
3. Could the authors contextualize the work within the broad landscape of chemical search using ML ?

### Soundness
2

### Presentation
1

### Contribution
1
