# Micro-Learning for Learning-Hard Problems

- Decision: Reject
- Scores: 0, 6, 6, 4, 2, 4, 4

## Abstract
ML increasingly faces high complexity nonlinear data whose noise, imbalance, or small sample size thwart conventional models. We formalize this difficulty through the notion of Learning Hard Problems (LH-Ps), tasks that (i) defeat the vast majority of models, yet (ii) admit at least one high‑quality solution if the relevant label-aware  structural knowledge is appropriately incorporated during training. To address this, we introduce Micro‑Learning (MiL), a principled framework that constructs traininglets: small, knowledge‑fused subsets of the training data with demonstrably low complexity and infers a deterministic local model for each that collectively form a global predictor. We prove that the decision version of optimal traininglet selection is NP‑complete, establishing a strong theoretical foundation for MiL. MiL dramatically reduces overfitting risk by eliminating irrelevant or noisy samples, while retaining interpretability and reproducibility through deterministic optimization in a RKHS space. Experiments in benchmark domains, from music information retrieval to medical proteomics, show that MiL solves LH-Ps  and outperforms deep learning and classical baselines, especially on imbalanced or small-sample datasets, with negligible overfitting. Moreover, our work provides (i) the $1^{st}$  definition of LH-Ps, (ii) a Learning-Hard Index to quantify task difficulty pre-training, and (iii) theoretical guarantees on traininglet optimality and complexity, enriching learning theory and ethical AI.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper presents Micro-Learning (MiL), a framework for learning from high-complexity categorical data. MiL creates traininglets (small representative clusters derived from the dataset's class labels), splitting the training process into learning over distinct data regions with theoretical guarantees on complexity and generalization bounds. To motivate MiL, the authors present a theoretical analysis of a class of problems they call Learning-Hard Problems (LHPs) and a Learning-Hard Index (LHI) to quantify the difficulty of a categorical dataset. Experimental results indicate performance improvement across a range of high-LHI tasks compared to traditional ML and DL techniques.

### Strengths
- Theoretical analysis and guarantees for Micro-Learning. Key properties of MiL are defined in detail mathematically, and the paper's overall structure flows well.

- Well-defined problem statement, supplementary provides detailed proofs.

- Open source code for MiL on the ovarian dataset.

- Decisions on traininglets with SVM are interpretable, a key differentiator over black-box
DL baselines.

- A thorough number of baselines (15).

### Weaknesses
- It is unclear where this work sits in the scope of preprocessing/complexity reduction techniques. The results section compares MiL to ML and DL based methods, but does not cover class imbalance techniques, dimension reduction techniques, or other methods commonly used to stratify difficult datasets.

- Definition 1 describes latent solvability as the risk of a hypothesis composed with a knowledge-injection operator. However, the paper doesn’t compare MiL to ML and DL based methods with any label-aware projection or re-sampling operator, which they define as knowledge-fusion. It could be the case that with proper data preprocessing, the baseline results could improve

- Furthermore, the method by which results are generated in Figure 3 is unclear. The paper reports hyperparameter selection and CV; however, these results are not presented anywhere or discussed again. The work should include these metrics to verify that the results in Figure 3 represent a fair comparison. It would also be beneficial to include this process in the open source code.

- No empirical results table is reported for Figure 3. The figure is way too small to read effectively, and the contrasting MiL vs. baselines visualization makes it very difficult to interpret how MiL performs. I would suggest plotting an MiL bar alongside the baselines
for a more interpretable analysis of the results. Additionally, no empirical results are presented for CASIA and SAVEE outside of subplot d of Figure 3, which does not describe exact metric values. Expanding Table 2 to include results across baselines would go a long way in conveying the strengths of MiL, or at least including these results in the supplementary material.

- The paper could use another read over. Many terms, especially in definitions and theorems, are referenced without being defined beforehand. Lines 301 and 302 have question marks due to a (presumably) unresolved LaTeX reference to an equation.

- All figures are too small to read and need to be made much larger. References should be fixed.

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the concept of Learning-Hard Problems (LH-Ps), defined as tasks where most models fail but a solution exists if domain knowledge is incorporated. To address LH-Ps, the authors propose Micro-Learning (MiL), a framework that constructs small, customized subsets of the training data ("traininglets") for each query point and trains a local deterministic model (e.g., an SVM) on them. The paper provides a theoretical analysis, including the NP-completeness of optimal traininglet selection, and presents empirical results on several benchmarks showing MiL's superiority over various classical and deep learning baselines, particularly on imbalanced or small-sample datasets.

### Strengths
The paper proposes a formal definition for "Learning-Hard Problems" (LH-Ps) and a corresponding "Learning-Hard Index" (LHI), which is a valuable conceptual contribution for characterizing difficult learning scenarios.

The analysis of the computational complexity of traininglet selection (NP-complete) and the provided generalization bounds (e.g., based on local Rademacher complexity) add theoretical rigor to the proposed method.

The paper includes experiments on multiple diverse benchmarks (music, speech, medical) and compares against a wide array of baselines, demonstrating consistent performance improvements, especially on imbalanced and small-sample datasets.

### Weaknesses
The central mechanism of MiL---"fusing domain knowledge"---is not clearly defined or operationalized.

- The paper claims domain knowledge is fused before model induction, but the described process (e.g., intersecting metric balls, label-aware t-SNE) appears to be a form of sophisticated, data-dependent sample selection and preprocessing. It is unclear how this constitutes the injection of external, human-curated domain knowledge (e.g., expert rules, ontological relationships). This conflation of terms creates significant confusion about the method's novelty and true contribution.

- Relatedly, the description of how test data information is fused into training (as illustrated in Figure 2b) is vague. The method seems to use the test query to select a relevant training subset, but the specifics of how this "fusion" is implemented beyond a nearest-neighbor-like selection are not sufficiently detailed, especially considering practical applications with high-dimentional data like images.

The core idea, as understood, involves training a new model (like an SVM) on a custom data subset for each individual test sample. The paper briefly discusses complexity (Sec. 4, "MiL Complexity") but does not adequately address the profound practical limitations this imposes. For large-scale datasets with millions of test points, this per-query training cost would be prohibitive, making the method's scalability and real-world applicability a major concern. 

The manuscript contains several notation inconsistencies and typos that hinder comprehension. For instance, in the paragraphs surrounding Eq. (3) and in Lines 301-302. These issues, while seemingly minor, accumulate and detract from the paper's professionalism and readability.

### Questions
Please clarify what is meant by "domain knowledge." Is it external, human-provided knowledge, or is it knowledge automatically extracted from the data structure and labels? The methodology should be rephrased to accurately reflect its actual operations. Furthermore, provide a detailed, step-by-step explanation of how Figure 2b is implemented, specifically the "fuse test data information into training" step.

The computational complexity of MiL is a critical drawback. The paper should include a more honest and thorough discussion of this limitation. Please also discuss potential strategies for mitigating this cost (e.g., approximate methods, caching).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the concept of Learning-Hard Problems (LH-Ps), a novel class of machine learning tasks characterized by two simultaneous properties: (C1) near-universal failure, where almost all models in a broad hypothesis space perform poorly, and (C2) latent solvability, where at least one high-quality solution exists when appropriate domain knowledge is incorporated during training. To address LH-Ps, the authors propose Micro-Learning (MiL), a principled framework that constructs query-specific "traininglets"—small, knowledge-fused subsets of training data—and trains local models on them rather than learning a single global model.

The paper makes four primary contributions. First, it provides the first formal definition and characterization of LH-Ps, introducing the Learning-Hard Index (LHI), a data-centric metric computed via locality-preserving embeddings (t-SNE/UMAP) and clustering that quantifies dataset complexity pre-training. Tasks with LHI ≥ 0.80 are classified as learning-hard. Second, the authors prove that the decision problem of finding optimal traininglets is NP-complete (Theorem 1), establishing strong theoretical foundations. Third, they develop Precision Traininglet Construction (PTC), a practical four-stage algorithm comprising probing learning, training sanitization, meta-traininglet fusion, and precision pruning. The paper provides theoretical guarantees showing that PTC reduces local Rademacher complexity (Proposition 1), enables traininglet-trained models to exceed full-data models for Bayes-optimal prediction (Proposition 2), and contracts the train-test distribution gap (Proposition 3). Fourth, comprehensive experiments across five benchmarks—IRMAS (music), CASIA and SAVEE (speech emotion), COVID-19 (medical triage), and Ovarian (proteomics)—demonstrate that MiL outperforms 15 baselines including classical methods (SVM, Random Forest) and modern deep learning architectures (CNNs, LSTMs, CapsNets). Statistical validation using Mann-Whitney U-tests shows MiL achieves median performance of 0.97 versus 0.77 for the best deep learning baseline (p < 2×10⁻⁸, Cliff's δ ≈ 0.77), with particularly impressive results on extreme class imbalance (Ovarian: 98.15% accuracy with 100% sensitivity at 98.5% imbalance). The framework maintains interpretability and reproducibility through deterministic SVM optimization while dramatically reducing overfitting by training on compact, noise-reduced subsets rather than full noisy datasets.

### Strengths
Originality:
The paper demonstrates exceptional originality across multiple dimensions. The formalization of Learning-Hard Problems (LH-Ps) through Definition 1 is genuinely novel, introducing a problem class characterized by dual conditions—(C1) near-universal failure and (C2) latent solvability—that captures a paradox practitioners encounter but lacked theoretical language to describe. This is not reframing existing concepts but creating new taxonomy for understanding when and why machine learning fails.The Learning-Hard Index (LHI) represents the first pre-training, model-agnostic diagnostic metric for problem difficulty. Unlike Rademacher complexity or VC dimension that measure model capacity, LHI is data-centric and computable before training begins. This shift from "how complex is my model?" to "how hard is my data?" is conceptually innovative. The MiL framework's query-specific traininglet construction represents a paradigm shift from global model learning. While local learning methods exist, MiL's execution is original: (1) knowledge fusion via multi-metric intersection, (2) principled noise removal guided by self-prediction, (3) meta-traininglet fusion combining four complementary views, and (4) SVM-micro-CNN-let hybrid maintaining determinism while gaining expressiveness. This creative combination of ideas from kernel methods, meta-learning, and deep learning is novel.

Quality:
Technical quality is strong across theory, algorithm design, and empirical validation. The NP-completeness proof (Theorem 1) establishes fundamental computational limits, justifying the heuristic approach. Propositions 1-3 provide constructive guarantees: Proposition 1 guarantees low-capacity "sweet-spots" via local Rademacher complexity; Proposition 2 proves traininglet-trained models can exceed full-data models; Proposition 3 shows PTC contracts train-test distribution distance. These results connect learning theory to algorithmic design principally. The PTC algorithm demonstrates high-quality design with each stage theoretically motivated: Stage 1 uses data-driven hyperparameter selection; Stage 2 reduces LHI by 6-20% through noise removal; Stage 3 ensures label completeness; Stage 4 removes residual outliers. The progression from NP-hardness to practical heuristic to theoretical guarantees exemplifies bridging theory and practice. Experimental quality is comprehensive with proper statistical methodology: Mann-Whitney U-tests, Cliff's delta effect sizes (δ≈0.77), Bonferroni correction, and 5-fold CV repeated 5 times. The 15 diverse baselines span classical ML and modern DL paradigms. Five benchmarks across music, speech, and medical domains with varying characteristics validate generalizability.

Clarity:
The problem motivation is clear—Figure 1 effectively illustrates how raw data appears inseparable but label-aware embedding reveals structure, immediately conveying why standard methods fail. Mathematical formulations are precise with well-defined definitions and consistent n;otation. The experimental setup clearly presents dataset statistics (Table 1) and results (Table 2) with transparent statistical validation. The paper's structure logically progresses from problem identification (LHI) to theoretical analysis (complexity) to algorithm design (MiL/PTC) to empirical validation. Figure 2's comparison clearly illustrates the paradigm shift from global to local functions. However, Algorithm 1 being in supplemental rather than main text and dense writing in Sections 3-4 hinder accessibility.

Significance:
The significance is substantial across multiple dimensions. Scientifically, LH-P formalization fills a genuine gap in ML taxonomy between "easy" and "impossible" problems, enabling precise communication about problem difficulty. The LHI metric has immediate practical value—practitioners can diagnose whether specialized methods are needed before investing in model development. For high-stakes applications (medical diagnosis, fraud detection), the significance is profound. MiL addresses small samples, extreme imbalance, and interpretability requirements simultaneously—Ovarian cancer results (98.15% accuracy, 100% sensitivity at 98.5% imbalance) demonstrate life-saving potential with regulatory-compliant deterministic predictions. Theoretically, the NP-completeness result establishes fundamental limits on optimal data selection. The distribution contraction guarantee offers new perspective on why local learning can outperform global learning. Methodologically, MiL opens research directions in unsupervised settings, approximate construction for larger datasets, and integration with neural architecture search. The empirical results are substantial—15.8 percentage point improvement on IRMAS over prior state-of-the-art represents significant progress. Stochastic dominance over all 15 baselines (p<2×10⁻⁸) demonstrates robust superiority. The framework challenges the "bigger data, bigger models" paradigm, with implications for sustainable AI, democratizing ML where large datasets are unavailable, and maintaining interpretability on imbalanced data.

### Weaknesses
Scalability Limitations and Missing Computational Analysis:
The most significant weakness is the O(Mn²) preprocessing complexity that fundamentally limits MiL to small/medium datasets. The largest dataset tested is IRMAS with n=6,705—no experiments validate scalability beyond 10K samples. The paper acknowledges this limitation but only suggests "FPGA/GPU acceleration" without concrete analysis. Critical missing information includes: (1) wall-clock time and memory usage for each dataset, (2) exploration of approximate nearest neighbor methods (FAISS, Annoy) to reduce distance computation costs, (3) empirical analysis of quality-speed tradeoffs with approximations, and (4) clear guidance on maximum practical dataset size. For a method targeting real-world applications, the absence of computational cost analysis is a major gap. The paper should provide concrete numbers (e.g., "IRMAS preprocessing takes X hours on Y GPU with Z GB memory") and demonstrate whether ANN-based approximations maintain performance while improving scalability. Without this, practitioners cannot assess MiL's feasibility for their problems.

Absence of Meta-Learning Comparisons:
A critical weakness is the lack of empirical comparison with meta-learning methods despite discussing MAML (Finn et al., 2017) and Prototypical Networks (Snell et al., 2017) in related work. Meta-learning directly addresses small-data problems—the same domain MiL targets. The paper dismisses these methods briefly ("cannot escape the original hypothesis space") without empirical validation. This is problematic because: (1) meta-learning is the most relevant baseline for few-shot scenarios, (2) the distinction between MiL's data-selection approach and meta-learning's initialization-learning approach needs empirical support, and (3) readers cannot assess whether MiL's paradigm shift actually outperforms the established meta-learning paradigm. At minimum, the paper should include MAML or Prototypical Networks as baselines on at least 2-3 datasets to validate MiL's superiority over this competing approach.

Limited Failure Case Analysis:
The paper focuses exclusively on successes without analyzing when or why MiL fails. This is problematic for several reasons: (1) COVID-19 has LHI=78.5%, below the 0.80 threshold, yet is included without explanation of why it still works, (2) no discussion of borderline cases (LHI 0.75-0.85) where the diagnostic might be unreliable, (3) no analysis of scenarios where traininglet construction might fail (e.g., no good neighbors found, all classes equally noisy), (4) no discussion of adversarial robustness, and (5) no guidance on when practitioners should use MiL vs. standard methods beyond the LHI threshold. The paper should add a "Limitations and Failure Modes" subsection with: (1) at least one dataset where MiL underperforms or fails, (2) sensitivity analysis showing performance degradation as LHI decreases below 0.80, (3) discussion of failure scenarios (insufficient data, all samples noisy, adversarial examples), and (4) decision framework for method selection.

Presentation Density Hindering Accessibility:
The paper's dense presentation significantly limits accessibility. Specific issues: (1) Algorithm 1 (PTC pseudocode) is in supplemental, not main text—this is the core algorithmic contribution and should be in Section 4.2, (2) notation inconsistencies create confusion (T_{x'}, T^{PTC}{x'}, T^{(j)}{x'}, U_{x'} without clear relationship), (3) abstract is overly dense (150+ words in complex sentences), (4) Sections 3-4 assume high familiarity with learning theory without sufficient intuition, (5) figures are too small (Figure 1b-d t-SNE plots hard to read, Figure 3 subplots cramped), and (6) no notation table despite heavy mathematical content. These issues prevent readers outside core ML theory from fully understanding the contributions. Specific improvements needed: (1) add Algorithm 1 with complexity analysis to main paper, (2) create notation table in appendix, (3) simplify abstract to 100-120 words, (4) add intuitive explanations before technical sections, (5) enlarge figures and improve captions, and (6) add visual flowchart for PTC pipeline.

### Questions
Scalability and Computational Cost:
The O(Mn²) preprocessing complexity is a major concern that limits practical adoption. Can you provide:
- Wall-clock time and memory usage for each of the five datasets (including hardware specifications)?
- Concrete analysis of the largest dataset size MiL can handle practically (e.g., what happens at n=50K, n=100K)?
- Have you explored approximate nearest neighbor methods (FAISS, Annoy, HNSW) to reduce the O(n²) distance computation cost? If so, what is the accuracy-speed tradeoff?
- Can the PTC stages be parallelized across multiple GPUs or distributed systems?
Without this information, it's unclear whether MiL is limited to toy problems or can scale to real-world applications. Demonstrating even approximate scalability would significantly strengthen the contribution.

2. Baseline Fairness and Hyperparameter Details
The paper states baselines were tuned via "nested grid search" but provides no specifics. Please provide:
- Complete hyperparameter specifications for all 15 baselines (grid ranges, final selected values)
- Computational budget allocated to each baseline (GPU hours, number of trials)
- Architecture details: CNN depth/width/kernel sizes, LSTM/GRU hidden dimensions, number of layers
- Training details: batch sizes, learning rates, optimizers, number of epochs, early stopping criteria
- Did deep learning baselines use data augmentation, dropout, batch normalization?

Specific concern: You report CNN achieving ~60% on IRMAS, but Yu et al. (2020) achieved 68.5% with a specialized architecture. Did you try similar architectures, or only vanilla CNNs? This 8.5 percentage point gap raises questions about whether baselines were given fair opportunity.
Without these details, reviewers cannot assess whether MiL's superiority stems from algorithmic innovation or simply better tuning. This is critical for accepting the empirical claims.

3. Meta-Learning Baseline Comparison
The paper discusses MAML (Finn et al., 2017) and Prototypical Networks (Snell et al., 2017) in related work but doesn't compare empirically. Can you:
- Add at least one meta-learning baseline (MAML or Prototypical Networks) to the experiments?
- Explain why meta-learning methods were excluded from the comparison?
- Provide theoretical or empirical analysis of why MiL's data-selection approach should outperform meta-learning's initialization-learning approach?
Meta-learning is the most natural comparison for small-data problems. Without this comparison, the positioning of MiL relative to the most relevant prior work is incomplete. Adding this could either strengthen your claims or reveal complementary approaches worth discussing.

4. Failure Case Analysis
The paper focuses on successes but doesn't analyze failures. Can you:
- Provide at least one example where MiL fails or performs poorly?
- Analyze what happens at borderline LHI values (0.75-0.85)? The COVID-19 dataset has LHI=78.5%, below your 0.80 threshold—why does MiL still work well here?
- Discuss scenarios where traininglet construction might fail (e.g., no good neighbors found, all classes equally noisy)?
- How does MiL handle adversarial examples or distribution shift at test time?
Understanding failure modes is as important as understanding successes. This would help practitioners know when NOT to use MiL and would demonstrate intellectual honesty.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Micro-Learning (MiL), a framework that trains local classifiers on traininglets (small, knowledge-fused subsets of the data) instead of fitting a single global predictor. The authors formalize Learning-Hard Problems (LH-Ps), propose a Learning-Hard Index (LHI) to quantify task difficulty, and provide theoretical guarantees on the optimality and complexity of traininglets. Experiments across five datasets demonstrate that MiL outperforms conventional baselines, particularly on noisy, imbalanced, or small-sample tasks.

### Strengths
- The proposed framework is novel and interesting, offering an alternative to global learning by leveraging local, knowledge-fused traininglets.

- Empirical results across diverse small or imbalanced datasets demonstrate consistent gains over conventional baselines.

### Weaknesses
- Line143 LHI is claimed to be computable “before any training,” yet it depends on an embedding and clustering pipeline (e.g., t-SNE/UMAP + k-means). It’s unclear whether LHI then measures dataset difficulty or the difficulty specific to these representation/cluster choices.

- Provide a broader empirical study of LHI across varied regimes (dataset size, modalities, class counts, imbalance) and correlate LHI buckets with baseline performance. This would better justify the 0.80 threshold and show how conventional models degrade as LHI rises.

- The work focuses on scarce/imbalanced settings; however, modern large pretrained models often perform well in these regimes. Add comparisons or discussion against strong pretrained baselines to position MiL fairly.


- The paper would benefit from an additional proof reading:
  - Line042: “be as above” appears before the referenced objects are defined.
  - Line063: “IRMAS” dataset is introduced with little context. A short description would be helpful.
  - Fig. 1’s puzzle motif is visually unclear in how it maps to the IRMAS example. 
  - Line301: equation numbers are missing.

### Questions
- L247 Is $\cal{T}_{x’}$ independent of $x’$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript formally defines a category of Learning-Hard Problems (LH-Ps) characterized by non-linearity, noise, imbalance, and small sample sizes, which pose significant challenges to generalization. The authors argue that the failure of deep learning and other standard models on LH-Ps stems from their inability to incorporate domain knowledge. To address this, they introduce the Micro-Learning (MiL) framework, which performs localized learning by extracting domain-informed subsets from the training data for each query point—a process reminiscent of test-time learning. This design effectively filters out irrelevant or noisy samples, thereby mitigating overfitting risks. Experiments demonstrate that MiL outperforms deep learning and other standard baselines on LH-P tasks. Additionally, the authors propose a Learning-Hard Index to measure the difficulty of a task, and provide theoretical guarantees regarding the optimality and complexity of their approach.

### Strengths
The initial problem analysis and the introduction of this work are well-articulated. The proposed concept of a problem difficulty index and the motivation for using traininglets are particularly compelling and may offer valuable insights for the domain generalization research community.

### Weaknesses
The manuscript suffers from significant presentation issues, with the methodology being ambiguously explained and key definitions omitted. Crucially, the process of incorporating domain knowledge remains unclear. The experimental design lacks rigor: comparisons against specialized state-of-the-art deep learning methods for imbalanced or small-sample problems are absent, undermining the claimed advantages. Furthermore, the evaluation is incomplete due to the lack of ablation studies and assessments across datasets of varying difficulty levels.

### Questions
1.The manuscript requires a precise operational definition of "appropriate domain knowledge," clarifying how it is quantitatively or qualitatively incorporated into the MiL framework.

2.Line 92, please define "modern MIR system" or provide a canonical reference.

3.Significant writing issues persist, particularly in mathematical notation. Symbols (e.g., f_dm in Line 146) must be formally defined upon their first appearance. 

4.Equation (1): The definition of the Adjusted Mutual Index (AMI) is absent. Furthermore, the rationale for selecting the threshold 0.8 requires explicit justification.

5.The methodology for obtaining t-SNE embeddings (X_r) from high-dimensional image data needs elaboration. Please specify the feature extraction pipeline (e.g., whether a pretrained DNN was employed) to ensure reproducibility.

6.To properly demonstrate the meaningfulness of the Learning-Hard Index (LHI), it should be quantitatively evaluated on established benchmark problems with known difficulty characteristics (e.g., domain shift vs. balanced classification).

7.Line 180: The computational procedure for the norm in F_r(f) is unspecified.

8.Propositions 1 and 2 establish that, for a given hypothesis class H, MiL identifies the function with minimal local Rademacher complexity, thereby reducing its overfitting propensity relative to other functions in H. A pertinent question remains: how does the framework guarantee that this particular function is also effective at solving the underlying task, rather than merely being the least overfit?

9.Line 203: Contains a typo "S_pS"? Formatting issues: Excessive spacing (Line 295), incorrect equation references (Line 301), and inconsistent notation (e.g., "radius," "neighbourhood," "data split") require resolution.

10.Section 4.2 requires restructuring for logical coherence and readability. Line 311: The impact of test batch size on performance needs explanation. The protocol for single-sample inference should be specified. The design rationale and fusion strategy for the four meta-traininglets lack motivation.

11.The visualization of results is suboptimal, with figures being too small and analyses lacking depth. For instance, in Figure 4, if the intent is to demonstrate that traininglets possess superior separability, it is unclear what baseline they are being compared against. Critical parameters such as batch size are omitted. Including test samples in the visualization would more effectively illustrate whether the traininglets generalize well.

12.The proposed method involves numerous nuanced design choices and hyperparameters, raising concerns about its general applicability. The core mechanism for domain knowledge integration remains conceptually unclear throughout the technical exposition.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new concept calls Learning-Hard Problems which are problems that are: 
1. Every model fails to generalize the entire dataset
2. Solvable once additional knowledge \phi is added to the model
These problems can be identified by using the Learning-Hard Index (LHI) that indicates the difficulties of learning on the full data
These problems are said to have solvable regions in the data where the learner f can generalize well to test data. These are compact regions within the dataset that have strong locality and knowledge representation that are called traininglets. However, finding these subsets are NP-complete and requires a greedy algorithm.
The algorithm is called Precision Traininglet Construction (PTC) pipeline with probing learning, sanitization, meta-traininglet fusion, and precision pruning.

### Strengths
This paper has a very strong theoretical backing that explains the problems and its solutions. The paper finds a very niche type of datasets that require a different approach than conventional methods. Overall, it has strong novelty with a lot of theoretical backings.

### Weaknesses
Although many models are used for experiments, they seem to be too generic. The method should be compared against the best attemps at solving the datasets mentioned in the papers. Generic methods like CNN have many variations with different degrees of effectiveness and might need special modifications for specific tasks. Overall, it is not clear that the baselines models are the best attempts at the datasets.

The threshold 0.8 LHI for LHP classification seems arbitrary since a dataset like COVID-19 with LHI < 0.8 still fit the characteristics of LHP where most models fail to generalize.

Part 3 of PTC does not make sense: The 4 sets are not specified.

All of the graph figures are too small to be read, they should all be enlarged. However, figure 3 and table 2 should be merged since they all represent the experimental results while figure 3 should have all methods on the same side for easier reading or be made into a tables like table 2 where just listing the results of MiL in table 2 does not serve any purpose.

Writing errors in lines 203 - S_pS instead of S_p, 301 and 302 are missing references

In section 4.2, NTC and PTC are described where NTC is a naive step that PTC is built upon, therefore NTC should be its own subsection like PTC.

### Questions
What is the significant of the 0.8 threshold?

Many methods mentioned in the related sections do not appear in the experiments suchs as MAML, SAM, etc. Are they the best attemps at the datasets?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 7

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the concept of Learning-Hard Problems (LH-Ps), a class of tasks at which most models fail, even though there exists a high-quality solution that could be reached through the use of domain knowledge. The authors formalize this by proposing a data-centric metric called the Learning-Hard Index (LHI) that can identify such problems before training.
The main contribution is the Micro-Learning framework (MiL), which aims at solving LH-Ps. In contrast to training one global model, MiL learns, for each test instance, a new, local, interpretable model (e.g., an SVM). This local model is learned on a so-called "traininglet", a small low-complexity subset of the original training data that is optimized for the given query. The paper theoretically analyzes this process, proving that computing the optimal traininglet is NP-complete, and then proposes a practical heuristic pipeline, Precision Traininglet Construction (PTC), to efficiently generate them.

### Strengths
1. Formalization of LH-Ps is a significant conceptual contribution. It provides a clear definition and a diagnostic tool-the LHI-for a class of problems that practitioners often encounter but for which they lack the vocabulary to describe formally.

2. The MiL framework represents a break in the dominant paradigm of training large, global models. Indeed, the basic intuition of "localizing the solution" by training instance-specific traininglets is intuitive and powerful, particularly in datasets with complex, nonlinear structures or high levels of noise and imbalance.

3. The rigorous theoretical work by the paper provides bounds to strengthen the claims. The intractability of the problem in question is clearly established by the NP-completeness proof for optimal traininglet selection, thus justifying heuristics like PTC.

### Weaknesses
1. The main weakness of the paper is the ambiguity between the theoretical claims and the experimental demonstrations. Particularly, the methodology proposes a generalizable framework viewpoint for solving LH-Ps but the experiments are conducted on trivial timeseries, tabular datasets which critically limits the scope of the work. Datasets like iNaturalist, CUBS, ImageNet etc. are naturally imbalanced which should be experimented with. If not, the scope of the paper should be adjusted accordingly.

2. The paper targets model overfitting as a resultant of imbalance in real-world scenarios. However, no methods combatting challenges in longtail imbalance are cited or contrasted. Eg. GLMC (Du etal., 2023), PaCo (Cui etal., 2021) etc. 

3. The clarity in the figures are very poor in the current version - The included text in Fig. 1, mathematical equations in Fig. 2 etc. are barely visible. In Fig. 3 and Fig. 4, the color scheme should be chosen such that each instance is distinguishable.

4. In Fig.3 the goal is to contrast MiL against other learning strategies however MiL performance numbers are placed as a separate figure (per dataset) which makes it hard to compare. I would suggest putting this figure as a horizontal one covering the full textwidth to improve clarity. Table 2 and Fig. 3 seem to convey the same metrics which can be combined.

5. In lines 301 - 302 the reference to the equation is missing.

### Questions
1. A specific LHI threshold of 0.80 is proposed to identify a task as an LH-P. The paper would benefit from a more detailed justification or sensitivity analysis for this specific value to show it is not arbitrary.

2. Is there a correlation between LHI and the class imbalance in real-world datasets ? 

3. As stated in lines 298-299, a label rebalancing is performed during the naive traininglet construction. Is there any justification on why a rebalancing is necessary ? 

4. All datasets adopted in the paper are classification datasets. Is the current method scalable to auxiliary tasks eg. regression, retrieval etc. ?

### Soundness
2

### Presentation
1

### Contribution
3
