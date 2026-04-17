# ​ImmunoGraph: Accelerated and Equitable Representation Learning for Large-Scale Immune Networks​

- Decision: Reject
- Scores: 4, 2, 0

## Abstract
Comparative analysis of adaptive immune repertoires at population scale is hampered by two practical bottlenecks: the near-quadratic cost of pairwise affinity evaluations and dataset imbalances that obscure clinically important minority clonotypes. We introduce ImmunoGraph, an end-to-end pipeline that addresses these challenges by combining antigen-aware, near-subquadratic retrieval with GPU-accelerated affinity kernels, learned multimodal fusion, and fairness-constrained clustering. The system employs compact MinHash prefiltering to sharply reduce candidate comparisons, a differentiable gating module that adaptively weights complementary alignment and embedding channels on a per-pair basis, and an automated calibration routine that enforces proportional representation of rare antigen-specific subgroups. On large viral and tumor repertoires ImmunoGraph achieves measured gains in throughput and peak memory usage while preserving or improving recall@k, cluster purity, and subgroup equity. By co-designing indexing, similarity fusion, and equity-aware objectives, ImmunoGraph offers a scalable, bias-aware platform for repertoire mining and downstream translational tasks such as vaccine target prioritization and biomarker discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors claim that a core computational bottleneck in population-scale immune repertoire analysis is the pairwise similarity computation, which grows quadratically with the number of receptor sequences. Despite prior efforts in subquadratic similarity computation, the authors identify three practical gaps: (1) many high throughput systems overlook domain constraints important for biological validity, (2) there exist systematic omission of low prevalence but biologically important antigenic classes, and (3) existing work have reproducibility issues due to omitting configurations. To address these issues, the authors propose ImmunoGraph, a pipeline that combines antigen-aware, near-subquadratic retrieval with GPU-accelerated similarity kernels, learned multimodal fusion of alignment and embedding signals, as well as fairness-constrained clustering. ImmunoGraph achieves superior performance at higher throughput compared to existing methods for immune repertoire analysis.

### Strengths
1. The appendix contains fairly rich results, which is a pleasant surprise given the less exciting main text. In return, this could be considered a “weakness” on the presentation.
2. The performance of ImmunoGraph in Table 1 looks compelling.
3. The ablation study (Table 2) provides good justification of the architectural components, although it’s not presented in the best form.
4. The (theoretical) time complexity assessment is helpful and complements the (empirical) throughput results.

### Weaknesses
I do not seem to fully understand the big picture after reading this paper. 

When the authors mentioned "immune repertoire analysis", my impression is comparing multiple repertoires of different individuals so that we can understand how different sets of BCR/TCR determine whether a person experiences immune response. But the results in Table 1 is performed on 10K sequences, which suggests that this is still a subset of a single repertoire. I would like to hear some clarification by the authors on what machine learning task they are doing, detailed in the next bullet point.
1. What graphs are being constructed? What are the nodes and edges and what do they represent?
2. What tasks are being performed (graph classification, node classification, link prediction, or not graph tasks at all)? What are the input and output?

Another major issue is that the presentation of this paper is undermining the contributions of this work. I wonder if the authors were writing this paper in the last minute. I will go over the major issues below.

3. With the entire suite of architectural components, the paper will benefit from a clear and illustrative architecture figure in the main text (I am looking for something much better than Figure 3 in appendix, since that figure is very text-dense and uninformative).
4. The figures in the main text are subpar. They are creating a lot of unnecessary white space, and they are very casually made without good considerations on background, color, text size, spatial organization, etc.
5. Some figures in the appendix (for example, Figure 7) deserve to appear in the main text more so than Figures 1 and 2.
6. The tables can be largely improved too. In Table 1 and 3, if you allow more width on the first column, it can immediately be better looking and taking less space. In Table 2, the ablation study can be a lot easier to follow if which components are included/excluded can be better shown: if they are added/removed one by one, make sure that is clearly communicated; if there are multiple combinations of components, show that in a configuration grid with check marks. Table 4 is a bit unnecessary given the low information density.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a retrieval approach for antigen-specific T-cell receptors (TCRs) by adapting the MinHash algorithm to account for amino acid sequence similarity and TCRs that may be underrepresented in population samples. Experimental results on 10K sequences demonstrate competitive performance relative to baselines in terms of retrieval and computational efficiency.

### Strengths
The paper focuses on representation learning of TCR repertoires, which is an important problem.

### Weaknesses
The paper is poorly written. It's unclear what problem the paper is trying to address. While the paper describes an antigen-aware MinHash algorithm for retrieval tasks, the input and output of the proposed approach are not clearly defined. Additionally, the methodology of the proposed approach (Section 3) is not self-contained. The proposed approach doesn't seem well-motivated compared to alternative approaches. Most of the model specifications are undefined, e.g.: i) Eqn 1: What are X and Y? and ii) Eqn 2: What is the MetaNet?

Given the lack of clear descriptions of the actual problem, the rest of the paper reads like a technical report that is an amalgamation of various techniques without a clear connection or motivation. The experimental results are also underwhelming. Tables and figures are presented without a clear description of the experimental setup or analysis of the results.

### Questions
I encourage the authors to provide a clear motivation for the problem, including what the inputs and outputs are. What are the challenges, and how is the proposed approach suited to solve the problem? Given the limited methodological aspects, additional extensive and clearly motivated experiments would also strengthen the submission.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
My best guess at what this paper is trying to do is to create a graph from immune repertoires in a computationally efficient fashion. It should be mentioned that the paper and abstract are rife with meaningless buzz words and that first, the authors never define what an immune repertoire is and what the principle of comparing two immune repertoires would be (what kind of a distance would be appropriate). Without these basics it is very hard to understand this paper.

### Strengths
The problem seems to be compelling. The immune repertoire is the complete set of all unique T-cell receptors (TCRs) and B-cell receptors (BCRs) within an individual's adaptive immune system. T cells can generate on the order of 10^18 unique receptors in a human body, B cells generate around 10^3. Comparing such vast sets of sequences is a very difficult domain specific distribution comparison or optimal transport problem. But this is not explained by the authors at all.

### Weaknesses
The problem is never defined.  What a immune repertoire is---a basic definition--- is never given in this paper. If it was given it would become obvioius that this is a problem of comparing on the order of 10^18 sequences between people. This is a highly complex task which likely requires advanced extensions of distribution distance or discrepancy methods. None of this is addressed in this paper. Section 3.5 on graph construction simply refers to a "similarity matrix", how do you deem that two sets of repertoires are similar, not pairs of sequences! On the other hand maybe the authors are trying to create a graph from a single repertoire by comparing pairs of sequences, but what is done downstream from this? How is this useful? 

Overall i think this paper is highly confusing, misleading and rife with strange buzz words. If this is to be a computational contribution then I would start with the basic problem you are solving and put it in a mathematical form and then explain the techniques that are being used.

### Questions
What exactly is the problem you are solving? 

Why is it useful?

### Soundness
1

### Presentation
1

### Contribution
1
