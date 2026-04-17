# Quantum-inspired benchmark for Intrinsic Dimension Estimation

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Machine learning models can generalize well on real-world datasets. According to the manifold hypothesis, this is possible because datasets lie on a latent manifold with small intrinsic dimension (ID). There exist many methods for ID estimation (IDE), but their estimates vary substantially.  This warrants benchmarking IDE methods on manifolds that are more complex than those in existing benchmarks. We propose a Quantum-Inspired Intrinsic-dimension Estimation (QuIIEst) benchmark consisting of infinite families of topologically non-trivial manifolds with known ID.  Our benchmark stems from a quantum-optical method of embedding arbitrary homogeneous spaces while allowing for curvature modification and additive noise. The IDE methods tested were generally less accurate on QuIIEst manifolds than on existing benchmarks under identical resource allocation. We also observe minimal performance degradation with increasingly non-uniform curvature, underscoring the benchmark’s inherent difficulty.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper discusses a quantum-inspired algorithm for
intrinsic dimension estimation (IDE) in the context of the "manifold
hypothesis" on the set of data. This is an interesting and important
question.
The idea is to constructs infinite
families of manifolds starting from classical homogeneous spaces
as quotients of semisimple real or complex Lie groups (e.g. spheres,
grassmannians, flags, Stiefel manifolds etc). The authors start from what
they call "quantum optical embeddings" of homogeneous spaces, though they
do not explain and I do not understand the word quantum used in this context.
Starting with the ground truth of ID, from the manifolds (datasets) they
construct, then they proceed to give IDE with their method QuIIEst and
then they benchmark it against more known methods, showing its potential.
They also add some noise and make considerations regarding distortion
and possible topological complexity.

### Strengths
As far as I know this idea is new, though I do not understand
the quantum component of it, I think it is worth exploring and can
be of aid in IDE, especially when topology, curvature and more generally
geometry plays a role in data distribution. Moreover the authors
are able to provide a playground for others with "toy datasets" that
may be of help in answering other questions (eg. knowledge transfer).
Their results show performance degradation of existing IDE methods,
validating their QuIIEst model and method.

### Weaknesses
There is the surprising fact that the geometry of the manifold
(curvature, density etc) plays no role: actually this should be better
motivated, because it may hide a shortcoming in the theoretical treatment.
Moreover a comparison on this point with other IDE methods is mandatory.

The paper shows that IDE methods perform worse on QuIIEst
but does not deeply analyze why, eg.
which geometric/topological/other features cause errors (link with
previous weakness).

The experiments appear very limited in scope, though there may
be promising as scalability goes.

At some point code and generation scripts must be released for
reproducibility.

### Questions
How exactly are the “quantum embeddings” implemented? This
part is not fully clear, it would benefit from more details.

Can QuIIEst generate non-homogeneous manifolds? Can their method
work in reducing data parameters and dimensionality (as for any
homogeneous space a group action can be of help).

Are there known analytic expressions for curvature and metric tensors
in these embeddings and after distorsion? (this is actually an
interesting mathematical question, how geometry changes with noise
addition).

How computationally expensive is manifold generation as ID increases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work focuses on building benchmarks for intrinsic dimension (ID) estimation. Pointing on that usual benchmarks for ID estimation are simple, the authors build a set of more complex physics-inspired manifolds. Building off of some tractable, well-understood manifolds (e.g., Stiefel manifolds, Grassmanians), the authors apply different embeddings of these classes into higher-dimensional spaces and put these forward as an ID benchmark. A suite of ID methods is evaluated on these benchmarks.

### Strengths
- The work is well-motivated - it is true that current benchmarks focus on very trivial manifolds that have no relationship to real-world data (though I'm not sure those proposed here have much relationship either - see weaknesses).
- This paper is well-written and I enjoyed reading it. This may be strange to say in a review, but the authors' passion for the subject really shines through in the level of detail, connections, and examples they give throughout.
- I do think the experiments are executed nicely. The analysis is unique and interesting, comparing performance to properties like curvature and noise (in addition to ID).

For these reasons, I am opting for an accept score.

### Weaknesses
- It's not clear at all whether these benchmarks are representative of real-world ID estimation problems. The work briefly mentions how Grassmanians and Stiefel manifolds have been studied in ML, but presumably not in a context when their IDs were unknown and needed to be estimated.
- Although some interesting quantitative analysis is performed showing *when* estimators fail, no insight is provided into *why* these failure cases exist on a per-estimator basis. I don't think I found in the work any suggestions on how to improve ID estimators on the basis of this benchmarking work either. (The notion of generating different embeddings of real-world manifolds is mentioned in the conclusion, but it's not at all obvious what this would entail.)
- Most benchmark manifolds are constructed using the "Gilmore-Perelomov" coherent-state method. But there is no accessible explanation of this concept for a general ML audience. There is a highly technical description in Appendix D.1 which I cannot follow. All I can glean is that it is a piece of math used somewhere in the process of generating embeddings.
	- You need to provide more accessible details about this. Is it or does it provide a function of some sort? Surely you can black-box most of the technical details while providing us with a precise description of its purpose here.

Side note: the citation style violates the ICLR 2026 formatting instructions: https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip

### Questions
- Can you give a less technical explanation of what you are doing with the coherent state method?

### Soundness
4

### Presentation
4

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
This paper introduces QuIIEst (Quantum-Inspired Intrinsic-Dimension Estimation benchmark), a benchmark designed to evaluate intrinsic dimension estimation (IDE) methods on synthetic manifolds with known intrinsic dimensions. The authors construct several manifold families, including Stiefel, Grassmann, Flag, and Pauli manifolds, by leveraging the structure of homogeneous spaces (quotients G/H). The benchmark allows controlled sampling and noise perturbation, and the paper compares multiple IDE methods such as PCA, MLE, TwoNN, DANCo, ABID, and CorrInt on these manifolds.

### Strengths
- The benchmark provides a consistent way to evaluate intrinsic dimension estimators on known manifolds.
- The experimental section is well-organized, comparing several established IDE methods under controlled settings and providing clear quantitative results.

### Weaknesses
- Fundamentally, the work is unrelated to quantum mechanics. All constructions are standard results from Lie group representation theory and homogeneous space geometry. The “quantum-inspired” terminology appears unnecessary, as the methodology is fully explainable within classical differential geometry. To me, the quantum story is merely a coverage.  
- All manifolds used in the benchmark are homogeneous manifolds such as Stiefel, Grassmann, and Flag manifolds generated by Lie group actions. Non-homogeneous manifolds are not included. This limits the benchmark’s ability to assess IDE methods on more realistic and asymmetric data manifolds.  
- The description of the quantum background, especially in Appendix D, is insufficient and lacks rigor. Several notations are either undefined or unclear.  

Overall, the paper mainly reuses known results on homogeneous manifolds to generate a dataset for IDE evaluation. The datasets comprise existing matrix manifolds. While the benchmark may be useful in practice, its conceptual novelty is limited.

### Questions
see wk

### Soundness
2

### Presentation
2

### Contribution
2
