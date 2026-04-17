# Low-rank Interpretable Cell–Cell Hidden Interactions from Embeddings

- Decision: Reject
- Scores: 2, 8, 8

## Abstract
Multicellular organisms rely on continuously changing cell–cell interactions that govern critical biological processes as cells modify their internal states and trajectories in space over time. Studying these interactions is critical to understand development, homeostasis, and disease progression. Live-cell imaging provides a unique opportunity to directly observe these dynamical events; however, current computational approaches often fail to model complex, time-varying events involving diverse populations and spatial contexts. Here, we present LICCHIE, a model designed to infer time-changing, feature-based cell-cell interactions, applicable across systems and conditions. Our approach represents each cell with a dynamic multi-feature vector, and interactions are modeled as spatially constrained, directed influences between cell pairs, evolving over time. We optimize the model using an iterative scheme balancing data fidelity, interactions smoothness, and low-rank sparse structure. We validated LICCHIE’s ability to capture cellular interactions across populations in a controlled synthetic setting and applied it to real-world 3D live-cell imaging of patient-derived tumor organoids to (1) identify components with interpretable structures that capture interaction type and directionality, and (2) suggest modulation strategies that may accelerate Natural Killer (NK) polarization and tumor cell death.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LICCHIE — a low-rank dynamic model designed to infer time-varying cell–cell interactions” from live-cell imaging data. Each cell is represented as a temporal feature vector derived from morphological and spatial descriptors, and pairwise influences between cells are modeled as time-dependent linear transformations. These interactions are constrained to be low-rank, sparse, and temporally smooth, aiming to balance interpretability and expressiveness. The authors validate LICCHIE on synthetic datasets and apply it to live-cell microscopy of tumor–NK co-cultures, claiming that the learned interaction matrices reveal biologically meaningful influence patterns and potential regulatory mechanisms.

### Strengths
**Readable presentation:**

The manuscript is well-written and clearly organized, with intuitive figures that make the mathematical setup easy to follow even for non-domain experts.

**Mathematical clarity:**

 The formulation is compact and well-structured, combining ideas from linear dynamical systems, low-rank decomposition, and temporal smoothness regularization in an interpretable way.

### Weaknesses
**Conceptual and biological validity concern**

The paper defines cell–cell interactions from live-cell imaging sequences, yet the experimental setup appears to involve isolated cells or loosely adherent co-cultures rather than structured tissue or organoid environments.  In such conditions, cells are not embedded within a continuous microenvironment, and long-range or contact-based signaling is largely absent.  Therefore, what the model captures is more accurately morphological co-variation or motion correlation among nearby cells, rather than bona-fide cell–cell communication.  This distinction should be made explicit, as it fundamentally affects the biological interpretation of the inferred interaction matrices. 

**Lack of molecular or mechanistic validation**

 The inferred interaction matrices are not validated against known ligand–receptor signaling pathways, transcriptional profiles, or perturbation responses.  Without molecular or experimental corroboration, it remains unclear whether the model captures meaningful communication or merely statistical dependencies.  This weakens the biological interpretability of the proposed “interaction components.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents LICCHIE, a low-rank, interpretable framework for modelling cell-cell interactions from live-cell imaging data.  It represents pairwise influences between cells as combinations of shared rank-1 “interaction motifs,” enabling biologically interpretable analysis of how cellular features dynamically affect one another over time.

### Strengths
- The problem is very well articulated and follows a very coherent flow. 
- The approach is biologically grounded and is well-motivated at every architectural component.

### Weaknesses
- In the abstract, NK is not defined. 
- Ambiguity in biological grounding and temporal consistency: The decision to avoid explicit cell tracking and instead operate purely in “feature space” is only loosely justified in Section 3. Without clear temporal correspondence between specific cells across frames, it is uncertain how the model distinguishes genuine dynamic interactions from coincidental correlations among transient feature observations.
- Lack of clarity in feature-space formulation and identity handling: While the authors justify avoiding explicit cell tracking by operating in “feature space,” it remains unclear how temporal consistency is maintained when cell identities are not preserved. Without explicit linking across frames, it is ambiguous how the model distinguishes genuine temporal evolution of a single cell from population-level variability or measurement noise.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper develops LICCHIE, a model for inferring cell-cell interactions from live imaging of biological cell data. The model learns time-varying interaction matrices that are constrained by spatial proximity and a low-rank regularization. Optimization is performed by iterative linear regression and canonical polyadic tensor decomposition (PARAFAC).

### Strengths
1.	The formulation of the biological problem is, to my knowledge, unique. I have not seen any papers tackling this problem previously. This is an important biological problem, however, and an initial approach to solving it is an important contribution.
2.	The model formulation is sensible and parsimonious, but also interesting from an ML perspective.
3.	The biological insights gained from running the model on real data are quite interesting. It seems that this method provides new important new insights into these kinds of live-cell imaging datasets.
4.	Evaluations using simulated data show that the method accurately recovers ground truth interactions compared to baseline models.

### Weaknesses
1.	Baselines used for simulated data comparison are a bit simplistic. I understand that there are not really competing methods to compare with here. But, it seems that you could for example ablate your model more to understand which parts of the objective are most important for strong performance.
2.	Comparisons with baselines on the real data are missing.

### Questions
1.	Using a tensor decomposition here is an interesting approach. You imposed a low-rank constraint to obtain rank-1 components. Does it make biological sense to consider a tensor decomposition with a core tensor instead (Tucker decomposition)? This could give a different perspective on the structure of the cell interactions.
2.	The paper mentions cell state several times, but this seems largely aspirational, unless I misunderstood the details of the real dataset. Can you clarify what “cell state” means in the real data application presented here? If you’re not modeling it here, what sort of data would you need to do true cell state modeling?

### Soundness
3

### Presentation
3

### Contribution
4
