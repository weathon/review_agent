# MarS-FM: Generative Modeling of Molecular Dynamics via Markov State Models

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Molecular Dynamics (MD) is a powerful computational microscope for probing protein functions. However, the need for fine-grained integration and the long timescales of biomolecular events make MD computationally expensive. To address this, several generative models have been proposed to generate surrogate trajectories at lower cost. Yet, these models typically learn a fixed-lag transition density, causing the training signal to be dominated by frequent but uninformative transitions. We introduce a new class of generative models, 
**MSM Emulators**, which instead learn to sample transitions across discrete states defined by an underlying Markov State Model (MSM). We instantiate this class with Markov Space Flow Matching (MarS-FM), whose sampling offers more than two orders of magnitude speedup compared to implicit- or explicit-solvent MD simulations. We benchmark Mars-FM ability to reproduce MD statistics through structural observables such as RMSD, radius of gyration, and secondary structure content. Our evaluation spans protein domains (up to 500 residues) with significant chemical and structural diversity, including unfolding events, and enforces strict sequence dissimilarity between training and test sets to assess generalization. Across all metrics, MarS-FM outperforms existing methods, often by a substantial margin.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a new class of generative models called MSM Emulators, which is designed to overcome the issue in previous approaches where learning transition densities with fixed time steps may cause the training signal to be dominated by frequent but uninformative transitions. During training, the proposed model, dubbed as MarS-FM, leverages flow matching to learn state transitions defined by the discrete transition matrix of preconstructed Markov State Models (MSMs) rather than those based on fixed time steps, thereby enhancing its ability to cross energy barriers and access different macroscopic states. Experiments on the tetrapeptide and MD-CATH datasets demonstrate that MarS-FM outperforms existing methods across all metrics by a substantial margin, while exhibiting a large improvement of sampling from rare metastable states.

### Strengths
- This paper presents clear motivation and well-structured logic. Aiming to address the limitation of existing methods that learn transition densities at fixed time steps, it proposes a generative model based on the Markov State Model (MSM) that directly learns state transitions from the probabilistic transition matrix, enabling the model to cross energy barriers and access different macroscopic states.
- Extensive evaluations on the tetrapeptide and MD-CATH datasets have been conducted. The results demonstrate the model’s out-of-distribution (OOD) generalization to unseen biomolecular systems and its substantially superior ability to access diverse metastable states compared with existing methods.
- The training objective is simple and effective, making the method easy to reproduce. Remarkably, even with a single objective, the model achieves impressive experimental results, demonstrating the effectiveness of the proposed approach.

### Weaknesses
- As a comparison to existing models that learn transition densities at fixed time steps, MarS-FM appears to lack an analysis of dynamical observables for the generated trajectories, such as the decorrelation time of torsion angles. Evaluating the model solely using thermodynamic metrics does not allow one to distinguish it from an unconditional generative model that targets the Boltzmann ensemble, and therefore fails to highlight the advantages of the conditional generation approach.

### Questions
1. Equation (5) presents the flow-matching-based training objective, but I would like to know more about certain training details. For two macroscopic states $i$ and $j$ with corresponding samples $x(t)$ and $x_1$, the conformational difference between them may be substantial, making the conditioning on $x(t)$ potentially uninformative for generating $x_1$. Would this increase the training difficulty, and does the conditioning network indeed provide effective guidance in such cases?
2. Lines 264–278 describe two generation strategies built on MarS-FM. I would like to know whether, and under what conditions, each strategy can unbiasedly reproduce the Boltzmann distribution, which is crucial for MD simulations.
3. The Tree sampling strategy mentions the number of samples $n$ generated in parallel at each iteration. I would like to understand how the choice of $n$ affects the resulting ensemble under a fixed sampling budget. Intuitively, a larger $n$ may concentrate generated conformations within local states and miss global diversity, whereas a smaller $n$ may explore macroscopic states more broadly but lack local detail. The authors are encouraged to provide an ablation study on $n$ and investigate whether there exists a hyperparameter setting that achieves an optimal trade-off between local accuracy and global coverage.
4. As mentioned in the Weaknesses section, the authors could introduce dynamical metrics for further evaluation to verify whether MarS-FM can capture the true dynamical behavior of the MD process.

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
This paper introduces a novel strategy to improve modeling of molecular dynamics (MD) based on the introduction of a Markov State Model to better represent state transitions at training time. The paper introduces MarS-FM as a model in this space based on Flow-Matching. MarS-FM  sampled molecular trajectories are compared against samples from MDGen and traditional MD simulations to demonstrate that the proposed method can achieve superior fidelity, even when unfolding simulations of molecules not observed during training.

### Strengths
* The paper tackles a practical problem in molecular modeling, demonstrating the effectiveness of the proposed MarS-FM solution against a relevant method in literature. 

* The paper reports convincing results on a large set of peptides demonstrating the generalization capabilities of MarS-FM on a disjoint set of trajectories using a wide range of metrics. 

* The paper is overall well-motivated and effectively explains limitation of existing MD Emulators with clear illustrations.

### Weaknesses
* The key part of the method and its training objective is quite compressed in a few paragraphs in section 4.1. Without any additional pseudocode, illustrations, and references, the details are quite hard to follow. 
   * The notation for the noise sample $x_0$, initial $x(t)$, next $x_1$ and interpolated $x_s$ frames is quite confusing. The time index $t$ is dropped in $x_1$ (and $x_s$), which should depend on $t$ and $\tau$.  

   * The paper reports that $x_1$ is sampled uniformly withing the state $S_j$. In practice, I expect that the density $p(x_1|S_j)$ is far from uniform on its domain. Please clarify the distinction between the empirical and modeled density in this paragraph. 
   
    * A methodology comparison with MDGen is mentioned but not fully detailed. The method and background sections describe MSM constructions and input representations but the notation and flow-matching objective are hardly introduced, which makes following the objective in equation 5 more challenging. 

* The paper reports extensive empirical comparisons with MDGen but limited or no ablation on the effect of the modeling choices such as the lag time $\tau$ or number of MSM states. The reported metrics focus on statistics of the equilibrium distribution, but no comparison or ablation evaluates the fidelity of the transitions learned by MarS-FM.

### Questions
1. Section 4.1 reports that the MSM construction is performed once per dataset. Does this mean that different molecules share the same state space or that only the TICA projection are performed on the whole dataset but the clustering and transition modeling are molecule-specific? In the first case, how can a small state space (e.g. 10 x 10 transitions) effectively capture the dynamics of multiple molecules?  
 

2. Can the authors clarify the sampling strategy described in lines 248-251? The paper reports that $x_1$ is sampled uniformly withing the state $S_j$. In practice, the density $p(x_1|S_j)$ is far from uniform on its domain. Do the authors sample uniformly from the data-points that are assigned to $S_j$ in the training trajectory? 

 3. How are the intervals $\tau$ picked in practice at training time? The parameter $\tau$ is implicit in $p_T(x_1|x(t))$ but it seems quite a relevant detail to balance the focus on different temporal horizons. Furthermore, this parameter may affect  the accuracy of the MSM model since a limited number of states (e.g. 10) may not be sufficient to describe smaller lag times since the transitions. How does Mars-FM work for varying $\tau$ and number of MSM states? 

 4. How does the transition distribution modeled by Mars-FM compare against MDGen or other models in literature that model a specific transition interval $\tau$? Do the conditional statistic also improve or does Mars-FM improve only in terms of the statistics of the equilibrium distribution?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors developed a model called MarS-FM for combining molecular dynamics (MD) modeling using flow matching and a traditional method markov state model. The work is build upon MDGen, and using similar dataset to test the performance of the generated MD.

### Strengths
There are not many work in machine learning conference that combines the MSM with MD generation models. This MarS-FM model sets an good example for combining traditional model and recent developments.

### Weaknesses
The benchmark is only on short trajectories, which highly limit the ability of MSM model (which desiged for long trajectories) and the application of this model.

### Questions
1. Authors should test their models on long MD trajectories including DESRES fast folding proteins and dynamic proteins (including WW domain and BPTI). Using short trajectories from MDGen is not convincing for either MSM models and conventional simulations
2. This model should tested on large proteins compared with models like AlphaFold subsampling and BioEmu. Only small protein is limited to toy datasets
3. Authors should also report multiple quality metrics including the protein validity, the bonding, the interaction distances, the Ca-Ca distances, etc.
4. MSM models are the best fit for generate protein folding trajectories and pathway between rare event. I suggest authors add several case studies on these examples.
5. A large issue for the tICA representation of conformation generation models are that incorrect conformation and unfold conformation are mixed in the same place. Authors should give multiple examples of the generated conformations based on k-means clustering, to show the possible generated incorrect conformations.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes to use Markov state models to augment the training of generative models that sample molecular dynamics (MD-Emus) by targeting the distribution of transitions as defined by the MSM transition matrix. Such models are called MSM-Emus. The method is validated to yield comparable or improved results on tetrapeptide simulations and improved results on the mdCATH dataset of protein simulations.

### Strengths
The paper poses an insightful question for training MD rollout emulators - how do we more effectively learn from MD trajectory datasets beyond simply training on observed frame pairs? The approach is sound and seems to lead to useful empirical results.

### Weaknesses
**Clarity**

My chief concern is the clarity of the writing. First, the authors write that they "learn to sample from the Markov chain transition induced by an MSM, rather than emulating the noisy temporal dynamics induced by MD." This phrasing seems to suggest that the the MSM transitions somehow differ and improve upon, by design, the dynamics in the MD. In truth, the MSMs are themselves fit to the "noisy temporal dynamics induced by MD" by enforcing coarse-graining and Markovianity approximations. Second, MSMs are not usually taken to be representations of fine-grained dynamics, yet the present model still learns full structural dynamics. Third, the authors argue that the time lag is a limitation of MD-Emus, yet MSM-Emus also have a specific lag time - that used in the construction of the MSM.

A better way to express the proposed idea is as follows: Instead of learning the transition density faithfully, MSM-Emus are trained to predict next frames as a mixture distribution corresponding to MSM states in proportion to the MSM transition probabilities. While this corrupts the underlying dynamics, it allows the model to focus on learning the state change probabilities rather than paying attention (and potentially overfitting) to the exact starting frame within the state and for rare starting states to be identified and upsampled during training. This ultimately and empirically leads to better emulation of long-timescale dynamics (in the absence of the MSM) at inference time.

**Justification for score**
I would support to accept this paper if the authors can convincingly tidy up the presentation.

### Questions
No specific questions.

### Soundness
3

### Presentation
1

### Contribution
3
