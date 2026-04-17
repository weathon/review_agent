# EnerBridge-DPO: Energy-Guided Protein Inverse Folding with Markov Bridges and Direct Preference Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Designing protein sequences with optimal energetic stability is a key challenge in protein inverse folding, as current deep learning methods are primarily trained by maximizing sequence recovery rates, often neglecting the energy of the generated sequences. This work aims to overcome this limitation by developing a model that directly generates low-energy, stable protein sequences. We propose EnerBridge-DPO, a novel inverse folding framework focused on generating low-energy, high-stability protein sequences. Our core innovation lies in: First, integrating Markov Bridges with Direct Preference Optimization (DPO), where energy-based preferences are used to fine-tune the Markov Bridge model. The Markov Bridge initiates optimization from an information-rich prior sequence, providing DPO with a pool of structurally plausible sequence candidates. Second, an explicit energy constraint loss is introduced, which enhances the energy-driven nature of DPO based on prior sequences. This enables the model to effectively learn energy representations from a wealth of prior knowledge. It can also directly predict sequence energy values, thereby capturing quantitative features of the energy landscape. Our evaluations demonstrate that EnerBridge-DPO can design protein complex sequences with lower energy while maintaining sequence recovery rates comparable to state-of-the-art models, and accurately predicts $\Delta \Delta G$ values between various sequences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
EnerBridge-DPO introduces an energy-guided protein inverse folding framework that integrates Markov Bridges with Direct Preference Optimization to generate low-energy, high-stability protein sequences while preserving structural fidelity.

### Strengths
+ Combines Markov Bridge generative modeling with Direct Preference Optimization, introducing energy-based fine-tuning into protein inverse folding for the first time.
+ Incorporates explicit energy constraints and ΔΔG prediction, aligning learned representations with biophysical energy landscapes.
+ Demonstrates lower energy, stable protein designs, and competitive recovery rates across multiple benchmarks with solid ablation analyses.

### Weaknesses
+ The model’s energy improvements rely on computational predictors (FoldX, Rosetta, BA-Cycle) without experimental or molecular dynamics confirmation.

+ DPO fine-tuning depends on precomputed or predicted energy scores, which may introduce bias and limit generalization to unseen proteins.
+ The paper lacks discussion on computational cost, hyperparameter sensitivity (e.g., β in DPO), and robustness across large or diverse protein complexes.

### Questions
1. How sensitive is EnerBridge-DPO to the β hyperparameter in the DPO term? Does higher β harm diversity?

2. Could the model generalize to de novo backbones not present in the training set?

3. Is the energy predictor differentiable and updated during DPO, or fixed as an external oracle?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EnerBridge-DPO, a protein inverse-folding framework that unifies Markov bridge generative modeling with Direct Preference Optimization (DPO) and Boltzmann-aligned energy constraints.
The method attempts to address a current inverse-folding limitation — they optimize for sequence recovery but neglect thermodynamic stability. In this paper they focus on binding free energy of protein complexes. 

EnerBridge-DPO proceeds in two stages:
Markov Bridge Pre-training: Trains AdaLN-Bias and Cross-Attention Adapter layers on top of frozen ESM2 weights to iteratively transition the predicted PiFold sequence into the true native sequence.

Bridge-DPO Fine-tuning: Fine-tunes the Markov Bridge model using DPO guided by ∆∆Gbind values for protein complexes. Here, DPO steers the bridge prediction towards a lower-energy sequence.

Experiments on BindingGym, SKEMPI, and PDB benchmarks show improved sequence recovery, lower predicted binding energies, and accurate ΔΔG prediction compared to existing baselines like ProteinMPNN and Bridge-IF.

### Strengths
Combination of Markov Bridge and DPO.
The combination of Markov bridges (for structured stochastic refinement) with DPO (for preference-based alignment) is new and elegant. The paper demonstrates a theoretically consistent formulation where probabilistic modeling (via bridge processes) and preference learning (via DPO) jointly improve protein design.

Interesting empirical results.
Ablation studies confirm the necessity of both DPO fine-tuning and energy supervision in various downstream results.

Clean mathematical formulation.
The paper provides a clear derivation of the bridge process and adapts the DPO loss to the inverse-folding domain, including justification of each term. 

Computational efficiency.
Training with only T = 25 timesteps, a cosine noise schedule, and simple Adam/Noam optimization shows practical efficiency — feasible for wider adoption.

### Weaknesses
Energy-aware learning objective.
The inclusion of a Boltzmann-aligned energy loss attempts to ground the generative process in physical thermodynamics in order to make the model interpretable and biologically relevant. However, the assumption that model probability strongly correlates with free energy is a poor decision by the author for several reasons. The biggest reason is that the datasets used for ∆∆Gbind are very noisy datasets with heterogenous analytical methods used for data collection, they often use proxies rather than actually measuring ∆∆G, these measurements are often dependent on the temperature, buffer, etc making it hard to aggregate data between different labs and proteins. 

Ablations on DPO temperature and preference data.
It is unclear how sensitive performance is to the β parameter or the construction of winner–loser pairs. More analysis here would help assess robustness.

Limited experimental scope.
Evaluation focuses mainly on sequence recovery and ΔΔGbind prediction; it would strengthen the paper to test downstream structure quality (e.g., AlphaFold2-refolded RMSD) or binding specificity. Using stability ∆∆G data would better demonstrate effectiveness of the DPO fine-tuning.

### Questions
Ambiguous difference in sequence pairs.
How many mutations are there between the positive and negative pairs in the BindingGym data used for DPO? I didn't see any stats for this in the paper. I know most of SKEMPI is single point mutations. Being able to show performance improvements for larger sequence differences (double/triple/etc mutants) compared to single point mutants might be an application that this method makes a meaningful improvement. Separating improvements for single point vs higher order mutants would improve benchmarking and evaluation of the method. 

Better downstream evaluation results are needed. 
While the method in itself (DPO + Markov bridges) is interesting, the downstream results are not very impressive, thus, questioning its utility. Table 2 shows modest improvements but the std is so large that I doubt it is statistically significant. Additionally, 3-fold cross validation is known to have serious data leakage and the performance improvements are marginal. In summary, seems like a lot of method development work for unimpressive results that are marginally better than baselines and are most likely the result of significant hyperparameter tuning. Please provide additional experiments that demonstrate true, unquestionable performance improvement in a downstream protein task. 

Generalization: 
Does this method work if you use ProteinMPNN or some other inverse folding framework? I don't see any ablations for changing the input prior sequence distribution.

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
4

### Summary
The authors propose EnerBridge- DPO, an inverse folding framework focused on generating low-energy, high-stability protein sequences. The method first pretrains a generative diffusion bridge mode that refines a structure-conditioned prior sequence from PiFold, then fine-tunes with an energy-guided DPO that prefers lower-energy sequences. An explicit energy constraint loss is introduced, compelling the model to learn and predict quantitative energy features.  The Experimental results demonstrate that EnerBridge-DPO designs protein complex sequences with lower energy compared to existing methods, while maintaining comparable sequence recovery.

### Strengths
1. The motivation is well grounded: effective sequence design should explicitly favor lower-energy sequences.

### Weaknesses
1. The methodological novelty appears very limited. The bridge-based generative component and several architectural/training choices (e.g., Markov Bridge formulation, PLM backbone with AdaLN-Bias and structural adapters, frozen base weights) closely track Bridge-IF [1], and the added DPO fine-tuning for lower energy reads as a relatively incremental extension rather than a fundamentally new framework.

2. The evaluation omits designability metrics, which are critical alongside stability/energy. Assessing only recovery/perplexity and energy leaves an incomplete picture of practical design performance. The authors should report standard designability measures (e.g., diversity, success rate under structure prediction, foldability metrics such as pLDDT/TM-score distributions for generated sequences) to substantiate claims about usable sequence design.

3. The ΔΔG prediction gains over BA-DDG in Table 3 are modest and appear incremental.

### Questions
For ΔΔG prediction, what fraction of the SKEMPI pairs overlap structurally or sequentially with training data used for pretraining or DPO?

### Soundness
2

### Presentation
2

### Contribution
1
