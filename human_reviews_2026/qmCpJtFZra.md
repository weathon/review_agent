# Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 6, 4, 8, 10

## Abstract
Protein interaction modeling is central to protein design, which has been transformed by machine learning with applications in drug discovery and beyond. In this landscape, structure-based de novo binder design is cast as either conditional generative modeling or sequence optimization via structure predictors (``hallucination''). We argue that this is a false dichotomy and propose Proteina-Complexa, a novel fully atomistic binder generation method unifying both paradigms. We extend recent flow-based latent protein generation architectures and leverage the domain-domain interactions of monomeric computationally predicted protein structures to construct Teddymer, a new large-scale dataset of synthetic binder-target pairs for pretraining. Combined with high-quality experimental multimers, this enables training a strong base model. We then perform inference-time optimization with this generative prior, unifying the strengths of previously distinct generative and hallucination methods. Proteina-Complexa sets a new state of the art in computational binder design benchmarks: it delivers markedly higher in-silico success rates than existing generative approaches, and our novel test-time optimization strategies greatly outperform previous hallucination methods under normalized compute budgets. We also demonstrate interface hydrogen bond optimization, fold class-guided binder generation, and extensions to small molecule targets and enzyme design tasks, again surpassing prior methods. Code, models and new data will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Complexa, a novel framework for fully atomistic protein binder design that aims to unify two dominant but previously separate paradigms: generative modeling (like RFDiffusion) and sequence "hallucination" (like BindCraft). The authors argue that this dichotomy is false and that performance can be maximized by combining a strong generative prior with scalable inference-time optimization.

To train a strong base model, the paper makes two key contributions. First, it introduces Teddymer, a new, large-scale dataset of synthetic binder-target pairs, cleverly constructed by splitting predicted multi-domain monomer structures from the AlphaFold Database into interacting domain pairs. Second, it extends a state-of-the-art flow-matching generative model (La-Proteína) to be conditional on a target's structure, pretraining it on monomers and the new Teddymer dataset.

The framework's main novelty is in its use of this trained generative model as a prior for inference-time optimization. The authors adapt multiple test-time search algorithms (e.g., Best-of-N, Beam Search, MCTS) to steer the generative process toward high-quality binders, using scores from external structure predictors (like AlphaFold2) as rewards.

Complexa achieves new state-of-the-art results on computational binder design benchmarks. Its base model outperforms prior generative methods, and its inference-time optimization significantly outperforms prior hallucination methods under normalized compute budgets. The authors also demonstrate the framework's flexibility by extending it to small molecule targets, optimizing for interface hydrogen bonds, and guiding generation by protein fold class.

### Strengths
1. Solves a Key Data Bottleneck: The Teddymer dataset is a very clever and practical solution to the lack of large-scale binder-target complex data. The ablation study proves it is essential to the model's success.

2. Extremely Strong Empirical Results: The paper demonstrates clear and significant superiority over existing SOTA methods (RFDiffusion, BindCraft, BoltzDesign) in fair, compute-matched comparisons.

3. Thoroughness and Versatility: The authors are not satisfied with just one result. They show the method works for both protein and small-molecule targets, can optimize for physics-based properties (H-bonds), and can be controlled (fold-class conditioning). This demonstrates the framework's power and flexibility.

### Weaknesses
1. System Complexity: This is an "everything but the kitchen sink" model. It involves a VAE, a flow-matching model, and complex test-time search algorithms that in turn rely on other large models (AF2, RF3) as reward oracles. This makes the system as a whole extraordinarily complex and computationally expensive to train and run, even if it is more efficient than baselines.

2. Reliance on External Oracles: The performance of the test-time optimization is fundamentally coupled to the quality of the external structure predictors (AF2/RF3) used as reward functions. If the oracle is wrong or has blind spots, the search will be steered to exploit those flaws, not necessarily to find biologically viable binders.

3. Teddymer Interface Quality: The central assumption is that the interfaces in Teddymer (from predicted monomer domains) are good proxies for real binder-target interfaces. While Fig. 3 is a nice qualitative example, the paper would be stronger with a quantitative analysis comparing the geometric and biophysical properties of Teddymer interfaces versus a ground-truth set from the PDB.

### Questions
1. The ablation study for "Generate & Hallucinate" (G&H) in Fig. 12 shows that this simpler approach is less effective than the fully integrated search methods (like Beam Search). Why do the authors think this is? Is it because the search methods can correct the generative trajectory early and often, whereas G&H can only fix a single complete, and potentially flawed, sample at the end?

2. The "translation noise" ablation (Table 7) shows it is critical for performance. The hypothesis is that this forces the model to learn global positioning. Is it also possible that without this noise, the model simply overfits to the "centered target" convention used in training, and the noise acts as a data augmentation/regularization? Could the authors comment on this alternative interpretation?

3. For the "Generate & Hallucinate" (G&H) experiments (Sec I.3), the paper ablates using BindCraft stages 2+3+4 vs. stage 4 only. Why does the simpler "stage 4 only" (discrete optimization) perform better on easy targets? One might expect the gradient-based logit optimization (stages 2+3) to be more powerful.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Complexa, a generative framework for fully atomistic protein binder design that claims to unify generative and hallucination-based paradigms. The model builds upon La-Proteína (Anonymous, 2025) using a partially latent flow-matching architecture and introduces (1) a new synthetic binder–target dataset called Teddymer derived from domain–domain interactions in AlphaFold Database monomers, and (2) test-time optimization strategies such as beam search, Feynman–Kac steering, and Monte Carlo Tree Search. The authors argue that this unification improves binding success rates under normalized compute budgets.

### Strengths
The paper provides a comprehensive engineering system integrating generative pretraining, dataset synthesis, and inference-time optimization.

The construction of Teddymer—though synthetic—could be a useful large-scale dataset for structural learning.

The writing is technically detailed and clear, with ablation studies and code-release commitment.

The topic of scaling test-time compute for binder design is timely and of interest to the ICLR community.

### Weaknesses
Unclear contribution beyond La-Proteína.  
The proposed architecture directly extends La-Proteína’s partially latent flow-matching model with an added conditioning token for target residues. While the authors emphasize “unifying generative and hallucination methods,” this mainly translates into using La-Proteína’s backbone plus inference-time sampling and optimization (beam search, MCTS) already well-known from diffusion models. The methodological novelty appears incremental, not conceptual. It is unclear what fundamentally distinguishes Complexa from La-Proteína aside from adding conditioning and test-time heuristics.

No rigorous comparison with latest state-of-the-art works.  
Despite citing BoltzDesign (Cho et al., 2025) and BindCraft (Pacesa et al., 2025) in the related-work section, the experiments do not include direct quantitative or qualitative comparisons with these models under standardized benchmarks. These are the leading methods for atomistic binder design that already implement optimization over structure predictors and incorporate differentiable docking. Without such baselines, it is impossible to judge the claimed “state-of-the-art” performance.

Questionable novelty of “test-time scaling.”  
The adaptation of test-time compute scaling (Best-of-N, Beam Search, MCTS) is straightforward and mirrors what is standard in diffusion and flow-based generative modeling in language, vision, and molecule generation. There is no specific algorithmic innovation tailored to protein structures, nor theoretical analysis of why such scaling improves binder discovery.

Marginal improvement and unclear biological impact.    
Reported “unique success” metrics show numerical gains over older baselines such as RFDiffusion or APM, but do not exceed or even match the scale and biochemical relevance achieved by recent multimodal systems like AlphaFold 3, Boltz-2, or Chai-2. There is no wet-lab validation, no docking energy correlation, and no experimental evidence that the generated binders are meaningful beyond AlphaFold confidence metrics.

Ambiguous conceptual framing.  
The paper repeatedly emphasizes “bridging generative and hallucination approaches,” yet the implementation merely combines a pretrained generator with structure-score-guided search. This framing risks overstating what is essentially a hybrid of established techniques. Moreover, the term “scaling atomistic protein binder design” is misleading—there is no clear demonstration of scaling laws, compute efficiency, or emergent capability analyses.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Complexa, a partially latent flow-matching framework for binder generation across protein and small-molecule targets. To enlarge the training dataset, this paper also introduces Teddymer, a newly constructed dimer dataset derived from TED with complete CATH annotations. The authors also design and analyze various test-time scaling strategies to improve generative performance. Experimental results on multiple datasets demonstrate the method’s strong capability and generalization in protein–ligand and protein–protein binding tasks.

### Strengths
1. This paper contributes a valuable dataset named Teddymer, which can serve as high-quality binder training data and benefit future studies.

2. The proposed Complexa framework leverages La-Proteína, a partially latent flow matching framework, for binder generation, extending the scope of the original model.

3. This paper thoroughly analyzes test-time scaling strategies, opening an advanced and promising direction.

4. The experimental evaluation is extensive, covering diverse scenarios with clear explanations and detailed ablations.

### Weaknesses
While the technical contributions are solid, this paper contains several typos or inconsistencies that slightly affect readability. Specific examples include:

- Line 292: interference-time -> inference-time
- Inconsistent alternates between "CATH annotation" and "CAT labels"
- Line 1633: MTCD -> MTCS

Moreover, as this paper introduces a relatively complicated framework, the reviewer recommend adding an overview paragraph to go through the entire pipeline at the beginning of Section 3, including the data construction, latent flow matching framrwork, and test-time scaling strategies.

### Questions
1. There is a recent work [A] attempting to solve both small-molecule and protein binder generation within a single model. Within Complexa’s partially latent formulation, could a single shared model cover both cases? A brief discussion of this possibility would strengthen the paper’s positioning.
2. For very hard cases, it would be informative to include scaling curves similar to Figure 13/14 to illustrate the generation difficulty and scaling behavior quantitatively.

[A] Kong, Xiangzhe, et al. "UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design." Forty-second International Conference on Machine Learning.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This work introduces Complexa, which extends La-Proteina for binder design. Authors introduce the Teddymer dataset of synthetic protein dimer pairs: using annotations from TED, domain-domain interactions are retrieved from AFDB. For base model training, the partially latent La-Protein model is extended by representing targets as atom37 features, alongside sequence identity and binary hotspot tokens. A translation noise objective is used. At inference time, the model is steered using interface confidence scores from structure predictors or h-bond energies as rewards, with hotspots pre-specified. Authors experiment with different techniques (beam search, MCTS, etc.) Baselines are examined for both protein binders and small molecules.

### Strengths
Really nice exploration of inference-time steering strategies. Conditioning on both small molecule and protein binders is a great framing. Rewards are very sensibly chosen. The inference-time optimization analyses, complete with the inference-time compute scaling plots and compute normalized comparisons, will be a great addition to the literature. Not requiring explicit sequence re-design is elegant, and it’s great to set this new standard for the field. It’s great to see that the authors did not cut corners by choosing an easy task or weak baselines. Technically strong paper with strong presentation and very comprehensive appendix. This is a polished work and should be highlighted at the conference.

### Weaknesses
* The Teddymer dataset is quite sensible given the data-limited regime in biology. I do wonder if recapitulating the inter-domain distribution will be as useful for real-world uses such as mini-binder design. It also means that we are again massively overrelying on AlphaFold both for distilling the training data, post-training, and for evaluation. This could conflate evaluation numbers, especially for anything that still relies on AF2, and any biases stemming from AF2 would not be picked up until there are wet-lab (or other orthogonal) validations
* As with any optimization procedure, we’re beholden to the quality of the reward signal. ipAE is not a perfect signal, though I appreciate it’s one of the best solutions available.

### Questions
* Not a necessary additional experiment per se, but I’m curious if the inference search strategies are helping move us away from the pretraining distribution to something better, or if it’s helping us better interpolate within the pretraining distribution? Curious to hear the authors’ intuitions on this.
* It looks, from Figure 7, that the metrics continue to increase with GPU hours. What if we ran one of the methods for even longer? Do we expect to see an eventual plateau? Could this possibly imply reward hacking, given that we don’t see a plateau in the current plot? 
* When optimizing for ipAE, I’m curious if authors saw any signs of adversarial optimization. What if we plotted orthogonal metrics (e.g. Rosetta energy or h-bond counts) on the x axis with ipAE on the y axis for all generations? Do these metrics correlate well?

### Soundness
4

### Presentation
4

### Contribution
4
