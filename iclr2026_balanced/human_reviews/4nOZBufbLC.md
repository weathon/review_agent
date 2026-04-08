## Human Reviewer 1

### Summary
This submission proposes Count Bridges, a novel approach to finding a probability path between end-marginal distributions on integer values. This is partly achieved by modeling what resembles the diffusion term in an SDE as a difference of independent Poisson processes. 

Using an EM algorithm, the method is extended to the setting where the observations obtained from one of the marginals is an aggregate of samples, which then is used to solve important problems in biology. There are multiple real and synthetic data experiments in the paper.

### Strengths
This is one of the best papers I have read this year. The only reason I did not reward the submission with the highest presentation score is because some concepts were not explained/taken for granted (I discuss this in Weaknesses).

The originality and quality of the paper is top-class, especially considering the very clever use of the EM algorithm to solve the deconvolution problem. Furthermore, modeling probability bridges directly on the space of count data is of great importance in genomics and transcriptomics. Needless to say, Count Bridges can have a substantial impact on the field of computational biology *and* the method is novel, non-trivial and elegant. 

Although the results are missing error bars and the quantitative analysis of the results could be more thorough, Count Bridges are evaluated on multiple important biological tasks, outperforming for example a fine-tuned version of the DeepMind produced Enformer. 

Count Bridges will surely be an appreciated addition to the class of generative models by the ML community.

### Weaknesses
### Distinction to SBs
Why should Count Bridges not be considered as an instance of Schrödinger Bridges? SBs are not constrained to continuous measures, and there is no statement about why CBs are not SBs (there is instead a note on the connection between SBs and CBs in line 172).

I think this requires some clarity to avoid the risk of artificially distancing CBs from SBs in order to promote novelty.

### Clarity
Surprisingly, I was mainly concerned with the clarity in Sec. 3. I found that key concepts were not defined. 

* I found it quite frustrating that $\Lambda_{-}$,  $\lambda_{-}$,  $\Lambda_{+}$ and  $\lambda_{+}$ were not defined. The definitions/models for these should be in the main text.

* line 200: "We employ the standard U-statistic estimator (Gneiting & Raftery, 2007; De Bortoli et al., 2025)." This is not informative: do you mean that you use the weighting scheme as in De Bortoli et al. (2025)? They do not refer to their scheme as standard or a U-statistic **estimator**, nor do they reference Gneiting and Raftery so I could not figure out what the above sentence implied. 

*  "Let $P^{\kappa}_\text{ref}$ be the joint induced by the birth-death kernel." What is this kernel? It has not been defined at this point?

* The distributions and their parameters pop up in Proposition 3.1 without explanation of where and how they were derived. I found this information in the appendix, but the reader needs to be guided there.

### Related work
This is a minor, but I think DestVI might be worth mentioning in the context of ST deconvolution methods. Otherwise I believe the literature review is very comprehensive. Please double check how citations are delivered here, for instance the references between lines 309-310 after (CTMC) should be in parentheses. 

### Experiments
Are the results averaged over multiple runs? Or are these single runs? For a stochastic algorithm like Count Bridges I would expect to see error bars and reported averages.

### Typos
* line 45: adresss
* line 170: "CUDA kernel implementing (Devroye, 2002)". Either it should be **implementation** or described what in Devroye 2002 has been implemented.
* line 173: Léonard (2013) should be in parentheses. 
* JDS and RMSE are not defined properly. 
* line 360: The sentence "STDeconvolve Miller et al. (2022)" is on the loose.
* In the Applications section there is inconsistencies in how CBs are referenced: count bridges (line 366), CB, Count bridges and Count Bridges. 
* The introduced abbreviation CB is only used in Sec. 6.1?

### Questions
* How do you choose $\Lambda_{-}$,  $\lambda_{-}$,  $\Lambda_{+}$ and  $\lambda_{+}$?
* Could CBs be used to find a bridge between pairs of bulk data distributions?
* Could CBs be used to find a bridge between pairs of unit-level data distributions? Where the unit-level counts are inferred using your EM algorithm?

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes Count Bridge, an integer-native diffusion framework for count data that replaces Gaussian additive noise with two Poisson processes (birth/death). This yields sequential ±1 jump dynamics that preserve integer structure. The denoiser is distributional and trained with a strictly proper energy score suited to count-space geometry. The authors derive closed-form intermediate conditionals (Binomial / Hypergeometric / Bessel components) and implement principled local-bridge sampling. They extend Count Bridge to deconvolution by treating unit-level X0 as latent and using a generalized EM scheme: a projection-guided diffusion E-step (the projection is applied at each reverse timestep to guide local-bridge sampling under the aggregate constraint) and an M-step that trains the model via an aggregate-level energy score loss. Empirical results on synthetic data and several real deconvolution tasks (single cell -nucelotide expression, bulk RNA data deconvolution, spatial spot deconvolution) show competitive performance versus baselines such as CFM and DFM or Enformer, CIBERSORTx, STDeconvolve.

### Strengths
1.	Introduces an integer-native diffusion process (birth–death Poisson kernel) that naturally preserves counts.  
2.	Derives analytic intermediate conditionals (Binomial / Hypergeometric / Bessel), which support principled local-bridge sampling.
3.	Practical deconvolution pipeline: Gives a workable EM-style approach (projection-guided diffusion + aggregate-level loss) to infer unit-level counts from aggregate observations. 
4.	Broad evaluation: Tests on synthetic and multiple real-world deconvolution tasks, with comparisons to relevant baselines.

### Weaknesses
1. Identifiability and aggregation scale. Pure aggregate supervision is intrinsically ill-posed; performance can degrade as group size increases or between-unit heterogeneity decreases. Please provide quantitative sensitivity analyses showing performance vs. group size and vs. within-group heterogeneity (e.g., varying variance of unit distributions), and explain the practical limits (aggregation scales)  where the proposed EM is reliable. 
2. For the nucleotide-level gene expression modeling task, the authors compare to Count Bridge, which reportedly improves on the fin-tuned Enformer’s direct sequence to cell type-specific expression. I observe that the MSE for the 'plasma' cell type in Appendix E is several orders of magnitude higher than for other cell types. This is concerning and requires clarification. 
3. Please include an ablation over projection and discretization choices (KL rescale + multinomial, rounding/min‑distance, learned Πψ, etc.) and report sensitivity of downstream metrics.

### Questions
1.	Enformer-like models are typically valued not just for prediction accuracy but as tools to study sequence-to-expression regulatory mechanisms (e.g., variant effect prediction, attribution maps, motif/TF signal localization). The manuscript reports that Count Bridge outperforms a fine-tuned Enformer on cell-type-specific expression prediction in single-cell data. A key question for me is whether Count Bridge can support the same kinds of downstream regulatory analyses that make it useful in genomics.
2.	In the spatial transcriptomic deconvolution task, I strongly encourage the authors to validate their spatial deconvolution results on genuine spatial transcriptomics data. Specifically, please evaluate the method on at least one sequencing-based platform (e.g., 10x Visium) rather than only on spots synthesized from MERFISH single-cell imaging. 

The reviewer wrote the review. LLM was employed only to correct grammatical errors.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper introduces Count Bridges, a novel stochastic bridge framework for modeling integer-valued count data, extending diffusion-style generative modeling to discrete domains such as single-cell RNA-seq. The method formulates a birth–death bridge process that connects two count distributions through continuous-time dynamics, allowing for closed-form conditional laws and exact likelihood estimation. The authors further propose an EM-style learning scheme that enables deconvolution of aggregated transcriptomic data—e.g., recovering single-cell distributions from bulk or spatial measurements. Experiments on synthetic and real datasets show that Count Bridges outperform existing discrete flow and diffusion models in both modeling fidelity and biological interpretability.

### Strengths
Count Bridges constructs a birth–death bridge with a closed-form conditional distribution, enabling precise likelihood estimation for integer data. The EM-style extension for transcriptomics holds practical value: it provides an actionable workflow for deconvoluting single-cell distributions from aggregated observations derived from bulk or spatial sequencing. Experimentally, the method demonstrates robust improvements over relevant discrete baselines across synthetic and real-world data tasks presented in the paper. The paper is thorough in both mathematical derivations and implementation details, ensuring the work combines theoretical rigor with practical applicability.

### Weaknesses
- While being compared against CFM, DFM, and some biological baselines, a direct comparison with other recent count - specific or general discrete diffusion models (beyond Blackout Diffusion) on the proposed tasks could provide a more complete picture.
- The computational complexity of Count Bridges versus discrete diffusion or flow models is not reported; scalability to large transcriptomic datasets is unclear.
- While deconvolution results are promising, the paper provides limited biological case studies (e.g., linking recovered cell states to known pathways).

### Questions
- Many single-cell datasets are zero-inflated. Does Count Bridges handle this naturally via the birth–death process, or is special preprocessing required?
- How does runtime and memory scale with dimensionality (number of genes) and group size in aggregate deconvolution?
- Can the learned bridge latent representations be linked to biological factors (e.g., cell states, differentiation trajectories)?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper introduces Count Bridges, a stochastic bridge framework for modeling and deconvolving integer-valued count data, particularly motivated by RNA-seq and spatial transcriptomics. The proposed model is based on a Poisson birth-death process on the integers, yielding closed-form conditionals that enable exact sampling and tractable training. The framework supports both generative modeling of count distributions and aggregate-to-unit deconvolution via an Expectation Maximization procedure treating unit-level counts as latent variables.
The authors demonstrate applications to synthetic benchmarks and two biological tasks: deconvolving bulk RNA-seq data and inferring single-cell profiles from spatial transcriptomic spots. The method achieves strong empirical performance, outperforming baselines such as flow matching, discrete flow matching, CIBERSORTx, and STDeconvolve.

### Strengths
- Novelty and conceptual clarity: The introduction of a bridge-based model tailored to integer-valued data is original and addresses an underexplored area between diffusion-style generative models and biological count modeling. The stochastic birth–death formulation provides a principled way to interpolate between discrete distributions.

- Mathematical rigor: The theoretical development is clear and well-supported, including proofs and connections to Schrödinger bridges and optimal transport. The paper demonstrates careful thought in defining the discrete bridge process and the distributional training objective.

- Relevance to biology: Extending generative modeling to integer-valued transcriptomic data and demonstrating utility in deconvolution tasks is timely and important for large-scale single-cell and spatial genomics.

- Empirical results: The experiments convincingly show both generative and deconvolution performance, with clear benchmarks and biological relevance. The model’s ability to outperform reference-free and even reference-based baselines is notable.

### Weaknesses
1. Presentation and flow

While mathematically solid, the readability could be improved. Some definitions (e.g., lines 130–140 in the original text) appear abruptly, and notation (such as $A(X_0), Π𝜓$) could be introduced more gently. Reordering to first build intuition before formalism might help readers from outside the generative modeling community.

2. Clarity of deconvolution setup

The biological deconvolution experiments are appealing but could use more methodological clarity. For example:

- How exactly are unit-level latents initialized or sampled during the EM procedure?

- How are aggregates simulated from real datasets (e.g., in MERFISH) and how sensitive are results to the aggregation scheme?

- What metrics are used to evaluate count-level accuracy beyond JSD and RMSE?

3. Comparative baselines

While the model outperforms selected methods, the comparison to continuous diffusion models (e.g., Gaussian-based or Poisson-approximated variants) could be expanded, since this would better position Count Bridges as a discrete alternative rather than a purely new category.

4. Experimental depth

The evaluation focuses mainly on summary-level results. Including visualizations of reconstructed single-cell profiles, or ablations on the projection module $Π𝜓$, would strengthen the empirical case.

5. Theoretical framing

The connection between Count Bridges and Schrödinger bridges is very interesting but only briefly touched upon. Expanding this connection, perhaps through an interpretation in terms of entropy-regularized optimal transport, would make the theoretical section more accessible and conceptually richer.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 5

### Summary
The paper introduces Count Bridges, a generative framework for modeling and deconvolving integer-valued count data. The core of Count Bridges is a stochastic bridge process on the integers, defined by a Poisson birth-death mechanism. The model is trained using a distributional scoring loss, which is necessary to respect the discrete geometry of the data, as opposed to simpler point estimates used in continuous diffusion. For deconvolution, the authors extend this framework via an Expectation-Maximization (EM) algorithm that treats unit-level counts as latent variables: the E-step uses a projection-guided sampling process to impute unit-level data consistent with an observed aggregate sum, while the M-step trains the model by evaluating its performance at the aggregate level. The authors demonstrate the utility of their method on two large-scale biological applications: predicting single-cell gene expression from DNA sequence (outperforming a fine-tuned Enformer model) and deconvolving both bulk RNA-seq and spatial transcriptomic spots into single-cell profiles, benchmarking against methods like CIBERSORTx and STDeconvolve.

### Strengths
Mathematically, I find the paper very sound and compelling. A lot of papers have been released that train diffusion-type models on categorical data, but little research has extended these concepts to counts, which are the main type of modality in transcriptomics. The method appears correct and elegant, and potentially serves as a relevant first step for follow-up research in the field. I also really appreciate the connection between the model and the EM algorithm, which I find very original and effective to guide generation towards a certain aggregated profile.

### Weaknesses
Overall, my main points of criticism are in the readability and experimental evidence. I think the paper could be slightly more rigorous in presenting some concepts, and the biological experiments are not as comprehensive as they could be. Nonetheless, I hold a good opinion about the paper, and I am happy to discuss it during the rebuttal phase. 

**Text.**

- L126-136: I recommend being a bit more thorough in defining the terms. For example, define $A$ as a covariance and not *noise*, make the sentence in L132-135 more clear (what is the subject?). 

- I would also define the concept of ±. When inspecting the math, it is clear that it represents the addition and removal of counts following the Poisson process, where both adding and removing counts have their own time-resolved rate parameter. However, you can make it explicit from the text. Specifically, the introduction to the model is a bit abrupt and, to me, could benefit from a bit more natural language. I would explain that the counterpart to the continuous formulation of bridges is a birth/death-based Poisson process, with subsequent introduction to the individual terms. This could potentially improve the flow. 

- A similar abrupt introduction is the connection with Schrödinger Bridges. The $\kappa$ parameter and $\pi$ notation for the process are previously undefined. I believe this breaks the flow of the read. Also, the concept of iteration, does it refer to an iterative proportional fitting? I find this connection quite unclear. 

- For the distributional scoring loss, I recommend potentially using another symbol than $w$, as it is used to the success probability of $N_t$ before. A similar concept holds for $A$, it is used both as a covariance for the diffusion term and as an aggregation function. Also $\Pi$ is interchangeably used as the correcting projection for endpoint $x_0$ and the count bridge joint defined in 175. I personally feel all these aspects hinder the ease of reading a little bit. 

- Typo L 159: $(X_s)_{t\in[0,1]}$, I think $t$ should be replaced by $s$. 

- L258: Is the notation $A(\textbf{X}_{g0})$ correct? $A$ is defined as an aggregation, but here it is applied to a single entry.  

**Content.** 

- It's not clear to me how the bulk deconvolution task works. How do you match single cells to bulk profiles? Do you build a synthetic dataset? How do you match endpoints to train the bridge? I feel that all these details should be briefly introduced in the main; otherwise, the results are hard to interpret. Also, deconvolution is a very established approach with other methods like MUSiC [1]. I am not suggesting an exhaustive benchmark, but maybe a more comprehensive comparison would be appropriate. 

- In general, I feel the experiment presentation is a bit disorganized. The metrics are not introduced, the captions are very short, and the benchmarks are very restricted. It would have been interesting to see some predictions of important genes performed by the model on the spatial slides, or to plot some 2D embeddings to validate that the generated cells are realistic. 

[1] Wang, X., et al. (2019). MuSiC: bulk tissue cell type deconvolution with multi-subject single-cell expression references. Nature Communications.

### Questions
- What type of metric $\rho$ do you use in the count space?

- How does the experiment for RNA prediction from DNA work? What I understand is that, for every base pair on the DNA context, you condition noise-based generation of counts with the genomic context obtained via Enformer. Is it true? How do you run Enformer then, and how are the two settings comparable?

- You mention that you are using images to condition the bridge parameterization in the spot experiment. Did you try ablating the conditioning on the images and evaluating the performance after this? I find it unexpected that including image information serves as a performance boost for this task.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
2