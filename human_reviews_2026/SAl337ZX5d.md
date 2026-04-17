# RefineStat: Efficient Exploration for Probabilistic Program Synthesis

- Decision: Accept (Oral)
- Scores: 8, 6, 4, 8

## Abstract
Probabilistic programming offers a powerful framework for modeling uncertainty, yet statistical model discovery in this domain entails navigating an immense search space under strict domain‐specific constraints. When small language models are tasked with generating probabilistic programs, they frequently produce outputs that suffer from both syntactic, and semantic errors, such as flawed inference constructs. Motivated by probabilistic programmers’ domain expertise and debugging strategies, we introduce RefineStat, a language model–driven framework that enforces semantic constraints ensuring synthesized programs contain valid distributions, well‐formed parameters, and then applies diagnostic‐aware refinement by resampling prior or likelihood components whenever reliability checks fail. We evaluate RefineStat on multiple probabilistic-programming code-generation tasks using smaller language models (SLMs) and find that it produces programs that are both syntactically sound and statistically reliable, often matching or surpassing those from closed-source large language models (e.g., OpenAI o3).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces RefineStat a new framework for automatically synthesizing probabilistic programs using "smaller language models" (around 8B parameters). RefineStat introduces two checks for generating better probabilistic programs: 1. semantic and syntax checks to ensure the language models generates valid PyMC code (this is done using constrained decoding). 2. statistical diagnostic checks that ensure the results from posterior inference are reliable.

Empirically the paper presents results on synthesizing programs for some PosteriorDB datasets. RefineStat is evaluated using multiple open source language models as different base models.

Overall, I am in favor of accepting the paper because I think it provides a useful building block for setting up an automated workflow for synthesizing probabilistic programs. I do have some reservations about the empirical validation though which i would like the authors to address in the rebuttal.

### Strengths
The paper is very well written. The main arguments are easy to follow and the methodology is explained clearly. Both checks (semantic/syntax checks and statistical diagnostics) are sensible for any automated workflow in building Bayesian probabilistic models. I think the paper can serve as a useful stepping stone for building automated model building workflows.

### Weaknesses
There's a couple of weaknesses which I would like the authors to respond to in the rebuttal period. I don't think any of them should disqualify the paper from being accepted but I would appreciate some further discussion on the points listed below.

1. The paper motivates the use of small language models by the potential large API cost of using bigger language models. However, there is not a rigorous cost comparison of the RefineStat approach vs just calling the OpenAI-o3 model.  At least for the OpenAI model it would be good to provide the cost of the API calls.

2. Similarly, why is there no comparison with OpenAI-o3 in Table 1?

3. I appreciate the fact that the paper explicitly accounts for the fact that the empirical validation might be impacted by the fact that the PosteriorDB models could be contained in the training set of the language models. However, I find the empirical checks to be a bit unconvincing (although I am not an expert in the dataset memorization literature). Looking at the Table in Appendix G there is sometimes quite a lot of variation in the achieved scores (e.g. for Dugongs, GP and Surgical) and additionally, this check was only done on the LLama model whereas the evaluation in Table 3 uses the DQ-7B model as a base model. So the whole experimental setup is a bit inconsistent.

At the same time, dataset leakage is an unsolved problem as far as I am aware. So while this evaluation might be flawed, I think it shouldn't disqualify the paper from getting accepted.

4. The OpenAI-o3 model seems to be competitive with the RefineStat approach (on 8 schools, dugongs and surgical, if I am reading Table 3 correctly). So the big question is whether we really need RefineStat or whether training a bigger model will also improve probabilistic program synthesis performance. I don't see a reason though why RefineStat couldn't be used with a big model (other than cost) so it would have been nice to see some results for using OpenAI-o3 as a base model in RefineStat (not sure whether the constrained decoding is possible with a closed source model and/or it might be too expensive).

### Questions
L130: If this definition is taken from Vehtari et al, 2017 then the reference should be part of the definition.

Algorithm 1 misses any description for how $\mathcal{D}$, $\mathcal{P}$, and $\mathcal{L}$ are initially sampled.

Table 1: What does it mean for a value to be bold in the table?

Table 3: The table caption should state which metric is shown, I assume it is ELPD LOO? Additionally, could you show the standard deviation of the metrics (if I understand correctly these values are averages over 5 runs?)

### Soundness
3

### Presentation
4

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
The paper proposes REFINESTAT, a probabilistic program synthesis framework that allows small open-weight language models to automatically generate Bayesian models that are both executable and statistically trustworthy.

REFINESTAT has two core components:

- Semantically constrained generation.
Instead of allowing an LLM to free-generate PyMC code, REFINESTAT enforces a set of domain-specific validity predicates at generation time. These constraints check: parseability, valid distribution names, correct parameter names/signatures, shape/type compatibility, support (e.g. positive-only scales), and dependency ordering between random variables. If a generated fragment violates one of these constraints, that fragment alone is locally rejected and resampled, rather than discarding the entire program. This prunes away statistically nonsensical or sampler-hostile code before it ever reaches inference.

- Diagnostic-aware refinement.
For each candidate model, REFINESTAT runs inference in PyMC (using NUTS/HMC) and computes standard Bayesian workflow diagnostics. These are aggregated into a Bayesian Workflow Reliability Score, which counts the number of diagnostics that pass. If the score is below a threshold, REFINESTAT selectively resamples just the likelihood block or just the prior block, keeping the rest of the model fixed, and tries again. A model is accepted only if it reaches sufficient reliability and good predictive performance.


The framework is tested on five canonical PosteriorDB problems (Eight Schools, Dugongs, Surgical, Peregrine, Poisson GP) using several open models and compared to:

- naive prompting (unconstrained generation),
- syncode (syntax-only constrained decoding),
- multi-agent GPT-4 systems such as BoxLM,
- expert-written Stan models,
- and proprietary baselines like OpenAI o3.

### Strengths
Prior constrained decoding work in LLM code generation typically enforces syntactic validity or DSL grammar. REFINESTAT goes further and embeds statistical semantics: valid distribution calls, correct parameter names and shapes, valid supports (e.g., positive scale parameters), and dependency ordering between latent variables and likelihood terms. 

Instead of treating generation as one-shot code synthesis, REFINESTAT adopts the classical Bayesian workflow loop: propose a model $\rightarrow$ run inference $\rightarrow$ diagnose $\rightarrow$ revise. It encodes this loop in a way that is easily enforceable by turning convergence/efficiency diagnostics.

Competing systems like BoxLM rely on multi-agent GPT-4 setups (generator + critic). REFINESTAT shows that structured guidance, combined with principled diagnostics, can enable a single, small open model to match or beat these heavier pipelines on some benchmarks.

REFINESTAT increases run rate from $\approx$10–11% (unconstrained)/ $\approx$21% (syntax-only) to $\approx$45–50%. Ablation studies show that removing specific semantic checks (especially parameter validity) sharply degrades success, establishing that each constraint is important.

A weaker open model (DeepSeek-R1-Distil-Qwen-7B), which fails with naive prompting, becomes capable of producing usable models on all datasets when run through REFINESTAT.

### Weaknesses
All experiments are on five classic PosteriorDB tasks (Eight Schools, Dugongs, Surgical, Peregrine, Poisson GP). These are relatively low-dimensional, reasonably clean, and heavily studied.
The author doesn’t use any messy, modern, high-dimensional problems:
- hierarchical GLMs with a large number of predictors,
- data with missing values or covariate drift,
- settings where NUTS is brittle.

The paper claims progress toward automated statistical model discovery, but current evidence is confined to highly curated, small-scale Bayesian exemplars.

All experiments are in PyMC. The authors do not demonstrate portability. The semantic predicates (distribution names, argument schemas, support typing, etc.) are currently PyMC-specific.

The refinement loop fixes models by resampling either the likelihood or the prior block when diagnostics fail, and continues iterating until it passes a threshold. This is a clever local repair heuristic, but it also has a limitation: it assumes the proper global structure (hierarchy, link function, GP terms, etc.) was already present in the initial model draft.

The paper has to quantify how often the system gets stuck in such local minima (tinkering with scale priors on a fundamentally wrong structure) or how frequently it must abandon a structure entirely.

The thresholds are hand-picked. The paper does not provide a sensitivity analysis to show robustness.

The method repeatedly runs full NUTS/HMC inside the loop to evaluate diagnostics and to drive refinement. That’s the expensive part.
The paper does discuss token usage (and claims REFINESTAT avoids GPT-4-scale inference costs by using only small open models). Still, it does not report: (i) the number of complete inference runs per dataset per accepted model, and (ii) wall-clock time.

### Questions
During refinement, can REFINESTAT introduce qualitatively new structure (e.g., add a GP term, change link functions, etc.), or does it mostly resample priors/likelihoods within the same initial skeleton? Empirically, how often does refinement change the model family, versus just adjusting hyperparameters? If it does not change structure mid-loop, do you consider that a limitation or is it a deliberate design choice?

When diagnostics fail, what feedback does the LLM get?

Are the thresholds fixed globally across all datasets/LLMs, or tuned per task?

Can you quantify average resource requirements? How many complete HMC/NUTS inference runs does REFINESTAT perform per dataset, on average, before it finds an acceptable model? What’s the typical wall-clock time order of magnitude per dataset on your hardware (minutes, hours)?

For a moderately larger dataset, is the loop still tractable, or does it become prohibitively slow?

It would be helpful to understand how naive prompting + the same diagnostic-aware refinement loop, but without semantic constraints, does. This ablation would isolate the marginal benefit of semantic-constrained decoding and strengthen the argument that both components are essential.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method for generating a probabilistic program using a "small" LLM, where during generation process certain semantic checks are enforced. This method can potentially improve the reliability of the final output. A variety of techniques are currently being proposed to check correctness at decoding time,  this paper differs due to its focus on probabilistic models and their properties. The method is evaluated on a  set of benchmarks against baselines including unconstrained generation by powerful LLMs as well as prior syntax-constrained generation methods.

### Strengths
The problem is well motivated and the basic approach is an appealing way to address the challenges.

### Weaknesses
The authors cite a number of papers proposing constrained decoding for classical programs, but a meaningful comparison is missing. The technique, as a result, seems incremental. The experimental results are limited and not particularly impressive.

### Questions
1. Can you explain the key differences between this and constrained decoding (e.g. "Type-Constrained Code Generation with Language Models" in PLDI 2025). Are there new challenges or just adaptation to probabilistic settings.
2. Can you explain how synthesis algorithms works using an illustrative example?
3. Based on the experimental results, the method doesn't seem "ready" for real-world applications. That's ok, but can you articulate what advances will be needed for this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The submission proposes a framework for constrained generation of probabilistic programs using a small language model and is based on both static and inference-based statistical checks to resample and improve the program. The framework is evaluated on several language model using a set of public datasets from statistical literature and shows solid performance.

The claimed contributions are 

* a method for SLM based generation of probabilistic programs ensuring synatcic and semantic correctness and predictive performance

* the use of constrained decoding for generation of probabilistic programs

* iterative resampling of probabilistic programs using an unmodified small language model

* an evaluation baseline of probabilistic program generation based on a number of criteria outlining syntactic, semantic, and statistical correctness.

### Strengths
The submission presents a non-trivial view on incorproration of Bayesian workflow into Large/Small language model pipeline. It is good to see Bayesian workflow to be formalized and automated for data-driven probabilistic program search. The method is described in detail, supported by clean source code to reproduce experiments, and thoroughly evaluated.

### Weaknesses
Major:

1. I am concerned about evaluating any language model based method on datasets broadly represented in the statistical literature. While the authors argue that through ablation studies they demonstrates that the strengths of RefineStat come from learning rather from memoization, independent empirical evidence of this does not seem to be very convincing. 

2. I am wondering whether PyMC is special in any way to specifically support the approach, how easy to apply the approach to other inference frameworks (Stan, Pyro/NumPyro, DynamicHML.jl, Turing.jl etc). I do not think PyMC is  sufficiently justified in the paper as the 'natural' choice.

Minor: while the authors make a very respectable effort to define everything, the clarity of definitions and notation throughout the paper can be improved. For example:

Lines 130-132 define PSIS-LOO in terms of Pareto smoothed ways. Lines 134-136 that follow mention the fitted Pareto shape parameters k_i, not defined previously. Either more details should be provided about Pareto smoothing in the definion, or fewer details in what follows. As it is now, the reader stumbles on this and unless happens to be familiar with pareto-smoothed cross-validation, has to read the cited paper to understand what is going on.

Lines 173-180 define the score as the sum of 7 'binary indicators'. The 'indicators' are defined as booleans rather than indicators, and booleans cannot be summed up in mathematics (they can in Python). Some sound notation should be used so that s_j is 1 if the condition holds and zero otherwise. Then, in lines 187-192 are quite confusing, at least on the first reading, giving an example when elpd cannot be computed and then immediately saying that the higher the score is, the more the value of elpd can be trusted.

Lines 220-223 talk about how std= is invalid and RefineStat correctly replace std with sigma, but in the accompanying Figure 2 there is sd= rather than std=.

Pedro Domingos appears as Domingos in line 440 and as Pedro in lin 442 in citation anchors.

These are just a few examples, there are quite some more such glitches, and this impairs the reader's ability to understand the paper. The paper should be proofread and fixed.

### Questions
How easy/challenging is it to retarget RefineState to Stan?

### Soundness
3

### Presentation
3

### Contribution
3
