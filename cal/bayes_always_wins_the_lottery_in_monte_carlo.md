=== CALIBRATION EXAMPLE 14 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is catchy but only partially accurate. The paper is not showing that “Bayes always wins” in any general sense; it studies Bayesian training of pruned networks under specific small-scale settings, and the strongest claims are about HMC/SVI on certain LeNet models and sparsity levels.
- The abstract states the problem and high-level approach, but it overstates the theoretical contribution substantially. Phrases like “generalized framework,” “Bayesian-based theoretical grounding containing convergence guarantees that ensure the optimal initialization distribution is found,” and “always win the lottery” are much stronger than what the paper actually establishes.
- The abstract also claims “predictive performance equivalent to or exceeding” lottery-ticket initialization generally, but the experiments are limited to small CNN/MLP models on MNIST/CIFAR-10, with several caveats in the Results/Limitations sections. This should be qualified.
- The reported “best-case accuracy improvement of 5% over random initialization samples and 3% over the original lottery-ticket initialization sample” is a concrete experimental claim, but the paper later reports mixed results and limited convergence for some settings, so the abstract reads more definitive than the evidence supports.

### Introduction & Motivation
- The motivation is clear at a broad level: lottery ticket performance is initialization-dependent, and the authors want a way to decouple performance from the original initialization. That is a legitimate question.
- However, the introduction does not clearly identify the precise gap in prior work. The paper treats “Bayesian training” as if it directly addresses initialization dependence, but it never convincingly explains why sampling from a posterior over weights should recover the specific mask-initialization interaction central to lottery ticket training.
- The stated contribution that HMC provides “the optimal initialization distribution” is not accurate as phrased. HMC samples from a posterior given a chosen prior and likelihood; it does not discover an “initialization distribution” in the lottery-ticket sense.
- The introduction over-claims by suggesting that HMC removes the need for the lottery initialization sample and yields “optimal weights” regardless of initialization. That is not what Bayesian inference guarantees, and the paper later concedes strong dependence on prior choice, warmup, and convergence behavior.
- The connection between “pruning mask generated from any initialization” and Bayesian inference is conceptually interesting, but the paper does not yet articulate a rigorous hypothesis explaining why the posterior should compensate for the loss of the original initialization.

### Method / Approach
- The method description is not yet reproducible at the level expected by ICLR. Key implementation details are missing for both HMC/NUTS and SVI: likelihood specification, prior parameterization, exact HMC/NUTS settings, thinning, chain counts, acceptance rates, diagnostics, optimizer settings for SVI, and how predictions are aggregated from posterior samples.
- The core conceptual leap is under-justified: the paper moves from “weights are initialized from a distribution” to “the weight itself is a distribution” and then uses HMC to find the true posterior. This is a standard Bayesian framing, but the paper does not rigorously connect it to the lottery ticket mask problem. In particular, the mask is fixed, but the Bayesian inference is over weights conditional on that mask; the paper never formalizes the induced posterior or what quantity is being optimized.
- There are logical gaps in the theoretical claims. HMC does not “ensure the optimal initialization distribution is found,” and convergence to a posterior does not imply convergence to a global optimum of test accuracy or loss.
- The text suggests that by “taking enough samples, the average of these should give us the most likely weight values that lead to the best performance.” That is not generally true: posterior means need not correspond to MAP solutions, and “best performance” on test data is not guaranteed by posterior averaging.
- The role of the mask is underspecified. Are masks applied once and frozen? Are the pruned weights removed from the state space, or simply zeroed? Is inference done only over remaining weights? The paper needs a precise formulation to justify the claimed generality.
- Edge cases and failure modes are only partly addressed. The paper notes memory limits and convergence issues for LeNet5, but does not discuss sensitivity to prior choice, multimodality, or whether HMC can meaningfully explore the posterior in the highly sparse regime.
- For the theoretical claims, the statements about HMC convergence are too strong. Even if HMC is asymptotically correct under ideal conditions, finite-sample convergence in neural-network posteriors is highly nontrivial, and the paper does not provide proofs or diagnostics to support the claimed “convergence guarantees” in this setting.

### Experiments & Results
- The experiments do test the central claim at a basic level: whether Bayesian training can match or exceed lottery-ticket initialization performance on fixed pruning masks. That is appropriate.
- However, the experimental scope is too narrow for the strength of the claims. The main evidence comes from LeNet300-100 and LeNet5 on MNIST/CIFAR-10, which are relatively small and dated architectures. This is a weak basis for claims about a general solution to lottery-ticket initialization dependence.
- The comparison to the lottery-ticket baseline is relevant, but the evaluation protocol is not fully clear. For example, the paper reports “highest tested pruning percentage” results in Table 1, but does not justify why the maximum sparsity point is the most informative summary statistic.
- The baselines are incomplete for an ICLR-level paper. There is no comparison to other strong pruning-aware training strategies, post-pruning fine-tuning methods, or modern Bayesian/neural compression baselines beyond a limited SVI variant and deterministic random/lottery initialization comparisons.
- The paper’s strongest claims rely on HMC, yet HMC is not evaluated with the diagnostics typically expected for MCMC-based work. There is no report of effective sample size, R-hat, trace plots, acceptance rates, or posterior predictive checks. This is a serious omission because the paper itself notes that some runs may not have converged.
- The results appear cherry-picked in places. The text emphasizes the best-case HMC gains, while Table 1 shows mixed outcomes across metrics and models. For example, SVI outperforms HMC on LeNet5 accuracy and ResNet-18 accuracy, while HMC is sometimes better on loss; this nuance is not discussed carefully enough.
- The ResNet-18 result is especially problematic as a claim about scalability. The paper says HMC could not even load ResNet-18 due to GPU memory limits, while SVI was tested with hyperparameter tuning. This makes the comparison asymmetrical: the method that “wins” on ResNet-18 is not the method central to the paper’s theoretical claims.
- The reported 20 random initializations for deterministic and SVI comparisons are useful, but there are no confidence intervals or statistical significance tests for the main HMC-vs-lottery results. The random-init baselines do have mean ± std in Table 1, but not the Bayesian methods.
- Dataset and metric choices are standard for a proof-of-concept, but the paper should be more cautious in interpreting classification accuracy on MNIST/CIFAR-10 as evidence for broader claims about sparse Bayesian training.

### Writing & Clarity
- The overall narrative is understandable, but the methodology section contains several conceptual ambiguities that impede understanding of the actual contribution.
- The most serious clarity issue is the paper’s conflation of three distinct objects: initialization distribution, prior distribution, and posterior distribution. This makes it difficult to tell exactly what is being inferred and what is being compared.
- The explanation of HMC/NUTS is too informal for the claims being made. The paper needs a clearer statement of the target distribution, the role of the mask, and how predictions are formed from posterior samples.
- Figure 1 is referenced as explaining the lottery-ticket procedure, which is fine, but the text does not clearly relate the figure to the Bayesian modification introduced by the paper.
- Table 1 is helpful in summarizing final performance, but it does not include uncertainty for HMC/SVI, and the labels “Lotto,” “Rand.,” and “HMC” are not enough to understand protocol differences without re-reading the methods.
- The paper is otherwise readable; the main issue is conceptual clarity rather than prose quality.

### Limitations & Broader Impact
- The limitations section is unusually candid about compute and memory constraints, which is good.
- It does acknowledge that HMC is expensive and that the experiments are limited to small datasets/models. This is an important limitation and directly affects the strength of the conclusions.
- However, the paper misses a more fundamental limitation: the theoretical interpretation of “convergence guarantees” is overstated. Convergence to a posterior is not the same as guaranteeing the performance of a pruned network, and the paper does not separate these notions.
- The paper also does not adequately discuss sensitivity to priors, posterior multimodality, or the fact that pruning masks may interact with the posterior geometry in ways that make HMC difficult even when memory permits.
- Broader impact discussion is absent. While this is not necessarily fatal for an ML paper, the work does touch on resource-heavy training methods and claims around “winning the lottery” for sparse models, so at minimum the paper should note the computational cost and accessibility tradeoffs more explicitly.
- There is also a practical limitation not fully acknowledged: if the key advantage depends on expensive posterior sampling, the method may not be a realistic substitute for the original lottery-ticket procedure in settings where model compression is most needed.

### Overall Assessment
This paper asks an interesting question: can Bayesian training recover or surpass lottery-ticket initialization performance when the pruning mask is fixed? The experiments provide some suggestive evidence on small networks, especially for HMC on LeNet300-100, but the paper’s central claims are overstated relative to the evidence and the theory is not yet soundly formulated. In particular, the manuscript conflates priors, initializations, and posteriors; makes unjustified claims about convergence implying optimality; and lacks the diagnostic and ablation evidence needed for an ICLR-level Bayesian/MCMC contribution. The result is a promising but currently insufficiently rigorous proof-of-concept rather than a convincingly established method.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes training lottery-ticket-pruned neural networks with Bayesian inference, primarily using Hamiltonian Monte Carlo (HMC), to avoid dependence on the specific initialization that originally generated the pruning mask. The empirical claim is that, on small vision models and datasets, HMC-trained Bayesian neural networks can match or modestly exceed the performance of the original lottery-ticket initialization, and that stochastic variational inference (SVI) can sometimes offer a scalable alternative.

### Strengths
1. **Interesting problem framing: decoupling masks from initializations.**  
   The paper targets a genuine open question around the lottery ticket hypothesis: whether the apparent importance of the original initialization is fundamental or merely an artifact of deterministic training. This is a potentially valuable direction for ICLR, especially because it connects pruning and Bayesian learning in a novel way.

2. **Empirical comparison across multiple training paradigms.**  
   The experiments compare HMC, SVI, deterministic training from random initialization, and deterministic training from the original lottery-ticket initialization. This comparison is useful because it directly tests whether Bayesian training can recover performance without the privileged initialization.

3. **Some evidence of improved performance on small models.**  
   The reported results suggest HMC can outperform the lottery-ticket initialization on LeNet300-100, especially on CIFAR-10, and can improve over random initialization by a nontrivial margin. The paper also reports that SVI can be competitive and may scale better to ResNet-18, which is an interesting practical observation.

4. **Acknowledges computational and scalability limitations.**  
   The limitations section is fairly candid about HMC’s memory and time costs, and the authors explicitly discuss why HMC was limited to small models. This is helpful for contextualizing the claims and shows awareness of where the method may and may not apply.

### Weaknesses
1. **The main theoretical claims are not established rigorously.**  
   The paper repeatedly states that HMC “finds the optimal initialization distribution,” “is sure to converge to the true posterior,” and therefore can recover optimal weights regardless of prior. This is a substantial overstatement. Convergence to a posterior does not imply the posterior mode/mean corresponds to an optimum for deterministic test accuracy, nor that the posterior induced by a pruned network is well aligned with the original lottery-ticket solution. For ICLR, this level of theoretical justification is too loose.

2. **The paper’s core conceptual leap is under-argued.**  
   It is not fully clear why posterior sampling over weights should “replace” the original lottery-ticket initialization. Lottery ticket performance is about optimization from a particular starting point under gradient descent, while Bayesian inference optimizes a probabilistic objective over distributions. The paper asserts equivalence between these viewpoints without adequately bridging the gap or justifying why a posterior over weights should preserve the same pruned-subnetwork advantages.

3. **Experimental evidence is limited in scale and depth.**  
   The models are small by modern ICLR standards, with the largest being under half a million parameters for HMC, and the datasets are limited to MNIST and CIFAR-10. The only larger-scale model is ResNet-18 under SVI, not HMC, which weakens the main claim that HMC can broadly “always win the lottery.” ICLR typically expects stronger evidence on more challenging benchmarks or at least a more convincing scalability story.

4. **Ablation and statistical methodology are insufficiently developed.**  
   The paper reports some averages and best-case numbers, but there is limited evidence of robust statistical testing, confidence intervals, or clear protocol details for pruning rates, repeated runs, posterior sampling settings, and model selection. In particular, several claims appear to rely on best-case outcomes, which is weaker than reporting mean/median performance with variance across seeds.

5. **Interpretation of results is sometimes inconsistent.**  
   The paper claims HMC consistently outperforms and “breaks the lottery,” but the table shows cases where SVI outperforms HMC, especially on LeNet5 and ResNet-18, and where HMC is unavailable for the larger model. The narrative overgeneralizes from favorable results on a limited subset of settings.

6. **Clarity and precision need improvement.**  
   Some mathematical exposition is imprecise or conflates distinct ideas, such as treating network weights as distributions in a way that is not carefully formalized. The paper also mixes conceptual explanations with algorithmic claims in a way that can be confusing for readers familiar with Bayesian neural networks and MCMC.

7. **Reproducibility is only partial.**  
   Although the paper mentions sample counts, warmup, and datasets, many key implementation details are missing: exact priors, likelihood formulation, HMC/NUTS settings, how masks were generated and fixed, tuning criteria, and whether test-time predictions use posterior samples or a point estimate. This makes it difficult to reproduce the results reliably.

### Novelty & Significance
The paper has moderate novelty in combining lottery-ticket pruning masks with Bayesian neural network training, particularly using HMC to remove dependence on the original initialization. That said, the broader idea of training pruned networks with Bayesian methods is not fully new, and the paper’s strongest novelty is more in the specific empirical question than in a fundamentally new algorithmic contribution.

In terms of significance, the results are interesting but not yet strong enough for a high-confidence ICLR acceptance under typical standards. ICLR generally looks for either a clear methodological advance, a strong theoretical insight, or compelling large-scale empirical evidence. This paper offers an intriguing hypothesis and some promising small-scale experiments, but the theoretical case is underdeveloped and the empirical scope is limited. As written, it feels more like an exploratory study than an ICLR-level contribution.

### Suggestions for Improvement
1. **Tighten the theoretical claims.**  
   Replace statements about HMC “finding the optimal initialization distribution” or guaranteeing optimal weights with more precise Bayesian language. Clearly distinguish posterior convergence from optimization of test accuracy, and explain what can and cannot be inferred from the posterior in the pruned-network setting.

2. **Add rigorous statistical evaluation.**  
   Report mean and standard deviation across many seeds for all methods, not just random initialization. Include significance tests or confidence intervals, and avoid emphasizing only best-case numbers unless clearly labeled as such.

3. **Strengthen the experimental scope.**  
   Evaluate on at least one larger and more standard benchmark for modern pruning/Bayesian work, or provide a stronger scaling study. If HMC cannot scale, make that limitation central and frame the contribution accordingly, rather than implying generality.

4. **Clarify the Bayesian prediction procedure.**  
   Specify exactly how predictions are made from the posterior: number of posterior samples, whether they are averaged, whether a single sample is used, and how uncertainty is handled. This is essential for both reproducibility and interpretation.

5. **Include stronger ablations.**  
   Test the effect of different priors, pruning levels, sample counts, warmup length, and NUTS/HMC settings. Also compare against additional baselines such as SGHMC or Laplace-style approximations, if feasible.

6. **Improve the conceptual bridge between lottery tickets and Bayesian inference.**  
   The paper would benefit from a clearer explanation of why a Bayesian posterior over weights should emulate the privileged initialization of a winning ticket. If the claim is that Bayesian training makes initialization less important, show this experimentally and theoretically in a more direct way.

7. **Be more cautious in the conclusion.**  
   The conclusion should avoid overclaiming that the method “breaks the lottery” in a universal sense. A more defensible claim is that Bayesian training can partially or sometimes substantially reduce dependence on the original initialization for the evaluated pruned networks.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison against standard Bayesian/pruned-training baselines on the same masks: Laplace/last-layer Bayesian approximations, SGHMC/SGLD, and plain fine-tuning from random init on the pruned subnetworks. Without these, the claim that HMC is special for “winning the lottery” is not convincing.

2. Report the performance of the *same pruning masks* across multiple pruning levels and multiple independently generated masks, not just the highest sparsity table. ICLR reviewers will expect to see whether the effect is consistent or only appears at one cherry-picked sparsity.

3. Compare against non-Bayesian sparse training baselines that are more relevant than Lottery Ticket initialization alone, such as random mask + reinit, random mask + retraining, and strong-ticket / data-efficient pruning methods. Otherwise it is unclear whether the gains come from Bayesian inference or just from using any decent sparse mask with enough training.

4. Add a compute-matched baseline: deterministic training and Bayesian training should be compared under equal wall-clock time or equal gradient evaluations. The current claim of superior performance is not meaningful if HMC is orders of magnitude more expensive.

5. For ResNet-18, include HMC or a principled explanation of why it is infeasible relative to SVI, plus a scaling study over model size. The paper’s core contribution is about Bayesian training of pruned networks, but the evidence is mostly limited to toy-scale models.

### Deeper Analysis Needed (top 3-5 only)
1. Clarify the Bayesian interpretation of a lottery-ticket mask. Right now the paper conflates priors, posteriors, and initialization distributions; it needs a precise statement of what distribution is being inferred and why sampling from it should recover the lottery-ticket initialization effect.

2. Provide convergence diagnostics for HMC/NUTS: effective sample size, R-hat, trace plots, acceptance rates, and sensitivity to warmup/sample count. Without these, the claim of “convergence guarantees” is not empirically grounded for the actual neural network posteriors used here.

3. Analyze whether the improvement comes from the posterior mean, a sampled model, or test-time ensembling. This matters because the paper claims “training” equivalence, but Bayesian methods can improve performance through averaging rather than finding a better single set of weights.

4. Quantify variance across seeds for all methods, including HMC and SVI, with confidence intervals. The reported gains are sometimes only a few points, so without full uncertainty estimates the main performance claims are not trustworthy.

5. Disentangle the effect of the prior choice from the effect of the pruning mask. The paper suggests glorot-like priors help, but does not show whether performance depends more on prior matching, mask quality, or posterior approximation quality.

### Visualizations & Case Studies
1. Add posterior trace plots and pairwise parameter/mode visualizations for HMC on a pruned network. This would show whether the chain is actually exploring a stable posterior or just drifting among poor solutions.

2. Visualize accuracy and loss versus number of HMC samples/warmup steps. The paper currently asserts convergence-related claims without showing whether performance saturates, oscillates, or is still improving.

3. Show layerwise mask sensitivity or neuron importance under Bayesian training versus deterministic training. This would reveal whether the Bayesian method is genuinely compensating for initialization or merely being less sensitive to mask imperfections.

4. Include failure cases where a random pruning mask remains poor under HMC/SVI. If the method truly exposes mask quality, these examples are necessary to show when it does and does not work.

### Obvious Next Steps
1. Extend the method to larger models with stochastic-gradient MCMC variants and test on more modern benchmarks. At ICLR, a contribution framed as “Bayes always wins the lottery” needs evidence beyond MNIST/CIFAR-10 and small LeNet-style networks.

2. Test whether the approach can identify or exploit better masks, not just retrain fixed lottery-ticket masks. That would move the paper from “retraining under Bayes” toward a genuine contribution on sparse model discovery.

3. Replace the current narrative about “optimal initialization distributions” with a mathematically correct treatment of sparse Bayesian inference under pruning masks. As written, the theoretical story is too loose to support the strong conclusions.

4. Evaluate calibration and uncertainty, not only accuracy. Since the paper’s premise is Bayesian training, it should show whether the method improves predictive uncertainty in a way consistent with Bayesian inference.

# Final Consolidated Review
## Summary
This paper studies whether lottery-ticket-pruned networks can be trained without the original winning initialization by using Bayesian inference, mainly Hamiltonian Monte Carlo (HMC), and compares against stochastic variational inference (SVI). On small MNIST/CIFAR-10 LeNet models, the authors report that HMC can sometimes match or slightly exceed the original lottery-ticket initialization, while SVI is faster and sometimes competitive on larger models like ResNet-18.

## Strengths
- The paper asks a genuinely interesting question: whether the apparent importance of the original lottery-ticket initialization is actually necessary once training is reframed probabilistically. That is a plausible and potentially useful angle for sparse model training.
- The experimental setup does include the right high-level baselines for the core question: HMC, SVI, deterministic training from random initialization, and deterministic training from the lottery-ticket initialization on the same pruning masks. This makes the main comparison interpretable.
- The reported results provide at least some evidence that Bayesian training can recover performance on fixed sparse masks, especially for LeNet300-100, where HMC is reported to outperform both random and lottery initializations in the strongest sparsity setting. The paper also candidly notes that SVI can be much faster and scale better than HMC.

## Weaknesses
- The theoretical story is substantially overstated. The paper repeatedly suggests that HMC “finds the optimal initialization distribution” or that convergence to the posterior implies optimal weights/performance, but posterior convergence does not guarantee test-optimal solutions, and it is not the same object as the lottery-ticket initialization distribution. This is not a minor wording issue; it undercuts the central claim.
- The Bayesian/masking formulation is conceptually muddy. The paper conflates initialization distributions, priors, and posteriors, and never formalizes the inference problem for a fixed pruning mask in a way that makes the claimed generalization convincing. As written, it is not clear what distribution is being optimized or why that should recover the lottery-ticket effect.
- The evidence is limited to small, dated architectures and datasets, with the main HMC results on LeNet300-100/LeNet5 and MNIST/CIFAR-10. HMC is explicitly unable to scale to ResNet-18 in their setup, so the most theoretically interesting method is also the least scalable. That makes the universal-sounding title and conclusion unsupported by the actual experiments.
- The HMC results lack the diagnostics needed to trust the convergence narrative. The paper itself notes variance and possible non-convergence on LeNet5, but provides no effective sample size, R-hat, trace plots, acceptance rates, or posterior predictive checks. Given the strong claims about convergence, this is a serious omission.
- The comparison on larger models is asymmetrical: ResNet-18 is only evaluated with SVI, not HMC, so the paper’s stronger Bayesian claim is not actually tested where scalability matters most. This weakens the significance of the SVI results as evidence for the central thesis.

## Nice-to-Haves
- Add a clearer description of how predictions are formed from the posterior samples: number of samples used at test time, whether outputs are averaged, and whether a single posterior draw or ensembling is used.
- Include uncertainty estimates and repeated-run statistics for HMC and SVI, not just for random initialization.
- Test a broader range of priors and pruning levels to show whether the effect is robust or mostly prior-dependent.

## Novel Insights
The most interesting insight is not that Bayesian methods can train sparse masks at all, but that fixed lottery-ticket masks may carry enough structure that a Bayesian learner can exploit them even without the original lucky initialization. The paper’s own results suggest a subtle split: HMC may help on smaller problems where posterior sampling is feasible, while SVI may be the more practical path when scale matters, even if that means giving up the clean convergence story. That tension—between theoretical sampling correctness and practical scalability—seems more central than the paper’s current “Bayes always wins” framing.

## Potentially Missed Related Work
- SGHMC / SGLD-style Bayesian neural network training — relevant because the paper itself raises stochastic-gradient MCMC as a possible scalable alternative, and this is a natural baseline for sparse Bayesian training.
- Laplace approximation or other posterior approximations for pruned networks — relevant as a lighter-weight Bayesian baseline to test whether full HMC is actually necessary.
- Random-mask retraining / strong sparse training baselines — relevant because the main question is whether the gains come from Bayesian inference specifically or simply from training a reasonably good sparse subnet.

## Suggestions
- Replace claims about “optimal initialization distributions” and “always winning the lottery” with precise Bayesian language: what posterior is being inferred, under what fixed mask, and what empirical claim is actually supported.
- Report HMC diagnostics and seed variance for all methods. Without convergence evidence, the strongest claims are not credible.
- Add compute-matched comparisons and stronger sparse baselines. Right now the paper shows that expensive Bayesian sampling can work on small models, but not that it is the best or most informative way to train pruned networks.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0]
Average score: 1.0
Binary outcome: Reject
