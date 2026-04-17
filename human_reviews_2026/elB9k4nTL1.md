# Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Hyperparameter tuning can dramatically impact training stability of large-scale models.
Recent works on neural network parameterisations, such as  μP, have shown that layer types and sizes should dictate how global hyperparameters should be rescaled in order to achieve efficient transfer across model sizes.
On the other hand, the established practice for hyperparameter optimisation search is to look for optimal global base values that apply at some fixed model scale.
We transfer hyperparameters across all scaling axes: width and depth, using an extension of CompleteP (Dey et al., 2025), training horizon, and batch size.
Our study covers all optimisation hyperparameters of modern models: learning rates, Adam parameters, weight decay, initialisation scales, and residual block multipliers.
Lastly, we demonstrate that hyperparameter transfer holds even in the per-layer hyperparameter regime.
We characterise the empirical challenges of navigating the high-dimensional hyperparameter landscape, and propose practical guidelines for tackling this optimisation problem.
We suggest a simplified parameterisation of the hyperparameter space that reduces the dimensionality of the search-space at no performance cost.
Our experiments demonstrate training speed improvements when applying transferred hyperparameters to Large Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper extends and refines existing parameterisation schemes for hyperparameter scaling, most notably CompleteP. The authors introduce/combine several ideas: a continuous-time analysis of SGD to derive scaling rules for the learning rate, weight decay, and other hyperparameters across batch size and token horizon; corrections to inconsistencies in the original completeP work (AdamW $\eps$); per-module and per-depth hyperparameters. They empirically demonstrate that these refinements allow efficient hyperparameter transfer across scaling axes, including model size, batch size, and token count. Impressively, they do this with not only learning rate, but also Adam parameters, initialization scales, and multipliers.

### Strengths
Tackling hyperparameter transfer across many scaling axes (width, depth, batch size, and training horizon, etc.) is very ambitious. As far as I know, no prior work has attempted such comprehensive scaling experiments.  

The derivation of batch-size and weight-decay scaling from the SDE limit of AdamW is conceptually elegant. The pragmatic “trust-region random search” for finding the per-module multipliers $\alpha_g$ and $\delta_\ell$ seem effective and fairly efficient, showing that the resulting configurations generalize well to large models.  

The experiments are extensive and to have combined several ideas regarding hyperparameter scaling make the paper overall very impressive.

### Weaknesses
The paper would benefit from a concise quantitative table comparing Complete(d)P vs. CompleteP across representative settings. This would make the incremental gains clearer. Further, it is not totally clear what one ought to compare to, because there is no dedicated related-work section. This makes situating the contribution relative to prior works difficult. There seem to be missing citations, such as [1] (I would also expect to see a quantitative comparison to [1]).

The paper would be more useful to practitioners if it provided recommended base hyperparameters (e.g. base learning rate, initialization scale, etc.) for typical model sizes. Table 1 gives the scaling rules but not concrete base values. In practice, we don't want to run these grid searches (if possible).


[1]: https://openreview.net/pdf?id=DZ6iFdVDrx

### Questions
- Can you provide an approximate compute budget (GPU-days) for the per-module hyperparameter search? Section C.1 gives partial details but not the full scope.  
- Do you believe these scaling ideas could extend to mid- or post-training adaptation (e.g. LoRA fine-tuning)? Would the same SDE-based scaling rules still hold?  
- For practitioners pretraining Transformers from scratch, could you recommend a minimal set of “good defaults” derived from *Complete(d)P*, avoiding the need for exhaustive search?

### Soundness
4

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
The paper investigates hyperparameter transfer for pre-training of large Transformer models as model size scales. The proposed approach builds upon μ-parameterization (μP) and its variants, such as Depth-μP, and extends these frameworks to incorporate more recent architectural components like QK-normalization. In addition, the authors broaden their analysis to cover the effects of scaling batch size and training token budget.

The work further examines per-layer hyperparameters, showing that optimizing them independently can yield additional performance improvements. Finally, the paper introduces a local search strategy designed to more effectively navigate the sharp cliffs present in the hyperparameter landscape.

### Strengths
- Extending the state of the art (Depth-μP) to include more recent architectural advances such as QK-normalization makes the proposed hyperparameter transfer framework significantly more practical and up to date.

- Addressing hyperparameter transfer across batch size and token budget is both sensible and novel, as this dimension of scaling has received limited attention in prior work.

### Weaknesses
- Clarity and notation: Certain sections of the paper are difficult to follow, with inconsistent or undefined symbols. For instance, the variable θ denotes model weights in Section 2 but appears to represent hyperparameters in Section 3. The second paragraph of Section 3.1 in particular is confusing and requires clearer explanations and consistent notation.

- Questionable argument against existing HPO methods: The paper claims that established hyperparameter optimization techniques (e.g., Bayesian optimization) would not meaningfully work in this setting. However, this assertion is not convincingly supported. Even if the loss landscape contains cliffs, Bayesian methods can still locate high-performing regions without fully modeling these discontinuities exactly. This holds particularly for Bayesian optimization methods that do not use Gaussian processes. The authors should provide empirical evidence or an ablation study to substantiate this claim.

- Limited downstream evaluation: The primary metric used is the final validation loss on pre-training data, which the authors argue correlates with downstream performance. While reasonable, showing results on actual downstream tasks (e.g., fine-tuning accuracy or benchmark performance) would strengthen the claim that the transferred hyperparameters provide practical benefits.

### Questions
- Line 241: Where does the term \eta * e^k originate from?
- Line 244: What is \Theta_\eta2?

### Soundness
3

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
3

### Summary
The authors propose Complete(d)P, a refinement of the CompleteP parameterization that extends to modern Transformer components like Query-Key normalization. The key contribution is demonstrating that hyperparameters optimized at a per-module level on small proxy models (50M parameters) can transfer successfully to larger models (1.3B parameters). The paper covers all major optimization hyperparameters, and authors also provide practical guidelines for optimizing the challenging high-dimensional per-module hyperparameter landscape.

### Strengths
1) The extension to per-module hyperparameters represents a meaningful advancement beyond prior work on global hyperparameter transfer.

2) Novel weight decay scaling rule for batch size (κ scaling) derived from SDE analysis is theoretically motivated and empirically validated

3) Comprehensive experimental coverage across multiple scaling dimensions (width, depth, batch size, token horizon). Empirical results show a speedup improvement of up to 27%.

### Weaknesses
1) Model sizes are relatively limited - while the sizes of the models used in this paper are comparable to CompleteP, µP was evaluated on larger models. Addressing the question of scalability would have strengthened the guildelines proposed by the authors.

2) The evaluation does not include the model's performance on downstream tasks. CompleteP, for example, includes this kind of evaluation. Given that this paper builds directly on said paper, not including this in the evaluation reduces the reader's ability to assess the approach's applicability.

3) Evaluation is conducted only on a Transformer-decoder architecture, so transferability to other Transformer architectures (while certainly possible) is not analyzed.

### Questions
1) In Figure 1, what is the "optimal global" baseline? Is this the best global HP found through the random search?

2) You mention that removing QK-norms doesn't completely break transfer (Figure 14), contrary to Dey et al. (2025). What implementation differences might explain this discrepancy?

3) Can you provide error bars or results across multiple random seeds for the main speedup claim (Figure 1)?

4) Feel free to address any of the weaknesses listed above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work investigates the transfer of optimization hyperparameters of language models from a lower to larger scale. The key contribution is to optimize hyperparameters at the low scale on a per-module-type basis. The per-module optimization, empirically, leads to a speed-up of 27% over global optimization in the large scale training run.

### Strengths
* How to set hyperparameters for large models is a crucial question, and investigating per-module hyperparameter optimization on the lower scale is an important idea.
* 27% speedup in the large scale training is a substantial improvement and emprirical evidence of per-module HP transfer being useful is important to a larger audience.
* The most related work is closely discussed throughout the paper and ideas well placed in the literature
* The paper is mostly well written

I did not closely check some of the more theoretical ideas in the paper and may be missing some strengths there.

### Weaknesses
* The paper makes a point about Bayesian optimization being an ill fit for the problem setting at many points, but does not include an actual empricial comparison.
* Saying "Our study covers all optimisation hyperparameters of modern models" in the abstract is too bold of a statement. This already breaks down when other optimizers or learning rate schedules are considered, or, e.g., the choice of optimizer becomes a hyperparameter. I suggest making a slight adjustment here.
* Experimental rigor could be improved by considering more seeds and quantifying measurement error, e.g., by showing error bars as the standard error around the mean.
* "Even with this modification, however, we found that this trust-region random search quickly plateaued with a relatively high variance in the final loss values." Showing these results in the paper would be beneficial.
* How much optimization of your search method did you do with respect to your final evaluation metric (validation loss on the 1.3B model). Seeing how your method transfers to scales / models that you have not touched during method development would be good.
* No code is provided to reproduce the experiments. Some details are provided in the Appendix, but I doubt I could reproduce the study.
* The authors do not discuss the limitations of their work.
* Figure 1 "Best grid search loss" is hard to read (even without a red-green blindness). More explanation in the caption would be helpful.
* Figures should make it clear that validation loss is shown
* I would like to see a comparison where only the parameterisation is changed and the search method etc. is kept the same.

### Questions
* What do the multiple solid red lines in Figure 1 represent?
* What does "optimal global" refer to, e.g., in Figure 1? Is there a direct comparison to Depth-muP and CompleteP? 
* Evaluation is done with respect to final validation loss. Is this also being used to optimize the hyperparameters at the lower scale? If so (maybe additionally) showing results on hold-out dataset would be good.

### Soundness
1

### Presentation
3

### Contribution
2
