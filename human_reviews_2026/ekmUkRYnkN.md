# Scaling Multi-Task Bayesian Optimization with Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
In multi-task Bayesian optimization, the goal is to leverage experience from optimizing existing tasks to improve the efficiency of optimizing new ones. While approaches using multi-task Gaussian processes or deep kernel transfer exist, the performance improvement is marginal when scaling beyond a moderate number of tasks. We introduce **BOLT**, an initialization-only transfer strategy that distills prior BO runs into an LLM which proposes candidates for new tasks, while the surrogate at test time remains single-task. The LLM is periodically fine-tuned on top solutions from completed runs, creating a closed loop where better BO outputs yield better initializations over time. This decoupled design scales to roughly 1500 tasks without the saturation observed for shared-surrogate MTBO and adds only a small, amortized overhead relative to the BO inner loops. We evaluate on two domains: database query optimization and antimicrobial peptide design. We demonstrate that LLM-generated initializations steadily improve and accelerate BO, and with sufficient fine-tuning, a few LLM samples often match or surpass full ''from-scratch'' BO with far fewer oracle calls.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
While this article claims to consider multi-task Bayesian optimisation, really it's about transfer learning within a class of Bayesian optimisation problems. The latter is really important, and the proposed solution is promising. The idea is to tune an LLM to map from task features to likely high performing parameter selections for that task, and to use these suggestions to initialise Bayesian optimisation when a new task comes along. The idea is good, and the results are interesting.

### Strengths
In the problem sets considered, the LLM is able to learn the strongly performing parameters from early runs of Bayesian optimisation. It learns so well that the initial suggestions for later tasks are often better than the final discoveries by single task BO on those tasks. However further refinement by BO provides even better solutions.

The avoidance of any need to learn a cross-task performance model makes multi-task BO much more feasible than in the traditional settings we consider.

### Weaknesses
The paper does not fully explore why this works. My intuition is that if you provide a few really strong examples to initialise BO, the acquisition function should then take the performance at these points as a baseline and go and search in the areas where it does not yet have any examples (ie far from the initialisation). 

The number of BO reps is high. We are initialising with 50 observations and the x axis in Fig 1 goes to 2000 (so far as my eyes can make out). This is a lot of samples.

### Questions
Is my hypothesis correct on the consequence of initialising only with high value parameter values? If not, why not? Would it be better if we also transferred hyper parameters, or some other way of conveying the baseline performance level?

Given that we can find good initial points using the LLM, and apparently have access to a large number of samples, would local search not be likely to perform better? I'm not convinced that BO is the right approach once we have a well-fitting LLM.

How dependent is this on the particular BO method you have used? The BOSS algorithm (Moss et al, NeurIPS 2020) is explicitly designed for optimising over string spaces and would be a useful comparator. Is it that the *constrained* version of LOL-BO is really important (see my previous points). In which case, more should be made of it - perhaps an ablation where we use an unconstrained BO to demonstrate the difference.

Minor quibbles:
- I think you have a typo in line 177. You are not taking the argmax over alpha values. It should probably read argmax_x alpha(x,GP)?
- I disagree with your analogy to self-play in RL in line 243. In RL self-play we bring in new data from the environment when the algorithm plays itself. In self-instruction, and in what you do here, there is no further extrinsic information introduces at all.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
&nbsp;

The authors introduce Bayesian optimization with LLM transfer (BOLT), a framework for multitask Bayesian optimization (MTBO) that leverages supervised finetuning (SFT) of LLMs to propose initial candidates for unseen tasks. The method is applied to database query planning and peptide design. Although the idea is interesting, I have concerns regarding the overall experiment design as well as the implementations of the baselines. Furthermore, I have concerns over the reproducibility of the work given that the codebase is not released. If these issues can be addressed in the rebuttal I will be willing to raise my score.

&nbsp;

### Strengths
&nbsp;

The method is novel and timely, leveraging advances in LLMs for multitask Bayesian optimization (MTBO). It is conceivable that LLMs can be strong meta learning tools for MTBO albeit the tradeoffs against existing methods in terms of performance and/or compute cost are unclear. 

&nbsp;

### Weaknesses
&nbsp;

I summarize my comments in major and minor points. In particular points 2,3, and 4 are worth prioritizing for the authors to receive an upgraded score.

&nbsp;

**__MAJOR POINTS__**

&nbsp;

1. In terms of the structure of the paper, the background on the problem settings on page 3, namely antimicrobial peptide and database query plan optimization would be better placed in the appendix. Given that the authors are introducing a general purpose optimization methodology at a machine learning conference, readers are more likely to be interested in comparable optimization methods rather than the specifics of the applications considered. Given that the authors leverage LLMs, an expanded related work section on meta learning and/or transfer learning in BO as well as other attempts to use LLMs for BO may be more appropriate.

2. The absence of an anonymous GitHub or similar in the submission raises concerns for the reproducibility  of the current work e.g. the description of the architecture for the network in Section C.1 is not complete. What was the choice of activation function?

3. The problems considered by the authors feature optimization over structured input spaces via latent space BO. Following on from point 2 above some of the details of VAE training are missing. Do the authors periodically retrain the VAE? As I understand the authors use the same initialization set as the BAO algorithm for database query planning? Why not use random initializations and report errorbars? For the peptide problem the authors mention they curate L=1000 sequences partitioning 100 of these to the validation set. Is the VAE trained on the 900 sequences comprising the training set?

4. It would be worth adding a random search baseline across the VAE latent space as a sanity check.

&nbsp;

**__MINOR POINTS__**

&nbsp;

1. In terms of the opening statement, "Multi-task optimization seeks to use related, previously solved tasks to accelerate the optimization of new ones." I would disagree with this definition. Related tasks do not necessarily need to be "solved" to provide benefit to the optimization of a new task.

2. When citing domains in which multitask optimization problems occur, it would be useful to provide reference works for the domains mentioned.

3. Line 64, the acronym MTBO for multi-task Bayesian optimization is introduced before it is defined.

4. Line 64/64, the source papers for Optformer and LLAMBO should be given upon first mentioning the methods.

5. Line 86, when introducing Bayesian optimisation it would be worth citing the originating papers for the methodology [1,2] as discussed in [3] in place of the references given.

6. Line 89, the notation $y$ should be defined as a noise-corrupted version of $f(\mathbf{x})$.

7. The source paper for the VAE [4] should be cited when introducing it on line 96.

8. On line 97 when introducing the idea of latent space Bayesian optimization, the source paper should be cited [5], as well as close follow-up works on the topic [6, 7] which were published before Eissman et al.

9. Line 124, the citation to Leis et al. should be parenthetical e.g. (Leis et al. 2015) as opposed to narrative since the author's name is not part of the sentence. It would be worth correcting this across the manuscript.

10. When describing structured input spaces it would be worth explaining what the authors mean by "structured" as this may not be apparent to layreaders. Something akin to a "non-numerical, discrete input" may be appropriate together with examples such as images, molecules, or amino acid sequences.

11. Line 148, the acronym for LLMs has already been defined earlier in the manuscript.

12. In Algorithm 1, the notation $X_t^*$, $X_{init}$, $y$ etc. should be defined. $y_{init}$ should be denoted as a vector. Additionally, $y$ and $y_{next}$ should also be denoted as vectors for generality in the batch setting. On line 177, $x$ should also be a vector. $\alpha$ should not be a subscript on the argmax.

13. In Algorithm 2, it would be better not to use the variable $T$ for the number of iterations since it overloads the use of $T$ as the number of tasks. Additionally, $x$ and $y$ should be bolded as vector quantities $\mathbf{x}$ and $\mathbf{y}$.

14. Line 198, $x$ should be a vector.

15. There is a missing full stop at the end of Equation 2.

16. There are missing capitalizations in the references e.g. "bayesian" in place of "Bayesian".

17. There is a missing arXiv reference for Eggensperger et al. 2020.

18. There are missing conference references e.g. Haluptzok et al. was published at ICLR 2023.

19. It would help the reader if the authors described roughly what the DKT and FSBO methods were instead of forcing the reader to read the source papers.

20. In the related work on language models as optimizers it would be worth discussing the relation of the current work to [8].

21. The source paper for AdamW [9] should be cited given that it is used.

&nbsp;

**__REFERENCES__**

&nbsp;

[1] H.J. Kushner (1962). [A Versatile Stochastic Model of a Function of Unknown and Time Varying Form. Journal of Mathematical Analysis and Applications](https://www.sciencedirect.com/science/article/pii/0022247X62900112) 5(1):150–167.

[2] H.J. Kushner (1964). [A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise.](https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/86/1/97/392213/A-New-Method-of-Locating-the-Maximum-Point-of-an?redirectedFrom=fulltext) Journal of Basic Engineering 86(1):97–106.

[3] Garnett, R., [Bayesian optimization](https://bayesoptbook.com/). Cambridge University Press. 2023.

[4] Kingma and Welling, [Auto-encoding Variational Bayes](https://openreview.net/forum?id=33X9fd2-9FyZd&source=post_page---------------------------), ICLR 2014.

[5] Gómez-Bombarelli, R., Wei, J.N., Duvenaud, D., Hernández-Lobato, J.M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T.D., Adams, R.P. and Aspuru-Guzik, A., 2018. [Automatic chemical design using a data-driven continuous representation of molecules](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572). ACS Central Science, 4(2), pp.268-276.

[6] Griffiths, R.R. and Hernández-Lobato, J.M., 2020. [Constrained Bayesian optimization for automatic chemical design using variational autoencoders](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc04026a). Chemical Science, 11(2), pp.577-586.

[7] Kusner, M.J., Paige, B. and Hernández-Lobato, J.M., [Grammar variational autoencoder](https://proceedings.mlr.press/v70/kusner17a.html?ref=https://). ICML 2017. PMLR.

[8] Ranković, B. and Schwaller, P., 2025. [GOLLuM: Gaussian Process Optimized LLMs--Reframing LLM Finetuning through Bayesian Optimization](https://arxiv.org/abs/2504.06265). arXiv preprint arXiv:2504.06265.

[9] Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7), ICLR 2019.

&nbsp;

### Questions
&nbsp;

1. On line 155, the authors state that each training task comprises the top-K observations from the trajectory for each of $t$ tasks. Are these the top-K optimal observations or simply the top-K observations identified in a BO procedure. If the latter, how does one ensure consistency in the efficacy of the BO procedure used to generate the traces for each training task? Update: This is clarified to be the latter on page 4 of the paper. In this case it may be worth running a sensitivity analysis on the BO runs used to initialize the LLM with the top-K observations albeit this would be costly from a compute perspective.

2. In Figure 1, it is not clear how different tasks are treated. How does the number of oracle calls given in the x-axis related to the number of test tasks? What are the errorbars reported over?

3. I take it that BOLT-10, BOLT-20 etc. refer to the number of training tasks the LLM has been fine-tuned on? It might be worth emphasizing this for the reader in the experient section or the figure captions since the meaning of the BOLT-T notation appears only to be provided in Section 3. Update: This appears to be explained in the caption of Figure 2 but not in Figure 1.

4. For self-augmentation I fail to see the point of saving time on BO computation under the assumption that querying $f(\mathbf{x})$ (scoring under the problem's oracle in the authors' words) is a more expensive process relative to re-training the BO surrogate. Obviously the authors' BO runs were computationally intensive but is there an implicit assumption that for self-augmentation to be useful, evaluating $f(\mathbf{x})$ must be relatively cheap compared to the cost of running BO?

5. What is the task for the results reported in Table 4? The database task presumably?

&nbsp;

### Soundness
2

### Presentation
3

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
The paper presents BOLT, a multi-task Bayesian optimization method that transfers information between tasks by fine-tuning a language model. Unlike other approaches, information transfer is done at the initialization level, such that initial values for the Bayesian Optimization procedure are chosen increasingly effectively as information from other tasks are incorporated into the language model. The authors demonstrate substantial improvements over earlier methods - in particular, they show better scaling with the number of tasks.

### Strengths
**Originality**. To paper introduces a novel way to use language models to improvement multi-task BayesOpt. 

**Quality**. The method is described in detail, the empirical results support the stated claims. 

**Clarity**. The paper is well-written, clearly explaining the idea. A reproducibility statement is included to clearly state what code will be made available upon acceptance.

**Significance** The paper reports significant improvements over the state of the art. The method is also quite simple, and is thus likely to find real-world application.

### Weaknesses
Although the paper presents several succesful applications, it does not provide a clear picture of when it fails. Limitations are discussed in broad terms, but the paper would benefit from e.g. a more careful analysis of how sensitive the performance is to the similarity between tasks. For example, one could imagine an experiment on the peptide design task where performance was reported as a function of the degree of similarity between peptides, to see if it broke down with very low similarities.

As is explicitly stated in the paper, the method focuses on large-data domains. However, it remains unclear how much data is typically necessary, and whether the approach could find application in small-data regimes with more restricted fine-tuning procedures. Any insights on this matter would be useful for a reader that is considering implementing the method.

### Questions
### Questions
line 190. *"we extract the top-K observations from each of the T runs completed so far."* Don't you risk getting very similar solutions using this approach. Wouldn't it make sense to impose some diversity criterion?

line 268. *"we filter out characters that do not correspond to strings of integers or valid amino acids for the respective tasks."*. How frequently do such non-valid sample errors occur? It this a problem in practice?

line 280. *"l points are sampled using a temperature parameter of 0.7 unless otherwise specified."*. For the two problems you use two different temperature hyperparameters. How are these tuned, and is the performance very sensitive to this choice?

### Minor comments

line 257. I found it difficult to assess whether a budget of 200,000 oracle calls is reasonable in practice. You could consider adding a note on this. 

Figure 1. The labels in this figure are difficult to read, both due to the colours and the small font. Considering changing this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of scaling with multi-task Bayesian optimization and proposes the Bayesian Optimization with LLM Transfer (BOLT) method. BOLT is a simple approach that uses LLMs to generate candidates for new tasks, which provides strong initializations for BO. BOLT scales MTBO without saturation, effectively handling MTBO settings up to 1500 tasks.
The experiments on database query optimization and antimicrobial peptide design showed that BOLT is effective in providing high-quality initializations, which yield strong few-shot performance, outperforming LLM-based MTBO methods.

### Strengths
-  The proposed method BOLT is simple and modular, i.e., can be plugged into any BO loop.

- BOLT avoids saturation observed in common-shared GP methods and provides scaling.

- BOLT is empirically evaluated on diverse real-world use-cases, particularly, two high-throughput domains: database query optimization and antimicrobial peptide design.

### Weaknesses
-  BOLT requires a task description context that can be used in an LLM prompt to define the task. This excludes common MTBO settings.  Although the authors raise this point as a limitation, they still state BOLT as broadly scalable MTBO. A more precise scope statement or a counter-example domain would help to clarify this better.

- How sensitive is BOLT's performance to the outer loop? The ablations on top-k size for fine-tuning and fine-tuning frequency would have provided a better sensitivity analysis.

- It would strengthen the empirical effectiveness of BOLT better if the methods are analyzed under a fixed compute budget. That is, in terms of the compute budget, would BOLT still outperform if the total cross-task compute, including oracle evaluations and finetuning cost, is set to the same budget across methods? 


- MINOR:

- - Line 64-65, the references for Optformer and LLAMBO should be added.
- - Line 457: "illustrates" should be corrected as "illustrate".

### Questions
- How would the performance of BOLT change if task contexts are perturbed? Could authors discuss robustness to context perfurbations? To test LLM's generalization performance vs. memorization?


- See also the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
