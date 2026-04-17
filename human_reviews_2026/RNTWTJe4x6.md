# Compositional Generalization through Gradient Search in Nonparametric Latent Space

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 4, 8

## Abstract
Many state-of-the-art methods in deep learning fail at systematic reasoning in settings which require compositional generalization. To address this, we propose a novel architecture which uses a nonparametric latent space, information-theoretic regularization of this space, and test-time gradient-based search to achieve strong performance on compositional meta-learning tasks such as program induction, Raven's progressive matrices, and linguistic systematicity tasks.  Our proposed architecture, Abduction Transformer, uses nonparametric mixture distributions to represent inferred hidden causes of few-shot meta-learning instances. These representations are refined at test-time via gradient descent to better account for the observed few-shot examples, a form of variational posterior inference which allows Abduction Transformer to solve meta-learning tasks that require novel recombinations of knowledge acquired during training. Our method outperforms standard transformer architectures and a contemporary test-time adaptive variational approach, indicating a promising new direction for neural networks capable of systematic generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the problem of compositional generalization and proposes the 'Abduction Transformer' as a solution. The model's core mechanism involves inferring Dirichlet Process Posterior parameters to effectively handle few-shot problems. The proposed algorithm consists of a 2-stage process: 1) An Encoder generates an initial hypothesis based on all few-shot examples, and 2) A separate Decoder architecture receives this hypothesis and performs gradient search (fine-tuning) to refine it into a hypothesis that best explains the examples. The authors validate their methodology on benchmarks such as ARC-LIKE REASONING and RAVEN’S PROGRESSIVE MATRICES.

### Strengths
This few-shot learning framework is generalizable, so can be applied to any benchmark.

### Weaknesses
This paper suffers from critical flaws in its experimental validation, primarily concerning benchmark and baseline selection.

1. Omission of Core Baselines (MAML): The authors acknowledge that their methodology is a ""method of fine-tuning to minimize loss for a given few-shot input,"" and is fundamentally a meta-learning (learn-to-learn) approach. Therefore, the validity of this methodology must be directly compared against at least classic meta-learning algorithms like MAML. They should have verified whether their 'gradient-based hypothesis refinement' mechanism is superior to MAML's 'gradient-based model adaptation' on standard few-shot benchmarks (e.g., tieredImageNet). If their methodology indeed has such generalization capabilities, it must at least outperform classic meta-learning methods. Their method is, in fact, generally applicable. However, the paper intentionally avoids this crucial comparison, presenting only 'strawman' baselines like GPT and standard Transformers, which were not designed for meta-learning. This fails to prove any practical comparative advantage.

2. Use of Arbitrary Benchmarks: The authors used a niche, self-created benchmark called ""1-D ARC-LIKE REASONING"", which they only state is ""inspired by"" the standard 2D ARC benchmark. A search for this benchmark reveals it is extremely niche. Furthermore, even the standard 2D ARC benchmark is not frequently used. On the contrary, on the 2D ARC benchmark—which should be more difficult—some existing LLMs have achieved nearly 70% accuracy. How does this discrepancy arise? Moreover, seeing that modern LLMs like GROK achieve nearly 0% on that benchmark suggests the benchmark has high variance between LLMs and is a task that LLMs are inherently bad at. Why does high performance here validate this model? For example, VLMs perform poorly at depth estimation. A very simple depth estimation model will easily outperform a VLM. What value does that simple model have? This merely proves the VLM's inability on that task; it does not guarantee the new model's performance. This appears to be an attempt to avoid direct comparison on recognized benchmarks, severely damaging the credibility and generalizability of the experimental results.

3. Unrealistic Inference Cost (Test-Time Training): The most significant problem is that the method performs 100 steps of gradient search (fine-tuning) to solve a single test query. This is not fast few-shot inference. This is, in effect, Test-Time Training. This approach of training a model for 100 steps on a single problem renders any comparison against MAML (which takes only a few steps) or GPT (which requires no extra training) fundamentally meaningless. This is an unrealistic inference cost and not a fair comparison."

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors build on recent work (LPN) that investigates learning compressed latent spaces that can be searched at test time to reduce the amortization gap from amortized inference. In this work, the paper proposes the Abduction Transformer, which uses multiple mixture distributions (via Dirichlet Processes), instead of a single Gaussian distribution, to learn the approximate posterior distribution. The method incorporates information-theoretic regularization and test-time gradient-based search in a nonparametric latent space. It is evaluated on OOD compositional meta-learning tasks such as ARC-like program induction, Raven’s progressive matrices, and linguistic systematicity tasks.

### Strengths
- Targets a critical weakness in LPN: which does not necessarily handle compositionality in the latent space due to a single latent vector.
- Approach to tackle compositionality is simple and scalable
- Explores performance on a range of experiments, including ARC-like program induction, Raven’s progressive matrices, and linguistic systematicity tasks.
- Uses appropriate baselines, like single-vector LPN, standard transformers, and GPT-5 Thinking.

### Weaknesses
- Lack of ablations on the method. Since the Abduction Transformer performance is supposedly derived from the larger number of distributions compared to LPN it is important to perform an ablation that scales this axis to see when compositionality emerges. Specifically, a graph with number of distributions on x-axis and test-time gradient search and non-gradient search performance plotted in the graph.

- There is no evaluation on non-compositional data.

### Questions
1. Is it possible to control the number of distributions in the method? If so, how does performance of your method scale with number of distributions starting from 1 and increasing beyond what you investigated, to greater understand what is the key component generating compositionality. Any additional / alternative ablations that could give more insight into this would strengthen the paper also.

2. How does performance compare on non-compositional datasets or datasets that do not explicitly target the requirement of compositionality?

3. Can you give examples of problems solved and failed by LPN and the Abduction Transformer to understand more clearly the failure and success modes of each?

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
This work addresses program synthesis tasks, with a focus on out-of-distribution (OOD) and compositional generalization. They introduce a new architecture named the Abduction Transformer, whose goal is to infer a latent cause (hence, abduction) of the observed specification for a given task, and use it to predict the solution of the considered task. This two-stage process implicitly defines a probabilistic graphical model corresponding to a deep variational Bayesian model, of the kind used in variational autoencoders (VAEs).

The main contributions of the paper include:
- a VAE formulation for neural program synthesis
- the representation of the latent space as a nonparametric Dirichlet Process (DP)
- gradient-based search in the latent space to maximize the likelihood of the specification
- training with 1 gradient step, using the ELBO loss

The authors experiment with the Adbuction Transformer on three different benchmarks: a 1D ARC dataset, Raven's Progressive Matrices, and a linguistic systematicity task. They highlight performance improvements due to latent search and some OOD generalization at test-time.

### Strengths
The paper is well written, the method is promising, and the experimental setups are somewhat well explained. I particularly enjoyed Figure 1, which details the method very well (although the DP equations could be skipped in this figure for higher clarity). In my opinion, the largest novelty in this work is the modeling of the latent space as a DP.

I believe the paper tackles a crucial aspect of machine learning, i.e., OOD generalization in the context of program synthesis, and demonstrates the proposed architecture quite well with a set of 3 different benchmarks. The related work section successfully connects the Abduction Transformer to recent works such as Test-Time Fine-Tuning [Hübotter et al., 2025] or the Latent Program Network [Macfarlane & Bonnet, 2025]. Overall, the work tackles an important problem in machine learning, and the proposed architecture seems promising to overcome some of the limitations of current SOTA systems.

### Weaknesses
There are several weak points I would like to address in this review.

## Novelty
The authors carefully relate this work to the framework of VAEs [Kingma et al., 2013] and the baseline Latent Program Network (LPN) [Macfarlane & Bonnet, 2025]. After carefully analyzing the [LPN paper](https://arxiv.org/abs/2411.08706) and comparing it to this work, it is not clear to me what novelty this work brings that is not covered in the LPN paper. This is the most important weakness to me; it seems that the LPN paper covers most of what is "introduced" in this work, in its current version.
- VAE network for program synthesis: in LPN.
- Training loss: identical to LPN, i.e., max likelihood and KL to prior.
- Training dynamic: 1-step of gradient descent in latent space, no backprop through latent update.
- Test-time search: gradient descent in latent space, maximizing likelihood of the specification.

A notable difference from the LPN baseline is the DP process used to parameterize the latent space (the baseline is a standard normal Gaussian distribution). If this is an important novel contribution from this paper, I would like the authors to demonstrate it (see ablations below).

## Experiments
The three experiment setups presented in this paper remain on the toy level. Since ARC-AGI is mentioned, it would improve the contributions of this paper to demonstrate the generalization abilities of the Abduction Transformer on the ARC-AGI v1 or v2 benchmark. Alternatively, the [ConceptARC benchmark](https://arxiv.org/abs/2305.07141) could be a more accessible version to try it on.

## Baselines
In addition to the LPN baseline, it would improve the paper to include other program synthesis baselines, especially if ConceptARC or ARC-AGI is considered to be added to the experiments.

## Ablations
The parameterization of the latent space as a DP seems interesting and would need to be ablated to be justified. The number of latent space gradient steps during training does not seem to be ablated or justified either.

### Questions
1. What made you go for a DP to model the latent space? What justifies moving from the standard implementation of Kingma et al., 2013?

2. What are the novel aspects introduced in this work that you would like to emphasize?

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose the Abduction Transformer to achieve better compositional generalisation on out-of-distribution (OOD) meta-learning tasks. The key idea is to represent each few-shot task with a nonparametric latent hypothesis, modelled as a Dirichlet-process (DP) mixture over latent vectors produced by the encoder. This allows the model to adaptively allocate a variable number of latent components to capture the compositional structure of the task. The authors evaluate their approach on three tasks: 1-D ARC-like program induction, symbolic Raven's Progressive Matrices (SRAVEN), and linguistic systematicity and show competitive performance with fewer parameters (1.2M), outperforming standard transformer baselines and comparable to GPT-5 Thinking on certain tasks.

### Strengths
- The paper has a clear probabilistic framing, casting few-shot meta-reasoning as posterior inference over a task-specific hypothesis, trained with a variational free-energy bound and refined by test-time gradient search.
- The DP prior over the latent space acts as a strong architectural prior that naturally aligns with the compositional learning premise.
- The authors show the results of extensive ablation studies on modeling choices such as KL-regularisation over the DP posterior and test-time gradient search, indicating both are crucial.
- The latent-space visualization suggests unseen compositions embed near their constituent primitives, which is qualitatively consistent with the paper’s goal of compositional reuse.

### Weaknesses
- From the paper, it seems that there are three types of composition: in data (function composition) as in 1-D ARC, in hypothesis space (production) as with the linguistic systematicity, and in features as independent submodules (parallel composition) as in SRAVEN. The issue is that the paper conflates these fundamentally different kinds of composition under a single formalism. Although all are expressed with the same symbolic notation $H_1(H_2(x))$, they correspond to distinct cognitive and computational challenges. Without explicitly distinguishing these regimes, it becomes unclear whether the reported improvements arise from genuine compositional reasoning or simply from learning parallel or modular mappings.
- The paper relies on approximate KL and truncating DP samples ($\kappa_0 = n + 1$), but does not quantify the approximation or truncation error in training or test-time search. Without rates of approximation or stability results, the “information-theoretic regularisation” claim remains empirical.
- Why does gradient descent on latent representations lead to better hypotheses? It lacks convergence analysis and could benefit from more insights into the geometry and identifiability of the latent optimization: when and why gradient updates align with true compositional structure rather than overfitting to spurious patterns.
- The method could benefit from even a small-scale analysis on a real compositional regime such as code editing, or some visual entities from ARC-AGI. Would it be possible to show such results?
- The baseline comparisons for methods that perform test-time adaptation are limited to just one: single vector LPN and there are no comparisons to other compositional generalisation methods (for some examples which might not be up to date, neural module networks, program synthesis approaches, neural interpreters etc). Even if some methods need adaptation or don't directly apply, discussing why would strengthen the paper. 
- The notation is confusing since the paper uses $H$ both as a _function_ (the hypothesis mapping in compositional generalisation) and as a _latent variable_ (a mixture distribution in the nonparametric encoder). This can be made clearer by using, for eg, $h$ as the latent variable and $H_h(\cdot)$ as the functional mapping. 
- The writing of the central claim can be made a bit clearer in my opinion. The 1-D ARC and SRAVEN setups are synthetic, controlled, and compelling for diagnosis, but the core gains are still modest, and comparisons to “GPT-5 Thinking” are zero-shot and prompt-based versus a trained model with test-time search making not like-for-like in compute or adaptation. 
- Appendix B claims denoising attention "subsumes" regular attention for _discrete distributions_ (categorical variables with additive noise), not for the _continuous Gaussian_ case used in the model. Without specifying assumptions like zero noise variance or small-variance limits, it is not clear the conditions under which this holds. 


I am happy to update the score of the paper if my questions and concerns can be addressed.

### Questions
- Under what assumptions on the decoder does the test-time objective exhibit a landscape such that gradient search converges to semantically correct hypotheses rather than spurious modes?
- It's interesting to see the model maintain high accuracy even on less than 7 samples from the linguistics task. How can you disentangle prior-based guesses from genuine compositional recovery?
- Maybe I am missing something but can you explain why the number of gradient steps while training is 1 but is 100 for evals?
- Why is it appropriate to average samples from different distributions at the encoder's output? Why is averaging done here rather than using a product of experts or other combination method?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces the Abduction Transformer (AT), a variational, test‑time‑adaptive architecture for systematic compositional generalization. AT encodes each few‑shot episode into a nonparametric latent space—a Dirichlet‑process (DP) mixture over a set of vectors—regularized by an information‑theoretic KL to a DP prior. A transformer decoder then cross‑attends to this latent mixture via denoising attention to generate outputs. Crucially, at test time the model refines the latent hypothesis $H$ by gradient descent on the few‑shot examples before answering the query, turning inference into iterative hypothesis testing. Across three tasks including 1‑D ARC‑like program induction, symbolic Raven’s (SRAVEN), and a linguistic systematicity benchmark, AT surpasses standard transformer baselines and a prior test‑time latent‑search model, and is competitive with zero‑shot GPT models under the authors’ protocol.

### Strengths
* This paper proposes a clear mechanism for compositionality through the procedure of (1) encoding few‑shot pairs, (2) inferring a DP posterior over latent hypotheses, (3) sampling a mixture $H$, (4) decoding, (5) refining $H$ with gradients on the few‑shot loss, and (6) answering the query. This operationalizes “search over hypotheses” rather than memorization.
* t‑SNE visualizations in fig. 7 show clear clustering of primitives and sensible placement of unseen compositions near their components, consistent with the intended semantics of the latent hypothesis space.
* Instead of a single latent vector, AT projects encoder outputs into DP parameters $(\mu,\sigma^2,\alpha)$ and samples a mixture with a variable effective number of components, which is an upgrade from fixed-size latents and aligns with the transformer’s token‑proportional outputs.
* Strong OOD results on diverse tasks are achieved with small models. AT achieves 25.1% perfect solve on strictly OOD 1‑D ARC compositions and 46.1% on SRAVEN with only 1% of rule combinations seen during training. The results match or surpass that from GPT-5 (Thinking) with only a 1.2M parameter budget.

### Weaknesses
* It is not very clear why denoising-attention is used in the AT decoder, and how much of the performance gain should be attributed to this design.
* The comparison to GPT-5 can be a bit unfair, as it uses a "low effort" setting, while a large reasoning token budget is quite important for getting good performance in such reasoning tasks. Also, being able to update the latent hypothesis at test time could be a huge advantage.

### Questions
* In table 2, why does the gradient-based refinement of $H$ harms the solve rate when trained on 90% of the possible compositions?
* In table 1 and 2, how does the performance of GPT scale with the reasoning token budget?
* How is the computational overhead compared to baseline methods such as Single Vector LPN?

### Soundness
2

### Presentation
3

### Contribution
3
