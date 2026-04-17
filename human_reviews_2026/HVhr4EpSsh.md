# Path Planning for Diffusion Language Model Sampling

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Any order generation of discrete data using masked diffusion models (MDMs) offers a compelling alternative to traditional autoregressive models, especially in domains that lack a natural causal ordering of data. However, current popular masked diffusion models depart from their successful continuous diffusion model counterparts with simplified masked inference wherein unmasked tokens cannot be iteratively refined- even if there is a mistake. In this paper, we extract the full power of MDMs by introducing a novel inference sampling strategy termed Path Planning (P2) that decomposes each generation step into two sub-stages: planning and denoising. Under P2, the planner at every step selects appropriate tokens that are marked to be updated, which can then be sampled using the denoiser. We demonstrate that P2 generalizes all existing sampling strategies for MDMs and critically enhances generative quality through the new capability of 
refining and updating existing unmasked tokens. We theoretically prove that P2 establishes a (new) expanded evidence lower bound (ELBO) on the log marginal likelihood of data. We instantiate P2 with a family of planners including: 1.) Self-Planning, 2.) BERT-Planning, and 3.) Trained-Planning with a learned planner leading to SOTA generative performance for MDMs on a suite of domains. Specifically, solely using P2 inference, we observe relative improvements of $22\%$ in protein sequence foldability, $8\%$ in RNA sequence pLDDT, $4\%$ in math reasoning, $68\%$ in story generation (ROUGE score), and $33\%$ in code generation for the challenging pass@1 metric.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a planned denoising algorithm for masked discrete diffusion models, which is an important topic of study in this area. Here, instead of choosing to denoise the masked tokens at random, and keeping them that way till the end, the planner model chooses to unmask the tokens with most confidence and re-mask possibly erroneous tokens (a feature found in uniform diffusion but not in masked diffusion).

Three variants are introduced for training such a planner, and they are shown to improve the generation quality for protein folding, RNA sequence generation and various language/math/coding tasks. The authors differentiate the current work from Kim et al, which uses heuristic planning algorithms for masked Diffusion models, and Liu et al which works mainly with uniform diffusion instead of masked diffusion.

### Strengths
This work considers the well motivated problem of planned denoising in discrete diffusion models. Better inference time algorithms can significantly improve the efficiency and accuracy of diffusion models, as amply demonstrated in the literature. The algorithm is simple and shows strong gains across multiple datasets. This work demonstrates the utility of a trained planner for masked diffusion models, and shows that it can lead to better results than inference time heuristics such as topK-marginal sampling.

### Weaknesses
- Most of the ideas in the paper seem straightforward and introduced in some form in prior works. I therefore am not very positive about the novelty of this work. (Kim et al, Liu et al, Campbell et al etc)

**Missing Baselines**

- In the space of planning and constrained generation, the authors need to compare to the remasking strategy used in Algorithm 2 in https://arxiv.org/pdf/2410.14157

- Kim et al seems like an important baseline to this work, and is easily implementable on any MDM since it is an inference time heuristic. The authors only compare to this work in the protein folding task. The evaluations seem incomplete without comparison in the language modeling and the RNA sequence generation tasks as well. 

- The DDPD algorithm of Liu et al can be modified to work with pretrained masked diffusion models, as clearly acknolwedged in this work. Since the authors themselves claim that this is the closest work, I am surprised that there is only a single comparison with this method, that too only in the appendix. I do understand that this requires extensive training, comparable to the current work. I believe that such a comparison will make the current work's case much stronger. 


**Minor points:**

"Specifically, solely using P2 inference, we observe relative improvements of in 22 protein sequence foldability,  8 in RNA sequence pLDDT, 4 in math reasoning,  68 in story generation (ROUGE score)...."

Usually absolute improvements are reported in this literature. I request the authors to change this to absolute numbers. For instance,  in table 2, the foldability improvements seems to be 10% absolute instead of 22% relative.

### Questions
See weaknesses section.

- Does the change in the path measure given by the stochasticity parameter $\eta$ change the final marginal distribution of the samples? Is there any reference to a theoretical analysis with arbitrary $\eta$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The crucial bottleneck in existing MDM at inference is that they can not recorrect what tokens have been already unmasked, resulting in error propagation and suboptimal generation quality. The paper introduce a path planner to decide which token to unmask and which token to remask, allowing to refine exising unmasked tokens. The authors provide a theoritical support on the evidence lower bound of the framework and empirally instantiate planer from various source: self-planing (model itself), BERT-planning model, and trained-planning model. The method demonstrates SOTA performance on conducted benchmarks, such as biological sequence, math reasoning, and code generation.

### Strengths
1. The method is technically sound and indeed novel since it equipping the diffusion model a error-correction capacity in a clean and disentangled way by a denoiser and a planner.
2. The method is well motivated with a easy-to-read presentation.
3. The quanitative results are strongly compelling.

### Weaknesses
1. The name of planner is a bit confusing: mask planner denotes denoiser, unmask planner refers to the "actual" planner.
2. The rationale of using BERT as planner remains ambiguous. It would be better to provide insight on why BERT predicted probability is reliable indicator.
3. Training details of trained planner is missed.

Exposition: 
- Related work: Paragraph should start with a capital word.

### Questions
Q1. Since discrete diffusion itself can be used a planner, what happen if we use a large discrete diffusion to server as a planner. Expect the performance will be better?

Q2. In Table 3, what type of planner is used? I assume this is self-planner.

Q3. Do the author have a table for comparing different sampling strategies on math and coding performance? Similar to what have done in Table 5 (protein generation) and Table 6 (infilling).

### Soundness
4

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
5

### Summary
This work formalizes the problem of selecting unmasking and remasking positions of MDMs. The authors provide extensive experimental results to support their method's empirical usefulness.

### Strengths
The authors put effort into formalizing the unmasking and remasking predictors with ELBO, and provide experimental results on various domains.

### Weaknesses
I believe there are a few crucial flaws and limitations in the author's theoretical claim. (I'd be happy to be further clarified by the authors on these)

- **Claim on remasking**: Beyond P2-Train, in P2-Self and P2-Bert, the authors incorporate the training-free remasking strategy into the unmasking schedule, which they argue as an unification. However, the information that they use for remasking has never been learned. For example, in P2-Self, the authors use the pretrained MDM's logits to select which positions to remask, but a model has never gotten the training signal on clean token positions, so it isn't that much different from randomly remasking clean tokens, as done in (Wang et al, 2025)
. Given that investigating the unmasking order using logits has also been previously studied, e..g, Kim et al, MaskGIT, just incorporating the remasking scheme into unmasking shouldn't be the core contribution.

- **Explanation on P2-train**: Indeed, P2-train is the trainable approach to get an unmasking and remasking planner, in contrast to prior work that focuses on a training-free approach. Given that P2-Self and P2-Bert don't differ that much from the prior approach (as I claimed above), the authors should've dealt more carefully with P2-Train. Below are the reasons why I don't think the author's account on P2-Train is inappropriate.

- *Training algorithm*: In Algorithm 2, the authors argue that they use the indicator function $\mathbb{1}[z^i = x_0^j]$ as a label. But given the sampling process of D_t, xt, and z should have the same clean token at the clean positions. Doesn't it mean that this label is always one for the clean token position? If we follow this algorithm, $G_U(z,x_t)$ always outputs one for the clean token position, which is apparently what we don't want, since the training signal given to $G_U$ is always one at clean token positions. It is even unclear in the code whether we use the same backbone to model $G_M$ and $G_U$. Also, the ELBO loss in Proposition 1 tells that $G_M$ should $z^{-i}$ takes an input. I was wondering how the authors addressed this input discrepancy. This is because it seems tricky to deal with since it takes $z$ as an input, but behaves like it doesn't take $z^i$, even in the recent work (Zhao et al, 2024) uses an entirely new architecture to address this issue.

- *No detail on planner design*. I failed to find the training details of the planner, which I think is the most important part of the paper. To clarify, I'm not referring to the specific implementation detail but the basic training configuration of how to train $G_U$ and $G_M$. 

- *No theoretical guarantee*. The MDM training objective itself has the theoretical guarantee (if the loss is exactly minimized, it recovers the per-token posterior). But the authors don't provide that guarantee; it's even unclear what the planners are aiming towards. (Note that the ELBO loss cannot explain this). 

- **Paper writing**: Last but not least, the paper is extremely hard to follow (at least for me). Although the idea (adding additional modules to learn the unmasking and remasking positions selection) is really simple, the authors formulated it in an unreasonably difficult way, further decreasing the readability. I personally had a really hard time digesting the author's proposed formulation and realized that it can be even more simplified.

### Questions
Please refer to the Weakness section!

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a planning module that specifies what tokens to unmask on priority and also what tokens to re-mask for a masked diffusion model given a partially masked canvas in an intermediate decoding step. Prior to this decision, the planner module (one for marking and other for unmaksing) first take existing canvas an input and also the sampled tokens applying denoiser to decode all positions at once. If the planner module suggests unmasking the sampled tokens are put in and if the module asks to remask, that position is masked again and if the planner module does not suggest anything , then the previous input token is retained. 

Contrary to many other planning related works before (chiefly Liu et. al. 2024  and Kim et. al. 2025), this enables remasking and works for MDM in a general fashion. Another insightful contribution is that the sum of planner module training (for both remasking, unmasking) losses and denoiser training loss is a valid ELBO lower bound in the limit. Authors provide a CTMC viewpoint of masked diffusion and show the ELBO result from this CTMC viewpoint.

Authors instantiate planners using 1) BERT models 2)  denoiser's own logits is used to obtain planning information and 3) training a new planner. Authors show that such planner training with frozen denoiser and applying at inference time on top of existing Masked Diffusion Models improve performance across datasets - molecule generation, RNA generation and downstream evals for language generation.

### Strengths
1) Paper's ELBO justification for the additional masking and unmasking planner losses with an expanded treatment (with CTMC viewpoint of MDM) is very valuable. I went through the proofs at a quick pace. They appear correct.

2) The experiments span molecule generation and language generation. So in that sense it is comprehensive. 

3) Comparisons and discussion about close baselines has been provided (although I have some reservations here mainly)

4) Paper admits re-masking while most MDM planners before only plan for masking strategies.

### Weaknesses
1) Liu et. al 2024 already trained planners (but in uniform diffusion and not masked diffusion). Authors contention (I did read Section D.4 ) is that there is no umasking based planning. To be fair to the Liu et. al 2024 paper , given that it is uniform diffusion, there is no need for specific unmasking planning - because the only decision that needs to be made is whether to flip or not from a current token. In fact, roughly in the author's algorithm that is the final decision being made (but my combining two planners since you have to account for the absorbing state). So in that sense, the essential idea in non absorbing diffusion has already been made.

2) Comparison with Kim et. al 2025 has been made in some tables (Top-K Marginal). I agree that re-masking brings considerable benefits in the RNA and molecule generation tasks which are heavily (and this is rather well known at this point) constrained. However, Top-K Marginal comparison and Top-K Margin (another method that Kim et. al 2025) propose has not been compared with for the language tasks. Except infilling stories task (which is a bit tailor made for diffusion models), coding and many other evals only show modest gains compared to other baselines even at 7B scale for language modeling.

3) RDM sampling already does re-masking and masking based planning according to the authors own table. The only thing missing is stochastic control which is just additional hyper parameter tuning to mix these two (as in (7) in the paper). In fact in Table 5, the P2 (self) which used the denoiser logits to plan, compares barely within a 1-2% margin from RDM sampling for the protein sequence generation task.
Only this would be a fair comparison because RDM uses its own denoiser to plan as additional compute in train and inference is used by the best P2 models in Table 5. 

So the relative benefit over RDM is also not very clear.

### Questions
I have asked all my questions regarding comparisons and novelty with respect to Kim et. al. 2025, Liu et. al. 2024 and RDM Sampling in the weakness section. 

1) RDM Sampling is really close but for stochastic control in methodology. The gain seems to be marginal in compute matched settings. Idea of planners have been further explored in Kim et. al. 2025 and Liu et. al 2024. 

2) The language modeling experiments do not show any big difference except in story infilling tasks and misses comparison to Kim et. al. 2025.

Given 1) and 2) (and points elaborated in the weakness section), except for the elaborate theoretical contribution, with respect to novelty of technique I find a lot missing given the claims in the paper. I would like the authors to respond to this.

I will consider raising my scores based on the author's response.

### Soundness
3

### Presentation
3

### Contribution
2
