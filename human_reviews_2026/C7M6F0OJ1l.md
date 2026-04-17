# Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures

- Decision: Reject
- Scores: 2, 4, 4, 8, 4

## Abstract
Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability. Additionally, the architecture leads to significant improvements in computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces and evaluates a new Sparse Autoencoder (SAE) architecture designed to capture Hierarchical relations in the data. This method brings a mixture of experts approach to an architecture similar to Matryoshka-SAEs. The authors explore the interpretability of the features recovered by this new method and compare it to standard SAEs along commonly used metrics.

### Strengths
- Adressing the limitations of SAES is an important line of research
- Combining ideas from mixture of experts with an approach like Matryoshka-SAE seems like a promissing direction.

### Weaknesses
Unfortunately, the paper fails to make a convincing case for the advantages of H-SAE over existing methods due to the following limitations of the evaluation methodology.

**Sparsity as a confounder:** The authors use metrics from SAEBench but do not incorporate the following evaluation practice prescribed in the SAEBench github page. This makes it unclear whether H-SAE improves performance at the same sparsity level, plots for the key metrics over sparsity levels at the same dictionary size seems crucial to establish the advantages of H-SAE.

>'Recommended Evaluation Practice
When evaluating new SAE methods, we strongly recommend training multiple SAEs across a range of sparsities (e.g. L0 ∈ [20, 200]) alongside directly comparable baselines. Many evaluation metrics correlate strongly with sparsity, so assessing performance across multiple sparsity levels is essential to avoid misleading conclusions.

>This practice helps ensure that observed improvements are real and not just artifacts of sparsity or statistical noise. It also makes it easier to determine whether a method actually improves the Pareto frontier on target metrics.'

Very limited comparison to the most similar existing method (Matryoshka-SAE). Figure 4 would be more convincing if it were a comparison with Matryoshka-SAE or at least BatchTopK-SAE at the same sparsity level with other examples in the Appendix to show this one isn't cherry picked.

### Questions
I think the H-SAE approach is reasonable, but the evaluation of the method has severe limitations. To increase my score I would need to see:

1) Plots comparing existing SAEs (ideally including Matryoshka and BatchTopK) to H-SAE along different sparsity levels for the same dictionary size.
2) Reporting of sparsity levels for the different SAEs in the current results/figures and making sure they are similar for a fair comparison. If the sparsity levels are different, the l1 coefficient or topk values should be adjusted to line them up.

Beyond this I would also like to see a more detailed evaluation and discussion of the pros and cons of this method when compared to existing SAEs. While I think the approach is interesting and worth pursuing, given the magnitude of the changes needed in the evaluation I would recommend taking some time to improve the paper and resubmitting rather than trying to make all these changes in the rebuttal, but I could change my mind if new convincing evaluations are provided.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a hierarchical sparse autoencoder approach.  This includes a novel architecture and 3 tweaks to the training process (which are tested in ablations in the appendix), and is inspired by "The Geometry of Categorical and Hierarchical Concepts in Large Language Models" - Park et al., ICLR 2025.  

The basic idea is this: 
- There is one top-level SAE, and many low-level SAEs, one for each feature in the top-level SAE. 
- You first run the top-level SAE, and see which features are active.
- Then for each active feature, you run the corresponding low-level SAEs.
- The reconstruction of an input is given by the sum of the outputs of all of the (top-level and low-level) SAEs that were run on that input. 

The low-level SAEs also include projection matrices for dimensionality reduction(/expansion).

The authors highlight:
- The conceptual motivation, novelty, and significance.
- Improvements in reconstruction and interpretability.
- Computational savings.

### Strengths
I find the contribution of the paper compelling.  It seems well-motivated, significant, and effective.
The lack of feature hierarchy in SAEs does seem like an important issue, especially in light of Park et al. (2025).

The paper is quite clear and well-executed overall.  The main exception to this is that Matryoshka SAEs are not given enough attention; this is a critical weakness to me at the moment, since I think it's essential to clearly demonstrate advantages and novelty over the most closely related work.

### Weaknesses
- (major) I’d like to see the Matryoshka SAE featured more prominently.  It's plausibly a more important point of comparison than a standard SAE.  So I'm surprised to see the experiments being kept to the appendix and being more limited (e.g. no comparison in terms of explained variance).  I would also usually expect it to be introduced in sufficient detail to understand how it works without reading their paper (e.g. including key equations).  And I think the claim that H-SAEs have “fewer uninterpretable features” needs to be explicitly supported when it is made.  I’m also not sure if the comparison between the two methods is fair in terms of parameters and compute; this should be clarified.
- (minor): I'd like to understand the connection with Park et al. more thoroughly,
- (minor): Incorrect characterization of background / related work in the first sentence of the abstract: “Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract
space.”  But this is not necessarily the goal of sparse dictionary learning.
- (minor): I'm unconvinced by the way this work is framed as related to causal representation learning.  I’m not aware of any reason to expect the features learned by a sparse autoencoder to be causal.
- (minor): “In particular, they show that every categorical concept” I think “show” is a bit strong here; I believe this is in the context of a particular formalism, and it’s unclear the extent to which LLMs universally mirror it empirically.
- (nit): “feature absorption” is not defined

(just FYI) Some typos:
- “We also highlight Switch SAEs (Mudide et al., 2024), which introduce a mixture-of-experts type routing mechanism shares some structural similarities to our approach.”
- citeps in first paragraph of related work
- “The H-SAE with 64 sublatents per expert is on par with the standard SAE with 32 top-level features” → “The H-SAE with 8k top-level features and 64 sublatents per expert is on par with the standard SAE with 32k top-level features”

### Questions
While the work stresses computational efficiency gain, it seems like it also potentially increases the number of parameters, which is worth noting.  Relatedly, I’m not quite convinced by the claim that: “because the memory cost of a batch gradient step scales with the number of activated parameters, not the total number of parameters, this efficiency also applies to (per-step) training.”  Indeed, within a (large) batch, I’d expect many different top-level atoms to be active, requiring the loading of many low-level SAEs.  Furthermore, I’m not sure if parameters can be moved on and off GPU quickly enough to make this practically useful (e.g. if memory is the bottleneck on your GPU, then I think you would need to reduce the number of top-level atoms or move low-level SAE’s parameters on and off the GPU with every batch). I’d like to see versions of Figure 3 with parameters and compute cost on the x-axis.

An important limitation which I don’t see mentioned is that the model is restricted to a 2-level hierarchy.  Would it be straightforward to extend to arbitrary depth?

Does eqn (1) really need a LeakyReLU in it?  Won’t the top-K activations typically be positive anyhow?

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
5

### Summary
In this work, the authors propose a new architecture for Sparse Autoencoders to extract hierarchical concepts from data. They draw from previous theories on the hierarchical representation of concepts in LLMs to design an architecture with an inductive bias to find high-level parent features and low-level children features. Specifically, this is instantiated as a mixture-of-experts approach where a Top-K SAE finds the high level concepts present in a representation and then selective activates low-level Top-1 SAEs which determine the specific child feature pertains to the input.

### Strengths
* The authors provide a novel architecture and approach grounded in theories of hierarchical concept representations in LLMs
 * The method is simple and intuitive
 * Empirical results highlight the efficacy of the method in terms of reconstruction metrics as well as more nuanced benchmarks such as feature absorption and feature universality across language.
 * I think the absorption experiment is a great way to demonstrate the efficacy of this method, as it intuitively (and empirically) seems clear that hierarchy should help with the splitting problem.

### Weaknesses
(Apologies for any clarity issues, I bounce between saying high- and low-, top- and low-, etc. to describe your hierarchy of features).

The authors argue that this architecture is more useful for three primary reasons: 1) H-SAEs have better reconstruction, 2) H-SAEs are more interpretable, and 3) H-SAEs learn hierarchical semantics. However, I see some issues with these claims:
1. If I understand Figure 3 properly, you compare H-SAEs against SAEs with the same number of top- or total number of latents respectively. This means that, with the "expert" sub-SAEs, the H-SAEs have 32 or 64x as many parameters as compared to an SAE, which I believe is an unfair comparison. I think it would be more interesting if you can show that given the same computational or parameter budget, reorganizing your SAE into a hierarchical structure is better. This would convince me more strongly that this inductive bias is the right way to learn the representation structure as it would demonstrate it is more efficient at learning and better at reconstruction. Instead, I can't be sure that these gains in performance simply come from using a bigger model.
2. Its hard for me to interpret the quality of the numbers in Fig. 5b. If we are looking at set difference, it seems to me that the both SAEs differ on a significant proportion of features. If I were to frame it another way, out of the top-8 features for both SAEs on different translations of the same sentence, I can expect about 4 to differ, which is significant. It seems as though a difference of about 7 versus 9 is minor compared to the fact that both SAEs clearly exhibit disagreements across languages. I appreciate the introduction of a new evaluation for feature redundancy, and would love your comments on what you expect as a 'good result' for this evaluation.
3. I agree that by design, we should see 3) and it appears that this is happening qualitatively. However, the performance of this is briefly addressed in a small paragraph on line 459 and not much is discussed beyond qualitative examples. I understand that a common difficulty in interpretability research is providing quantitative or large-scale evaluations, but I think this is still quite brief given it is the entire goal of the paper. Most of the experiments are devoted to reconstruction and the 2 interpretability benchmarks, but I feel like this is the most important component of the paper. From a practitioner's perspective, I guess I would pick to train an H-SAE over a normal SAE because of the reconstruction results, but from a scientific perspective, a much more interesting question is using your method to evaluate the veracity of the hierarchical concept representation hypothesis. How hierarchical are LLM representations in practice? If this hierarchy is the right model of LLM representations, how will we know and what can we do with it? Instead, this paper's contribution seems to mainly be bumping existing metrics up with some qualitative examples of why it is better.

Other comments:
 * Additional loss terms: The discussion from lines 300-310 confused me a little bit. I read it as "we did X and Y because we think it is good, but its not clear if it actually is." Reading this part weakened the messaging of the paper, and it seems as though it was unnecessary. Maybe this just needs to be framed more confidently, but I think it might be better off removing these parts. When I read Algorithm 1, my first question was why you were including a sparsity loss given you were using top-k architectures, which don't use a sparsity penalty [1]. Later I understood that this is being used as an auxiliary loss term on the non-activating features, but it appears in Alg 1 to be applied to all features. Even so, why change from the usual auxiliary loss? It makes it harder to compare unless you can also run an ablation comparing these two design choices.
 * Efficiency claims: Similar to my point above on reconstruction metrics, how would the computational efficiency compare with an SAE trained with the same number of **total** parameters (so j + js features or just js features)? 

Typos/Formatting:
* Line 196, it seems as though you added too many parentheses in z_j = (LeakyReLU(Ex))
* Figures 3 and 5 are formatted quite badly. You should remove the subcaptions and include them all in the overall caption as "Figure 5: **(a)**: The standard SAE... **(b)** The H-SAE...". Instead, you have 3 captions that are all aligned differently and run off the page. Additionally, you should try to align the left and right subfigures vertically, and avoid having them break the margins if possible.
 * On line 367, do you mean 32k and not 32 top-level features?
 * The citation for "Unlocking Hierarchical Concept Discovery in Language Models..." is in all caps.

Overall, I really like this idea and the proposed architecture, but the evaluations fall a little short and are closer to the minimum required to evaluate a novel SAE. I think this work has huge potential for interesting applications, evaluations of the method, or evaluations of the hierarchical concept representation hypothesis and I would be willing to improve my score if these were explored or addressed but that would require pretty large changes to the existing submission. I also apologize for the long-winded review, I am more than happy to clarify anything during the discussion period!

[1] Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., ... & Wu, J. (2024). Scaling and evaluating sparse autoencoders.

### Questions
* Did you try having the low-level SAEs take in the residual only? I.e. instead of $SAE^j_1(\Pi x))$, doing  $SAE^j_1(\Pi (x-x_{high}))$.
 * I would be interested in seeing an ablation of Top-1 for the low-level SAEs. Why not try allowing for more low-level concepts? Maybe a representation contains information about a wedding AND a divorce. This relates to my point in the Weaknesses section -- this paper would be much more compelling with a thorough investigation of the hierarchical decompositions unlocked by your model and more evaluations or explorations of this component as opposed to the brief treatment they currently receive. 
 * Additionally, I would be interested in understanding the hierarchical structure of the language features explored in the redundancy experiment. You implicitly hypothesize that the top-level feature would be a multilingual syntax feature, with the low-level features describing language difference, but what if it were the other way around? Specifically, a high-level language feature, each with its own low-level syntax feature. I find this discussion interesting and recent work has looked at the relationship between syntax and meaning in SAE features and notably specifically modeled syntax as a low-level feature [2]. I would appreciate a deeper dive into the structure that explains Figure 5b and how your hypothesized hierarchy informs your evaluation design. I would also appreciate an explicit description of your hypothesized hierarchy (although you do slightly touch on this in the last sentence on page 8).
 * On that note, one thing [2] does is they evaluate their high- and low-level features separately. I think this sort of investigation could also improve the work: Would the feature redundancy be much lower if just looking at top-level features? How does absorption change when looking at top features or sublatents or both? Are you doing all of the evaluations in this paper on top-level features only?
 * In footnote 1, when you say primal and dual space representations, what do you mean? Why is a dual being introduced?

[2]: Bhalla, U., Oesterling, A., Verdun, C. M., Calmon, F., & Lakkaraju, H. Leveraging the Sequential Nature of Language for Interpretability.

### Soundness
4

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
4

### Summary
This paper tackles the well-known reconstruction-interpretability trade-off in SAEs, which often suffer from feature splitting as they are scaled up. The authors propose a new architecture, the Hierarchical SAE, which explicitly models the hierarchical nature of semantic concepts. The architecture uses a MoE-like design: 1. A top-level SAE identifies general "parent" concepts. 2. The activation of these top-level features gates smaller, specialized "expert" SAEs. 3. These experts model "child" concepts by operating in a dedicated low-dimensional subspace. 4. The final reconstruction is a sum of the high- and low-level representations, directly implementing the "corgi" = "dog" + "corgi-in-dog-space" structure.

When applied to LLM internal activations, this H-SAE architecture improves reconstruction performance. It matches a SAE 4x its size while being more computationally efficient. It also learns more interpretable and less redundant features.

### Strengths
The core idea of using a hierarchical, MoE-style architecture for SAEs is simple but practical. The gain in computational efficiency is a significant result on its own, potentially unblocking efforts to scale interpretability tools to frontier models.

The experimental validation is quite thorough. The authors combined standard reconstruction loss with metrics for downstream task performance like CE loss, as well as feature absorption and cross-lingual feature redundancy. The direct comparison to Matryoshka SAEs on multiple tasks including their own synthetic benchmark is convincing as well.

The paper's core mechanism is easy to follow from the architectural diagram (Figure 2) and the main equation (Equation 3). The qualitative examples provide concrete evidence that the model is learning the intended structure. E.g., Figure 1 shows a high-level "marriage" feature with "divorce" and "engagement" subfeatures.

### Weaknesses
One limitation is that proposed architecture implements a two-level hierarchy (parent/child). Real-world semantics are often deeper. The paper doesn't discuss the limitations of this two-level structure or how the architecture might be extended to model deeper, more complex hierarchies.

The subspace dimension $s$ is a key hyperparameter, set to 4 or 8. While this is motivated by the low-rank finding from prior work and benefits efficiency, there is no sensitivity analysis or further justification for these specific values. It's unclear how performance would change with a larger $s$.

Last but not least, since feature absorption is a central problem that the paper tackles, adding references to prior sota/influential work on this benchmark, such as https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ycscpaQAAAAJ&citation_for_view=ycscpaQAAAAJ:2osOgNQ5qMEC and https://www.alignmentforum.org/posts/zbebxYCqsryPALh8C/matryoshka-sparse-autoencoders, would better contextualize the contribution.

### Questions
Continuing from the weaknesses, H-SAE implements a two-level hierarchy. Have you considered or explored extending this to deeper, multi-level hierarchies (e.g., by making the low-level "expert" SAEs hierarchical themselves)? What challenges do you think will likely arises (e.g., optimization stability, vanishing activations)?

Could you clarify the impact of the $\mathcal{L}_{ortho}$ term in the main experiments? The appendix ablation suggests it's mainly for preventing dead latents. Does it have any negative or positive impact on the quality of the learned hierarchy or the reconstruction performance on the residual streams, compared to just using a different dead-atom mitigation technique (like the one from Gao et al.)?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel SAE architecture that models learned features in an explicit hierarchy. Specifically, it employs a mixture-of-experts method that activates sub-features within activated features of the model. The authors empirically explore how the architecture improves the “reconstruction-interpretability” tradeoff, finding that HSAEs improve reconstruction and feature absorption.

### Strengths
- Hierarchical encoding is a very interesting topic and an important component of modeling LLM representations. Hierarchical structure is inherently quite interpretable, and intuitively resolves many issues with existing SAEs (such as splitting and absorption).
- Additionally, this work is grounded in existing literature on the hierarchy of representations in language models, and the MoE architecture is an intuitive operationalization of hierarchies.
- The results are promising, particularly for efficiency and for absorption.

### Weaknesses
- Given that for each high-level feature, there are 16 or 64 sub-features, shouldn’t the authors  be comparing to standard SAEs with 16x or 64x the number of normal latents (essentially the size of all of the sub-features from HSAEs combined) to ensure overall width is the same? If they are already doing this, please ignore this comment but clarify this point in the paper.
- Why not always compare to both TopK and Matryoshka SAEs? For each experiment, the authors only use one or the other as the baseline, making it difficult to determine where HSAEs improve upon current baselines.
- The interpretability gains are not currently very clear to me. Given the qualitative examples provided, hard to tell which is better (e.g. Figure 4, App D.2). Is there a way to qualitatively measure improvement? While automated interpretability is known to have limitations, results on this metric would be helpful. Similarly, evaluations of downstream interpretability applications (such as steering) would also help back of the claims of improved interpretability.

- Minor comments:
    - Please move Figures 3 and 4 closer to their references in the text.
    - Please provide standard deviations for Figure 5b.
    - The captions of Figures 3 and 5 are not within the margin requirements, please add them to the normal caption with the tags (a) and (b) inline.

### Questions
- To measure compositionality, please see Archetypal SAE proposed metrics.
- Did the authors explore any examples of improved downstream utility for HSAEs over existing SAE architectures?
- The implementation of hierarchical SAEs feels very similar to the notion of sparse group sparsity from prior sparse coding and dictionary learning literature. Did the authors try the optimization techniques from prior works on group sparsity?
- What is the motivation for the “small ℓ1 sparsity penalty on the latent values on both the top and low-level features outside the top k”? Won’t this simply result in downscaling?

[1] Fel, Thomas, Ekdeep Singh Lubana, Jacob S. Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. "Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models." ICML (2025).

### Soundness
3

### Presentation
2

### Contribution
3
