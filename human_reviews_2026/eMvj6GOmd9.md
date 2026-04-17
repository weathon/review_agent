# Low Rank Weight Bases for Visual Analogies

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 8

## Abstract
Visual analogy learning enables image manipulation through demonstration rather than textual description, allowing users to specify complex transformations that are difficult to articulate in words. Given a triplet $\\{\mathbf{a}, \mathbf{a}', \mathbf{b}\\}$, the goal is to generate $\mathbf{b}'$ such that $\mathbf{a} : \mathbf{a}' :: \mathbf{b} : \mathbf{b}'$. Recent methods adapt text-to-image models to the analogy task using a single Low-Rank Adaptation (LoRA) module, but they face a fundamental limitation: attempting to capture the diverse space of visual transformations within a fixed adaptation module constrains generalization capabilities. Inspired by recent work showing that LoRAs in constrained domains span meaningful semantic spaces that can be interpolated, we propose LoRBa, a novel approach that specializes the model to each analogy task at inference time through dynamic composition of learned transformation primitives, informally, choosing a point in a "*space of LoRAs*". We introduce two key components: (1) a learnable basis of LoRA modules, to span the space of different types of visual transformations, and (2) a lightweight encoder that dynamically selects and weighs these basis LoRAs based on the specific analogy pair. Through comprehensive evaluations, we demonstrate that our approach achieves state-of-the-art performance and significantly improves generalization to unseen visual transformations. Our findings suggest that LoRA basis decompositions are a promising direction for flexible visual manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors propose a new approach, called LoRBA, which specializes in the model for image transformation tasks by leveraging the learned low-rank adapter parameters. The method follows a visual analogy framework, where the model is required to generate a new image by applying the same visual changes observed in a reference image pair to a target image. It contributes to improving the model’s generalization across diverse unseen image transformation tasks through a new architecture and provides detailed component-based analyses.

### Strengths
•	The paper provides a detailed ablation analysis that affects performance. 

•	A variety of methods have been utilized to evaluate the results.

### Weaknesses
•	The paper employs the term “visual analogy” primarily in the context of style transfer, object addition, and related image transformation tasks. However, visual analogy refers to a broader process of relational knowledge transfer, extending beyond image manipulation. 

•	The visual analogies were referred to as Visual Prompting and Visual Relations in Section 2. While these are related concepts, visual analogy generally encompasses a broader conceptual scope and is not strictly identical to the other two. It may be helpful for the authors to distinguish these terms more clearly.

•	The paper proposes to decompose visual analogy learning through the LoRBA architecture and includes edit prompts in experiments that describe the intended transformation. In visual analogy, however, the transformation is inferred from the relation between the reference pair (A → A′). When the text prompt explicitly defines this relation, the reasoning aspect of analogy is effectively replaced by instruction following. In that case, the task aligns more closely with prompt-conditioned image transformation that follows the analogy structure (A→A′ :: B→B′), but not the reasoning process itself.

•	The quantitative results for Preservation VLM, Edit Accuracy, CLIP, and LPIPS show that the performance of LoRA and LoRBA are quite close. Similarly, in the qualitative comparisons in Figure 4, LoRA appears to produce visually comparable results. Given that the observed performance gap between LoRA and LoRBA is relatively small, the reviewer is uncertain about why the community utilize this architecture instead of LoRA?  What is the main reason that makes LoRBA better than LoRA?

### Questions
•	In Section 2, what type of transformations are inferred? 

•	Figure 2 is missing to show the edit prompt in the process.

•	How is LoRBA conceptually and architecturally different than inspired work of Dravid’s?

•	Regarding the test data created with 18 community LoRAs, were the outputs manually reviewed or verified by human evaluators? How are quality of the results ensured?

•	Table 1 requires a clearer explanation, as it is currently difficult to interpret without additional context or guidance.

•	The variations of capacity effect evaluation, such as {N = 32, r = 4}, are missing or not described well in Table 1.

•	Figure 7 is missing the results of LoRA.

•	What are the general tasks such as style transfer, background replacements, object insertion, object displacement etc. that LoRBA fails to generate accurate results and what might be the reason for that?

•	While the paper is motivated by the concept of visual analogy, the use of explicit edit prompts (e.g., “Turn this photo into an architectural rendering”) defines the transformation in advance and bypasses the reasoning process of visual analogy. It would be valuable to include an experiment without explicit transformation prompts to assess whether the architecture itself can work visual analogy task by inferring and applying the relation purely from the exemplar pair.

•	One of the related works using the same exemplary-based image editing method with LoRA is PairEdit. An additional experiment can be conducted to evaluate PairEdit on the same test set and compare its results with LoRBA?
PairEdit : https://arxiv.org/abs/2506.07992

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the task of visual analogy learning. The authors propose a two-stage framework: first, they train a set of LoRA modules alongside a corresponding set of learnable combination weights. In the second stage, these LoRA weights are combined to generate an image that fulfills the analogical relationship.

### Strengths
**High-Quality Visual Results:** The method produces results of high visual quality that demonstrate strong adherence to the analogical prompts when compared against baseline methods.

### Weaknesses
**Lack of Methodological Clarity:** The core weakness of this manuscript is the clarity of the methodology section. The description is ambiguous and lacks a clear, end-to-end overview of the training and inference processes. Furthermore, the mathematical notation is inconsistent and potentially confusing (e.g., the relationship between `a/b` as inputs and `A/B` as LoRA weights make reader confusing), which hinders a complete understanding of the proposed technique.

### Questions
A comprehensive evaluation of this work is contingent upon a clear understanding of the methodology. Could the authors please provide significant clarification on the following points?

1. **Elucidation of the Training and Inference Pipeline:** The current description of the pipeline is difficult to follow.
    
    - **Recommendation:** To resolve these ambiguities, I strongly recommend that the authors include a detailed diagram or, ideally, a **pseudo-code algorithm** that explicitly outlines both the complete training and inference procedures.
        
    - Given a single training instance with three inputs (`a`, `a'`, and `b`), could you please detail the process used to train the full set of N=32 LoRAs? The mapping from one training example to a large set of distinct LoRAs is not intuitive.
        
    - How are the learnable combination weights (`e_i` in Equation 4) incorporated into the training loss and updated during the optimization process?
        
    - Why use the 2x2 grid as the Flux’s input?
        
2. **Compare with LLM-based Methods:** Recent approaches, such as those leveraging Visual Language Models (e.g., "Nano-Banana"), have also been applied to analogy tasks. Could you please discuss the comparative advantages and potential limitations of your LoRA composition framework relative to these LLM-based methods? A discussion on aspects like inference speed, training cost, would be insightful.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LoRBA, a visual-analogy editing method that replaces the single-LoRA adapter common in prior work with a learnable basis of LoRAs. A small encoder (frozen CLIP + projection) embeds the analogy triplet \{a,a’,b\} and routes softmax mixing coefficients over the basis to create a task-specific “mixed LoRA” at inference time; the diffusion backbone (FLUX.1-Kontext) receives the full triplet via extended attention while CLIP features are used only for LoRA selection. 
Experiments on Relation252k (train) and a custom validation split show improved trade-offs between edit accuracy and preservation (VLM-based) and higher pairwise win rates vs. several strong baselines; ablations support design choices (basis size, softmax vs. tanh, CLIP routing input).

### Strengths
The paper motivates that a single LoRA under-represents the space of transformations, while a learned basis + router can specialize per analogy at inference. This is grounded in prior observations that LoRAs can span a semantic space.

LoRBA pushes the Pareto front on VLM edit-accuracy vs. preservation and wins user-study & VLM pairwise comparisons vs. most of baselines, indicating better edits without sacrificing identity.

### Weaknesses
Three prior-art baselines run on FLUX.1-Dev, whereas LoRBA (and a capacity-matched single-LoRA baseline) run on FLUX.1-Kontext. This makes it hard to attribute all gains solely to the LoRA-basis design rather than backbone differences. A fairness note is warranted or re-runs on the same backbone are needed.

Because Relation252k’s test set is unavailable, the authors build their own validation suite (Unsplash images, LLM-generated prompts, and community LoRAs). While thoughtfully constructed, this pipeline can encode distributional choices that favor the method; public release and stress tests would help.

Both scalar metrics (edit-accuracy & preservation) and pairwise selection use Gemma-3 prompts. Although there is a user study, the paper would benefit from reporting VLM–human correlation and sensitivity analyses (prompt variants).

Experiments cap the long edge at 512 and focus on FLUX.1-Kontext; it’s unclear how the approach scales to higher resolutions or transfers to other diffusion backbones. 

Typo: “mosiac” → “mosaic” in Fig. 3 caption. Please standardize notation (e.g., keep e_i for coefficients consistently across text/equations) and explicitly reference Eq. (2–4) where the “Mixed LoRA” is injected.

### Questions
Clarify whether the custom validation set (images, prompts, LoRA list) and code will be released to enable reproduction

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
3

### Summary
This paper introduces method to jointly train 1) multiple LoRA modules and 2) a learnable encoder that assigns the coefficients for each LoRA module for image editing with visual analogy pairs.The paper compares against previous baselines and reports state-of-the-art on quantitative (CLIP score similarity + VLM-as-a-judge) and qualitative (user study) metrics.The paper also runs a suite of ablations to identify the core of the improvement.

### Strengths
1. The paper is well written. I’m not too familiar with the topic, but was able to understand the motivation and the setup clearly.
2. Strong performance gains against the baselines. The model is better at making accurate edits while preserving the original image. The results are validated with qualitative and quantitative metrics.
3. Test-time inference is efficient. There is no need to train a separate module / separate set of coefficients for a new task at test-time. The images only need to go through a CLIP model to retrieve the query vector.

### Weaknesses
1. It is unclear whether the out-of-domain tasks are truly distinct from the training analogy types. From how it is mentioned, it seems like the authors only sampled from LoRA modules for samples where the base model was unable to make edits for. Does this mean these analogy types are disjoint from the training analogy types? Were there manual checks? 
2. Related to 1, does not test the limit of generalization (lines 482-483). Are there specific analogy types in the authors’ validation set that the model performs better/worse on?
3. Limited analysis of scalability. Any experiments on increasing N? Is it because of the limited data? Would the performance plateau?

### Questions
1. It seems like the validation data from Gong et al. (RelationAdapter) have now been released (https://huggingface.co/datasets/handsomeWilliam/Relation252K-unseen/tree/main). Could you also validate your method on the validation set just to remove any confounding factor from constructing your own validation set from a different pipeline?

2. Nit-picky but were the images presented to the survey participants randomized in order to remove any positional bias?

### Soundness
3

### Presentation
4

### Contribution
3
