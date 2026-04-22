# SSD: A Sparse Semantic Defense Against Semantic Adversarial Attacks to Image Classifiers

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 0, 6

## Abstract
Adversarial attacks to image classifiers pose a major threat to machine learning models. However, existing defenses against such attacks have been designed mostly for unrealistic image threat models, such as bounded $\ell_p$-norm image perturbations. In this paper, we focus on defending against more realistic semantic adversarial attacks, which modify semantic image concepts (e.g., make it in snow) that are irrelevant to the underlying classification task (e.g., classify a dog). Intuitively, a classifier that is robust to semantic attacks should rely only on concepts that are relevant for the task. Therefore, the proposed Sparse Semantic Defense (SSD) uses large language models to build a dictionary of visual concepts that are relevant for a given visual recognition task, and large vision-language models to embed images and concepts into an aligned, shared latent space. Sparse coding is then used to decompose the image embedding as a sparse combination of the text embeddings of relevant concepts plus a residual term that captures irrelevant concepts, including semantic attacks. We provide a theoretical justification for why sparse coding can separate irrelevant semantics from the resulting sparse code.
A simple linear classifier on the sparse code is then used. Note that SSD is robust to semantic attacks by design because it relies only on semantic concepts that are relevant to the task. SSD is also interpretable by design because it relies on task-relevant visual concepts. Experiments on ImageNet show that SSD performs favorably with respect to other baselines in terms of robust accuracy against semantic adversarial attacks while maintaining interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper defends in semantic space rather than in pixels: it builds a task-relevant concept dictionary with WordNet+GPT, embeds images/text with CLIP, then uses LASSO to get a sparse “concept code”; a linear head predicts from that code. The residual is meant to soak up irrelevant semantics/attacks. Results on ImageWoof/ImageNet against Instruct2Attack/ColorFool look solid for a training-free defense, and the method is naturally interpretable.

### Strengths
1. SSD is among the first works to explicitly design a semantic-space defense grounded in language–vision alignment.
2. Proposition 1 provides a clear rationale under RIP conditions for why sparse coding can separate relevant vs. irrelevant semantics, which is conceptually elegant.
3. Compared to adversarial training or diffusion-based defenses, SSD avoids expensive semantic adversarial generation and adds minimal inference overhead.

### Weaknesses
1. Current prompts (“make it in snow/at night”) cover appearance; consider adding other semantic editors to back Fig. 3’s orthogonality story.
2. The paper reports robust accuracy but provides no certified robustness (e.g., smoothing bounds). This matters because the defense is deterministic at test time and thus a natural target for adaptive optimization.
3. The paper references an exact prompt (Fig. 5) and a filtering step (Appx D.4), but the main paper doesn’t quantify final |C|, coverage, or diversity metrics; small changes in prompt/filtering could materially change robustness/interpretability.

### Questions
1. How large is the dictionary (|C|) on ImageNet, typical nonzeros per image, and per-image solve time for Eq. (3)? 
2. How do you pick λ and the nonzero cap (100/200)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes Sparse Semantic Defense (SSD): represent each image as a sparse combination of task-relevant, text-encoded concepts and classify using those sparse codes. Off-dictionary perturbations are absorbed into a residual, keeping key concept coefficients stable—improving robustness to semantic edits without adversarial training and yielding concept-level interpretability. Limitations include reliance on the concept dictionary’s quality/coverage and limited testing against fully adaptive semantic attacks.

### Strengths
This method doesn’t rely on ℓp bounds and holds up across varied semantic edits (weather, lighting, color)，aligning better with real-world shifts.

### Weaknesses
Although the method proposed in this paper involves a series of complex components and processes including GPT and CLIP，however, I believe it is meaningless. For classifier defenses, there are already many simple and effective approaches such as image pre-processing and adversarial training methods, all of which consume far fewer resources than the method in this paper. Even if one insists on leveraging large language models, there exist numerous simple yet effective solutions.

For example, I directly input the adversarial image from Figure 1 into GPT-5. GPT-5 not only produced the correct classification but also provided a clear and interpretable explanation—demonstrating better interpretability than the method proposed in this paper. Below, I quote GPT’s answer:

“Q: Tell me the name of this kind of dog

GPT-5 Answer: Looks like a Shih Tzu—a small, flat-faced toy breed with a long, fluffy double coat. (Sometimes people mix them with Lhasa Apsos, but the shorter muzzle and rounder head point to Shih Tzu.)” 

Thus, since GPT alone can already achieve the defense, the series of so-called post-processing steps proposed by the authors on top of GPT become entirely unnecessary. The proposed method is not only much less efficient than traditional adversarial defense techniques, but also less effective than simply using GPT.

In summary, compared to existing approaches, this paper’s method offers no advantages and instead increases computational cost. So I believe this work is meaningless.

### Questions
See the weakness part of my review.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Sparse Semantic Defense (SSD), a defense method against semantic attacks. The main idea is to use a language model to generate class-related visual concepts, encode them with a CLIP text encoder to form a semantic dictionary, and then decompose image features into a sparse combination of these concepts via LASSO. The approach enhances robustness while maintaining interpretability. The paper also provides a theoretical explanation and demonstrates improved robustness on ImageWoof and ImageNet. It’s a clean, interpretable take on semantic defenses, and the overall idea feels fresh.

### Strengths
1. The method is simple and intuitive, combining LLM-generated concepts and sparse coding for enhanced interpretability.

2. SSD shows consistent robustness gains under several semantic attacks and is more efficient at inference compared to diffusion-based defenses.

3. Theoretical intuition and empirical evidence together help clarify why sparse semantic representation improves robustness.

### Weaknesses
1. Theoretical assumptions (orthogonality and RIP) are strong and not validated on larger or more realistic datasets.

2. Experimental coverage is limited, i.e. semantic attack diversity and adaptive comparisons are lacking.

3. The concept dictionary still contains noisy or irrelevant entries, weakening interpretability claims.

### Questions
1. Could the authors validate the "orthogonality between attack residuals and dictionary space" assumption on a larger dataset, even with simple statistics?

2. Would it be possible to design a lightweight approximation of adaptive attacks to more fairly test SSD against stronger defenses?

3. Could the authors show robustness results under additional prompts (e.g., different scene or style edits) to demonstrate generalization?

### Soundness
2

### Presentation
3

### Contribution
3
