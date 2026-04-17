# De-hallucinating CLIP Embeddings to Improve Brain-Vision Mapping

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Recent advances in vision-language models, such as CLIP, have enabled their widespread use in brain encoding and decoding, where global image embeddings serve as anchors linking visual stimuli to voxel-level brain responses. However, we observe that CLIP's global visual embeddings often exhibit hallucinatory semantics: they encode objects not explicitly present in an image but inferred from prior associations. This imaginative bias poses a significant challenge for brain-vision mapping, particularly for natural scenes containing multiple annotated objects, where human neural responses are constrained to what is actually perceived. To address this issue, we propose a framework that suppresses CLIP's visual hallucination by integrating object- and concept-level representations. First, we extract object-centric embeddings using segmentation masks, isolating visual features tied to explicitly present objects. Next, we stabilize these diverse segment embeddings with a concept bank of text-derived CLIP embeddings, aligning bottom-up perception with top-down categorical knowledge through cross-attention. The resulting concept-stabilized object features act as corrective signals to be fused with global scene embeddings to form de-hallucinated visual representations. Finally, these representations are used for voxel-wise regression. Experiments on the NSD dataset demonstrate that our method generates representations that better align with category-selective brain regions (bodies, faces, food, places, and words), leading to more accurate and reliable neuro-based image generation compared to standard CLIP regression. These results highlight the importance of suppressing model imagination in bridging human perception with multimodal foundation models and offer a new direction for robust, biologically grounded brain-vision alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a lightweight learning pipeline to address hallucination problems of CLIP model (before voxel-wise regression to fMRI responses). Specifically, the proposed method first extracts object-centric CLIP features using segmentation masks and then stabilizes these diverse segment embeddings through attention modules with a concept bank of text-derived CLIP embeddings. The final step is fusing the concept-stabilized object features with the global embedding. Experiments on the NSD dataset show consistent improvements compared to standard CLIP regression.

### Strengths
* The manuscript identifies a concrete failure mode of CLIP-based brain encoding and decoding tasks, and uses segmentation, concept bank, and cross-attention to address it, which is straightforward, interpretable, and well-motivated.

* This work addresses an important and timely problem, i.e., bridging the semantic gap between foundation-model embeddings and human neural representations before brain encoding and decoding.

* While recent work [1] addressed a similar issue (semantic misalignment between supervision signals and neural recordings), the perspectives differ: [1] argues neural signals lacking full semantic information of supervision signals (CLIP embeddings), whereas this work treats this as the hallucination problem of CLIP. In both, the shared underlying goal is to close the semantic gap for better brain-model alignment.

  [1] Bridging the Gap between Brain and Machine in Interpreting Visual Semantics: Towards Self-adaptive Brain-to-Text Decoding, ICCV 2025.

### Weaknesses
* The pipeline relies on the image segmentation ground truth. In many natural datasets, segmentation may be unavailable or noisy. The paper does not analysis the sensitivity of the proposed method to segmentation errors and number of segments, such as using image segmentation methods instead of ground truth.

* While the paper emphasizes that CLIP’s semantic hallucination leads to misalignment with human brain responses, a closely related work (Mind-SA [1]) has already identified a similar semantic mismatch problem between brain and model representations. Despite the differing assumptions, both aim to bridge the same semantic gap between models and the brain before alignment. The paper does not explicitly discuss or compare with [1].

  [1] Bridging the Gap between Brain and Machine in Interpreting Visual Semantics: Towards Self-adaptive Brain-to-Text Decoding, ICCV 2025.

* The paper does not provide qualitative examples (as well as discussions) of image generation, and primarily compares its approach against a single baseline (BrainDiVE).

### Questions
How sensitive are results to prompt engineering or to the granularity of categories?

### Soundness
4

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
4

### Summary
The authors present an interesting paper where they demonstrate that CLIP suffers from visual hallucinations (presence of concepts in images that are not actually there but either can be heavily associated in real world image statistics, but also might be quite erroneous and unexpected). The authors demonstrate a method they say effectively removes these visual hallucinations from the CLIP embeddings and then this provides a cleaner signal to link to fMRI data (specifically NSD) and that standard ridge regression results can be more informative as previous experiments that use non de-hallucinated CLIP embeddings might be making erroneous links via linking to information not actually presented in the image.

### Strengths
I really liked the goal of this paper. When I first read the abstract, it caught my attention and I felt that if the paper were to continue to fully investigate this, that this would be a paper I'd definitely be sharing with my colleagues and raising this point that I wasn't really aware of before now. I have previously examined this fMRI dataset using CLIP embeddings and it caused me to critically review how visual hallucinations might have affected some of these results. These papers do often get accepted into ICLR and this contribution I would deem to be highly relevant. Most notably, it would cause a lot of submissions to be critically analysed in light of these findings and that would be very relevant for progressing scientific knowledge on this field of research.

Having said that, the paper fell short in a few places and I really tried to find reasons to see past them, but some of them were quite important to me.

### Weaknesses
There are quite a few weaknesses I wish the authors had considered before submission, but I am hoping that by pointing them out, there will be a chance to make an updated version of the paper much better. My main problem is the evaluation of the method and the choice of experiments used to back up the claims of the paper. Let's take Figure 1, which very nicely shows the problem of visual hallucinations and how erroneous elements can creep into the predictions of constituent elements. I was fully expecting at this point to be seeing figures in the result section that compares and contrasts the old method with a de-hallucinated version of CLIP that could be specifically pointed at and where it is obvious that the new method removes the erroneous detection of these features. This would be an impactful easy win that is so intuitive, that actually its absence in the paper made me think this MUST HAVE been tried and didn't actually work. I can't contemplate why else such a comparison would be left out of the paper. This point needs to be clarified. 

I also have a bit of an issue in understanding the logic behind the proposed multi-step de-hallucination method. I understand the extraction of the object-centric embeddings that are in the image and how you tied those to the concept bank of text embeddings, but why does simple fusion mean that the original problem no longer contains the hallucinated elements? You are expanding the representation and maybe adding more information from what you know is there, but that confounded information lurking in the background might still very well be there. This is where and exactly why I needed to see something visual to see that this method actually works. This is later described as a "corrective signal" but this logic needs to be explained more directly as I just don't see how it corrects incorrect hallucinations in any strong and direct way. At best it might upweight information related to the real present concepts, but the choice of language and claim is much stronger in what you have presented.

Your figure captions need a lot of work. Look at Figure 2. Consider the amount of information you have in there and look at that absolutely generic caption that tells me nothing new. I have circled the brain image in the top middle in my own notes and written, "What is this??" and I still have no idea what it represents. You have the details of the concept text encoder there, lots of letters and introducing variable names like "X^{s}" but I just don't know what these mean. I can make a guess but I shouldn't have to. I need to be able to be guided through a figure of such detail by relying on the caption. 

Experimental details are often missing. Section 3.1. in the Dataset section, "all models are trained on a per-subject basis with a classic train/test split." - When I read this, I immediately went hunting for the specification of this because you absolutely cannot leave this information out in a paper like this. I have no idea what a classic train/test split because I've seen every manner of train/test splits and my experience tells me there is no such thing as a classic train/test split. Again, I shouldn't have to guess if this is 80/20 or something else because the issue of confounds across fMRI splits is also a huge issue in neural decoding. 

Lines 356-357 highlight the core of the testing strategy and I just think this is pretty weak, forcing into only five categories and making a determination on something so highly intricate as to whether there is the presence of visual hallucinations. The test doesn't even look at the presence of this information, but rather increased scores on another task. 

402-405: the category preferences are taken from Stigliani's localizers used in NSD, right? These are useful but hardly gold-standard mappings of where these categories are represented in the brain. I think this point should be considered and perhaps tone down the high claims of finding the "true" brain signal. 

There is a lot of talk about image generation as a method of evaluation here, but I am pretty surprised to not see any visualised. It would be very helpful to see this, because otherwise I am just left to assume that the visualisation of these images is so poor that it was felt that not putting them in the paper was better than putting them in (and this raises some questions/doubts for me). 

Also, there are some areas of language usage that need to be cleaned up a bit to make it more appropriate for scientific writing (e.g. "these evidences", "for 8 subject", "transparentizes", "the whoe model", "rather than cmodel omplexity").

### Questions
1. Can you clarify when talking about the global embedding, what exactly you mean. Are you taking the CLS token of the final layer of the vision encoder or doing some merging of all the patch tokens. This is important to know and it's not specified in the paper.

2. Can you explain why you didn't visually contrast hallucinating CLIP with the de-hallucinated version showing that the de-hallucinated version has a more sensible ranking order of elements that are detected (an adaption on the problem that Figure 1 demonstrates)?

3. Can you provide a more extensive / simpler / extended (whatever) explanation of how adding more information to CLIP's global embedding effectively "de-hallucinates" the imagined concepts? I am really not clear how this works. I really hope it does, but the lack of a direct visual contrast (see (1)) makes me quite suspicious. I want this to be the case, but I need to be convinced first. 

4. I presume you used ridge regression in the brain-vision mapping but I actually see no mention of this in the equations. I don't know a single other work that has had success in brain-vision mapping without providing some sort of extra constraint in the regression. I can't believe you didn't use one as it's so non-standard, but it's not mentioned in the equation. Can you please clarify this point? If you didn't use ridge regression, I'm going to need an extremely strong and well-defended motivation as to why this choice was made. 

5. I just wanted a direct answer to the question of whether this paper is the one introducing the idea of visual hallucinations in CLIP and this isn't mentioned in other work that you are building on? The reason is that this idea in itself is quite big and I thought probably needs way more investigation to support the claims and it's an incredibly relevant result to the community. I'm very eager to see this finding spread out in the community, but it almost seemed like a secondary finding in this submission. That is fine, if it's not your choice to want to lead with that claim, but I needed to be sure of this myself during my review. [EDIT] : I see now this is explained in A.1. but this is not linked directly in the main paper, causing my earlier confusion. This absolutely needs to be amended because it's almost deliberately unclear in the current formulation. 

6. On Line 165, you say the entire scene can be decomposed into into the labelled segments, but is that truly the case the entirety of each image is fully segmented such that every pixel belongs to a single segment? I thought backgrounds and complex patterns would not be assigned to a specific categorical segment. If my understanding is correct, then it's not the case that: "[e]ach image can be decomposed as X = ....." (rest of line 165). Please can you clarify this point? 

7. Do you have any statistical tests to report for section 3.2? Do we just accept the trends in Figure 4 on varying 1-4% scales are meaningful? Looking at Figure 3 and having those high concentrations around zero with (also) no statistics makes me just wonder about the robustness of the results. 

8. The end of section 3.2. contains a claim that summarises that go some way to explain why the reported differences occur, but I want a more extensive rationale. Can you be specific, pointing specifically to the presence of hallucinated objects in CLIP representations, that would make scores worse on this test? I'm interested to understand if you can tie it to the additional corrected embedding and how this works. 

9. Can we see any of the generated images that were fed into the encoder to compare your method to BrainDiVE?

10. In Line 411-412, you say subject 6 had fewer training samples relative to other subjects. I went back to review your Dataset description and you say that each subject saw 10,000 images and there is no description anywhere else in the paper that leads to any information on why some subjects might have fewer data samples than others. What's going on here?

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
This paper focuses on the "hallucination issue" of CLIP visual embeddings in brain-vision mapping tasks (i.e., encoding objects not present in the image). It proposes a lightweight de-hallucination framework consisting of "object-level representation extraction + concept bank anchoring + global feature fusion" and validates, on the NSD fMRI dataset, that this framework enhances neural encoding accuracy and category-selective voxel activation.

The work integrates the de-hallucination issue of vision-language models with the need for brain representation alignment in neuroscience, making its research topic both innovative and of interdisciplinary value. The methodological design aligns with biological intuition (integrating bottom-up perception and top-down categorical knowledge), and the core conclusions are supported by experiments. However, there remains significant room for improvement in aspects such as the completeness of qualitative validation, the comprehensiveness of quantitative comparison, and the rigor of methodological details. Critical supplementary experiments are required to enhance the persuasiveness and generalizability of the conclusions.

### Strengths
1. The authors have identified the phenomenon of inconsistency between conceptual representations and image content in CLIP. Building on this issue, they propose a method that does not incur substantial computational overhead to mitigate such hallucinations.

2. Experimental results on neural coding demonstrate the effectiveness of eliminating hallucinations in CLIP for improving coding accuracy.

### Weaknesses
1. No supplementary comparative diagrams of hallucinations between "the proposed method vs. original CLIP" (e.g., improved Top-8 prediction results corresponding to Figure 1) are provided, making it impossible to intuitively judge changes in the ranking of correct objects and the reduction of spurious/contextual hallucinations.

2. For the generated results in Table 1, only numerical data (zero-shot accuracy) is presented, without visual comparisons of generated images between "the proposed method vs. BrainDiVE," making it difficult to reflect the practical significance of improved semantic consistency.

3. No dimensionality-reduced visualizations (e.g., t-SNE or UMAP) of the original CLIP global embeddings, object-level embeddings ($Z_{obj}$), and concept-enriched embeddings ($Z_{obj}^C$) are provided. The de-hallucination mechanism of the method becomes a "black box," resulting in insufficient interpretability.

4. No comparisons are made with classic methods (e.g., CLIP-Guided Decoding [1], CLIP-DPO [2] ) on general CLIP de-hallucination benchmarks (such as POPE [3] , object-level comparison tasks in Liu et al. (2024) [4] ), making it impossible to prove the generality of the de-hallucination capability.

5. Incomplete ablation experiments: Only the comparison of "image→image+object→full" is conducted, without ablating the concept bank, cross-attention, or object segmentation, nor verifying different fusion strategies. This makes it impossible to distinguish the contributions of individual modules.

6. The selection of the concept bank prompt ("a cropped centered photo of [object]") lacks justification: No comparison of the effects of other prompts is provided, nor is the generalization to "unseen categories" verified, which may affect reproducibility.

7. The simplicity of the fusion strategy ($z=(z_{img} + z_{obj})/2$) lacks support: No comparisons with alternative fusion methods (e.g., attention weighting, adaptive weighting) are made, nor is a sensitivity analysis of fusion weights conducted, making it impossible to prove its optimality.

[1] Seeing is Believing: Mitigating Hallucination in Large Vision-Language Models via CLIP-Guided Decoding

[2] CLIP-DPO: Vision-Language Models as a Source of Preference for Fixing Hallucinations in LVLMs

[3] Evaluating Object Hallucination in Large Vision-Language Models

[4] Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models

### Questions
Overall, this paper is good in both topic selection and quality. If the authors can address and supplement the issues I raised in the "Weaknesses" section, I will consider raising the score to 6.

### Soundness
2

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
4

### Summary
This paper introduces a novel de-hallucination framework designed to mitigate the semantic hallucinations present in CLIP’s global visual embeddings when applied to brain-vision mapping tasks. The method first extracts object-level embeddings from segmentation masks, then aligns them with a text-derived “concept bank” using cross-attention, and finally fuses them with global scene embeddings to form de-hallucinated visual features. Experiments on the Natural Scenes Dataset (NSD) show improved voxel-wise regression, stronger activation in semantically selective regions, and better semantic consistency in brain-driven image generation compared to BrainDiVE.

### Strengths
1. Original Problem Definition 
The paper clearly identifies and analyzes a real issue—hallucinatory semantics in CLIP embeddings—within brain-vision applications. This is an underexplored but critical limitation for neuroscience-aligned visual models. 

2. Strong Empirical Validation Results on NSD are extensive and multi-faceted: voxel-wise R² improvement, category-based image generation, and neural activation alignment. Demonstrates neuroscientific relevance.

### Weaknesses
1. Missing Formalization of the “De-Hallucination” Objective：

The authors introduce cross-attention between object features and a “concept bank” but do not specify what loss function or alignment constraint guides this process—e.g., whether it minimizes distance between object embeddings and concept embeddings, or uses contrastive or reconstruction objectives.

Without a defined objective, it’s unclear how the model learns to “de-hallucinate” beyond architectural heuristics. Adding even a small formal definition (e.g., minimizing semantic divergence) would make the contribution more rigorous.

2. Relation to Variational and Contrastive Frameworks Is Unclear：

The cross-attention formulation (Eq.6) suggests an implicit distribution alignment between noisy visual segments and stable semantic priors—conceptually similar to variational information bottleneck or contrastive predictive coding.
However, the paper does not clarify whether the model explicitly minimizes KL divergence, employs negative sampling, or estimates mutual information.

Clarifying this connection would strengthen the theoretical grounding of the “concept stabilization” process.

3. Quantitative Gains Are Modest and Require Statistical Analysis:

While the paper reports modest improvements (max average R² increase ≈ 1.5%), there is no mention of statistical testing (e.g., permutation test or paired t-test).

Given the small margins, readers cannot assess whether the observed trends are robust across subjects.
It would help to include confidence intervals or significance tests for voxel-wise gains.

4. Lack of Comparison Against Stronger Baselines

Only BrainDiVE is used for comparison. BrainDiVE (Luo et al., 2023) is an appropriate baseline, but recent NSD-based methods such as MindEye2 (Scotti et al., 2024b), CLIP-MUSED (Zhou et al., 2024), and NeuroCLIPs (Gong et al., 2024) should be included or at least discussed.

This would clarify whether the proposed improvements stem from de-hallucination or from differences in architecture or training scale.

5. Ambiguity in Image Generation Evaluation:

While CLIP-based evaluation is convenient, using the same model (CLIP) both for de-hallucination and evaluation may confound results.
Alternative metrics—such as human-annotated category accuracy or FID computed against category exemplars—would provide more independent validation.

6. Missing Details on Cross-Attention Implementation:

Readers need to know whether it is a single-head or multi-head attention, whether weights are learned or frozen, and how it is regularized.
Given that this is the core mechanism of the framework, a concise architectural schematic or pseudocode would enhance clarity.

7. Need for Perceptual or Behavioral Validation:

The argument that the proposed method produces representations “aligned with human perception” is well-motivated but not empirically validated beyond neural alignment.

Even a small-scale human annotation (e.g., asking whether generated images reflect the correct object categories) would substantiate the claim of perceptual fidelity.

8. Overlap with Recent Works:

While the paper distinguishes itself by focusing on hallucination mitigation, several related works also incorporate semantic grounding or object-based refinement.

A clearer discussion of differences (e.g., whether the concept-bank cross-attention introduces unique mechanisms) would help position this paper more distinctly.

9. Ablation Study Interpretation

Although adding concept-level features improves mean performance (54.0%), for some subjects (e.g., subj08), performance drops.
This should be analyzed—perhaps due to segmentation quality or concept-bank mismatch. A per-subject discussion would strengthen the robustness claim.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
