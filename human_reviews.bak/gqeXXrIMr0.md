# Rethinking Visual Counterfactual Explanations Through Region Constraint

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 5, 8, 6

## Abstract
Visual counterfactual explanations (VCEs) have recently gained immense popularity as a tool for clarifying the decision-making process of image classifiers. This trend is largely motivated by what these explanations promise to deliver -- indicate semantically meaningful factors that change the classifier's decision. However, we argue that current state-of-the-art approaches lack a crucial component -- the region constraint -- whose absence prevents from drawing explicit conclusions, and may even lead to faulty reasoning due to phenomenons like confirmation bias. To address the issue of previous methods, which modify images in a very entangled and widely dispersed manner, we propose region-constrained VCEs (RVCEs), which assume that only a predefined image region can be modified to influence the model's prediction. To effectively sample from this subclass of VCEs, we propose Region-Constrained Counterfactual Schrödinger Bridge (RCSB), an adaptation of a tractable subclass of Schrödinger Bridges to the problem of conditional inpainting, where the conditioning signal originates from the classifier of interest. In addition to setting a new state-of-the-art by a large margin, we extend RCSB to allow for exact counterfactual reasoning, where the predefined region contains only the factor of interest, and incorporating the user to actively interact with the RVCE by predefining the regions manually.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a method for visual counterfactual explanation which is based upon methods like latent diffusion counterfactual explanation by Farid et al. and modified the previous approaches by adding a spatial (region) constraint, which allows more precise target generation of new images containing features of the target class. Thereby, also an external classifier is used as guidance for the generation process, e.g. for a stable diffusion model. Furthermore, the authors added optimization techniques to overcome limitation with respect to generated artefacts, color balance and realism (measured via FID score). Furthermore, the authors introduced also an approach for generating the regions automatically to enhance the usability of the method. In the evaluation, the authors show improved realism via FID score, effectiveness via flip rate, similarity and sparsity in comparison to current state-of-the-art methods (e.g. ACE or LDCE). In addition, qualitative results are presented showing the applicability of the method for e.g. discovering complex patterns. The authors claim finally that they could improve on existing methods by a large margin.

### Strengths
- Well written text that is good to follow as a reader
- Despite there a different attempt to condition the generation process, the idea is novel enough when including the optimization steps of the method bringing XAI methods
- The theoretic description of the method is well formulated with respect to quality and clarity
- The structure of the evaluation has a high quality

### Weaknesses
- Only tested in a limited dataset, the authors should consider applying the method also on different datasets or domains
- From the results in the main paper, it could be confirmed that the method works on different generative models and external classifiers but claimed that results are available in the appendix. Therefore, I encourage the authors to include at least the numeric results in the main paper, so that a reader can obtain insights on the performance for different domains and models.
- The use of FID score is misleading the comparison to other methods, that do not limit the generation process by regions, since the limitation to certain spatial regions automatically reduced the FID score significantly. The authors should at least consider this limitation and think of better comparison, e.g. maybe a region-based FID score.
- In the paper the authors state that the method helps gaining more clear visual explanations where even a user can interact with the method to optimize the results. This is true with respect to how the method works, but no study with real users was performed to evaluate if such statement holds true. Either this claim should be removed or validated by a (small) user study.
- The other SOTA methods, e.g. ACE and DVCE are not really introduced, nor cited and not compared from a theoretical point of view. At least they must be cited and introduced.

### Questions
Apart from the recommendations in the weaknesses section I have the following questions and suggestions:
- Highlight the possible use cases of the method (e.g. class confusion) more so that the value of the method can be better understood
- Avoid using emphasizing phrases, e.g. by a large margin, and not bringing a quantitative results or at least a reference
- To many details and efforts with respect to the evalution are in the appendix weakening the main paper. I would suggest to include more details in the main paper from the additional evaluation results on different generative models and classifiers and reducing some of the visuals in the main paper + some very detailed analysis of the shown images and come up with a more condensed evaluation but including all the insights.
- What are the limitations of the method?
- It was not clear to me If there are not many possible image modifications for one sample yielding into a class change. How is that restricted to find the “best” modification and what are the optimal criteria?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a method for creating visual counterfactual explanations for classification tasks such as ImageNet classification. These explanations illustrate how an image could be modified to change the classification output. The approach is based on a combination of existing methods which select the most relevant parts of the image to change the decision and then mask and inpaint these regions to change the classification decision. A major contribution of the proposed method is limiting the masking/inpainting to a small connected spatial region in the image rather than allowing changes everywhere. The method outperforms other approaches on a set of metrics which aim to balance various factors desirable for visual counterfactuals (namely, changing the model’s prediction while keeping the output image similar to the input, keeping changes sparse, and ensuring the output is realistic-looking).

### Strengths
- The proposed method outperforms similar approaches on the proposed metrics, at least on a very small evaluation dataset (3 class pairs).
- The writing is very clear, and the results are very well organised and presented, with well-designed figures to illustrate the model's performance.
- The paper shows some interesting possible applications, such as allowing users to target specific regions of an image to see those regions would need to change to change the model's decision.

### Weaknesses
- The main contribution of the paper (limiting the inpainting to a specific region) also seems like a weakness, since the model cannot capture cases where the differences between two classes are changes in shape. (For example, this is shown the American chameleon <-> common iguana examples -- the main difference between the two types of lizard is the overall shape/size, but the model can only show how the texture/color differs between these classes.)
- The intro says that the goal of generating counterfactuals is to explain the decision-making process of an image classifier to a user, but it's not clear whether the proposed model is better than any other model at achieving that result. This approach performs best on the metrics, but simply maximizing performance across metrics seems like it would result in a "counterfactual" image that flips the classifier's decision with almost no visible changes to the image, which wouldn't be a useful counterfactual for human users (as they wouldn't be able to tell what changed).

### Questions
- The comparison models in the main experiment (Table 1) are not defined anywhere in the paper -- the abbreviations used in the table are never used in the text, and the table has no citations to tell you what the abbreviations might mean. I think I was able to work out what these models are through Googling but they should be named in the paper (though it's worth noting that there is another XAI method called "ACE").
- Since the evaluation dataset seems to be restricted to class pairs that favor the proposed method (cases where the two classes have the same shape but different surface color/texture), I'd be curious to know how the proposed model compares on a wider set of counterfactual pairs. Does the proposed method outperform ACE, etc. on arbitrary counterfactual pairings, where the key differences between the classes might include shape changes?
- Can this method be helpful for users to understand adversarial attacks (if the input image is a adversarial image, can the counterfactuals help explain why the model saw the image as the wrong category)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets at the task of VCEs and defines more refined task with region constraints to gain more precise explanations. Further, it also proposes the method RCSB for the newly defined task, which is built upon the I^2SB and solves the problem as a conditional inpainting task. The idea is simple and the experiments demonstrates its effectiveness. Nevertheless, the technical novelty is relatively limited.

### Strengths
1. The newly formulated RVCEs is reasonable. 
2. The presentation is good and the paper is easy to follow.
3. The experiments demonstrates the effectiveness of the proposed method.

### Weaknesses
1. The technical novelty is relatively limited. It formulates the task as a conditional inpainting problem and solves it by applying $I^2SB$.

### Questions
1. The paper claimed that the definition of VCEs has potential confirmation bias. However, it is not deeply investigated. Besides, the teaser example in Figure 1 also cannot indicate such confirmation bias.
2. There is no direct qualitative comparison between different methods, in Figure 5 and 6.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This manuscript studies the problem of visual counterfactual explanations (VCEs) and argues that region is important. It is an extension of the visual counterfactual explanation by adding a hard region constraint. The method is based on a recently developed diffusion-based image-to-image translation method, named Image-to-Image Schrodinger Bridge ($I^2$SB). The authors discover a good way to inject the gradient from the classifier into the generation process of $I^2$SB and make the $I^2$SB generate more plausible images that change the prediction of the classifier. It gives us a more flexible way to understand the model prediction process.

### Strengths
- Bring the region constraint to the counterfactual example generation somehow makes sense to better align with human perception.

- The authors provide a way to better inject the guidance from the gradient of the classifier f into the generation process of the inpainting model. From the visual examples, we can indeed observe a much higher quality.

### Weaknesses
- As the regions are the core input of the proposed method, how to define/find the regions is crucial. The method uses either the attribution map generation or the grounding method. Will these methods introduce additional bias in the understanding of the classifier? 

- From my perspective, the proposed method is more like just an image painting with specific guidance, where the guidance comes from the gradients of the classification model. The quality of the explanation ability somehow entangles the ability of the classifier itself and the inpainting model. The quality (FID, etc,) of the generated samples is also highly entangled with the base inpainting models. This might be a problem if we want to understand the behaviors of the classifier.

- The visual counterfactual explanation is similar to the problem of attribution map generation, where the goal is to detect the regions that have the most influence on the prediction of the model either positively or negatively. Actually, one of the region generation methods is attribution map generation (IG). If we use the attribution map generation method, we can also generate some examples with high FID while with different predicted labels. I wondered what's the core difference between these two setups.

### Questions
- I'm wondering what will be generated if we force the label of the generated image to a very different class. For example, if we want to change the output of the model given the image of an animal to the class of a dining table, what does the generated image look like?

- There are some cases in which the model indeed makes use of unusual visual cues to predict the label of the image, like the background. We may also want to know that the model has learned such a prior from the dataset (for example, the model links the snow background to the wolf due to dataset bias). How can the proposed method discover such cases?

- The visualization shown in Figure 5 is interesting. At the same time, I noticed that after changes, several images changed the image's attributes significantly toward the target class (like the Flamingo and Cauliflower cases). I'm wondering what can we learn from these changes to understand the model's behaviors, as the model's prediction on these changed samples is very reasonable and as expected.

### Soundness
3

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
4

### Summary
The authors propose a method for generating region-constrained visual counterfactual explanations, arguing that the region constraint leads to improved interpretability of the resulting counterfactual explanations (compared to unconstrained CEs). For this purpose, they leverage i) Image-to-Image Schrödinger Bridges, ii) automatically determined regions of interest, using either classical visual attribution methods or a text-to-object segmentation foundation model (LangSAM), and iii) several tweaks for stabilizing classifier guidance of the diffusion process. The proposed method outperforms several baselines on previously identified particularly challenging counterfactual questions on ImageNet.

### Strengths
- The paper is very well-written overall and generally a pleasure to read; the various technical aspects of the method are very well-motivated, rigorously and clearly presented, and demonstrated to be useful in ablation experiments.
- Region-constrained counterfactual explanations are certainly very useful (although I would argue that this approach is not completely novel, cf. my comments below) and I believe the explicit problem formulation and framing here can help move this field forward. The extensive and clearly explained examples are also much appreciated.
- The I2SB-based conditional inpainting approach appears to perform very well.
- An interesting approach for (ADAM-derived) gradient stabilization and normalization in classifier-guided diffusion

### Weaknesses
**The relationship to several prior works should be discussed more comprehensively**, and possibly some of these methods should be included as baselines. The explicitly novel contributions should be highlighted more clearly.
- Both [Jeanneret et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeanneret_Adversarial_Counterfactual_Visual_Explanations_CVPR_2023_paper.pdf) and [Weng et al. (2024)](https://link.springer.com/chapter/10.1007/978-3-031-73016-0_20) also use automatically determined region constraints and conditional inpainting to obtain region-constrained counterfactual explanations. In several places it reads as if the authors claim novelty on region-constrained CEs *per se*, or the use of conditional inpainting for this purpose.
- Concerning the challenge of obtaining meaningful gradients with respect to noisy (intermediate) images, as discussed on page 4, this has been addressed before in various approaches such as DiME (by computing gradients with respect to denoised images instead, which is computationally expensive). Both FastDiME and [He et al.](https://openreview.net/forum?id=o3BxOLoxm1) have (more?) efficient implementations of the Tweedie approach. 
- [Karunratanakul et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.pdf) is also very related and contains many of the same ingredients presented here. Again, the precise differences should be discussed.

**For SOTA claims, more extensive experiments and standard benchmarks would be needed.** 
- Performance is only evaluated quantitatively on three particular tasks previously identified to be particularly challenging for competitor methods. It seems unfair to specifically compare on the case where baselines are known to perform poorly, but not on the ones where they perform well. More generally, a SOTA claim for VCEs would require more extensive experiments. E.g., a comparison on CelebA-smile would already be interesting.
- As noted above, I would suggest adding several more baseline methods such as DiME, FastDiME, and possibly combinations of these methods. E.g. it should be relatively straightforward to implement an I2SB version of FastDiME. This would help further disentangle the benefits provided by the different method components (I2SB, different auto-masking compared to prior work, gradient estimation & stabilization) presented here.

**Overclaims regarding trustworthiness and 'causality' of the resulting explanations**. While I am generally on board concerning the utility of region-constrained VCEs, the authors do go quite a bit overboard in their claims about them, I believe.
- If a VCE is constrained to only change a specific part of the image, this may yield to more intuitively appealing explanations, but the model might actually still be looking at weird shortcut features in other parts of the image (outside the mask). One would then miss this crucial information. This could happen in the proposed method e.g. due to non-localized shortcuts (e.g. [Wang et al.](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_What_do_neural_networks_learn_in_image_classification_A_frequency_ICCV_2023_paper.html)), where every individual pixel of the background would receive a low importance score. It would also miss weak shortcut influence, where the shortcut doesn't completely dominate predictions but still subtly skews confidences scores, such as in racial/ethnic shortcuts in medical imaging ([Zou et al.](https://www.science.org/doi/full/10.1126/science.adh4260), [Yang et al.](https://www.nature.com/articles/s41591-024-03113-4)). Personally, I would be *very* careful with explanation methods that explicitly constrain the explanation to those parts of an image that we believe to be useful a priori. (Cf. e.g. the case study in Fig. 5 in regard to which it is claimed that "RCSB allows for causal inference about the model’s reasoning" - but if we constrain explanations to LangSAM segmentations, we might miss that the classification model is actually primarily looking elsewhere.) The whole setup could thus be seen as strongly *increasing* the risk for confirmation bias, which the authors claim to combat using their method. Also cf. the example in Fig. 1, where it could actually be considered very informative that the counterfactual explanation changes the copyright caption (the authors mention this as a flaw), because it might indicate potential shortcut learning! As a final example in this regard, the authors write that "In this case, RCSB allows for increasing trust in the model, as making the lemons more orange correctly modifies its decision." - but again, this trust is unwarranted: the importance of the object color might in fact be negligible compared to the importance of some background shortcut feature.
- Relatedly, the relative importance of different features is lost in region-constrained VCEs. The fact that one *can* change the head of a bird such that it is classified differently does not mean that these specific properties of the head are actually important to the model's classification. Again, this severely limits any claims of "inferring causally about the model's predictions" - we still don't know which features were actually important for the classification.

To reiterate, I am generally sympathetic to the approach pursued here. However, the discussion of its merits and limitations (!) should be much more nuanced.

### Questions
In addition to my questions & suggestions presented above under 'Weaknesses', just a few minor remarks:
- In Fig. 1, what is the "recent SOTA method" used? 
- In the caption of Fig. 2, the authors write that "Intermediate images of I2SB are much closer to the data manifold." - I assume that by this, the authors mean that most of the image is 'natural' throughout the whole generation procedure? I was a bit confused here at first, because the trajectory of the cat's face looks very similar in both rows (as would be expected).
- Did the authors consider using the *inverse* of their mask (i.e., only allowing the model to change the background) for assessing a model's robustness to shortcut features?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the task of Visual Contour Factual Explanations (VCE) by introducing regional constraints to enhance the specificity and clarity of the explanations. Additionally, it proposes a method, RCSB, to generate counterfactual results based on region-specific information. The effectiveness of the proposed approach is evaluated on both generated and manually designed regions.

### Strengths
The paper is clearly written, with most details of the proposed method presented effectively. It includes numerous qualitative results that demonstrate the method’s effectiveness and clearly define the problem.

### Weaknesses
1. The paper demonstrates that the proposed method achieves better performance when a semantic region constraint is applied. However, it would be helpful to explore the effects when the region is not semantically relevant—such as when using a randomly selected area with similar shape and size—to understand how critical semantic relevance is to the method’s effectiveness.



2. In Table 1, three configurations of the proposed method are presented as settings A, B, and C. While it appears these configurations differ based on specific hyperparameter choices, could you provide additional context or rationale for highlighting these particular settings? Are there theoretical or practical reasons to show them in this table?



3. The results for LangSAM-based regions have a considerably higher FID score than those for other regions, suggesting that this style of regions may pose unique challenges. Could you clarify why this region is so different or particularly difficult for the method? Additional explanation on this would provide valuable insight into the model’s limitations.



4. The paper briefly mentions "confirmation bias," but could you expand on how this bias is defined in the original problem and mitigated by the proposed method?

### Questions
See strengths and weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
