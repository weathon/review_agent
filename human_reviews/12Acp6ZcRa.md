# Evaluating the Robustness of Text-to-image Diffusion Models against Real-world Attacks

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 3, 5

## Abstract
Text-to-image (T2I) diffusion models (DMs) have shown promise in generating high-quality images from textual descriptions. The real-world applications of these models require particular attention to their safety and fidelity, but this has not been sufficiently explored. 
One fundamental question is whether the existing T2I DMs are robust against variations over input texts.  To answer it, this work provides the first robustness evaluation of T2I DMs against real-world perturbations.  Unlike malicious attacks that involve apocryphal alterations to the input texts, we consider a perturbation space spanned by realistic errors (e.g., typo, glyph, phonetic) that humans can make and adopt adversarial attacks to generate worst-case perturbations for robustness evaluation. Given the inherent randomness of the generation process, we develop novel distribution-based objectives to mislead T2I DMs. We optimize the objectives by black-box attacks without any knowledge of the model. Extensive experiments demonstrate the effectiveness of our method for attacking popular T2I DMs and simultaneously reveal their non-trivial robustness issues. Moreover, we provide an in-depth analysis of our method to show that it is not designed to attack the text encoder in T2I DMs solely.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers the scenario of attacking text-to-image diffusion models with realistic modifications of the input text. Authors perform comprehensive analysis of possible attack objectives, provide thorough theoretical justification and evaluate them with extensive experiments. The paper argues that popular text-to-image models such as Stable Diffusion are vulnerable to this type of attacks thus pointing out the importance of further studying this topic.

### Strengths
1. Originality: the authors study a novel problem setting. I am not aware of other works providing comprehensive studies of considered attack objectives for T2I models. 

2. Quality: conclusions in the paper are based on solid experimental justification, every step in excluding some of the objectives from further experiments is clearly justified. Experimental evaluation is rich and considers different metrics. Besides, the authors provide code for their experiments (although I did not examine it carefully). Results for Stable Diffusion model are provided which is relevant for the community due to its popularity.

3. Clarity: the paper is easy to follow, numerous illustrations facilitate understanding.

4. Significance: T2I approaches have significant impact in both research and industry, thus studying their possible vulnerabilities is valuable for the community.

### Weaknesses
1. Looking at the results in Figures 1, 6 and 7 one can observe that by analysing the output image one can see which word contains an error: “ice” or “cream” in Figure 1, “cat” in Figure 6 or “panda” in Figure 7. This clearly follows from the attack design selecting significant words (Section 3.3.2). However, this limits the “imperceptability” of the attacks. Modifying not the most significant words and still being able to perturb the image would be another interesting track.

### Questions
1. The results provided in the paper carry theoretical significance for understanding the work and limits of text-to-image models. This outcome is valuable on its own. However, it is not clear whether proposed attacks pose non-negligible security threat for some real-world applications. Up to my understanding, work with T2I models is usually not automated and controlled by a human being at all times: from providing the text input to controlling the output image. Thus a person seeing not what they were expecting to see will probably just inspect the textual query and modify it to delete any unwanted modifications. Are there any scenarios where a malicious attacker can really cause any harm with these attacks? An explicit analysis of this issue would provide additional context for the results and broader impact of this work.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces a method for finding minor modifications to text inputs in text-to-image generating models that induce errors in the generated image. Modifications take the form of typos, replacements with unicode characters, and replacements with similar-sounding letters. The errors the algorithm induces are images that are unfaithful to the prompt in some way. For example, one image is missing any depiction of ice cream when prompted for “A giant icae cream sculpture towering over a miniature town.” The method developed has three components. First, the most important words in the prompt are selected via word ablation. Then, random modifications are applied to that word (typos, etc.). Finally, the most successful modifications are adopted and the iteration repeats until a stopping condition is met. In order to instantiate this algorithm, the authors adapt and develop measurements for distance between distributions of texts and distributions of images. 

In the parlance of the adversarial examples literature, this can be considered an untargeted black box attack.

### Strengths
The paper’s main contribution is in adapting metrics for distance between distributions to the problem of generating “adversarial examples” for text-to-image models. While the final algorithm is quite simple at its core, the choice of metrics to use for attacking models that have non-deterministic outputs is not immediately clear. This paper examines the design space and its conclusions are useful reading for the community.  

It is also important to note that, to my knowledge, this is the first paper that strives to preserve the semantics of the input prompt in attacks on text-to-image generating models. Even though, as I point out in the Weaknesses section, how much those semantics are actually preserved is arguable, it is important that the method does not produce gibberish prompts.

### Weaknesses
The paper should better address the question of whether the perturbed text meaningfully alters the semantics of the text – to the point where the images produced remain faithful to the modified prompt, even if different from the intention of the original prompt. This has particular relevance since the paper presents an untargeted attack and untargeted attacks in image generation can have a very loose definition. This has three prongs:
1. The example prompt and image pairs that are selected in Figures 1, 6, 7, and 8 can reasonably be challenged on the grounds that many modifications, although minor in Levenstein distance, actually alter the semantics. For example, it is not obvious that the word “pamda” with an “m” should produce an image the animal panda. 
2. It is unclear what the following quantities are precisely: human labeler-provided consistency scores (in particular, what are N1 and N2 in Appendix D?), SI2T, and ST2I.
3. The prompts that humans typed in seem rather to be rather easy attempts that do not alter the text semantics as much as the auto-modified prompts do. But humans can conceivably try harder. Is there a way to normalize for the strength of the attack? 

Another point on which the paper can be improved is to develop methods for generating images based on the adversary’s chosen semantics. In many practical scenarios, it is important to understand if images can draw very specific violating concepts. For example, adversaries might want to generate specific public figures in compromising scenarios.

The authors might also improve the paper by including a discussion of the ability to undo the adversarial modifications with different tokenization, with hand-written rules, or with Large Language Models that undo typos and glyph and phonetic replacements.

### Questions
Can you define exactly how SI2T and ST2I and their adversarial counterparts are computed?

Can you define the human-provided consistency scores described in Appendix D (N1 and N2)? Why is a difference greater than 1 (as opposed to 0) the “pivotal” difference determining attack success rate?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to evaluate the robustness of the off-the-shelf text-to-image diffusion models r.w.t. the text perturbations including the typo, glyph, and phonetic. Different from other works that focus on textual adversarial attacks, this paper focuses on these text modifications that commonly happen in real-world applications. By proposing four different adversarial attack objectives, e.g., maximum mean discrepancy, and KL-divergence, this paper proposes to perturb the text condition to ensure these objectives. Experiments on several off-the-shelf T2I diffusion models show the attack objectives contribute to reducing the similarity between text and images.

### Strengths
1. This paper studies an interesting and important problem, i.e., the adversarial robustness of the text-to-image diffusion model. The text-to-image diffusion model has shown its power in generating images in the zero-shot setting but the robustness to the adversarial attacks is under-studied. This paper focuses on a significant problem to evaluate the robustness against the common perturbations.

2. The writing is easy to understand and the concepts are well illustrated with figures.

### Weaknesses
1. The main concern is the significance of the problem setting/definition. First of all, generally, the attack should be deployed with some malicious goal, while this paper considers the typos/glyph/phonetics as "real-world attacks" which could be controversial. In the real-world scenario, the existing typos/glyphs/phonetics are usually mistakenly caused by the users of T2I applications without malicious intentions, and even with a bad generation, the users can go back to check the mistakes in the prompts and fix them. It is hard to understand the purpose of these "adversarial attacks" in this scenario. Second, whether the robustness of the diffusion model to the text input is good or bad is another important question. In Figure 1, it is hard to say whether the generations based on these modified texts are "wrong". From another perspective, we can even conclude that these text-to-image models are "accurate" in generating images that reflect the text. Since the texts and their semantic meanings have changed, the generation "accurately" reflects the changes. It is also controversial that what criteria does the human evaluation adopts. Is it based on the difference between the image content and the original meaning of the text or the actual meaning of the text?

2. All the examples are based on the deviation of the objects in the texts. As one character change in the noun can lead to another meaning, it is normal to see that the T2I does not generate the correct objects. The question is does this "mismatch" only happen in the noun words? 

3. The details of the human evaluation are not clear. Also, even with the clean inputs, the mismatch in the existing T2I models happens. The authors should provide the human evaluation results on the original text and corresponding images as a baseline, otherwise the human evaluation results in Table 2&3 are meaningless.

### Questions
1. Is the human evaluation based on the difference between the image content and the original meaning of the text or the actual meaning of the text?

2. Does the mismatch only happen on noun words?

3. What is the possible application of these real-world attacks?

4. Can you provide the human evaluation on the original task as the baseline?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper evaluate the robustness of T2I DMs by adversarial attacks. Specifically, this paper treats realistic errors (e.g., typo, glyph, phonetic) that humans can make as minor perturbations and optimizes adversarial perturbations under distribution guidance. The method is validated on stable diffusion and DALL-E 2.

### Strengths
-- Robustness evaluation of T2I DMs is an important and interesting problem; this paper reveals that small perturbations to the prompt can result in significant changes in the generated images.

-- Realistic errors are used to attack DMs. This perturbation is more subtle and is more likely to occur in real-world scenarios.

-- Distribution guidance is used to optimize adversarial samples. Compared to targeting a single image, this approach is more novel and has been proven to be more effective by the authors.

### Weaknesses
This paper focus on an important and interesting problem, i.e., robustness evaluation of T2I DMs, and proposes a novel distribution guidance method for generating adversarial samples. However, its performance is not fully validated, and the authors need to provide enough details to guarantee its reproducibility.

-- The insight of the definition of robustness in the paper is that, after minor perturbation perturbations to the prompt, DMs should still generate similar images. However, based on the examples provided in the paper, successful attacks have led to changes in the subject of prompts (e.g., in Fig.7, ‘panda’ changed to ‘pamda’), and images generated from such prompts should inherently be different. The authors need to demonstrate the generation of significantly different images when the perturbed prompt and the original prompt should, in fact, result in similar images.

-- The authors mentioned that introducing distribution guidance is aimed at addressing the randomness. However, they only compare the performance of distribution guidance and single image guidance in terms of the decrease in distribution metrics. To better demonstrate the ability of distribution to handle randomness, the authors should include experimental results for single image guidance in the 'real-world attack experiment' section.

### Questions
Please see the two questions list in the weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
