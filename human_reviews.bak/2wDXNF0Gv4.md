# Prompt-Agnostic Erasure for Diffusion Models Using Task Vectors

- Decision: Reject
- Scores: 6, 8, 6, 5

## Abstract
With the rapid growth of text-to-image models, a variety of techniques have been suggested to prevent undesirable image generations. Yet, these methods often only protect against specific user prompts and have been shown to allow undesirable generations with other inputs. Here we focus on \textit{unconditionally} erasing a concept from a text-to-image model rather than conditioning the erasure on the user's prompt. We first show that compared to input-dependent erasure methods, concept erasure that uses Task Vectors (TV) is more robust to unexpected user inputs, not seen during training. However, TV-based erasure can also affect the core performance of the edited model, particularly when the required edit strength is unknown. To this end, we propose a method called \textit{Diverse Inversion}, which we use to estimate the required strength of the TV edit. Diverse Inversion finds within the model input space a large set of word embeddings, each of which induces the generation of the target concept. We find that encouraging diversity in the set makes our estimation more robust to unexpected prompts. Finally, we show that Diverse Inversion enables us to apply a TV edit only to a subset of the model weights, enhancing the erasure capabilities while better maintaining model utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of preventing style mimicry in text-to-image models by proposing an unconditioned approach to concept erasure, independent of user prompts. This approach uses Task Vectors (TV) for concept erasure, offering greater robustness to unexpected user inputs. Also, the authors introduce Diverse Inversion, a technique that estimates the required TV edit strength by identifying a broad set of word embeddings within the model’s input space, each capable of generating the target concept.

### Strengths
- Clarity and Structure: The paper is well-organized and clearly written, making it accessible and easy to follow, even for readers less familiar with the technical aspects of concept erasure and Task Vectors.
- Visualization Quality: The visualizations of generated images are well-crafted, effectively illustrating the model’s concept erasure capabilities and supporting the clarity of experimental results.
- Clear Literature Review: The related work section thoroughly covers relevant research on concept erasure and on jailbreaking generative models. This strong contextual foundation helps to situate the authors’ contributions within the broader field and underscores the necessity of robust model editing methods.

### Weaknesses
- Edit Block Selection: The rationale for editing the first three blocks is not fully explained. A discussion on why these specific blocks were chosen would strengthen the methodological foundation. I suggest that the authors provide a brief explanation of the model architecture and how the blocks relate to different levels of abstraction or functionality.
- Alpha Parameter Choice: The choice of α is not well-clarified. While Figure 4 mentions α, no figure or table apart from Figure 7 details the specific α values used. Since Diverse Inversion is intended to estimate the optimal strength of the Task Vector (TV) edit, it would be beneficial to provide explicit α values and clarify if the authors tested a range of α values to identify the best-performing option. I suggest that the authors include a table or figure to illustrate how they arrived at optimal strength.
- Figure Placement: Figure 1 appears on page 2, yet it is first referenced on page 4. It would improve readability and flow by moving the figure closer to its initial mention or adding an earlier reference to it in the text
- Table Clarity: In Table 2 (page 10), the acronym “SLD-Med” lacks explanation, and the term “UCE” is only briefly mentioned in the related work section (page 3). It’s unclear if SLD-Med and UCE refer to the same concept; clearer definitions would enhance comprehension. I suggest that the authors include a brief explanation of these terms in a footnote or in the table caption.
- Equation Definition: In Equation 4, the variables [a, b] and [c, d] are not clearly defined. While the meaning can be inferred from the surrounding text (Lines 341-343), each variable in the equation should be explicitly defined. I suggest that the authors consider adding a brief explanation of these variables immediately following the equation, which would maintain the mathematical formalism while improving readability. Alternatively, consider replacing the equation with a detailed textual description if it enhances clarity. 
- Typos and Formatting Issues:
  - Line 285: "Sec.3.2" should be "Sec. 3.2".
  - Line 343: "e.g. Van Gogh" should be "e.g., Van Gogh".
  - Line 354: "I.e." should be formatted as "I.e.," or, for clarity, replaced with "For example,".
  - Line 355-356: The sentence lacks a verb; it currently reads “we can the value of the edit strength α.” Please revise for clarity.
  - Line 360: "i.e. setting" should be "i.e., setting". 
  - Line 400: "In Figs" should be "In Fig".

### Questions
- Edit Block Selection: What was the rationale for choosing to edit only the first three blocks in the model? Would the authors consider expanding on why these specific blocks were selected for editing?
- Alpha Parameter Choice: The choice of the α parameter remains somewhat unclear, with few details provided outside of Figure 7. Could the authors specify the α values used throughout the experiments and clarify whether they evaluated multiple α values to determine the optimal edit strength?
- Figure Placement: Would the authors consider moving Figure 1 closer to its first reference on page 4 to improve readability and flow?
- Table Clarity: Could the authors clarify the meaning of “SLD-Med” in Table 2 (page 10) and confirm if it is the same as “UCE” mentioned briefly in the related work section? Including these definitions would improve comprehension.
- Equation Definition: In Equation 4, the terms  and  are not clearly defined. Could the authors provide explicit definitions for each variable, or alternatively, replace the equation with a detailed textual description if that would improve clarity?
- Typos and Formatting: There are minor typos and formatting inconsistencies (e.g., “Sec.3.2” instead of “Sec. 3.2”). Would the authors consider addressing these issues to enhance overall readability?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a novel method for concept erasure in pre-trained generative models. This method consists of two key components: (1) the development of a Task Vector Method for concept erasure; and (2) the selection of optimal parameters through novel Diverse Inversion procedure. Notably, this approach is input-independent and does not rely on specific pre-defined prompts that contain concepts. As a result, it demonstrates enhanced robustness against concept inversion when compared to previous methods, while maintaining comparable results on unrelated concepts generation tasks and within the "given prompt generation" setting.

### Strengths
- The authors clearly identify the problem of “input dependence” associated with previous methods and provide compelling evidence of these issues via the MNIST toy experiment, which emphasizes prompt complexity rather than using a fixed set of prompts. 

- They propose a method to address these challenges, which combines an existing concept-forgetting technique Task Vectors with a novel procedure called Diverse Inversion to optimize parameter selection for Task Vectors. 

- Although Task Vectors is an already existing technique, the authors unveil its previously unexplored property of Concept Inversion Robustness.

- The Diverse Inversion idea is an interesting approach that could be applied to other research areas, potentially enhancing our understanding of concept learning and erasure processes. 

- Overall, the text is straightforward and presents all ideas clearly and concisely.

### Weaknesses
- Certain aspects of the experimental workflow are not sufficiently detailed. For instance, the setup of the toy experiment on MNIST lacks information regarding the embedding grid search procedure. Additionally, the Diverse Inversion Set selection procedure may need more clarification, particularly regarding the number of restarts of the Concept Inversion procedure and a comprehensive step-by-step description.

- Furthermore, it appears that the vector from the Diverse Inversion set, which is utilized for selecting the parameter alpha, was also employed in evaluating the robustness of the methods against Concept Inversion. If this is the case, it would be helpful to report how the metrics would be affected if this vector were removed from the Diverse Inversion set.

- It would be beneficial to include additional visual examples to illustrate the results presented in Table 2.

### Questions
1. Was the vector from the Diverse Inversion set used in evaluating the robustness of the methods against Concept Inversion? If so, could you please provide information on how the metrics would change if this vector were excluded from the Diverse Inversion set?

2. Could you provide a step-by-step description of the Diverse Inversion Set selection procedure? Additionally, please include details on the number of restarts for the Concept Inversion procedure.

3. Why is the Control Task not utilized for selecting alpha, alongside the Diverse Inversion set?

4. Can you elaborate on the toy example, specifically regarding the embedding grid search procedure?

5. It would be beneficial to include additional visual examples to illustrate the results presented in Table 2.

### Soundness
3

### Presentation
4

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
The paper introduces a technique for erasing concepts from diffusion models. The method is based on using task vectors to erase the concepts, in combination with diverse inversion, a form of textual inversion. A key feature is that the erasure is prompt-agnostic and is designed to work with diverse prompts, especially adversarial ones.

### Strengths
* There is a decent initial analysis to motivate the approach and explain why it may be suitable.
* The method seems to perform well, maintaining the quality of the generated images for non-erased concepts, and successfully erasing the selected concepts.
* In general the paper is easy to follow.

### Weaknesses
* The evaluation is quite limited, it would be good if quantitative evaluation included diverse adversarial techniques in addition to concept inversion. There are some qualitative results for UnlearnDiffAtk and P4D in the appendix, but the paper would benefit from using these and maybe even others for more extensive quantitative evaluation. Also it would be good to show the method works also on other models than Stable Diffusion v1.4 specifically.
* The method seems to be primarily a combination of task vector technique and a version of text inversion, applied to the problem of concept erasure, so it may lack significant novelty.
* There are quite a few issues with the writing and presentation - the font is different than the standard one, this should be corrected; various typos, grammar issues or missing words, e.g. “jailbraking” L145, “might in some cases the usability might degrade” L358,  “Fig. 6 demonstrate” L410, “how how” L414, …

### Questions
* What could the prompts look like for a given complexity class L? Does it directly translate to the number of words?
* Can this method actually remove small parts of the image such as copyright logos? It was used in motivation but seems to not be tested?
* How well does the method work when using other adversarial techniques such as UnlearnDiffAtk and P4D - quantitative evaluation, not only qualitative that is already provided?
* Does the approach work well also on other diffusion models than Stable Diffusion v1.4?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents an interpretability study focused on understanding the second-order effects of neurons in CLIP. The authors propose a novel "second-order lens" to analyze neuron contributions that flow through attention heads to the model output.

### Strengths
1. The technical contributions are sound  and interesting.
2. The paper is well written. 
3. The paper included thorough evaluations.

### Weaknesses
1. Multiple concept erasure - How does the proposed method perform on multi-concept erasure? The baselines considered in this paper (UCE and ESD) evaluate their model on erasing multiple objects simultaneously. Therefore it is fair to compare this method for multi-concept erasure.
2. Missing baselines - Comparison to Selective Amnesia (SA) (a strong and very similar baseline in my opinion) is missing from the paper. I believe the proposed method lie under a similar umbrella as SA. 
3. Underperforms baselines on NSFW concepts—The authors state that TV only reduces nudity in 52% of images compared to SD1.4, which is worse than the baselines (ESD, UCE, etc.) considered in the paper. This is a major drawback of the method in a real-world setting.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
