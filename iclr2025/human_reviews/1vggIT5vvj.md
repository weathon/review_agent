## Human Reviewer 1

### Summary
This paper mainly focuses on the explainability of text-to-image diffusion model. The authors propose a new metric based on the cross-attention heads in the diffusion UNet to illustrate the correlation between each attention head and visual concepts. Based on the proposed Head Relevance Vectors, the authors further propose several applications including solving polysemous words problems and image editing.

### Strengths
1. The idea of correlating visual concepts with diffusion models is interesting.

### Weaknesses
1. I suggest the authors add textual description of the proposed HRV instead of directly showing Fig. 2 and Fig. 4 for better understanding.
2. I wonder why <SOT> and many <EOT> are required during update of HRV?
3. It would be better to used SDXL or some more recent models such as SD3 as primary model, given that SD1.5 is kind of outdated.
4. It would be better to add a random weakening baseline in Fig.3.
5. In Sec.5.1 the authors show that by utilizing HRV the SD can generate more proper concepts. I wonder if this method can be compared with using classifier guidance, where the model is encouraged to align the generated image with wanted concepts in terms of CLIP score.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper tries to understand cross-attention (CA) layers *regarding attention heads*.
* The authors introduce N head relevance vectors (HRV) for N visual concepts.
* The strength of an attention head to the HRVs represent the relevance of the head to the concept.

Above properties are interpreted by *ordered weakening analysis*.
* Sequentially weaken the activations of CA heads to observe weakened visual concepts.

Boosting and reducing the strength of different heads control the strength of visual concepts. It helps three applications: 1) correcting mis-interpretation of words in t2i generation, 2) boosting prompt-to-prompt, 3) reducing neglect in multi-object generation.

### Strengths
1. This paper provides a new perspective in understanding the features in text-to-image generation: different heads.
2. Qualitative examples (Figure 3a) and CLIP similarities (Figure3b) along weakening MoRHF and LeRHF clearly show the effect of weakening different heads.
3. The appendix provides extensive qualitative results to remove doubt for cherry-picked results.
4. The proposed method is useful for three applications: 1) correcting mis-interpretation of words in t2i generation, 2) boosting prompt-to-prompt, 3) reducing neglect in multi-object generation.
5. Discussions resolve natural questions: extension to SDXL and effect across different timesteps.

### Weaknesses
1. The paper should provide principles of the proposed approaches.
    * L224 Why should we count each visual concept having the largest value to update the HRVs?
    * This is the most critical weakness for not giving a higher rating. I think the perspective is worth noticing but a solid paper should provide why/how it works.
    * Answering this question with theoretical justifications or intuition would strengthen the paper.
2. HRV should be described more clearly.
    * L205 a concatenation of token embeddings // concat along which axis? I guess the result of concatenation is $N \times (d + H)$. Then the query Q does not match the cross-attention operation because $Q\in R^d$. Am I missing something?
    * L210 K1, ..., KN should be denoted in Figure 2.
    * Adding equations and proper notations would help readers to understand the operation.
3. Human evaluation should be explained in more detail.  Appendix C.2 is not enough. Adding a table with Number of participants, Number and types of questions, Number of questions per participant, and Any quality control measures used would strengthen the user study.

Misc.: Related works -> Related work

### Questions
I wonder the interpolation between different strengths of a head. For example, interpolating  material=[-2, 2]?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper proposes a method of constructing so called "HRV" vectors, which align with visual concepts. The authors leverage cross-attention layers of Stable Diffusion model to learn those vectors for predefined concepts. The proposed method helps to solve three known issues of the image synthesis task.

### Strengths
1. Good motivation and a clear idea
2. Comprehensive quantitative and qualitative comparisons with many other solutions
3. The experiments, settings, and other details are mainly clearly explained

### Weaknesses
1. It requires fixing a set of concepts beforehand for every HRV construction. Does not have a study of how the HRV matrix will be changed when some concepts are changed or replaced after the construction.
2. Manual settings, choice, and configuration are required for every concept (case) during inference (Sec 5.1, Fig 5). 
3. Lack of failed cases, there are no details about the limitations of this method.
4. Even though there is a section for bigger / novel models (SDXL), all experiments, studies, and comparisons are based on SD v1. New models might eliminate many issues the proposed method tries to solve.

### Questions
1. Could you give more details about why there are some irrelevant concepts after a certain point of ordered weakening (Fig 9)?
2. Could you give more details about how the "h" is chosen/computed in the method of HRV updates?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This work proposes Head Relevance Vectors (HRVs). HRs are an extension of the findings from previous works such as Hertz et al.'s P2P where cross attention maps were used to better understand t2i models and to edit images via prompts. HRV proposes using multiple concept words and concatenating them into a concept embedding matrix K which can then be applied to different heads of the cross-attention and by doing so, disentangle the different heads based on the concept they seem to be focusing on. The authors show this disentanglement of heads based on the concepts learned improved editing of images.

### Strengths
The motivation of the paper is clear and build on a well studied problem of understanding the role of cross-attention and what they learn in editing T2I models. The experiments are visually appealing and tell the story of the paper well, especially the weakening of HRVs that shows weakening based on the most and least relevant concepts / heads. The authors show that using HRVs to edit images works better than SDEdit,P2P, etc. They also show improvement over Attend and Excite for the problem of catastrophic forgetting in T2I models.

 While in the weaknesses, I do mention my thoughts on the originality of this work, I believe using previous findings around CAs and targeting different heads and their roles in generating different concepts would be interesting to the community.

### Weaknesses
I would argue that the work, while interesting, does not have new insight compared to what previous works such as P2P and Diffusion Self-Guidance have already already shown in regards to the role of cross-attentions. However, this work does take a step towards using those findings to narrow down on head-level manipulation of concept vectors. It goes without saying that T2I models could benefit from more comprehensive evaluation on larger set of generated images / human evaluation. However, I do understand the challenges this poses as well.

### Questions
There have been recent works that show the <SOT> and <EOT> CAs capture different concepts. I would be interested to see if the authors found anything interesting regarding HRV and these tokens. I am also curious as to how the weakening and strengthening would work on more complex images that share entangled objects and concepts. For instance, what would weakening of "melting" look like for "a plastic car melting". I think this would be an interesting experiment since adjective and verb concepts are entangled with an object in a given image and HRV might to better in these cases than the counterparts.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
5