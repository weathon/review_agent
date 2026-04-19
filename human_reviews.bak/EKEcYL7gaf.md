# Predicated Diffusion: Predicate Logic-Based Attention Guidance for Text-to-Image Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3, 6

## Abstract
Diffusion models have achieved remarkable results in generating high-quality, diverse, and creative images. However, when it comes to text-based image generation, they often fail to capture the intended meaning presented in the text. For instance, a specified object may not be generated, an unnecessary object may be generated, and an adjective may alter objects it was not intended to modify. Moreover, we found that relationships indicating possession between objects are often overlooked. While users' intentions in the text are diverse, existing methods tend to specialize in only some aspects of these. In this paper, we propose Predicated Diffusion, a unified framework to express users' intentions. We consider that the root of the above issues lies in the text encoder, which often focuses only on individual words and neglects the logical relationships between them. The proposed method does not solely rely on the text encoder, but instead, represents the intended meaning in the text as propositions using predicate logic and treats the pixels in the attention maps as the fuzzy predicates. This enables us to obtain a differentiable loss function that makes the image fulfill the proposition by minimizing it. When compared to several existing methods, we demonstrated that Predicated Diffusion can generate images that are more faithful to various text prompts, as verified by human evaluators and pretrained image-text models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed Predicated Diffusion, a comprehensive framework designed to articulate users' intentions effectively. This approach leverages predicate logic and utilizes pixels within attention maps as fuzzy predicates, with propositions serving as the textual representation. By employing this methodology, it transforms these intentions into a differentiable loss function. The experimental findings demonstrate a heightened faithfulness to the provided prompts. Furthermore, the paper introduces the concept of 'possession failure,' which expands the scope of inquiry to encompass the existence or non-existence of objects and attributes.

### Strengths
1.By using the Predicate Logic method, this paper easily converts the intentions of prompts into a differentiable loss function, which is a simple, intuitive, and effective method.

2.This paper proposes the term “possession failure” to describe the situation of missing attributes of prompts in T2I models. It provides a detailed direction for modeling the missing attributes question.

### Weaknesses
1. This paper represents an increment in the field, integrating additional predicate logics into the Attend-and-Excite framework. However, it's worth noting that the quality of the generated images appears to be subpar. For instance, Figure 1 illustrates issues such as a blurred bird in the first column, a cat with distorted textures in the second column, and a figure with missing eyes in the last column. These results suggest that the introduction of additional loss functions may have had a detrimental effect on the original model. Thus, a crucial question arises: How can we retain the benefits of the original model while addressing these shortcomings?

2. Despite the proposed method, the problem of "attribute leakage" persists. For example, in Figure 3, the model still generates two apples in response to the prompt "an apple and a lion," and this issue remains evident in Figure 4 with the prompt "a green balloon and a purple clock."

3. While this method effectively addresses the "possession failure" issue, it primarily focuses on Stable-Diffusion v1.4, rather than the latest SDXL model or the DALLE3 model. As a result, there may be limited instances of "possession failure," prompting a need to evaluate the overall contribution of this paper.

4. The paper conducts four distinct experiments to showcase the effects of integrating predicate logics. However, the question remains: If all predicate logics were learned simultaneously, what impact would this have on the original model's performance? Could it further deteriorate the model's results?

5. The proposed method lacks a discussion of its limitations. It is imperative to address and acknowledge the limitations of this approach in order to provide a comprehensive evaluation.

6. Several typographical errors require attention, such as the statement, "the existing models rarely fail to generate an object with a specified color." In subsection "One-to-One Correspondence" of the "METHOD" section, it appears that the intended message may be the opposite. Please clarify this statement for greater clarity.

### Questions
Please see the weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduced a novel approach to guide text-to-image diffusion models in order to improve relation consistency within a generative image given a prompt. To this end, the introduced guidance links predicate logic and attention maps of diffusion models. The authors motivated the utilization of logic based on four issues, namely missing objects, unintended mixture of objects, attribute leakage between objects, and possession failure. After introducing the methodology, these issues are used to assess the performance of the proposed guidance and other baselines.

### Strengths
- The paper addresses a currently unresolved issue of text-to-image diffusion models.
- The issues addressed are well explained and motivated.
- The implementation of propositions via attention maps is well introduced and well supported with examples, which makes it easy to understand.
- While the evaluation only considers Stable Diffusion, the approach can be transferred to other diffusion models without the need to adapt parameters and any additional training.

### Weaknesses
- While the methodology is well introduced, the experiments lack clarity. 
	- It is, for example, unclear how the prompts were selected. Are they extracted from existing datasets? 
	- In the case of experiment 3, why is the similarity metric missing? 
	- In Tables 3 and 4, is the fidelity corresponding to the rating from the user study? Are the reported values normalized? The authors describe that fidelity is assessed by human evaluators and two automated similarity approaches. However, it is not clear to which column these different metrics correspond. Further, it is unclear how the human ratings were aggregated; how many images were assessed by each human evaluator? What is reported, e.g., majority decision?

- The limitations are not well discussed. E.g., the compute overhead of the additional predicate logic-based guidance is unclear. Compared to, e.g., autoregressive image generative models, diffusion models’ inference time is rather slow. While approaches exist tackling these issues, I assume that the additional guidance introduced increases computation.

- Missing related work:
	- Universal Guidance for Diffusion Models. Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein. CVPR Workshops 2023.
	- SEGA: Instructing Diffusion using Semantic Dimensions. Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting. In Proceedings of NeurIPS 2023


Minor comment:

Typo: Section 3 second paragraph "Predicate Logic in Attention Map and Resulting Gauidance" -> Resulting Guidance

### Questions
Next to the questions raised above:
	
- Can you provide the computation costs you observed in your experiments, especially the additional overhead of using the introduced guidance?
	
- Which Stable Diffusion version is used in the experiments?
 	
- You mentioned that the text encoder causes the addressed issues. Did you evaluate your method on diffusion models not relying on the CLIP text encoder and instead using, e.g., a more complex LM such as T5? For example, IF or Stable Diffusion XL? And could the introduced guidance be utilized during training or fine-tuning the text encoder?
	
- Why is the fidelity increasing when using Predicate Diffusion? Is this because of resolving the issues of object mixtures?
- How were the human ratings aggregated? 
- Can you provide more details on the conducted user study? How many images were assessed by each human evaluator? What is reported, e.g., majority decision? Why are eight raters an appropriate and sufficient number of participants?

### Soundness
3 good

### Presentation
2 fair

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
This paper studies the misalignment between image and text in text-to-image generation. The paper proposes a framework that represents the input text prompt using predicate logic. The attention weight of each pixel is then considered as a continuous value that indicates the level of fulfillment of a pixel for a specific proposition. The intermediate image at each denoising step is then updated in order to maximize the level of fulfillment of the input prompt. Experiments show that the proposed method outperforms several baselines on generating more complete objects and objects with correct colors.

### Strengths
1. The paper proposes a novel framework for generating images that are faithful to the input text prompt. The framework is generic in that it covers various issues that have been studied in previous works, such as missing objects and mistakenly bonded colors.
2. The experiments show that the proposed method outperforms existing baselines on four evaluated settings.

### Weaknesses
1. Some of the assumptions that are used for representing text prompt as predicate logic do not make sense. For example, the prompt "There is a black dog" is interpreted as "There is a dog" AND "All dogs are black," which won't work for prompts such as "A black dog and a white dog." Similarly, prompts that have possession relationships such as "a man holding a bag" is interpreted as "all pixels of the bag is also part of the man," which is not necessarily correct.
2. The proposed optimization method will not guarantee that all predicates are satisfied. When multiple predicates exist in the text prompt, their conjunction is used as the objective function. However, since this is a multi-objective optimization problem, the optimization used in the paper is not guaranteed to find optimal solution for all predicates.
3. The visualized images in the paper seem to not have as good quality as the baselines. No metrics (either automatic ones such as FID or subjective evaluations) are reported in the paper.

### Questions
1. Should $P(x) \rightarrow Q(x)$ be $1-A_P[i] \times (1-A_P[i] \times A_Q[i])$?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Predicated Diffusion which combines predicate logic with the intuition of cross-attention layers in diffusion-based text-to-image. The paper draws connections between several propositions and attention map operations. Language prompts can be seen as a combination of these propositions and have corresponding loss functions that can be optimized in the diffusion process.  Experimental results show that the method outperforms several baselines including a recent SOTA method in this direction.

### Strengths
- The proposed method is novel. It is very interesting to see how first-order logic can be connected to compositionality in text-to-image generation, specifically the attention maps. Some of the propositions and losses are reasonable and interpretable. 
- The proposed method tackles a wide range of problems, including well-studied ones and also an underaddressed problem, i.e. possession failure. 
- Experimental results show that the method outperforms previous methods in many aspects.

### Weaknesses
- Some of the losses are not intuitive or cannot be easily verified. I am not sure if this is due to the presentation of the method section. For example, how does eq (2) prevent the two objects from highlighting the same pixels or regions? It would be better to give straightforward intuition behind the equations in terms of the behavior of attention maps. For example, if I understand correctly, eq 6 encourages the attention maps of "bag" to be partially overlapped with the attention maps of "man" yet does not force all pixels of "bag" to be part of the "man". 
- Predicated Diffusion requires manual or pre-defined use of different propositions for different prompts. As stated in Sec. 4, the authors applied different losses for different types of prompts. However, this is not practical for applications where prompts can be arbitrary. The authors manually extracted propositions for each prompt in Experiment (iv) which, I think, really downgrades the overall value of the work. Is there an automatic way to extract propositions for each prompt?
- Writing could be improved. I think Sec. 3 could be improved in structure and contents. Some paragraphs have too many logical equations that make them a bit hard to follow. Perhaps the authors could find a more organized way to explain every proposition (e.g. start with a simple derivation of logic equations, then provide the attention equation, and finally give some intuition in words. ). The authors could attempt to group contents into subsections to illustrate propositions from easy ones to hard ones and distinguish the novel propositions over A&E or SynGen. There are other trivial flaws like using "Experiment (x)" in tables/captions without specifying the experiment domain, making it hard to follow. 

While I really like the novelty and perspectives presented by the work, there are major weaknesses. I will adjust my rating accordingly depending on how well these concerns are resolved.

### Questions
See above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
