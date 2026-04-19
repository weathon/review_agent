# Is What You Ask For What You Get?  Investigating Concept Associations in Text-to-Image Models

- Decision: Reject
- Scores: 8, 3, 6, 6

## Abstract
Text-to-image (T2I) models are increasingly used in impactful real-life applications. As such, there is a growing need to audit these models to ensure that they generate desirable, task-appropriate images. However, systematically inspecting the associations between prompts and generated content in a human-understandable way remains challenging. To address this, we propose Concept2Concept, a framework where we characterize conditional distributions of vision language models using interpretable concepts and metrics that can be defined in terms of these concepts. This characterization allows us to use our framework to audit models and prompt-datasets. To demonstrate, we investigate several case studies of conditional distributions of prompts, such as user defined distributions or empirical, real world distributions. Lastly, we implement Concept2Concept as an open-source interactive visualization tool facilitating use by non-technical end-users. *Warning: This paper contains discussions of harmful content, including child sexual abuse material and NSFW
material, which may be disturbing to some readers.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Concept2Concept, a framework for auditing text-to-image models by analyzing the associations between generated images and prompts using interpretable concepts. It helps uncover biases, harmful content, and unexpected associations in models and datasets, demonstrated through various case studies.

### Strengths
- The paper is very well written and easy to follow. The authors handle the sensitive topics with the appropriate sense of responsibility.
- The selected applications of the method as well as the results are very interesting and I hope will spark a discussion in the communities using the respective datasets. 
- The presented framework increases evaluation robustness due to the three presented metrics, evaluating the relationship between prompts and generated images from different perspectives.

### Weaknesses
**W1:** To my knowledge, there is no anonymous code repository provided with the paper, neither for the experiments nor the tool. As a result, I am unable to comment on the tool's usefulness. It would be beneficial if the experiments could be replicated either within the tool or through a dedicated repository to also validate the correctness of the results. 
The selection and robustness of the results regarding the T2I and VLM detector models are in my opinion just weakly addressed: 

**W2:** Only in the Appendix, it is revealed that the audited T2I model is Stable Diffusion 2.1. This information should be part of the main manuscript as the results of Study 4 only apply to this model architecture. It would be interesting how results would change for other model architectures, as especially closed-source models are strongly safety fine-tuned. If I understand correctly your framework is model-agnostic and could also be applied to closed-source models accessed via API calls. Did you perform any experiments with other model architectures? And if not please argue in the manuscript why you restrict Application 1 to this specific model architecture.

**W3:** While the authors acknowledge that the detection model introduces uncertainty in the extracted concepts (Line 132), they do not address how sensitive the application results are to the choice of the detector model. Could specific concepts be overlooked if a different grounding model is used? Additionally, how does the safety fine-tuning of the detection model potentially conflict with the task of identifying sensitive concepts, such as in CSAM?

I am open to raising my score if the identified weaknesses are either adequately addressed through revisions to the manuscript or convincingly argued to be non-issues.

Comments:
- There is a closed bracket missing in line 187 „P(c“
- There is a closed bracket too much in line 236 „Figure 2)“.

### Questions
See weaknesses W1 to W3.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper systematically examines the associations between text prompts and generated image content in a human-understandable way. Its goal is to audit text-to-image models to ensure they produce desirable and task-appropriate images. The authors propose a framework called Concept2Concept that 1) extracts high-level concepts from generated images and 2) calculates concept distribution to uncover associations between prompts and generated images. Using this framework, the authors have identified potentially harmful associations in popular datasets like Pick-a-Pic.

### Strengths
The paper provides insightful and valuable findings concerning harmful associations present in popular datasets, offering critical observations that can guide future research and model development. The topic itself is highly relevant, and the authors’ motivation is clearly articulated, underscoring the importance of addressing these issues.

### Weaknesses
One limitation of this paper is that the overall framework still relies on human examination and investigation, which may impact its scalability. 

The technical and theoretical contributions are fair but could be strengthened. Further elaboration on the differences from existing work would help to clarify the novelty of this framework. As it stands, the paper resembles more of a technical application report than a traditional academic paper. To demonstrate the framework’s utility, the authors present five case studies that effectively showcase its application; however, they lack cross-model analysis, which would add depth to the evaluation. Using concepts as tools to analyze bias in text-to-image (T2I) models holds strong potential, and it would be beneficial for the analysis to extend into other domains, such as ethnicity, offering a more comprehensive evaluation across multiple models and datasets. The current five case studies, though useful, may fall short of meeting the quality criteria expected in a top conference.

Besides, why there is no information about the used T2I model in the main paper? and in the appendix, there is no discussion about the choice of the model and no discussion about the different comparisons of different models 

Additionally, there are minor typos (e.g., in Line 236, Figure 2) that could benefit from correction.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

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
This work introduces a framework for auditing T2I models and datasets. The framework uses discriminative models to find objects or concepts in generated images. Using the proposed method, the findings of several works that explore the biases of T2I models can be reproduced. Furthermore,

### Strengths
- The paper is well motivated and well written
- The proposed methods and metrics are simple and intuitive
- It is nice that the paper reproduces the findings of prior works using a different evaluation framework
- The paper has some important and worrying finding such as NSFW data in a human preferences dataset
- Open sourcing such a framework would be very useful for practitioners

### Weaknesses
- My main concern is that there is an existing work [1], that has not been acknowledged. It introduces a similar method that uses discriminative models to find co-occurences and biases in the generations T2I models, somewhat limiting the contributions of this work. Nonetheless, I think the other contributions and alaysis of this paper still have merit. 
- The method section has too much fluff and introduces too many concepts that are not used later on. For example, the concept co-occurence formula is never being used, and the concept stability is never explored in the main part of the paper.
- Figure 3: Methodologically, it is not clear what the prompt revision means. Are some concepts used as negative prompts?

[1] OpenBias: Open-set Bias Detection in Text-to-Image Generative Models, CVPR 2024

### Questions
-

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose Concept2Concept, a framework that characterizes the conditional distributions of vision-language models using interpretable concepts and metrics. This enables systematic auditing of both models and prompt datasets. Through case studies, they analyze various prompt distributions, including user-defined and real-world examples. Concept2Concept is also an open-source interactive visualization tool, making it accessible to non-technical users.

### Strengths
- This work addresses the important challenge of auditing text-to-image (T2I) models to assess their reliability, fairness, and bias.
- The authors introduce an interpretation of concept distributions, which forms the basis for their marginal and conditional distribution notations.
- Through various case studies—including bias analysis, disability representation, and model misalignment—the authors explore essential aspects of T2I model auditing.

### Weaknesses
- The primary innovation of the paper lies in interpreting distributions over concepts, leading to the marginal and conditional distributions defined in Equation 3 and summarization metrics in Equations 4-6. However, the connection between these two sets of equations is not well-explained, making it difficult to understand how they are conceptually or mathematically linked.

- Although marginal and conditional distributions are defined for continuous distributions, the summarization metrics—concept frequency, concept stability, and concept co-occurrence—are framed in discrete terms. The authors do not provide a derivation or proof to clarify the connection between continuous and discrete cases, leaving this foundational aspect unclear.

- The authors mention addressing uncertainty from off-the-shelf object detectors by sampling from a distribution of concepts. However, they provide little information on the practical implementation of this approach, making it challenging to interpret how this sampling is achieved or how effective it is in managing uncertainty.

- To address the uncertainty introduced by the object detector, the authors need a more comprehensive analysis, particularly in handling cases where the detector may be over-confident or under-confident. A systematic empirical study to quantify and validate this uncertainty would greatly improve clarity and demonstrate how well the framework manages these corner cases.

- The metrics introduced by the authors—concept frequency, concept stability, and concept co-occurrence—resemble the validity, proximity, and diversity metrics used for counterfactual explanations as defined in [1]. However, there appears to be no discussion connecting these proposed metrics to previous work on counterfactual explanations.

[1] Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations

### Questions
- Could you clarify the conceptual and mathematical connection between the marginal and conditional distributions in Equation 3 and the summarization metrics in Equations 4-6? An explanation of how these are linked would help in understanding the core framework.

- Since the marginal and conditional distributions are defined for continuous distributions, while the summarization metrics are based on discrete cases, could you provide a derivation or rationale that bridges these two? How do you address this foundational difference?

- You mention handling uncertainty from object detectors by sampling from a distribution of concepts, but the practical details of this approach are unclear. Could you elaborate on how this sampling is implemented and how effective it is in managing detection uncertainty?

- Given the similarities between concept frequency, concept stability, and concept co-occurrence and the metrics used in counterfactual explanations (e.g., validity, proximity, and diversity), could you discuss any connections or differences between your proposed metrics and those commonly used in counterfactual work?

### Soundness
2

### Presentation
3

### Contribution
2
