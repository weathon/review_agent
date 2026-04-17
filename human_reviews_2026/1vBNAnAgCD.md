# GRADIEND: Feature Learning within Neural Networks Exemplified through Biases

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
AI systems frequently exhibit and amplify social biases, leading to harmful consequences in critical areas. This study introduces a novel encoder-decoder approach that leverages model gradients to learn a feature neuron encoding societal bias information such as gender, race, and religion. We show that our method can not only identify which weights of a model need to be changed to modify a feature, but even demonstrate that this can be used to rewrite models to debias them while maintaining other capabilities. We demonstrate the effectiveness of our approach across various model architectures and highlight its potential for broader applications.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to utilize the internal representations of a Language Model (LM) to analyze and address biases embedded within its parametric knowledge.

The core methodology involves training an encoder-only LM, wherein tokens representing target interpretations or attributes are systematically masked. During this training phase, gradients are meticulously collected and subsequently analyzed to compute orthogonal directions that effectively decouple feature updates associated with biases from those of non-biased information. Finally, these derived orthogonal directions are intended to be deployed to mitigate biases during the actual training process of the larger Language Models.

### Strengths
1. This paper introduces a highly effective methodology for analyzing and potentially mitigating representational biases withi language models. The proposed method leverages the gradients collected during the training process of the target LM. Specifically, it employs a single neuron bottleneck encoder-decoder network to classify updates into bias-related and non-bias-related feature classes. The experimental results demonstrate that this approach is robust and significantly useful in isolating these components.
2. The innovative application of the encoder-decoder network extends beyond classification to bias mitigation, which is a particularly nuanced and valuable contribution. The network is trained by reconstructing the gradients of both bias and non-bias updates. By strategically modifying the decoder component, the approach can effectively modulate the model's propensity to learn or reinforce undesirable bias information during subsequent training.
3. This line of research is highly pertinent and contributes significantly to the current landscape of large model safety and alignment. Given the demonstrable harm caused when models inadvertently internalize and propagate toxic information or biased distributions found in real-world data, the work presented here is profoundly meaningful. In my assessment, studies focusing on the technical mechanisms of bias identification and mitigation, as detailed in this paper, are essential for promoting responsible AI development.

### Weaknesses
Given that fine-tuning remains the dominant paradigm in the modern development and deployment of Language Models (LMs), the current methodology presented in this paper appears to overlook its direct applicability within this context.

It would significantly strengthen the paper's relevance and impact to include a dedicated discussion on how the proposed method can be practically leveraged or adapted during the fine-tuning process. This discussion should address potential complexities, necessary modifications to the core technique, or provide a theoretical justification for its utility in a fine-tuning environment.

### Questions
1. The current description lacks sufficient detail regarding the gradient collection and utilization process, leading to confusion about the method's implementation. Specifically:

- Gradient Acquisition: It is unclear precisely how the gradients are collected in the proposed method. The paper must specify which layers (e.g., embedding, attention block, or final layer) are targeted for gradient emphasis and extraction, as this significantly impacts the captured feature information.

- Training Set Construction: The process of using the gradient differences to train the bottleneck encoder-decoder network is not clearly delineated. Authors must elaborate on the strategy used to split the collected gradients (or gradient differences) into appropriate training and validation sets for the reconstruction task.

2. The design choice for the bottleneck module warrants further discussion and justification.

- Dimensionality Analysis: Could the authors explore the impact of adding more neurons (i.e., increasing the dimensionality) to the bottleneck module? While a smaller bottleneck enforces stronger compression, it may limit the representational capacity needed to distinctly separate the bias and non-bias classes.

- Classification Performance: A detailed analysis or ablation study should be provided to demonstrate whether a slight increase in bottleneck size could lead to improved classification performance and better feature discrimination between the different update types.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This study proposes a new technique that uses model gradients to learn a feature encoding for societal topics like gender, race, and religion. The authors show that this encoding can be used to adjust the model's bias. The paper includes many experiments on different models. The results show that the method has a decent effect on changing model bias while keeping its capabilities.

### Strengths
The main strength is the new technique. Using gradients to learn a feature for bias is an interesting idea. The study also has a wide range of experiments. The authors evaluated on a vast set of models, which is good. The results show the method can change the models, which is a high impact.

### Weaknesses
- The paper's primary weakness is its presentation, which makes the methodology difficult to understand. The authors first explain the methodology with formal definitions and mathematics before providing a high-level overview. In its current format it is really hard for the reader to comprehend what you are actually doing. The paper would be significantly improved by first explaining the method conceptually and providing intuition about each step, and then diving into the formal definitions. An example after your formal definition would further improve the comprehensiveness.

- This lack of clarity extends to the experimental setup. As an experimental focused study, it fails to adequately give intuition about different parts of the methodology. For example, the creation of the data for racial and religious bias contexts is unclear (Lines 201-202). For gender, it is questionable whether the name-pronoun data (Lines 198-199) is enough to capture gender information, or if other contexts like gendered nouns or adjectives are needed.

- Methodologically, I am not convinced that this method is as effective for generative models. The approach seems more focused on encoder models. For generative models, the feature neuron is constructed from a partial section of the data (the left-side context), which may be insufficient to learn a robust feature representation.

- Finally, the evaluation of the results is not fully convincing. Looking at the individual outputs of the models in table 12, does not strongly indicate the superiority of this method. The aggregated results in Table 2 are used to make stronger claims, but the paper does not specify the methodology in detail on how you did calculate these aggregated ranks.

Minor presentation issues also detract from the paper:

- The referencing of (H1) and (H2) in lines 185-187 and 255 is inconvenient. The phrases should be written out. In Line 255, it is unclear if (H1) is part of the sentence.

- In Line 146, the manuscript uses incorrect starting quotation characters

### Questions
- What if the target words are not a single token? How does your methodology account for this?

- How did you ensure that there is enough information for the generative models in the left side of the masked token to compute meaningful gradients?

- Why did you present StereoSet SS and LMS scores separately? Given that you are measuring bias and model capabilities. Why you did not present the ICAT score?

### Soundness
3

### Presentation
2

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
The paper proposes GRADIEND, a simple encoder–decoder framework that learns interpretable feature neurons from model gradients. It encodes gradient differences tied to sensitive attributes into a scalar value and decodes them into weight updates that can rewrite model parameters to reduce bias. On seven transformer models, GRADIEND is tested to be widely applicable, learning clear feature directions and outperforming baselines when further combined with other post-processing debiasing techniques.

### Strengths
1. The idea of encoding feature semantics directly from gradients is elegant, bridging interpretability and parameter-level debiasing with weight-modifying.
2. The proposed method is mathematically well-formulated. The vivid illustration in Fig. 1 and explanations in Section 3 are very clear.
3. I appreciate the experiments. The authors evaluated the method on seven major transformer models covering both encoder- and decoder-only LMs. They conducted a systematic analysis across three bias domains and used multiple bias metrics for thorough benchmarking.
4. The empirical studies demonstrate effective bias modification without major performance loss. The performance can be further boosted when combined with other post-processing techniques.

### Weaknesses
1. The encoder–decoder setup might be viewed as a re-parameterization of gradient differences, lacking deeper theoretical justification or analysis of why it works indeed.
2. There is no theoretical analysis of how the task performance would be affected.
3. Binary gender assumption and limited race/religion classes make the fairness conclusions narrow.
4. There is no comparison with more recent causal or reinforcement-based debiasing techniques.

### Questions
1. Could you elaborate more on how the encoder can be trained jointly on multiple bias axes?
2. How stable are learned feature neurons across random seeds or pretraining checkpoints?
3. Can the method generalize to multimodal models (e.g., vision–language)?

### Soundness
2

### Presentation
3

### Contribution
2
