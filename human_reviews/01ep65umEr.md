# TeLLMe what you see: Using LLMs to Explain Neurons in Vision Models

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
As the role of machine learning models continues to expand across diverse fields, the demand for model interpretability grows. This is particularly crucial for deep learning models, which are often referred to as black boxes, due to their highly nonlinear nature. This paper proposes a novel method for generating and evaluating concise explanations for the behavior of specific neurons in trained vision models. Doing so signifies an important step towards better understanding the decision making in neural networks. Our technique draws inspiration from a recently published framework that utilized GPT-4 for interpretability of language models. Here, we extend and expand the method to vision models, offering interpretations based on both neuron activations and weights in the network. We illustrate our approach using an AlexNet model and ViT trained on ImageNet, generating clear, human-readable explanations. Our method outperforms the current state-of-the-art in both quantitative and qualitative assessments, while also demonstrating superior capacity in capturing polysemic neuron behavior. The findings hold promise for enhancing transparency, trust and understanding in the deployment of deep learning vision models across various domains. The relevant code can be found in our GitHub repository.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
They utilize GPT-4 to provide explanations from vision models based on the neuron activations and weights of the vision model. They use AlexNet and ViT as their vision models to show their explanations. They are comparing the neuron explanations  against Clip-Dissect. Also with their method, they have 2 different approaches one is using activations and captions to send to the LLM to generate the explanation. Another is using the weights and labels to use for the LLM. There is another variant to weights-label but instead of labels, they utilize CLIP.

### Strengths
An interesting approach in utilizing LLMs to help provide explanations from other non-LLM models. The vision models can be hard to interpret so utilizing LLMs can help provide more human-readable explanations.

It has the potential for good impact since this was stated before that it can help provide more readable explanations.

### Weaknesses
Writing:
Typos: 
in 3 Method, activations based method should be activations-based method. 
4.3 'we conducted additional experiments of our method an the last' should be 'we conducted additional experiments of our method at the last.'
Pseudocode for Algorithm 3 has a typo, please fix. It should not be activation-caption method since you use weights and labels. 
Also, with Assessment Prompt, you should not denote it as AssPrompt, AssessPrompt would be apt or AsPrompt.
In Figure 4, you mention alexnet but in other parts of the paper, you put AlexNet, please stick with one.

Experiments:
More experiments, you showed with ViT and AlexNet, but it would benefit to have around 3-5 models with around 3 real-world datasets. You showcase ImageNet, consider the CUB-200 dataset and Places to show the impact.

Another potential issue is that you use another LLM to validate the assessment of the explanations. It would benefit to have something that is not an LLM to validate the explanations. One idea is to get user studies to help reinforce that humans and LLMs would agree that the explanations are better than CLIP-Dissect.

Another idea is to use Network Dissection and to show which concept aligns with each neuron from the vision models. Only choose the ones that have a high intersection over union with the said concept and with the LLM generating the explanation if it includes the concept in the explanation then it is considered correct. Have accuracy as the metric and show that CLIP-Dissect has lower accuracy.

### Questions
Please refer to the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method for generating explanations of individual neurons in trained vision models like AlexNet and ViT with the help from Large Language Model. This helps explain the decision-making in these neural networks. The main contributions are to propose two techniques to generate short, human-readable explanations of neuron behavior using a large language model (LLM) like GPT-3.5. One is based on image captions and neuron activations and another is based on neuron weights and class labels. A scoring algorithm to quantitatively assess the quality of the explanations by correlating simulated and actual neuron activations/weights.

### Strengths
1. The paper is the first to leverage Large Language Models (LLMs) like GPT-4 for elucidating the behavior of individual neurons in vision models. 
2. The authors introduce a scalable scoring method to evaluate the quality of generated explanations. It provides an objective measure to compare and benchmark the effectiveness of their approach against other methods. 
3. The proposed method outperforms the included baseline techniques over quantitative and qualitative assessment. It also generates lucid, human-readable explanations. 
4. The generated responses show the ability to capture and explain the behavior of neurons with multiple semantic meanings. This is a challenging aspect of neural network interpretability.

### Weaknesses
1. Limited Value: While the paper presents a novel approach to the interpretability of vision models using LLMs, its practical application value appears to be limited. Bills et al. (2023) proposed a similar method that was valuable because it addressed the vagueness of highlighted sentences by using LLMs to generate clearer summaries. In the context of this paper, the top-activated images for a single neuron already provide a clear representation of the semantic meanings of such nodes. This raises the question: Is there a significant advantage to using the proposed method's explanations over directly observing the top-activated neurons?

3. Cost of Using LLM: The paper employs the LLM API, which is known to be costly, especially when applied to full networks. Given the financial implications, it's essential to justify the added value of this approach over other more cost-effective interpretability methods. In which scenarios would this technique be more beneficial than other cheaper solutions?

In all, the authors should provide a more compelling argument for the unique value proposition of their method. Specifically, they should address why their approach offers advantages over directly examining top-activated neurons or other interpretabiility strategies, which already provide clear semantic representations.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an interesting idea of using Language Model Models to explain the behaviour of neurons in vision models. While the idea is compelling, it could benefit from more robust evaluations and clarifications in several areas:

### Strengths
- The idea of explaining neurons with language models is timely and intresintg.

### Weaknesses
1. The decision to scale/round the activation values to integers in range [0-10] needs further explanation. Each floating-point value may carry important information, and the rationale for rounding should be experimentally justified.

2. The approach appears to heavily rely on GPT, and the assessment step (i.e., measuring the correlation scores) seems to be more a reflection of the GPT model itself rather than an evaluation of the explanations provided.


3. It would be helpful to understand whether using random integers instead of actual activation values for fine-tuning GPT would still yield meaningful explanations. Is Table 4 showing this?

4. An additional evaluation suggestion (faithfulness analysis) could involve zeroing out activations or weights associated with specific categories, such as identifying neurons related to cats and then zeroing them out. Subsequently, observing the impact on the image classifier's performance, specifically on cats and other objects or concepts associated with the affected neurons, could provide valuable insights. For example, Neuron 2 fires when it sees cats, trees, and apples. If we deactivate Neuron 2, probably the classifier should fail on many images of cats,  trees, and apples…

### Questions
Please see Weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a novel method to interpret neuron functions in vision models, enhancing the understanding of deep learning 'black boxes.' Adapted from GPT-4 interpretability techniques, this approach uses neuron activations and weights, providing explanations for neurons of an AlexNet and ViT model trained on ImageNet.

### Strengths
- The paper tries to adopt the latest LLM advancement as an interface to explain the internal neurons in deep learning models.
- Provide statistical analysis on the neuron assessment work to make the work more reliable. 
- The paper provides sufficient demonstration that the proposed method achieves acceptable results, in successfully explaining the neuron’s semantic encoding in language.

### Weaknesses
- The major weakness is that LLM doesn’t provide more information than the visual cue. This is different from the problem in explaining language models with language models since the visualization technique in the vision domain itself could demonstrate the receptive field of the neurons already and can be much more precise and immune from language model bias.
- Presentation is not clear and precise.
- ViT’s important information comes from the attention mechanism. How does the proposed work be used to examine the attention maps?

### Questions
- How does the metrics in section 4.3 demonstrate that the model explains the neurons correctly? More detailed description would be helpful.
- How could one assess the correctness of the GPT interpretation of the neurons?
- How does the GPT explanation help to understand the neural network’s internal decision problem? Deep learning models are known to be distributed representation, meaning that one neuron won’t determine the final decision. How could the proposed method be used to explain the cooperative behavior of the neurons in order to help people understand how the vision model arrives at its decision?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
