# GPT as Visual Explainer

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6

## Abstract
In this paper, we present Language Model as Visual Explainer (\texttt{LVX}), a systematic approach for interpreting the internal workings of vision models using a tree-structured linguistic explanation, without the need for model training. Central to our strategy is the collaboration between vision models and LLM to craft explanation. On one hand, the LLM is harnessed to delineate hierarchical visual attributes, while concurrently, a text-to-image API retrieves images that are most align with these textual concepts. By mapping the collected text and image to the vision model's embedding space, we construct a hierarchy-structured visual embedding tree. This tree is dynamically pruned and grown by querying the LLM using language templates, tailoring the explanation to the model. Such a scheme allows us to seamlessly incorporate new attributes while eliminating undesired concepts based on the model's representations. When applied to testing samples, 
our method provides human-understandable explanations in the form of attribute-laden trees. Beyond explanation, we retrained the vision model by calibrating the model on the generated concept hierarchy, {allowing the model to incorporate the refined knowledge of visual attributes}. To access the effectiveness of our approach, we introduce new benchmarks and conduct rigorous evaluations. The results unequivocally demonstrate the plausibility, faithfulness, and stability of our approach compared to existing interpretability techniques.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an LLM-based explainability approach for vision models. It uses LLMs to explain different visual concepts contained in the image in a tree-like manner. 
1) Given a predicted output from a vision model, LLM is used to explain predicted concept and its constituent parts using natural language
2) Text-to-image model is then used to identify the visual representations of the constituent parts. 
3) These visual representations are then passed to the vision model for prediction.

Then steps 1 - 3 repeat.

This type of recursive procedure helps to explain visual concepts in a hierarchical tree-like structure.
The authors also propose to prune infrequent nodes and expand the tree based on LLM prompting. In addition to that the paper also proposes to retrain the model based on the refined explanation trees to improve the model’s interpretability. Plausibility, faithfulness, and stability are the metrics used to evaluate the explanations against baseline approaches.

### Strengths
The paper has a number of interesting contributions:

1) It uses a combination of a variety of models such as vision, text-to-vision and LLMs to generate tree-like explanations for the vision models.
2) Semi-automatically curates annotated datasets for CIFAR10, CIFAR100 and Imagnet.

3) Shows that the explainability-guided regularizer can help with both model explainability and accuracy.

### Weaknesses
1) The paper seems a bit crowded with different contributions that do not read coherently.
2) Overall it is known that all models( text2image, vision and LLMs) have prediction errors. In this case it will lead to error propagation in the tree of the explanations which can result in error amplification. It would be good to study the impact of the erroneous predictions on the explanation tree. E.g. how much the errors from LLM model get propagated down to text to image and vision models.
3) The abstract seems a bit too crowded and could be refined and simplified. For example the following sentence: `This tree is dynamically pruned and grown by querying the LLM using language templates, … `.
It is unclear what `language templates` is meant here.
4) Figure 1 is hard to interpret, the order of the arrows is not very clear. I’d recommend using numeric numbers on the arrows. This will help to better understand the sequence of the actions.
5) It’s unclear why the authors choose `Concepts, Substances, Attributes, and Environments.` attributes.
6) The explanation tree can potentially become very large and there can be different ambiguous cases. It would be good to discuss the problems and solutions related to scale and ambiguity. A discussion section can be helpful.
7) The same concept can be described in different ways through text. It would be interesting to study and discuss those aspects in the paper.
8) The evaluation part is a bit unclear. It would be good to clearly showcase the use cases (examples) where other baseline approaches fail and proposed method is able to handle those challenges better.

### Questions
1) How was the quality of the annotated dataset established ? Since it is semi-automated it can still have a high error rate or there might be many ambiguous cases. How are the ambiguities handled ?
2) In terms of calibration that leads to accuracy improvement is it total accuracy or class based accuracy ? Overall accuracy might be high but subgroups might perform poor.

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
This paper aims to provide structured and human-understandable explanations for vision models and introduces a new and challenging task of generating visual explanatory tree. 
This work collects data used for the explainability for the existing dataset ImageNet and CIFAR complementing the lack of hierarchy annotation. 
The approach leverages LLM and text-to-image API as a bridge between language and vision domains. 
This paper also introduces new benchmarks and metrics for assessing the quality of predicted tree-structured explanations.

### Strengths
Compared to previous explainability approaches, by leveraging the strengths of LLM, this method can construct abundant parsing tree used for the explanation of the visual models. 
The building approach of the new dataset can automatically collect hierarchical annotations is significant.

### Weaknesses
As claimed as a work to generate human-understanable explanable parsing tree, this paper should include human evaluators results of assessing whether the generated results are reasonable. Without human judgment, these outputs cannot be properly evaluated.

### Questions
How does this model perform on out-of-domain categories? Can it still produce interpretable results if the category is not within ImageNet?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
