# Automatic visual concept rankings for large multimodal models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Ensuring the reliability of machine learning models in safety-critical domains such as healthcare requires auditing methods that can uncover model shortcomings. While traditional audits range from costly clinical trials to automatic benchmark evaluations, recent advances in automatic interpretability use AI systems to explain other AI models at scale. We introduce an algorithm for identifying salient visual concepts within large multimodal models (LMMs) and demonstrate that leveraging model internals yields more causally relevant insights than black-box approaches. Applying our method to two medical tasks (skin lesion classification and chest radiograph interpretation), we both uncover verifiable conceptual dependencies of LMMs and identify ways in which automatic concept labels may be misleading, highlighting both the promise of automatic interpretability for auditing and the continued importance of expert-in-the-loop oversight.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Visual Concept Ranking (VCR), an interpretability method designed to explain the output of large multimodal models (LMMs). The core idea is to identify concepts that are causally relevant to the model's output. The method first learns Concept Activation Vectors (CAVs) by probing an LMM's internal activations and then ranks these concepts by calculating the directional derivative of the model's output log-probabilities with respect to these CAVs. The authors demonstrate their method in medical tasks

### Strengths
- The paper's primary strength lies in its principled approach to identifying causal concept influence. By using gradient-based directional derivatives, it provides a more robust measure of concept importance.
- The VCR algorithm is clearly explained in four distinct steps (Fig 1) and appears technically sound. The scalability analysis (Fig 4) is also a useful addition.

### Weaknesses
- Although mentioned in the article, the key "shortcut" finding (Fig. 9) was not identified by the algorithm itself, but required manual human inspection of the surfaced images. This seems to severely contradict the author's claim of "automatic interpretability."
- Since there are so many existing methodologies related to concept bottleneck, it seems that there should be more comparisons with existing methodologies.

### Questions
- It would be helpful to compare your research with Kim et al [1]. This seems essential, as it's similar to a prior study that used llm and clip to automatically generate concepts.
- It would be better if we could see the quality evaluated directly by experts.


[1] Kim, Injae, et al. "Concept bottleneck with visual concept filtering for explainable medical image classification." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes VCR, a method for automatically interpreting large multimodal models by identifying which visual concepts actually drive their predictions. It uses a vision-language model to label image concepts, maps those to the model's internal activations, and measures how changes in each concept affect the model's output. The authors tested it on medical image datasets to demonstrate its effectiveness, though it still depends on the quality of automated labelling and requires human oversight for final interpretation.

### Strengths
1) The work presents a gradient-based concept-activation analysis for LMMs, extending LG-CAV. 
2) It proposes a label-free interpretability pipeline using OpenCLIP for automated concept generation, improving scalability for large concept sets. 
3) The work provides a scalable, generalisable framework for concept-level interpretability in multimodal models.
4) The method is mostly rigorous and reproducible. 
5) The paper is well-written and logically structured; it is reasonably accessible. 
6) The testing demonstrates practical value on real-world medical datasets, offering insight into model reasoning and shortcut behaviours in safety-critical domains.

### Weaknesses
1) Lack of stronger baselines: While the work mentions using methods such as MA-MONET, it is unclear exactly what that entails. Second, the work explicitly states that it extends Language-Guided CAVs (LG-CAV); therefore, it would be more convincing to include LG-CAV as a baseline as well.

2) Lack of understanding of the model size effect: While the work compares 3B and 4B models, we do not know how well the model will work with larger models. The comparison between the 3B and 4B OpenFlamingo models provides minimal insight into scaling behaviour. It remains unclear how the method performs on larger or different LLMs.

3) Human-in-the-loop: While the authors acknowledge the value of expert oversight to mitigate labelling errors, the paper lacks specifics on when and how human review would be incorporated, what criteria experts would apply, or how their input might quantitatively improve results. Clearer workflow definitions and evaluation of inter-rater reliability would strengthen this argument.

4) Lack of investigation of components: The framework relies exclusively on OpenCLIP for automatic concept labelling, but the authors do not explain why it was selected or whether they tested other vision-language models.

5) Causal inference is claimed but not fully established: This phrase kind of weakens the claim: "it's likelihood of calling a radiograph abnormal, which is another classic example of a 'shortcut." The statement implicitly acknowledges that the findings may reflect internal correlations rather than true causal reasoning. The method remains valuable for diagnostic interpretability, but causal claims should be moderated/toned down.

- Minor comment: There is no need to define LMMs multiple times.

### Questions
1) Why do you think the t-test is appropriate for the tests you have done? Could you just make the choice briefly in the paper/appendix?
2) Why were the significance tests not discussed in the paper? 
3) Are there any assumptions being made about the correlation of the concepts? If so, please list these alternatives and briefly discuss their implications.
4) What were the results like when tested with < 500 concepts, e.g., 20, 30, 50, 100, etc? Was there a specific reason for starting at 500? Please include the justification in the paper. 
5) What were the specific reasons for just testing 3B and 4B models? Are there plans to test with other LLMs?
6) Could you elaborate exactly how human oversight would solve the interpretability problem you are trying to address in the paper?
7) The work uses OpenCLIP. Did the authors explore other options? Why/why not?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Visual Concept Ranking (VCR), an algorithm for auditing large multimodal models by identifying which visual concepts causally influence their outputs. The method learns concept activation vectors (CAVs) by mapping LMM activations to concept scores from an external VLM (like OpenCLIP), operating without expert labels. VCR then ranks these concepts based on the gradient of the LMM's log-probability output with respect to each CAV. It is tested on OpenFlamingo-3B-Instruct and OpenFlamingo-4B on two medical datasets, with better performance than correlation-based concept selection and R^2-based concapt ranking

### Strengths
- The paper provides a critical extension of concept-based interpretability to LMMs
- VCR is "automatic" and does not require expensive, expert-annotated concept datasets
- The visualizations are clear and informative, making the audience understand the concepts quickly

### Weaknesses
- One major shortcoming is that the interpretability audit is only as reliable as the concept labels provided by the external VLM. The VLMs might have spurious correlation (like the purple ink marking example in line418-431), implicit bias, or lack the nuance for specialized domains. It's unclear how to mitigate this potential risk, especially given the main application is in safety-critical areas.
- The method relies on a predefined set of textual concepts and images. It's unclear how to select the set of text and images, and the effect of size and domain relevance.
- The title ("Automatic Visual Concept Rankings for Large Multimodal Models") suggests a general-purpose method, but all experiments are confined to the medical domain. While the authors suggest it could be applied to general data using vocabularies like Google's Trillion Word Corpus, no such experiments are provided. It is unclear how well this method performs on more abstract or general-domain tasks without this validation.

### Questions
- Could the authors elaborate on the novelty of VCR compared to LG-CAV? The CAV-learning pipeline seems to be a direct application of LG-CAV (without the three additional modules). 
- What is the exact definition of activation in step 2 (l118-126)?
- As general VLM might lack the nuance for specialized domains, how would VCR's findings change if using a VLM trained in medical data?
- The experiments would be significantly enhanced if more models (such as llava or qwen-vl) can be included.
- Could the authors add visual comparison of activating images of concepts for VCR and baselines?

### Soundness
2

### Presentation
3

### Contribution
2
