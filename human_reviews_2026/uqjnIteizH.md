# Uncovering Grounding IDs: How External Cues Shape Multi-Modal Binding

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large vision–language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as consistent within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism that explains how external cues enhance multimodal binding and offer both interpretability and practical improvements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work introduces Grounding IDs to explain why simple symbolic cues, lines, or partitions added to an image enable large vision-language models to align visual and textual information more accurately and perform structured reasoning. Through representation alignment, activation swapping, and attention analysis, the authors demonstrate that these region-specific identifiers play a crucial role in cross-modal binding and can markedly reduce hallucinations, improving the factual consistency and interpretability of model outputs.

### Strengths
Conceptual novelty: Grounding IDs are introduced for the first time, offering a causally verifiable account of how external visual structures enhance multimodal reasoning and alignment.

Rigorous analysis: A systematic investigation—combining representation alignment, activation intervention, and attention diagnostics—demonstrates the mechanism’s validity.

High practical value: The approach markedly reduces hallucinations while incurring negligible computational overhead and remains broadly applicable and extensible.

Well-designed experiments: The study integrates controlled synthetic data with standard MS-COCO benchmarks, balancing theoretical exploration and real-world validation.

### Weaknesses
1. The writing of the paper is somewhat informal, and key terms such as “symbolic indexing”, “latent identifiers”, and “abstract vectors” are not clearly defined early on. This may reduce the clarity and conceptual coherence, making it more difficult for readers to fully understand the proposed framework.

2. The experiments mainly focus on horizontal lines and symbolic structures, limiting generalization. Exploring other cues, such as color partitions, semantic regions, or dynamic elements, could provide a more comprehensive evaluation.

3. The limited dataset sizes, with around 100 samples for synthetic experiments and 200 for hallucination evaluation, may restrict the statistical reliability and generalizability of the results.

4. The current evaluation focuses on a single model size, leaving open the question of whether Grounding IDs can generalize to smaller (1B–3B) or larger (30B+) LVLMs.

### Questions
Since the grid lines or symbols and the scanning prompt were always presented together, it remains unclear whether Grounding IDs are primarily triggered by the visual cues, the textual instruction, or the combination of both. A 2×2 orthogonal ablation (visual-only, prompt-only, both, neither) would help disentangle these factors and clarify whether they are complementary or redundant. Could the authors provide such control experiments or justify why they were not included?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This method investigates the hypothesis that LVLMs associate "Grounding IDs" with certain partitions of data in different modalities, for example associating the symbol "&" with a certain sub-region of an image. This investigation is carried out with experimental testing of LVLMs in controlled situtations, in particular with images of coloured objects positioned in different regions of the image.

### Strengths
The hypothesis seems interesting and is investigated in a straight-forward and honest fashion. The question, ie what meaning do LVLMs give to abstract symbols which might describe image properties, seems important. The results seem convincing.

### Weaknesses
I found the paper difficult to understand, in particular Sections 3-4, not due to bad writing, but due to lack of precise details. I put specific examples of this below. Furthermore, I found the controlled experiments a bit limited. When the experiments went on to more real-world examples, using Qwen and LLaVA, I did not see where the notion of abstract symbols, which were hypothesised in the paper, intervened. It seems that you just add lines to the image, how does this show that these models actually use language symbols ?

However, I re-state that I found the work honest and to the point. Therefore, I would be willing to change my rating if the two following things were done:
- Significantly clear up and clarify sections 3-4, addressing the questions below
- Explain how the conclusions of sections 3-4 apply to the more real-world example with LlaVa and Qwen, in particular, how do you know that the abstract symbols are really used here (couldn't the model just look at the lines you have drawn ?).

### Questions
In Section 3, I do not understand which LVLM specifically is chosen here, it seems to be never specified. Are you using it pre-trained and frozen ? Is there any fine-tuning ? 

The "baseline" is not explained in Section 3. It is very difficult to trust any results if we do not know what this baseline is, is it more simple or more complicated than the structured input ? Does it contain similar types of data ? Same shapes ? etc... 

Section 4 is difficult to understand, again due to lack of details. I did not understand the process of swapping, what precisely was done. I got confused between swapping objects in an image and the swapping of activations inside the network. Which of these is done ? (one, both). What is "activation patching" ? You say "If we follow the standard setting without intervention ... the accuracy drops from 1.00 to 0.02". At this point I did not know what was being done exactly. If I understand correctly, you ask it to say what is in the row &, for example, but then switch the meaning of the symbols. So, the question is, was there any fine-tuning on the model? If so, then of course since you are forcing the model to associate this position with a symbol, and then when you carry out the inference, you change the meaning, so it will for sure not get it correct. If not, then I do not understand why it would give any meaning to that abstract symbol anyway, so we should not get such a change in accuracy. But since it is difficult to understand exactly what is done 

- p. 3, line 118: "the model is tasked ..." setence repeated twice
- p. , line 122: "each token must be paired with ..." do you mean that the llm must do this ? Not super clear.
- p.3, line 124: what is a "regular expression" ? This may be because I am not as familiar with the domain as others
- p. 3, line 129: what is the "baseline" ? Not clear or stated as far as I can tell.
- p. 4, line 186: "rows are labelled ... to prevent sequential order cues". If there is fine-tuning or any sort of training, then the model can simply associate these symbols with numbers. But it is not clear if the model is fine-tuned or not
- p. 5, line 223: "host context". What is a host context ?
- p. 5, line 217-218: I found the sentences "swap their corresponding patch activations ... the object ... is inserted" quite difficult to follow. What does "the object in row $ of $c'$ is inserted into row ..." mean ? Do you mean that you swap the values of the activations (neurons) of the LVLM at some layer ? If so, which layer ? Before, after attention ? Does it have anything to do with attention, or just the plain activations (neurons) ? This is far too vague for me to be able to correctly understand.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper explores how simple visual cues can influence the spatial and grounding capabilities of large vision–language models (LVLMs). The authors introduce the concept of Grounding IDs—latent identifiers induced by external visual structures that help bind objects to their corresponding symbolic or textual components. 

Through detailed representation and causal analyses, the study shows that these identifiers emerge as robust within-partition alignments in the embedding space and reduce the modality gap between image and text. Moreover, attention visualization and causal interventions reveal that Grounding IDs mediate cross-modal binding and improve grounding precision, effectively reducing hallucinations. Overall, the paper offers both theoretical insights into multimodal interpretability and practical improvements in multimodal robustness.

### Strengths
* The paper provides a novel conceptual contribution—the introduction of Grounding IDs—to explain how external cues can shape multimodal binding mechanisms.

* The authors perform detailed cross-attention and activation-swap experiments, demonstrating how symbolic annotations can directly modulate spatial perception and relational reasoning within LVLMs.

* The experimental methodology is carefully controlled, revealing causal links between visual structure and cross-modal alignment.

* The findings offer new interpretability insights, showing that external symbolic cues can serve as a controllable mechanism to enhance multimodal reasoning and reduce hallucinations.

### Weaknesses
1. The paper formatting deviates from the ICLR submission standard (notably excessive bottom margin spacing). I have encountered this issue in two of five papers I reviewed this year. Please fix it.

2. The motivation emphasizes precise grounding, yet the evaluation is limited to benchmarks such as CHAIR and POPE. Compared with works like VISER, this study lacks tasks involving object counting or visual grounding (e.g., referring expression comprehension, REC). Testing the proposed method on a stronger LVLM (e.g., Qwen-2.5VL) for these grounding benchmarks would make the claims more convincing.

### Questions
1. Are the Grounding IDs transferable across architectures or datasets, or are they model-specific phenomena?

2. Could the authors evaluate their method using a stronger LVLM such as Qwen-2.5VL to demonstrate generalizability?

3. How sensitive are the observed effects of Grounding IDs to the type and complexity of external cues (e.g., grid partitioning vs. text annotations)?

If these issues can be addressed or resolved, I would be willing to raise my overall score, as the paper presents an interesting and potentially impactful contribution to multimodal interpretability.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the relationship between external cues and multimodal binding in LVLMs. In particular, the paper shows that Grounding IDs (symbolic multimodal identifiers induced by external cues) can potentially strengthen attention between related components and improve multimodal grounding. Authors proposes that Grounding IDs are a key symbolic mechanism explaining how external cues enhance multimodal binding. Experiments and ablation studies are provided.

### Strengths
The strengths of the paper comes from the attempt to investigate how external cues (e.g., symbolic identifiers in this paper) can shape multimodal binding. The RQ of interest, how/why do external cues improve reasoning in LVLMs, is important and timely. The motivation from and connection to previous works (e.g., Binding IDs by Feng and Steinhardt, 2023, extension in LVLMs by Saravanan et al., 2025, VISER by Izadi et al., 2025) are presented relatively clearly. The central hypothesis (that structured symbols, either visually or textually, lead LVLMs to internally generate identifiers that bind objects to their designated partitions) is reasonable and aligns with takeaways of previous works.

### Weaknesses
### Writing

The paper can benefit from further improvements on the writing itself and clarifying on the interpretation of presented results (detailed in "Questions" section).

### Novelty and Significance

There is a concern on the novelty and significance of the proposed Ground IDs (detailed in "Questions" section).

### Questions
Questions and comments:

1. The paper can get difficult to parse from time to time. For instance, the interpretation of the figures are not easy to follow. I am not entirely convinced about authors' interpretation (of Fig. 5) that the model shifts toward predicting the intervened grounded object in the later layers (20--27), now that the similar uptick trends are observed both in baseline and structured (in Fig. 3a). The Fig. 6 can also benefit from additional clarifications (e.g., how each entry is calculated precisely), beyond lines 272--279.

1. The sentence "The model is tasked with generating ..." was repeated twice (lines 119--120).

1. The concern on novelty and significance. Overall, the results confirm previous claims (e.g., Schrodi et al., 2024, line 138), by extending scaffolds from horizontal lines to symbols. While I understand that the (color-coded) symbols may not be explicitly addressed by previous works (e.g., Saravanan et al., 2025), the effectiveness of textual references/cues are well-established. The RQ of interest itself (line 94, why do external cues improve reasoning in LVLMs) is important and timely, but the claimed benefits of Grounding IDs do not seem to be significant advancement over existing literature.

### Soundness
3

### Presentation
2

### Contribution
2
