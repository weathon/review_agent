# GQA-Q2Q: A Large-scale Dataset for Resolving Entity Ambiguity in Visual Question-Answering via Clarifying Subquestion

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Vision-Language Models (VLMs) have achieved remarkable results on various visual question-answering (VQA) benchmarks. However, their performance is significantly impacted by ambiguous questions in which the target entity in the image is not clearly identified. To address and evaluate this issue, it is essential to create a dedicated benchmark dataset that aligns ambiguous questions with a clarifying subquestion. However, constructing a large, high-quality benchmark dataset is costly, particularly when it relies on expert annotations. To efficiently construct such a dataset at scale, this paper presents a hybrid human-machine pipeline. This pipeline begins by generating a small initial set of subquestions using rule-based templates, which are then refined through human annotation. This initial annotated set serves as the foundation for training a subquestion generator and a validator, and the generator and the validator together allow automated construction of a large-scale dataset. As a result, this paper presents a new large-scale dataset, GQA-Q2Q, designed to disambiguate unclear entities in questions by providing clarifying subquestions. Furthermore, a VQA framework is introduced which utilizes the clarifying subquestions to resolve ambiguity before producing a final answer. The experimental results demonstrate that this approach significantly enhances VQA performance, validating the effectiveness of the proposed dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GQA-Q2Q, a 135K-scale dataset of yes/no clarifying sub-questions designed to resolve entity-level referential ambiguity in the GQA VQA benchmark. It proposes a human-machine pipeline for scalable, high-quality data construction and validates utility via a multi-stage VQA framework that detects ambiguity, generates sub-questions, and conditions final answers on sub-answers. The work is timely, technically solid, and fills a clear gap in VQA ambiguity handling. While not revolutionary in method, the dataset scale, quality control, and downstream gains make it a strong candidate for acceptance.

### Strengths
1.	The paper is well-structured and easy to follow.
2.	The paper introduces GQA-Q2Q, the first large-scale dataset specifically designed to resolve entity-level referential ambiguity in visual question answering.
3.	The paper proposes a modular, interpretable VQA framework with explicit ambiguity resolution.

### Weaknesses
1.	Reliance on GQA Scene Graphs Limits Generalization: The sub-questions construction pipeline on more VQA dataset need to be further disscused.
2.	The authors contend that "the construction of a large-scale, high-quality dataset of clarifying sub-questions is essential for training and evaluating VQA models capable of handling ambiguous entities" (Lines 52–53), which is precisely the core motivation behind building this dataset. However, this claim would be strengthened by empirical experiments demonstrating that training on existing ambiguous datasets leads to suboptimal performance, and evaluation on such datasets may yield misleading results—for instance, models with clearly divergent capabilities exhibiting similar performance scores due to ambiguous entities.
3.	Is the proposed VQA framework designed to work specifically with the GQA dataset? Given that ambiguous entities exist across many existing datasets, if the framework's applicability is limited solely to GQA, it would significantly diminish the generalizability and impact of the proposed method.

### Questions
Please see the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the critical issue of entity ambiguity in Visual Question Answering (VQA), where a question refers to an entity with multiple instances in an image, causing models to fail. 
To tackle this, the authors introduce GQA-Q2Q, the first large-scale dataset containing 135,846 clarifying sub-questions designed specifically to disambiguate these entities. 
The dataset was constructed using an innovative human-machine collaborative pipeline, which starts with a small, human-verified seed set to train a sub-question generator and validator for scalable, automated data creation.
The paper demonstrates the dataset's value through a novel VQA framework that leverages these sub-questions, showing significant accuracy improvements across multiple VQA models and proving its effectiveness over existing methods that do not explicitly resolve entity ambiguity.

### Strengths
1.  The paper introduces GQA-Q2Q, the first large-scale benchmark dataset specifically designed to resolve entity ambiguity in VQA through the generation of clarifying sub-questions. With 135,846 sub questions, this dataset addresses a significant gap in existing resources and provides a valuable asset for training and evaluating models on this specific challenge.
2. The work proposes an efficient human-machine collaborative pipeline to construct the dataset. This hybrid approach effectively balances quality and scale by first creating a high-quality initial set with human oversight and then using it to train a sub-question generator and validator to automate the large-scale annotation process.
3. The paper provides robust empirical evidence that integrating the proposed framework improves the accuracy of multiple strong VQA backbone models (e.g., BLIP-2, InstructBLIP, LLaVA) on ambiguous questions. Furthermore, the experiments demonstrate that the proposed method is complementary to, and can enhance, existing ambiguity resolution techniques like RepARe, highlighting its distinct and valuable contribution to the field.

### Weaknesses
1. The criterion for ambiguity detection is oversimplified. By defining ambiguity solely based on the co-occurrence of multiple entity instances, the approach fails to encompass a broader spectrum of more complex semantic ambiguities, such as those arising from referential expressions, unclear relationships, or attribute specifiers. This definition is explicitly stated in the main text.
2. The method of synthetically augmenting ambiguity, specifically by removing adjectival modifiers, risks introducing a distributional shift. The resulting ambiguities may not be representative of or equivalent to the subtleties and patterns of naturally occurring linguistic ambiguity.
3. The modest accuracy of the sub-question validator (82.81%) is a notable limitation, necessitating a high confidence threshold (τ) to mitigate the inclusion of unidentifiable sub-questions. While the appendix presents a trade-off analysis between precision and pass rate, a more systematic report on the end-to-end impact of this filtering strategy on the final quality and potential biases of the large-scale dataset is warranted.
3. The proposed VQA framework relies on a strong oracle assumption for its 'respondent' module, which is presumed to know the ground-truth target instance. This limits the framework's applicability in real-world, open-ended scenarios where such privileged information is unavailable.

[Minor]
1. The justification for the selected confidence threshold (τ=0.9) is based on dataset construction metrics, but the paper lacks a sensitivity analysis of how this choice impacts downstream VQA performance. An ablation study showing VQA accuracy across different values of τ would provide a more complete picture of the hyperparameter's influence and ensure consistent reporting between dataset quality and task performance.
2. The generalizability of the approach remains unclear, as it has not been evaluated on external VQA benchmarks known to contain naturally occurring ambiguities.

### Questions
Please refer to the Weakness section.

### Soundness
3

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
4

### Summary
This paper targets the problem of ambiguous questions in Visual Question Answering (VQA), where the visual target entity referenced by the question is not clearly specified. To address this, the authors introduce GQA-Q2Q, a new large-scale benchmark dataset designed to pair ambiguous questions with clarifying sub-questions. The dataset is constructed using a hybrid human–machine pipeline: a small human-annotated seed set is expanded automatically via a trained sub-question generator and validator.

### Strengths
1. Ambiguity in VQA is an underexplored yet critical challenge for real-world deployment of vision-language systems. Introducing a dedicated dataset for this issue is valuable for future research.
2. The experiments with many LLMs contribute to the understanding the current LLMs.

### Weaknesses
1. The paper does not clearly define what constitutes an “ambiguous” question. While examples are given, the formal rules or annotation criteria (e.g., whether ambiguity arises from multiple visual entities, vague attributes, or linguistic cues) are not detailed. A lack of precise definition undermines dataset impact.

2. The paper does not quantitatively or qualitatively compare GQA-Q2Q with existing datasets (e.g., GQA, VQAv2, or VizWiz). It would be informative to show the proportion of ambiguous questions already present in existing benchmarks and how GQA-Q2Q extends or complements them.

3. The experiments on current VLMs are not sufficiently rigorous to separate hallucination errors from ambiguity-related failures. Since large models often hallucinate facts even for clear questions, additional analysis is needed to differentiate whether observed errors stem from visual ambiguity or model bias.

4. While the dataset scale is impressive, there is little discussion of sub-question quality, such as linguistic fluency, correctness, or diversity after automatic expansion.

### Questions
NA

### Soundness
1

### Presentation
2

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
This paper presents a large-scale VQA dataset that disambiguates unclear entities through clarifying sub-questions. It also proposes a VQA framework that uses these sub-questions to resolve ambiguity before generating a final answer.

### Strengths
- A large-scale VQA dataset that disambiguates unclear entities is presented.

### Weaknesses
- Ambiguity issues may not always be resolvable. For example, in Figure 1, generating a sub-question to clarify the ambiguous entity is impossible without the ground-truth answer—the question itself cannot distinguish which "happy man" it refers to. If the ground-truth answer is required to remove ambiguity, this approach may has limited utility at inference time.
- Why is fine-tuning a VLM necessary for the sub-question generator and validator? A powerful VLM could perform both tasks with appropriate prompts. Moreover, fine-tuning may cause overfitting when annotated data are limited.
- The experiments lack strong evidence of effectiveness. Table 5 suggests that most performance gains come from RpeARe (Prasad et al., 2024) rather than the proposed method.
- Some terms are not defined, such as GQA-Q2Q. What does it stand for?

### Questions
- If the ground-truth answer is required to remove ambiguity, how will this approach work well at inference time?
- Why is fine-tuning a VLM necessary for the sub-question generator and validator?
- How do the results effectively demonstrate that the proposed method works?

### Soundness
2

### Presentation
2

### Contribution
2
