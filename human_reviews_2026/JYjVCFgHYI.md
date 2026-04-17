# ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant

- Decision: Reject
- Scores: 4, 6, 2, 6, 6

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated remarkable performance across a wide range of multimodal tasks. Recent advances in personalized MLLMs enable effective capture of user-specific concepts, supporting both recognition of personalized concepts and contextual captioning. However, humans typically explore and reason over relations among objects and individuals, transcending surface-level information to achieve more personalized and contextual understanding. To this end, existing methods may face three main limitations: (1) Their training data lacks multi-object sets in which relations among objects are learnable, (2) Existing models often neglect the connections between different personalized concepts, thereby failing to perform reasoning over them, (3) Their experiments mainly focus on a single personalized concept, where evaluations are limited to recognition and captioning tasks. To address the limitations, (i) We present a new dataset named ReGraP, consisting of 120 sets of personalized knowledge. Each set includes images, Knowledge Graphs (KGs), and Chain-of-Thought Question-Answering (CoT QA) pairs derived from the KGs, enabling more structured and sophisticated reasoning pathways. (ii) We propose Reasoning enabled Graph-based Personalized Large Language and Vision Assistant (ReGraP-LLaVA), an MLLM trained with the corresponding KGs and CoT QA pairs, where soft and/or hard graph prompting methods are designed to align KGs within the model’s semantic space. (iii) We establish the ReGraP Benchmark, which contains diverse task types: Multiple-Choice, Fill-in-the-blank, True/False, and Descriptive questions in both open- and closed-ended settings. The proposed benchmark is designed to evaluate the relational reasoning and knowledge-connection capability of personalized MLLMs. We conduct experiments on the proposed ReGraP-LLaVA and other competitive MLLMs. Results show that the proposed model not only learns personalized knowledge but also performs relational reasoning in responses, achieving the SoTA performance compared with the competitive methods. All the codes and datasets are released at: https://anonymous.4open.science/r/ReGraP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the solutions to the three main limitations of personalized MLLMs.
1) They present ReGrap dataset and the data generation pipeline for personalized MLLMs (Multimodal Large Language Models), including knowledge graph construction and CoT QA pairs generation from the constructed knowledge graphs.
2) They propose a new MLLM, ReGraP-LLaVA, which uses soft and/or hard prompts from knowledge graphs and CoT question-answer pairs during training.
3) They established the ReGraP benchmark, including Multiple-Choice, Fill-in-the-blank, True/False, and Descriptive questions, covering both open-ended and closed-ended settings.

### Strengths
1) This paper covers three main parts of personalized MLLMs development: datasets, models, experiments. They give detailed and comprehensive research content.
2) The formulas and figures used in the explanation parts are clear and concise.

### Weaknesses
1) Although the part “REGRAP-LLAVA: TRAINING FRAMEWORK” specifically introduces the “soft prompts” and “hard prompts”, the relationship between them and Figure 3 is not clearly mentioned. It is better to give concrete explanation of the example on the figure.
2) It is better to include the performance and cost optimization function in the TRAINING FRAMEWORK part to show a more complete optimization process.
3) There are only 2 other comparison datasets in the experimental part. More datasets should be provided to demonstrate the advantages of the dataset proposed and secure a high-level concept of the experiment. 
4) The ablation study results just show the performance of hard, soft and combined prompts, but ablation studies for number of objects and length of CoT QA pairs are also important. I suggest including the other 2 additional ablation studies in the main paper with a more detailed discussion.

### Questions
Please address the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the challenge of personalized MLLMs, where the response of the MLLM depends on the identity and relations within the image. The paper first builds a data generation pipeline to construct detailed knowledge graphs and CoT training data, using GPT4o. Then both the knowledge graphs and CoT is used for training the MLLM. During training, a graph neural network is trained to embed the the knowledge graph, as well as a hard prompting strategy to learn tokens that explicitly represent the graphs. Empirical results demonstrate the effectiveness of the method compared to both generalist and specialized baselines.

### Strengths
* The paper provide a complete framework for training a personalized MLLM, encompassing data generation, model architecture, and evaluation benchmarks. This framework enables the research community to easily adopt, extend, and improve any component of the pipeline.
* The method combine both embedding and token based approach to encode the knowledge graph information to the MLLM. Ablations demonstrate the effectiveness of this combination.

### Weaknesses
* It seems that CoT significantly boosts the performance. For the baselines, it would be more fair to evaluate them on CoT prompting setting instead of a prompting setting.
* The paper lacks comprehensive analysis on the different advantage and information provided by soft-prompting and hard prompting approach. Analysis on training time, inference time, and the evaluation accuracy on different types of problems would be needed to support the analysis.

### Questions
* How does the model perform in standard benchmark. Is there obvious catastrophic forgetting?
* Since the method is model independent, would finetuning on top of reasoning model like R1-Onevision or stronger model like Qwen2.5vl further improve the performance?

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
5

### Summary
This paper focuses on personalizing MLLMs and introduces a new dataset, ReGraP, which is constructed through a data generation pipeline that incorporates knowledge-graph–based chain-of-thought (CoT) question-answer pairs. The proposed benchmark includes multiple-choice, fill-in-the-blank, true/false, and descriptive question formats, covering both open-ended and closed-ended tasks. Experimental results on ReGraP demonstrate the effectiveness of the proposed personalization approach, ReGraP-LLaVA.

### Strengths
This paper identifies an evaluation gap in relational reasoning for personalized MLLM-based understanding. To address this limitation, the authors incorporate both knowledge graphs (KGs) and chain-of-thought (CoT) reasoning into multi-object personalized MLLMs. They further propose a data generation pipeline to construct a new benchmark dataset, ReGraP, supporting the evaluation of such personalized relational reasoning abilities.

### Weaknesses
1. **Limited scope and representativeness of the proposed dataset**. The diversity of concepts, relations, and scenarios covered in ReGraP remains narrow. Most scenes revolve around anime characters and personal items, resulting in a limited semantic scope. While such content may be common in personalization research, the benchmark lacks a clear definition or demonstration of “personalization.” In addition, the relational types are shallow. Most attribute or role associations, such as “who is the leader of the band”, are simple and do not capture more advanced or compositional reasoning. Consequently, the benchmark appears more like a combination of cosplay-style data and basic graph reasoning, with limited applicability to broader personalized MLLM tasks.

2. **Unclear motivation for integrating KG and CoT into personalization**. The rationale behind combining knowledge graphs and chain-of-thought prompting for personalization is insufficiently explained. Moreover, the experimental section lacks ablation studies to disentangle the contributions of each component. Notably, the method combining hard and soft graph prompting performs worse than a simpler design on the multiple-choice task, which weakens the claim that the proposed integration is beneficial.

3. **Benchmark construction suffers from a major data reliability issue**. The knowledge graphs and CoT explanations are automatically generated using GPT-4o without any grounding or verification procedure. Given that LLM hallucinations are well-documented, especially in structured knowledge extraction, the lack of human sanity checks or filtering introduces serious reliability concerns. A benchmark should minimize annotation noise to ensure validity of evaluation.

4. **Unfair experimental comparison**. The baselines in Table 2 do not incorporate knowledge graph information, whereas the proposed method explicitly leverages it. This results in a task setup inherently favorable to the authors’ method, offering limited insight into whether the approach provides a general improvement rather than exploiting task-specific priors. A more balanced comparison is needed to justify the claimed effectiveness.

5. **Formatting concerns**. The paper contains formatting issues. For example, table captions should appear above tables rather than below them.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel task to perform relational reasoning over personalized concepts using Multimodal Large Language Models (MLLMs). A framework based on soft and/or hard graph prompting is proposed to enhance the relational reasoning capabilities of MLLMs by leveraging knowledge graphs and Chain-of-Thought Question Answering (CoTQA) pairs. The paper also presents a new dataset, ReGraP, which contains images, knowledge graphs, and CoTQA pairs for training, and a benchmark to evaluate the relational reasoning between personalized concepts. Experimental results demonstrate that the proposed ReGraP-LLaVA model effectively learns personalized knowledge and performs relational reasoning, achieving superior performance compared to competitive methods.

### Strengths
1. It's a crucial problem to enable MLLMs to perform relational reasoning over multiple personalized concepts.
2. The proposed framework based on soft and/or hard graph prompting is well-designed to enhance the relational reasoning capabilities of MLLMs.
3. The paper develops a data generation pipeline for relational question answering synthesis, and also introduces a new dataset and benchmark named ReGraP, which are valuable resources for future research in this area.
4. The paper is well-written and easy to follow, with clear explanations of the proposed methods and experimental results.

### Weaknesses
1. The paper extends the idea of soft/hard prompting beyond previous works (e.g., Yo'LLaVA) by integrating reasoning over knowledge graphs. However, since prompting-based personalization has been explored before, the novelty mainly lies in using structured graph representations and CoT QA data, which could be better emphasized.
2. The paper lacks comparison with several related personalization methods such as RAP-LLaVA, UniCTokens and RePIC. Including or discussing these baselines would strengthen the experimental evaluation.
3. It's unclear how to select between soft and hard graph prompting in different scenarios. The paper should provide more insights or guidelines on when to use each approach for optimal performance.
4. The model needs to be trained on each concepts set, which may limit its applicability in real-world scenarios where personalized concepts may vary widely. The authors should discuss potential strategies to improve the model's generalization to unseen concepts without requiring retraining.
5. Some details should be clarified. For instance, how subgraphs are selected during inference (automatically or predefined). The specific GNN architecture used for graph embedding is also not detailed.

### Questions
1. Have the authors compared the performance of directly converting knowledge graphs into textual descriptions (without GNN encoding) versus using soft/hard graph prompting?
2. See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ReGraP-LLaVA, a multimodal large language model designed to perform relational reasoning over personalized concepts. The authors present a novel dataset (ReGraP) containing 120 sets of personalized knowledge, each with images, knowledge graphs, and Chain-of-Thought question-answering pairs. The method uses both soft prompting (via GNN-based graph embeddings) and hard prompting (via new reasoning tokens) to align knowledge graphs with the model's semantic space. Experiments demonstrate that ReGraP-LLaVA outperforms existing personalized MLLMs on tasks requiring both recognition and multi-step relational reasoning.

### Strengths
1. The motivation for this work is clear and compelling. The authors correctly identify that existing personalized MLLMs focus primarily on concept recognition and captioning, while neglecting the relational knowledge and reasoning capabilities that humans naturally employ when understanding personalized contexts.

2. The novelty of the approach is strong. To my knowledge, this is the first work to explicitly construct knowledge graphs for personalized concepts and use them to train MLLMs with relational reasoning capabilities. The dual approach of soft and hard graph prompting provides interesting alternatives for graph-MLLM alignment.

3. The experimental results are strong across multiple task types. ReGraP-LLaVA achieves substantial improvements over both prompt-based and finetuning-based baselines, with particularly impressive gains on difficult reasoning tasks (5.3% over the best finetuning baseline and 8.8% over the best prompt baseline on weighted average).

4. The paper includes useful ablation studies comparing soft vs hard prompting methods, and provides qualitative examples with attention visualizations that demonstrate the model's reasoning process. The human evaluation in Section B adds additional credibility to the results.

### Weaknesses
1. The evaluation setup and dataset descriptions are somewhat unclear throughout the paper. While the datasets are described in the main text (Section 5), the tables themselves do not clearly indicate which dataset is being evaluated. For instance, Table 2 does not specify that it evaluates on the ReGraP dataset, while Table 3 evaluates on Yo'LLaVA and MyVLM datasets with different tasks. The authors should add explicit dataset identifiers to table captions and within the tables themselves to improve clarity.

2. Given the strong motivation that prompting alone may not adequately capture personal and relational information, I would expect a more thorough exploration of advanced prompting baselines. The authors make an excellent point that lengthening prompts provide limited context for complex relational reasoning. However, the prompt-based baselines appear to use relatively straightforward description-based prompting. Recent work in visual compositionality has developed more sophisticated prompting approaches that could serve as stronger baselines. For example, Compositional Chain-of-Thought Prompting for Multimodal Large Language Models by Mitra et al. demonstrates how structured CoT prompting can improve compositional reasoning. Similarly, Davidsonian Scene Graph Prompting by Cho et al. shows how scene graph-structured prompts can enhance spatial and relational understanding. Including these or similar baselines would strengthen the evaluation and better demonstrate the necessity of the proposed training-based approach over advanced prompting techniques.

3. The paper would benefit from situating this work within the broader literature on visuolinguistic compositionality and scene graph reasoning. While the authors cite some related work on graphs with MLLMs, there is a rich body of work on compositional visual reasoning, scene graphs for visual understanding, and compositional generalization that seems highly relevant. For instance, work on using scene graphs for visual relationship detection, compositional visual question answering, and structured visual reasoning could provide useful context and potentially inspire additional baseline comparisons or evaluation protocols.

### Questions
Most of my feedback is constructive already, describing the perceived weakness and how to resolve it. So please refer to the Weaknesses section.


Can you clarify all the ways in which CoT QA samples are used in your method? From my reading, they appear to serve multiple purposes: (a) as training supervision where questions provide instructions and CoT answers provide target responses, (b) as a "hard-prompt formulation" of knowledge graphs, and (c) as part of your evaluation benchmark. It would be helpful to explicitly enumerate these uses and clarify whether they are also integrated into the graph representation itself in any way beyond being derived from the graph routes.

### Soundness
3

### Presentation
3

### Contribution
3
