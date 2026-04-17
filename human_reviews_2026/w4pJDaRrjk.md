# MegaScience: Pushing the Frontiers of Open Post-Training Datasets for Science Reasoning

- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the open-source community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present **TextbookReasoning**, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce **MegaScience**, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MegaScience, which significantly outperform the corresponding official instruct models in average performance (e.g., +3.24\% for Qwen3-30B-A3B). In addition, **MegaScience exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning**. We release our data curation pipeline, evaluation system, datasets, and nine trained models to the community to advance scientific reasoning research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces two new datasets designed to advance scientific reasoning in AI systems:

1. TEXTBOOKREASONING: A large, open-source, university-level scientific dataset comprising 650K challenging questions and step-by-step solutions. These are derived from over 12,000 scientific textbooks across a wide range of domains, including mathematics, biology, physics, economics, and more.
2. MEGASCIENCE: A comprehensive collection of high-quality, open-source datasets containing over one million data points.


**Key Contributions**

1. The release and open-sourcing of the TEXTBOOKREASONING and MEGASCIENCE datasets.
2. A detailed presentation and open-sourcing of the curation pipeline used to construct both datasets.
3. A thorough empirical evaluation demonstrating the effectiveness of these datasets in enhancing scientific reasoning capabilities of LLMs. The authors show that base models fine-tuned on these datasets  outperform their corresponding instruction-tuned counterparts.

### Strengths
1. **Relevance** The paper tackles the important problem of improving the scientific reasoning on LLM through the creation and open-sourcing of two very valuable resources.
2. **Impact/Significance** The open-sourcing of the new datasets and, more importantly, of the curation pipeline has the potential to spur more rapid progress in LLM-based scientific reasoning
3. **Presentation** Overall the paper is well organized, well written, and easy to read and follow.
4. **Experimental Evaluation** The ablation study clearly shows the importance of key components of the curation pipeline.

### Weaknesses
1. **Incorrect or Exaggerated Claims**
The authors make several incorrect or overstated claims regarding the strength of their empirical evaluation. For instance, in the abstract, they state:

> "Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MEGASCIENCE, which *significantly* outperform the corresponding official instruct models in average performance (e.g., +3.24% for Qwen3-30B-A3B)"

Similarly, in Section 4.2, they write:

> "Table 4 shows that Qwen2.5-7B, all Qwen3 models, and Llama3.1-8B trained on MEGASCIENCE *substantially* outperform their official instruction-tuned counterparts, demonstrating MEGASCIENCE ’s effectiveness in pushing the frontier in science."

These statements are not fully supported by the results presented in Table 4:

- **First**, Table 4 shows that *Qwen2.5-1.5B-instruct* and *Qwen2.5-3B-instruct* outperform their MEGASCIENCE-trained counterparts in overall performance.
- **Second**, *Llama3.1-8B-megascience* is only marginally better than its instruct variant in terms of overall average (+1.6%), and actually performs worse in *specific-avg* and *math-avg* metrics.
- **Finally**, while the Qwen3 MEGASCIENCE models do consistently outperform their instruct counterparts, the margins are relatively modest (+1%, +0.4%, +2%, +2.8%, and +3.2%). These improvements do not substantiate the use of terms like “*significantly*” or “*substantially*” as claimed by the authors.

2. **Heavy Reliance on LLMs**  
In the introduction, the authors critique the heavy reliance on LLMs in key aspects of existing scientific datasets. However, a central component of their own data curation pipeline (the Q-A Pair Refinement module) relies heavily on LLMs. Specifically:

- *DeepSeek-V3* is used to refine Q-A pairs with corresponding source documents to ensure that “questions include all necessary contextual information and answers provide comprehensive explanations with clear reasoning processes.”
- *Llama3.3-70B-Instruct* is employed to identify Q-A pairs lacking reasoning.
- *DeepSeek-V3* is again used to “enrich them with explanations and reformat their answers.”

Given the critical role of the Q-A Pair Refinement module (highlighted by the ablation study showing a 45% performance drop when it is removed), it would be prudent to rigorously evaluate how well these LLMs perform in the refinement tasks. A human evaluation of a small, randomly selected subset of the refined Q-A pairs could provide valuable insights into the quality and reliability of the refinement process.

### Questions
Why didn't you evaluate the performance of LLMs involved in various part of the curation pipeline? It seems that this could be done with a limited human evaluation on a small subset of a few hundred randomly selected instance data points

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the critical gap in open-source, high-quality datasets for scientific reasoning. It proposes high-quality TextbookReasoning dataset which is curated from large-scale scientific textbooks through rigorous data pipeline. Furthermore, it absorb high-quality data from existing open-source dataset to construct a large-scale and high-quality post-training dataset MegaScience. Extensive SFT experiments on MegaScience and TextbookReasoning demonstrates the good quality of the proposed datasets.

### Strengths
1. The writing and structure of this paper are clear and easy to understand.
2. The data curation process is rigorous and effective, which not only obtains high-quality data but also conduct strict deduplication and decontamination policies.
3. The experimental results are strong and convincingly demonstrate the high quality of the proposed datasets.

### Weaknesses
1. My main concern is that, in the refinement (Section 2.3) and solution annotation (Section 3.4) processes, the authors employ DeepSeek-V3 to generate solution trajectories. However, given that DeepSeek-V3 performs only moderately on scientific benchmarks (e.g., 59.1 on GPQA-D), its responses are likely to contain many errors and hallucinations. I think the authors should at least conduct some human verification to estimate the proportion of erroneous responses from DeepSeek-V3 to provide a warning for future users.

### Questions
1. Could you provide specific details on the LLM prompts or criteria used to filter "strictly copyrighted" textbooks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces *TextbookReasoning*, an open-source scientific post-training dataset containing over 650k reasoning questions derived from 12k university-level textbooks, and *MegaScience*, a large-scale mixture of open-source datasets with 1.2 million instances.

The work discussed existing critical challenges in scientific reasoning data, including unreliable benchmark evaluation, less rigorous decontamination, low-quality reference answers, and superficial knowledge distillation. Systematic ablation studies help being clear with effectiveness of data selection. Experiments demonstrate strong performance and training efficiency. The resources are released.

### Strengths
- The paper is clearly written with well-motivated research goals.
- The ablation study on data creation and combination is comprehensive.
- The evaluation covers diverse reasoning-intensive tasks across various domains.
- The data, prompts, and models are fully released.

### Weaknesses
- The paper needs more discussion and comparison with relevant works. For example, OpenThoughts[1] and S1.1[2] have released data and models, but their performance is not compared here. It would be valuable to compare with works that do not use MegaScience or TextbookReasoning.
- The paper focuses on post-training, but current paradigms commonly apply RL as post-training as well. This work lacks discussion of RL approaches for improving general reasoning performance, like General-Reasoner[3]. While it's acceptable to focus on SFT, comparing TextbookReasoning for RL or benchmarking against other RL-based works would be valuable.
- Since refinement quality is important (as shown in this work), the dataset creation assumes access to a strong model for data generation.
- Claims like "web content is now saturated with AI-generated text" should include quantitative evidence and citations.
- Copyright of textbook data might be a concern, but the author explained in the ethical statement.

[1] OpenThoughts: Data Recipes for Reasoning Models

[2] s1: Simple test-time scaling

[3] General-Reasoner: Advancing LLM Reasoning Across All Domains

### Questions
See the weaknesses section.

And I may have missed this—how was the subsample size in Table 1 determined?

### Soundness
3

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
4

### Summary
This paper made two contributions: 
1. It collects scientific textbook PDFs online, extracting questions from the textbooks, and curates a dataset called textbook reasoning with 650k examples of question and answer pairs. 
2. It conducts extensive ablations and proposes a data mixture called megascience that combines existing sources and textbook reasoning for post-training LLMs for scientific tasks. 

The authors show that fine tuning on the proposed mixture can improve the performance on various scientific question answering and reasoning tasks; and it conducted extensive ablation studies to validate their data filtering and mixing strategies for constructing MegaScience.

### Strengths
This paper made two contributions: 
1. It collects scientific textbook PDFs online, extracting questions from the textbooks, and curates a dataset called textbook reasoning with 650k examples of question and answer pairs. 
2. It conducts extensive ablations and proposes a data mixture called megascience that combines existing sources and textbook reasoning for post-training LLMs for scientific tasks. 

The authors show that fine tuning on the proposed mixture can improve the performance on various scientific question answering and reasoning tasks; and it conducted extensive ablation studies to validate their data filtering and mixing strategies for constructing MegaScience.

### Weaknesses
While this paper makes some good contribution, I think there are some limitations: 
1. I don’t think there’s significant novelty in this work – it uses standard methods to collect and clean the collected dataset, and constructs the proper data mixture. 
2. I think the performance improvement is somewhat limited: for example, in table 3 and 4, training on a new million scale corpus only yields 3 absolute point improvements, while one would expect bigger improvements (e.g., see the dataset scaling study in the openthoughts paper https://arxiv.org/abs/2506.04178)

### Questions
- Can you provide detailed stats for the TextbookReasoning dataset – i.e., breakdown of the domains, the average input and output token length.

### Soundness
2

### Presentation
3

### Contribution
2
