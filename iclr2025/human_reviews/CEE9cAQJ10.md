## Human Reviewer 1

### Summary
This paper proposes the Graph-based Synthetic Data Pipeline (GSDP), a scalable and cost-effective framework for synthesizing high-quality data by leveraging knowledge graphs to expand data and reveal implicit relationships. GSDP, shown to generate synthesis quality comparable to GPT-4-0613 at a fraction of the cost, achieves strong performance in mathematical reasoning tasks, with the GSDP-7B model reaching 37.7% accuracy on MATH and 78.4% on GSM8K.

### Strengths
1. strong experimental performance
2. interesting research problem

### Weaknesses
1. The motivation is not clear enough, authors point out the limitations of existing method, including limited scalability, high cost, and similarity to seed data. However, the similarity to seed data remains questionable and lack a quantitative investigation. Moreover, why the graph-based synthetic method can solve these limitations is not very clear in the introduction section.
2. In the KP extraction process, did authors check the quality of the generated KPs and the impact of each filter and clustering operation to the quality of KPs.
3. Would be nice to conduct study to present the diversity of the generated dataset is better than other synthetic methods.
4. The baseline results based on Qwen and Llama-3 models are missing, would be nice to present these results.
5. Would be nice to conduct case study to show how the KP graph works and show the superiority compared to existing methods.

### Questions
please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper presents a novel approach for generating synthetic data related to mathematics and mathematical reasoning using a graph-based synthetic data pipeline (GSDP). It automatically extracts knowledge points from seed data to create a knowledge point relationship graph. Utilizing the MATH training set of 7,500 problems and answers as seeds, the GSDP-MATH dataset expands to over 1.91 million pairs of math problems and answers. The authors report achieving accuracies of 37.7% on the MATH dataset and 78.4% on GSM8K when tested with several 7B LLM models. However, the dataset and models are not available for verification.

### Strengths
The idea of extracting knowledge points from seed data, forming a graph to explore their relationships, and then generating training data from these compressed concepts is intriguing. It aligns well with the principles of autoencoders. I believe this is probably a good angle for presenting your approach.

### Weaknesses
While the idea is plausible, it requires validation. Overall, I'm uncertain how this approach compares to the recent work on Critical Plan Step Learning, which appears to demonstrate greater generalization capabilities and performs better on cross-domain tasks. In principle, once you generate 1.91 million training samples, you may lose some generalization power. Additionally, the paper contains several typos.

### Questions
In principle, does generating 1.91 million data samples result in a loss of generalization power?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
Summary:

Training data comes at a premium when training generative language models to solve math (word) problems.  Semi-synthetic instance augmentation is commonly used in such cases.  This submission proposes math problem instance augmentation using "knowledge points".  I wasn't familiar with this term, but, going by the examples given, a "knowledge point" (KP) looks like a glossary entry of a mathematical term or concept like "Pythagorus theorem" or "completing the square".  These KPs can be connected by edges, based on whether/how they are related.  The proposed method samples small compact subgraphs of this KP-graph, and submits these to an LLM again to generate math problems that requires familiarity and expertise over these KPs.

### Strengths
Strengths:

* Identifies data scarcity problem in training LLMs to solve math (word) problems (but this is not unknown).
* Proposes selection of interconnected knowledge points as a way to synthesize high-quality, diverse math problems.
* Demonstrates that this form of data synthesis can be used to boost the performance of smaller language models.

### Weaknesses
Weaknesses:

* As in an increasing number of LLM-based papers, the "innovation" consists of creating a workflow and taking each task to an LLM with a suitably designed prompt.  E.g., a prompt is proposed for knowledge point (KP) extraction from a (problem, solution) pair.  There is very lightweight processing (based on cooccurrence) for graph formation among the KPs and the selection of small compact subgraphs from the KP-graph.  Then there is a prompt that sends this KP subset into an LLM again, asking it to generate a math problem involving those KPs.  Since LLMs are now basically wish-fulfillment machines, such LLM-based papers are incomparable to earlier NLP papers published at such venues.
* Perhaps as a partial consequence, the style of writing is quite foreign to students of pre-LLM ML and AI communities.  "Chain of thought" is itself a good example of how not to describe a "data structure" to a computer scientist, "knowledge point" perpetuates that style.
* We are not necessarily playing a number game here, but [https://paperswithcode.com/sota/math-word-problem-solving-on-math] lists the best MATH performance as 88.1 using a 72B LLM and 87.92 using GPT4-turbo.  In comparison the submission claims GPT-4-0613 is at 42.5 and gets it to 37.7 using a 7B LLM.  [https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k] lists the best GSM8K performance as 96.7 using a 72B LLM and 96.4 using a 7B LLM.  In comparison the submission shows GPT-4-0613 at 92 and gets it to 76.5 using a 8B model.  I have no experience in how to give credit for lower performance using smaller models.

### Questions
Comments and suggestions:

The title can be greatly improved. "Data" is too generic. If you are focusing on math word problems, say so in  the title itself.

Fig 2 caption should be much longer and explain the point of the diagrams without depending on distant text.

"Knowledge point" is too non-standard and nebulous.  Please be specific and precise right the first time you use this term and define it in terms of bits and bytes.

Overall, this may be better suited to a NLP or AI conference than an ML conference.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3