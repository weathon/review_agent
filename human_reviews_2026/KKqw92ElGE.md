# Explicit Column Relationship-Based Diffusion Model for High-Quality Synthetic Tabular Data Generation

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 0, 6, 0

## Abstract
Tabular data plays a vital role in critical applications such as healthcare, finance, and education. Its effective utilization in data-driven models is frequently hindered by data scarcity and privacy concerns. In response, synthetic tabular data generation has emerged as a powerful solution that provides privacy-preserving data mirroring real-world distributions. However, many existing generative models still struggle to preserve the complex column relationships within tabular data. Additionally, they often fail to account for the real-world constraints that are essential for ensuring the authenticity and practical usability of the generated data. In this paper, we propose ECR-DM, the Explicit Column Relationship-Based Diffusion Model for synthetic tabular data generation. In the forward diffusion process, we introduce the Noise Perturbation Mechanism, which enables the model to learn column distributions in a fine-grained manner. In the reverse diffusion process, we incorporate Constraint-Guided Recovery, which guides the model to recover inter-column dependencies and restore the true data distribution. NPM helps the diffusion model capture the detailed column-wise characteristics of the data, while CGR ensures the preservation of inter-column relationships and the high-quality synthetic tabular data generation. We validate the effectiveness of our approach through extensive experiments on six tabular data benchmarks. Our model outperforms state-of-the-art methods across seven evaluation metrics, particularly in downstream tasks. Code is available at https://anonymous.4open.science/r/ECR-DM-0C72.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a diffusion-based synthetic tabular data generative model. The model has a few components added to improve the diffusion model, such as adding column-specific noise and providing data constraints.

### Strengths
- The motivation of the work is clear - how to make synthetic data follow real-world constraints. 
- Projecting tabular data into a semantic space and aligning it with natural language representation of real-world data constraints is a novel idea

### Weaknesses
- This paper has several add-ons to the diffusion model. Not all components actually contributed to improving the performance.
- Transforming categorical columns or numerical columns into a unified semantic space is proposed in several previous works (e.g., https://arxiv.org/abs/2205.09328). This work fails to correctly cite.
- It is unclear how the columns’ interdependency can be well reflected by adding individual noise to each column. How does the learning distribution of each column effectively (with individual noise) lead to preserving the inter-column dependency? There is a logical gap. This is not clarified in the manuscript. Adding separate noises seems to be very empirical, rather than based on rationales.
- How to set the right noise for each column? Is it dependent on the column’s statistics? 
- It is unclear how the real-world constraint C in natural language’s representation space is well aligned with the semantic representation of tabular data. 
- It is unclear how to generate high-quality real-world constraints. What if such constraints or rules are not obviously visible and the rules are latent?

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes an approach to synthetic tabular data generation. Explicit Column Relationship-Based Diffusion Model (ECR-DM) introduces in the forward direction  a noise perturbation mechanism to learn column distributions and in the reverse direction constraint-guided discovery to adhere to inter column dependencies. The approach is evaluated on four classification and two regression datasets and against a variety of baselines.

### Strengths
The strengths are: 
- The paper tackles a practically important problem of modeling tabular data. 
- There proposed model is evaluated on a number of datasets and against a number of baselines.

### Weaknesses
The main weakness are:
- I had difficulty understanding the model. The problem set up did not include any optimization problem. Diffusion models themselves were not introduced. The technical exposition was heuristic and difficult to follow. 
- Critiques of alternative models lacked precision, rendering the contribution unclear. 
- The experiments seem promising but the lack of technical clarity made it hard to come to solid conclusions about the nature of performance. 

Minor:
- The acronyms NPM and CGR are not introduced before they are used. 
- Figure 1 is not helpful. It is too small, and the dependences do not require a picture. They would be more concisely described in words. (This is essentially what the figure does.)
- "They focus primarily on distributions and overlook the real, intricate relationships between columns" What does this claim mean? What is "overlook"ing in this context? Is there a quantitative claim that could more effectively make the point? 
- "How to construct..." This sentence is awkward. It appears to be missing a verb. How should we...How ought we...How can we...Also the sentence should probably end with a question mark. 
- "they still focus primarily on data distribution and fail to capture the complex inter-column dependencies" Again, it feels like there is a more precise way to word this claim. I don't really understand what is mean by it in the current form. 
- "Furthermore, they often overlook real-world constraints (e.g., Real-world Tabular Constraints of Fig. 1)" Are these different than the inter-column dependencies? How? Are real-world constraints different from Real-world Tabular Constraints?). 
- "Due to the unique reconstruction
process of diffusion models, they can incorporate realistic constraints, giving them a distinct advantage in this field. " <--- The beginning of this paragraph appears to be about how diffusion models don't capture these constraints? 
- The introduction is largely redundant with the abstract. There are more words, but they don't really ad much. 
- "We propose the Explicit Column Relationship-Based Diffusion Model (ECR-DM), which explicitly captures ..." double explicit
- The contributions are also repetitive. 
- I would consider putting the related work near the end rather than at the beginning. It is hard to understand what might or might not be related in an interesting way without knowing more about what are the technical innovations of the current work. 
- In the problem definition, I would have expected to see some kind of loss to minimized? I don't understand the problem based on the statement.

### Questions
Please see weaknesses.

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
3

### Summary
This paper proposes ECR-DM, a diffusion-based framework designed to generate realistic synthetic tabular data while preserving real-world constraints and inter-column dependencies. Unlike prior GAN or diffusion methods that mainly focus on statistical similarity, ECR-DM explicitly models column-wise relationships and incorporates domain-specific logical constraints through a constraint-guided recovery mechanism in the reverse diffusion process. Experiments on six real-world datasets show substantial improvements in constraint satisfaction (SA up to 98–99%) and downstream task performance compared to baselines such as CTGAN, TabDDPM, and TABDIFF.

### Strengths
The proposed approach is novel and well-motivated. It clearly improves both the logical consistency and fidelity of synthetic tabular data compared to existing baselines. The idea of integrating explicit constraints into the reverse diffusion process is elegant and practical, and the experimental results are convincing.

### Weaknesses
The model relies on predefined constraints, which limits general applicability. The computational cost and interpretability of the constraint-guided recovery step are not fully discussed.

### Questions
1. Could the constraint embeddings be learned jointly rather than predefined?
2. How does ECR-DM behave when constraints are incomplete or partially inconsistent?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Please read weaknesses

### Strengths
Please read weaknesses

### Weaknesses
In the first page of the draft (lines 033 through 041) there is a table that is located one line higher than the beginning of the section without keeping the corresponding section space from main text/figures/tables hence violating ICLR 2026 template
I recommend desk rejection

### Questions
Please read weaknesses

### Soundness
1

### Presentation
1

### Contribution
1
