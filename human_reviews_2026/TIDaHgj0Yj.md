# OSIRIS: Bridging Analog Circuit Design and Machine Learning with Scalable Dataset Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
The automation of analog integrated circuit (IC) design remains a longstanding challenge, primarily due to the intricate interdependencies among physical layout, parasitic effects, and circuit-level performance. These interactions impose complex constraints that are difficult to accurately capture and optimize using conventional design methodologies. Although recent advances in machine learning (ML) have shown promise in automating specific stages of the analog design flow, the development of holistic, end-to-end frameworks that integrate these stages and iteratively refine layouts using post-layout, parasitic-aware performance feedback is still in its early stages. Furthermore, progress in this direction is hindered by the limited availability of open, high-quality datasets tailored to the analog domain, restricting both the benchmarking and the generalizability of ML-based techniques. To address these limitations, we present OSIRIS, a scalable dataset generation pipeline for analog IC design. OSIRIS systematically explores the design space of analog circuits while producing comprehensive performance metrics and metadata, thereby enabling ML-driven research in electronic design automation (EDA). In addition, we release a dataset consisting of 87,100 circuit variations generated with OSIRIS, accompanied by a reinforcement learning (RL)–based baseline method that exploits OSIRIS for analog design optimization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work presents a synthetic pipeline that can generate large volumes of analog circuit layouts. This work also presents an RL design framework that can optimize circuit performance based on post-layout performance and show competitive performance compared to early SOTA MAGICAL and ALIGN. The entire framework will be open-sourced.

### Strengths
1. This work addresses an important problem for analog circuit design automation and provides an invaluable contribution
While the existing analog circuit dataset only contains basic netlists, this work is able to provide its layout, which is essential for determining the circuit's real-world performance. The entire synthetic pipeline is automated and can address the data shortage issue faced by the entire analog circuit design flow (front-end and back-end). I believe this work will drive the entire analog EDA filed forward. 
2. Excellent presentation and writing
3. Strong experiment results and supplement material support
The attached anonymous link contains all the material needed for determining this work's reproducibility.

### Weaknesses
1. More examples and results beyond the op-amp can further strengthen this paper

### Questions
see weakness

### Soundness
4

### Presentation
4

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
In order to address the issue of the lack of open-source high-quality datasets in the field of analog integrated circuits(IC), this paper presents a pipeline called OSIRIS, which is an end-to-end back-end framework for generating large quantities of analog layouts to advance ML research in circuit design. At the same time, it provides a public release of 64200 layout variants across four designs and an RL baseline.

### Strengths
1. Systematically generated a full-link dataset covering 4 types of typical amplifiers and 64000+ samples. All samples passed the “Design Rule Check” and “Layout-Circuit Consistency Check”, ensuring the “industrial-grade reliability” of the data. It is the first publicly available, reproducible, and fully annotated large-scale simulation IC layout dataset.

2. The process and methods have strong scalability. The automated processing flow has potential to handle other types of analog circuits and advanced manufacturing processing, without requiring significant modifications.

3. The supporting methods is highly consistent with the conference theme. The proposed two-level RL optimization framework  integrates machine learning technology deeply into the post-synthesis design process of analog ICs. It not only reflects the core direction of "machine learning-driven electronic design automation (EDA)", but also provides method validation for the practical value of the dataset, and is highly consistent with the positioning of ICLR, which focuses on innovative applications of machine learning.

### Weaknesses
1. Only covering 4 types of amplifiers and 130nm process, the circuit and process coverage is insufficient. 

2. The experimental part of the main text does not fully elaborate on the quality comparison of the data set and the model effect based on it, to prove the lightweighting. Without conducting verification in conjunction with specific simulation IC design tasks, there is a lack of quantitative results demonstrating the improvement in model performance of this dataset in actual tasks. As a result, the practical value and superiority of the dataset have not been fully verified through experiments, and the argumentation is not very persuasive.

3. Based on the data, the generation method is inefficient. It fails to meet the requirements of "millions of samples" in machine learning research or the "rapid iterative design" scenarios in the industrial sector. However, the article does not deeply analyze the core sources of the time-consuming bottlenecks, nor does it propose targeted optimization solutions.

### Questions
1. The dataset only covers 4 types of amplifiers. How can it be demonstrated that it has generalization capabilities for other analog circuits  and advanced processes? And It is advisable to verify the advantages of the dataset in some specific tasks.

2. The process of generating the dataset takes a considerable amount of time (more than 50 hours for a single circuit). Is it practical for large-scale machine learning tasks? Are there any optimization plans?

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
3

### Summary
This paper presents OSIRIS, a scalable dataset generation pipeline for analog IC layout design, producing over 64,200 circuit variations with detailed metrics to enable ML-driven research in EDA. It also introduces a reinforcement learning baseline that leverages the dataset for parasitic-aware layout optimization.

### Strengths
- The dataset is substantial, comprising more than 64,200 circuit variations, which could be highly valuable for future research.
- Since publicly available back-end analog circuit datasets are rare, this work has the potential to fill an important gap in the field.

### Weaknesses
- The main issue is that the paper is difficult to read. Given that ICLR is primarily an AI-focused venue, the paper should better explain the fundamental principles of analog back-end design and clearly describe the intended applications of the proposed benchmark. In its current form, it reads more like a technical report than a research paper.
- The experimental section is relatively short, even for a benchmark-oriented paper.

### Questions
- In Table 3, why does the Random method perform better than the open-source design tool MAGICAL?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The scarcity of open, high-quality datasets has constrained the use of machine learning in automating analog circuit design. This paper introduces OSIRIS, a scalable dataset-generation pipeline that uses reinforcement learning to systematically explore analog design spaces and produce DRC/LVS-clean layouts with comprehensive performance metrics and metadata, enabling robust benchmarking and generalizable ML methods.

### Strengths
1．	Introduces a dataset-generation pipeline for analog layouts and releases an open-source dataset augmented with post-layout simulations that guarantee the sample is LVS-, DRC-clean.

2．	Efficient design-space exploration. Proposes a reinforcement-learning-driven, iterative variant-generation method that enables efficient, performance-aware exploration of the analog layout space.

### Weaknesses
1．	Limited circuit type. The dataset currently covers only amplifier circuits at the 130 nm node.

2．	In Table 3, it’s not fair and confusing to compare with the MAGICAL and ALIGN, which are only analog layout generation tools without any design-space exploration.

3．	Constrained variant generation and diversity. Variants are created mainly by permuting device fingers and component placement within the halo, which limits structural diversity; some schematics permit fundamentally different layout topologies. Moreover, RL optimizes score and area only, without an explicit diversity objective, increasing the likelihood of many near-duplicate samples.

4．	Missing some usage examples. Include concrete examples of how the dataset can be used, e.g., training a post-layout performance predictor to guide place-and-route.

### Questions
1．	How do you handle cases where the RL agent fails to produce a valid solution after multiple iterations?

2．	Is component rotation included in the set of layout operations?

3．	Can the method scale to larger circuits like ADC/DAC?

### Soundness
2

### Presentation
3

### Contribution
2
