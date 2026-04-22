# OmniLayout: Enabling Coarse-to-Fine Learning with LLMs for Universal Document Layout Generation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, document layout generation, remains underexplored. A major obstacle lies in the scarcity of diverse layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate **OmniLayout-1M**, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce **OmniLayout-LLM**, a 0.5B model with a designed two-stage *Coarse-to-Fine learning paradigm*: 1) learning universal layout principles from OmniLayout-1M with coarse category definitions, and 2) transferring the knowledge to a specific domain with fine-grained annotations. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M^6^Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, models, and dataset will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper considers the problem of generating realistic document layouts of various complex types: e.g., newspapers, magazines, among others. Synthetically-generated layouts have been shown useful in the literature (not in this paper) to boost the performance of modern deep learning based algorithms for document analysis, and thus the generative problem has received a lot of attention in recent years.

The current paper makes two main contributions:
1. Curating the first million-scale dataset of document layouts (of six types). In comparison, existing datasets are smaller by at least an order of magnitude.
2. Devising an LLM-based method to generate document layouts in a gradual, coarse-to-fine manner. Experiments show that the new method produces considerably more realistic layouts (as measured by a number of geometric metrics) than previous speficially tailored methods for this problem; and perhaps slightly more realistic (as evaluated by my human eye, and by considering the numeric performance ) compared to general-purpose LLMs.

### Strengths
- The curation of a large-scale dataset of document layouts is important for the document analysis community.

- The outputs of the new method seem visually realistic and achieve competitive performance against previous methods, including powerful general-purpose LLMs. 

- The paper is well-written. Experiments are (to me) relatively adequate.

### Weaknesses
The main weakness is that, unfortunately, the contribution can become deprecated pretty fast given the good performance of general-purpose LLMs:
- The performance of the new method compared to general-purpose LLMs is not clearly much better. Given how good Claude Sonnet-3.7 performs on this task, for example, it seems realistic that the next generation would be at least as good, if not better than, the proposed method.

- The importance of large, human-curated datasets may be decreasing with time, given the huge amount of diverse data that modern general-purpose LLMs train on.

(Unrelated to the two above:)
Why are 5-shot results, which to me are perhaps most interesting, not included in the comparison in the main paper, only in the rebuttal?

### Questions
- Can you explain why you think this work will not become deprecated quickly in the presence of general-purpose LLMs?

- See question about 5-shot results in "weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the document layout generation task. A large-scale dataset with 6 common document types, i.e., OmniLayout-1M, is proposed from multiple sources. In addition, a two-stage Coarse-to-fine learning paradigm is proposed with the 0.5B OmniLayout-LLM. Experiments show the effectiveness of the proposed method for document layout generation.

### Strengths
The proposed coarse-to-fine paradigm is a reasonable solution for improving the performance of layout generation in a specific document domain. The constructed OmniLayout-1M dataset that contains several common document types and corresponding layout annotations could be useful for future research in the community.

### Weaknesses
1. The contribution of the proposed dataset is relatively limited. Many existing graphic design layout generation works also use the public document layout datasets like PubLayNet. As shown in Table 1, the only advantage of the proposed dataset is the scale, rather than the layout type. But the advantage of increasing the volume to 1M has not been fully demonstrated yet.  
2. I cannot agree with the statement that the proposed method is the first to extend document layout generation to complex and challenging domains. ContentGAN [1], which was neither cited nor discussed, was the first to model complex and challenging document types like fashion magazines and newspapers.
3. The effectiveness of the proposed OmniLayout-LLM with a coarse-to-fine learning paradigm is not very convincing. Based on the results in Table 4, the performance of the coarse-grained learning paradigm is much worse than the fine-grained paradigm. Note that most of the performance improvement comes from the fine-grained paradigm with existing specific datasets, rather than the coarse-grained paradigm with the proposed datasets. 
4. The effectiveness of the two stages should be evaluated on different model sizes, including 0.5B, 1.5B, and 3B.

[1] Zheng, Xinru, et al. "Content-aware generative modeling of graphic design layouts." SIGGRAPH 2019.

### Questions
1. The annotations are obtained in an automatic manner by employing MinerU. How to avoid annotation errors or noises during the construction process?
2. Since the datasets are collected from sources on the Internet, I was wondering whether the proposed dataset could be fully released without copyright issues.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies automatic document layout generation. Given the limitations of the existing datasets, the authors curate OmniLayout-1M, a large-scale dataset covering 6 document types from multiple sources. An automatic annotation process is proposed to obtain the element sequence. Based on the dataset, they fine-tune a 0.5B parameter LLM, resulting in OmniLayout-LLM. Experiments demonstrate superior performance of OmniLayout over existing layout generation methods and general-purpose LLMs.

### Strengths
The paper is generally well-written with clear motivation and methodology. It contributes OmniLayout-1M, which addresses a real need for diverse, large-scale layout data with automated annotation pipeline. The experiments are well-designed, covering 5 document types, 5 layout generation tasks, and different baseline models.

### Weaknesses
- During the dataset annotation, the authors employ MinerU to parse the elements from PDF files. Is it good enough to obtain high-quality labels for OmniLayout-1M? The performance of MinerU would largely determine the dataset quality. For example, if MinerU produces element boxes that severely deviate from the GT boxes, the resulting layouts may have potential issues, such as undesirable element overlap. Therefore, it is necessary to perform quality analysis on the dataset.
- The qualitative results of LayoutPrompter are missing. LayoutPrompter exhibits better evaluation metrics than other baselines, as shown in Table 2. It is necessary to include its qualitative results in Figure 4.
- The ablation studies are confusing in Table 4. For example, the 3B parameter model has the worst FID score compared to the smaller ones in the "C->S+P" task, which contradicts the general understanding of the scaling law. The results are also observed in the "C+S->P" and "Refinement" tasks. Furthermore, the FID value of the 3B parameter model for the refinement task is 67.24, significantly exceeding that of all other parameter sizes and task settings. However, from what I understand, the refinement task is relatively simple among the five tasks since it has the most conditions. Is this purely due to FID variance on small test sets, or are there overfitting issues?
- Some implementation details are not included. For example, how to perform deduplication and data filtering during data preprocessing?
- Ablation studies lack qualitative results comparison.
- The title of the paper claims "universal document layout generation". Have the authors tested on document types outside the six categories? According to the ablation results in Table 4, fine-grained learning is more important than coarse-grained learning, which indicates that the model's performance heavily depends on the fine-grained annotations for the target domain. This fundamentally contradicts the "universal" claim.
- How does performance degrade with very long sequences?
- The authors construct the coarse-grained data across five tasks with a ratio of 1:1:1:3:3. Are there any ablation studies on the mix ratio?

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on document layout generation.

The major contributions of the paper include: 1) a large-scale dataset (OmniLayout-1M) containing about one million layouts from diverse document domains; 2) an LLM-based layout generation framework (OmniLayout-LLM) with a two-stage learning scheme, where the LLM first learns universal layout knowledge from large-scale data with coarse-grained labels, and then is adapted to generate layouts of a specific domain using small-scale data with fine-grained labels.

### Strengths
1. Data scarcity in layout generation is an important problem to address.

2. The contributed large and diverse layout dataset could be valuable to the layout generation field.

3. The proposed two-stage learning scheme is shown to be effective, and the performance improvements of the proposed method over existing methods are noticeable.

### Weaknesses
1. The accuracy of layout annotations on OmniLayout-1M is questionable. Since the annotations were obtained automatically using existing models, without being manually checked by human subjects, it is unclear whether OmniLayout-1M has reasonable annotation quality, particularly for documents with a large number of elements and highly complex layouts. It would be better to see how good the layout annotations are through some quantitative scores. For example, it is possible to manually label a small subset of documents for each document domain, and compute some metric scores of the automatically annotated layouts against the ground truth ones.

2. The design of the layout representation introduced in Section 3.2 is evaluated. Unlike most existing LLM-based layout generation works, such as LayoutPrompter, this paper chooses to represent layouts as plain sequences, instead of in a HTML format that pretrained LLMs are familiar with. This may not well leverage prior knowledge in the LLMs. An experiment comparing different layout representations is missing in the paper, and should be added . 

3. References are incomplete. Some existing layout datasets for specific domains, such as mobile UI (RICO) and poster (CGL, PKU), are not discussed in the paper.
     -  RICO: Learning Design Semantics for Mobile Apps
     -  CGL: Composition-aware Graphic Layout GAN for Visual-textual Presentation Designs
     -  PKU: PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout

### Questions
What is the number of element categories on OmniLayout-1M?

### Soundness
2

### Presentation
3

### Contribution
3
