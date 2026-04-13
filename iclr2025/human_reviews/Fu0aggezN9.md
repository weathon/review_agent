## Human Reviewer 1

### Summary
In this paper, the authors introduced an approach to analyzing document structures using graph theory, which offers a unique perspective on understanding the complex relationships and patterns within documents. The primary contribution of this paper lies in its methodology that transforms textual content into graphical representations, enabling researchers and practitioners to visualize and analyze the hierarchical and relational aspects of documents more effectively.

### Strengths
1. The authors propose a method that represents documents as graphs, where nodes could represent sentences or paragraphs and edges could capture relationships between these elements based on semantic similarity or other linguistic features. This original perspective offers a fresh way of understanding the hierarchical organization within texts.

2. The authors provide a clear and systematic approach to converting textual documents into graph structures, followed by an analysis that leverages graph algorithms. They also discuss the advantages of their method over traditional text-based approaches, such as improved handling of complex document structures and enhanced interpretability through visual representations. 

3.  The clarity of the paper is good. The authors have structured the content logically, starting with an introduction to graph theory and its relevance to document analysis, followed by a detailed explanation of their method, implementation details, and results.

### Weaknesses
1. The paper could benefit from a more explicit discussion on potential weaknesses or limitations of the proposed method. This includes considerations such as scalability with large datasets, robustness to noise, and applicability across different types of documents.  

2. There are lack of comparison with MLLM based methods. The results of GPT4 should be reported for readers to compare.

### Questions
1. A higher IoU score suggests a better alignment between the predicted and actual bounding boxes.What is the IoU score when it falls between 0.7 - 0.95  and  0.8 - 0.95?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper defines a graph-based Document Structure Analysis task (gDSA) and builds a new Document Structure Analysis dataset, GraphDoc, based on existing datasets. By constructing a document layout graph structure that includes spatial and logical relationships, it enhances the depth of document structure understanding. Additionally, the paper presents a new end-to-end Document Relation Extraction model, DRGG, designed to extract features from the document layout and generate a relation graph that includes spatial and logical relationships among document elements.

### Strengths
The author compares the proposed DRGG framework with several existing document layout analysis and graph structure analysis methods, demonstrating through experiments that this model exhibits outstanding performance on the gDSA task, particularly in handling complex document structures, which is significant for achieving deeper document understanding.

### Weaknesses
1. Most document instances exhibit multiple relationships. The paper should provide additional experiments to evaluate the precision and recall of DRGG in capturing instances that have both spatial and logical relationships, compared to those with only one type of relationship.
2. In the dataset, the number of samples in the position relationship category is significantly higher than that in the logical relationship category. It would be better to consider adding experiments to demonstrate how to select the relationship confidence threshold to ensure evaluation accuracy in the context of imbalanced sample sizes among relationship categories.

### Questions
1. A performance bottleneck has been observed in detecting reference relationships in the experiments. Given the importance of reference relationships in document structure understanding, could you supplement with an analysis of the error cases in reference relationship detection, discussing the reasons for these errors and suggesting improvements.
2. It is suggested that the author can elaborate on the utility of the proposed dataset, evaluation tasks, and evaluation metrics in understanding complex document interactions. Is it possible to provide specific examples of how the GraphDoc dataset and DRGG model can be applied in real-world document understanding tasks? Highlighting of these advantages, such as Reading Order Prediction and Document Hierarchical Structure Analysis tasks mentioned in the paper.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces the **GraphDoc dataset**, a new dataset for document layout and structure analysis derived from the DocLayNet dataset by adding relation annotations. The authors propose a new task called **graph-based Document Structure Analysis (gDSA)** that involves not only detecting document elements, but also generating **spatial and logical relations** in the form of a graph. To address this new task, the authors also propose a model called the **Document Relation Graph Generator (DRGG)** that can generate relation graphs from document layouts, achieving 57.6% mAPg@0.5 on the GraphDoc dataset.  The authors hope that this graphical representation of document structure will be an innovative advancement in document structure analysis and understanding.

### Strengths
This is a large new dataset (80,000 single-page document images ; 1.10 million instances across 11 categories: Caption, Footnote, Formula, Listitem, Page-footer, Page-header, Picture, Section-header, Table, Text, and Title.) that extends DocLayNet Pfitzmann et al. (2022), and the relative location of the document elements is key to document understanding

### Weaknesses
This paper doesn't really have any major weaknesses. it is mostly a dataset paper and the dataset is strong. The authors also introduce a new model, which is rather straightforward but still interesting.

### Questions
Line 284: "most of the results have been manually verified and refined". What percentage of results have been manually verified and what percentage of results have been manually refined?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces a dataset which is built on top of the DocLayNet dataset to provide detailed annotation of both spatial and logical relation annotations for understanding the structure of single-page documents. As well as, a new framework, Document Relation Graph Generator, is proposed as a baseline for conducting layout and structure understanding in an end-to-end manner.

### Strengths
1. The research problem is interesting and novel which is the less explored in this area.
2. The dataset generation and the evaluation metrics look reasonable.
3. Some analyses are conducted to show the insight of the datasets such as the correlations and co-occurrence of different relation pairs.

### Weaknesses
1. Some key concepts are not clearly defined: e.g. why spatial relations are important, why only four spatial relation types are defined, Is proper all documents from different domains to adopt the same rule to define the relations between semantic entities.
2. Lack of analysis about rule-based generated relation and human refining details which is essential for demonstrating the high quality of the dataset.
3. It would be clearer if the detailed categorisation of parent, child, and reference relation types could be provided with examples (visualised would be much better) to enable the reader to understand definitions and tasks clearly.
4. The description of Document Relation Graph Generator is not clearly described to me. The overall workflow with the corresponding aim is relatively hard to understand. I have gone through it several times but still hard to understand including input to the framework, why the encoder and encoder structure, and how the decoder works during training and inferencing. 
5. The experimental section is underexplored. Analysis of different document domains, fine-grained relation type analysis (section->subsection, section->paragraph). 
6. Similar to quantitative analysis, more qualitative analyses are expected. 
I really like the idea of the paper, but more in-depth analysis is essential to maximising the contributions of the author desired.

### Questions
1. Why do you think spatial relation should be considered as one of the relation types of this dataset which might be directly acquired by leveraging the predicted bbox coordinated?
2. Could you provide more statistics details before and after human checking?
3. Do you think the rules might be updated based on different document domains?
4. For the relation prediction dataset, is there any reason you choose single-page datasets? It is more natural to use multipage-scenarios to establish the entire document's logical structure.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4