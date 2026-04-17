# ESMfluc: Predicting Flexible Regions in a Protein Using Language Models

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Proteins are dynamic molecular machines whose functionality emerges not merely from their static structures but critically from their intrinsic conformational flexibility. Understanding how a protein sequence encodes this flexibility is essential for deciphering the connection between sequence, dynamics, and biological function. While recent advances in deep learning and protein language models have significantly improved structural prediction, predicting sequence-encoded dynamics remains challenging. In this work, we introduce ESMfluc, a biLSTM model trained on molecular dynamics simulation data, utilizing embeddings from the Evolutionary Scale Modeling (ESM) architecture to predict local flexibility directly from protein sequences. Using fluctuation data derived from extensive molecular dynamics simulations, ESMfluc accurately identifies flexible residues without computationally expensive simulations while providing interpretability via attention maps. The model notably highlights distal flexible regions relevant for allosteric regulation and drug targeting. Our approach demonstrates substantial improvements over traditional flexibility proxies, offering researchers a computationally efficient method to reveal critical functional sites beyond active or binding regions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose to use a biLSTM model with an attention layer to predict flexible regions in proteins.
They model the problem as a per-amino-acid binary classification problem in two classes: rigid and flexible.
The model inputs are embeddings from a pretrained ESM2 model.

### Strengths
The authors make efficient use of existing models and technology.

### Weaknesses
The title talks about language models, but there are none in the paper.

In the end, the contribution of this work amounts to training a biLSTM+Attention model on a binary sequence-element classification task.
In my opinion, this is not enough to warrant reading by the ICLR audience.

### Questions
Why did you group all `N_{eq} > 1` into one class?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigate a sequence-based model for predicting local protein flexibility, using frozen ESM-2 embeddings followed by a lightweight BiLSTM and attention classifier.
The authors employ the ATLAS molecular dynamics dataset, deriving binary flexibility labels, and claim superior performance compared to structure-based predictors.
However, the entire framework is essentially a direct application offering no methodological novelty.
Furthermore, the experimental setup raises serious concerns about the scientific validity of the reported results.

### Strengths
This paper tries an interesting direction: linking protein sequence representations with dynamic flexibility signals derived from molecular dynamics simulations.

### Weaknesses
1. Predicting molecular dynamics–derived flexibility directly from sequence embeddings seems to be scientifically weak. While amino acid composition and local motifs indeed encode limited flexibility trends, MD-derived quantities such as RMSF or Neq reflect complex structure- and environment-dependent fluctuations that cannot be reliably inferred from sequence alone.
2. Methodologically, the paper presents no genuine innovation. The designed framework merely stacks a BiLSTM and a single attention layer on top of frozen ESM-2 features, without introducing any new architecture, loss formulation, or theoretical insight.
3. The experimental setup is also flawed.
    * the binarization of flexibility labels to 0/1 is arbitrary and discards most quantitative signal
    * random data splitting also raises potential issues of family-level data leakage
4. The writing quality is weak. References are frequently misused, with \citep and \citet incorrectly used throughout, resulting in broken sentence structures. In addition, this paper contains typos such as line 165 and line 332 suggesting inadequate proofreading.

### Questions
All my concerns about this paper is listed in the weakness part.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
ESMFluc is a protein sequence model that is trained to directly predict the dynamics of a protein. More specifically, it predicts the residue-level flexibility metrics derived from MD simulations. This enables the model to identify flexible regions without any structural inputs.

### Strengths
The authors show that ESMFluc outperforms the NetSurfP disorder predictor from static structures, showing the utility of sequence models over structure models for disorder prediction. The authors also provide classical machine learning baselines that validate the effectiveness of ESM features.

### Weaknesses
This paper is somewhat narrow in scope. It train and evaluate on the ATLAS dataset of all-atom MD trajectories but does not provide downstream applications for the flexibility predictor.

### Questions
Have you evaluated your flexibility predictor on downstream tasks, such as intrinsically disordered protein prediction [1][2]?

[1] Direct prediction of intrinsically disordered protein conformational properties from sequence. Jeffrey M. Lotthammer, Garrett M. Ginell, Daniel Griffith, Ryan J. Emenecker & Alex S. Holehouse.
[2] Critical assessment of protein intrinsic disorder prediction. Marco Necci, Damiano Piovesan, CAID Predictors, DisProt Curators & Silvio C. E. Tosatto.

Why are you using a biLSTM as opposed to attention?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ESMFluc, a model built upon ESM-2 to directly predict residue-level flexibility from protein amino acid sequences. The flexibility labels are preprocessed by binarizing the $N_{eq}$ values from the ATLAS dataset. Based on embeddings extracted from ESM-2, ESMFluc adds a biLSTM and an attention module to predict the binary flexibility class. Several prediction module designs (FC, LSTM, BiLSTM, and their combinations with attention modules) are evaluated. Comparisons with classical machine learning models, including logistic regression and random forests, demonstrate superior prediction accuracy. Compared with the structure-based approach NetSurfP, ESMFluc shows a clear advantage in both AUROC and Spearman metrics. Analysis of the attention weights reveals that a residue tends to attend to other residues with similar secondary structure classes and flexibility labels, even when they are distant in the sequence.

### Strengths
1. The analysis of attention homophily is interesting—it shows how residues with similar flexibility contribute to each other’s prediction results.
2. The paper clearly presents the methods, including detailed descriptions of the dataset and experimental setup.
3. The results reveal, to some extent, that protein sequences contain intrinsic information about structural flexibility. At least, a mapping between sequence and flexibility can be effectively learned using ESMFluc.

### Weaknesses
1. The paper focuses on binary classification during training but does not provide a strong motivation for binarizing the $N_{eq}$ labels from the original dataset. As such, the model demonstrates the capability to distinguish rigid versus flexible residues, but it remains unclear whether it can effectively capture different degrees of flexibility.
2. The evaluation lacks comprehensive baselines. Many pretrained protein models could be fine-tuned for the task, such as ESM-3. The conclusion that sequence-only modeling (as in ESM-2) is sufficient for flexibility prediction would be more convincing if additional backbone models—including those trained for structure prediction—were also evaluated.
3. Some details of the evaluation are missing. One important question is how NetSurfP was applied to the curated dataset. Was it evaluated in a zero-shot manner, or was it trained/fine-tuned using the $N_{eq}$ labels?
4. The paper’s current presentation lacks emphasis on its main contributions. For example, the methods section devotes extensive discussion to dataset preprocessing but much less attention to the design of the model architecture.

### Questions
1. How was NetSurfP used in the evaluation?
2. Since $N_{eq}$ is inherently a continuous variable, why was it necessary to convert it into a classification problem?

### Soundness
2

### Presentation
2

### Contribution
2
