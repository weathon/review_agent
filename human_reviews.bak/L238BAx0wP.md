# Learning to engineer protein flexibility

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 6

## Abstract
Generative machine learning models are increasingly being used to design novel proteins. However, their major limitation is the inability to account for protein flexibility, a property crucial for protein function. Learning to engineer flexibility is difficult because the relevant data is scarce, heterogeneous, and costly to obtain using computational and experimental methods. Our contributions are three-fold. First, we perform a comprehensive comparison of methods for evaluating protein flexibility and identify relevant data for learning. Second, we overcome the data scarcity issue by leveraging a pre-trained protein language model. We design and train flexibility predictors utilizing either only sequential or both sequential and structural information on the input. Third, we introduce a method for fine-tuning a protein inverse folding model to make it steerable toward desired flexibility at specified regions. We demonstrate that our method Flexpert enables guidance of inverse folding models toward increased flexibility. This opens up a transformative possibility of engineering protein flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript presents an deep learning based approach to predict protein flexibility and engineer protein sequence towards more flexibility. There are three steps in the manuscript. The first step is to compare metrics that quantify protein flexibility. The second step is to repurpose pre-trained protein language model for flexibility prediction. Here, the authors trained Flexpert-Seq (no 3D structure input) and Flexpert-3D (with 3D structure input) for the regression task. Last but not the least, the authors proposed a framework to train ProteinMPNN with flexibility instructions to generate more flexible protein sequence given the a input 3D structure.

### Strengths
The idea of predicting the flexibility of protein given structure or sequence is rather novel. The comparison between the metrics for evaluating protein flexibility is valuable. The framework proposed to inverse fold for better flexibility, Flexpert-Design, can be valuable for specific applications requiring more flexible proteins.

### Weaknesses
1. Writing can be improved. For example, in line 79-80, "From these simulations, Root Mean Square Fluctuations (RMSF)." is not a full sentence. Presentation can also be improved. In table 3 and 4, only MD results are bolded, whereas the ANM (MD str.) results are not. Authors can also add arrows to indicate if each metric is the higher the better or the lower the better.
2. The benchmark needs more explanation, especially the flexibility prediction one. It is not clear to me how AF2 and ESMFold are used for flexibility prediction (the regression task). The interpretation of the benchmarking results (table 3 and 4) is also vague.
3. There is not enough elaboration on why more flexible protein is needed. One major contribution of this paper is the Flexpert-Design framework, which tunes ProteinMPNN to generate more flexible protein given the prediction of a flexibility predictor. However, it is not clear to me (from reading the manuscript) the purpose of modifying the flexibility of a protein. The authors can focus more on the motivation. 
4. The Flexpert-Design can introduce bias from Flexpert-3D into the ProteinMPNN model through the flex loss. Also, the correlation between the input structure and the flexibility instruction can be low when the flexibility instruction is artificially increased, i.e. the input structure can never have high flexibility given by the flexibility instruction. In such case, the prediction of both flexibility instructed ProteinMPNN and Flexpert-3D can be unreliable.

### Questions
1. How is RMSF calculated? It would be better to note the RMSF equation in the manuscript.
2. Line 152-153, author claims higher pLDDT uncertainty indicates lower flexibility, which is kind of counter-intuitive. Can authors elaborate?
3. When benchmarking the flexibility prediction ability, how are AF2 and ESMFold used? Based on my understanding of the manuscript, the flexibility prediction is a regression task. Did authors attach a prediction head to baseline models? Please elaborate.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper the authors investigate the correlation of various in silico proxies to root mean square fluctuations of protein structures derived from molecular dynamics calculations. They posit that augmenting inverse folding methods for protein design to guide for RMSF as a proxy for flexibility is desirable for design applications. To this end, they introduce Flexpert-Seq, Flexpert-3D, and Flexpert-Design. They demonstrate that supervised models outperform other methods for predicting RMSF from sequence alone, and that they can guide inverse folding design to favor more flexible residues.

### Strengths
The paper introduces an interesting problem, explores the problem and relevant dataset before diving into the methodology, and shows that simulated dynamical properties of proteins can be predicted from sequence alone, or sequence and simple structural featurizations, offering important advantages in speed and use-of-use compared to other methods.
The discussion of sequence recovery as an incomplete evaluation of inverse folding methods, and the predominant usage of ProteinMPNN due to its extensive wet lab validation, is appreciated.
The Table 3 and 4 results are nice, clear demonstrations that the authors’ hypothesis is correct, and that a supervised model outperforms more indirect methods like pLDDT and ANM for predicting RMSF values.

### Weaknesses
The authors might provide a concise definition of “flexibility” earlier on in the paper. Flexibility could mean many things and be quantified and measured different ways, experimentally and in silico, as detailed in Section 3.1. 
Some discussion is needed of the incompleteness of the methods introduced in Section 3.1. pLDDT and B-factors can be confounded by many other covariates that are not flexibility, and RMSF calculations from MD may not accurately reflect the real world biophysics of dynamic proteins in a physiological context, where interactions with solvent and other biomolecules will affect effective flexibility. 
A small nit, say “MD trajectories” rather than MDs to clarify the source of reference datasets.
I would recommend capturing all the newly introduced methods under subheadings under a Methods section heading, and then going into results. With the current structure I expected to see Results after Section 4, but Section 5 introduces another method.
Without simple baselines (some ideas provided in the Questions below) it is difficult to assess the results presented in the design section.
The Conclusions section restates what was done, which is covered in the intro. Discussion of limitations and future directions would be appreciated.

### Questions
Can the authors support the statement in the abstract that accounting for flexibility is the major limitation of current protein design methods? There are many limitations of current methods; flexibility is certainly an important one, but it is not clear that it is a singular limitation.
Can the authors show some failure cases where predictions from Flexpert-Seq do not agree well on the ATLAS dataset? I’m curious if there are certain classes of protein or biophysical features that are difficult to predict well.
The enrichment for A, S, T, and G residues and depletion of bulkier residues on flexibility guidance makes sense - is there a “self-consistency” check that ensures that the guided designs retain the desired predicted backbone structure? How does Flexpert-design compare to running ProteinMPNN and using protein language model likelihoods to identify residues where mutations to smaller, more flexible residues are likely to be tolerated?
Relatedly, per-residue RMSF seems to be a “local” notion of flexibility, which causes a straightforward enrichment of a few specific, small residues. Can the authors imagine ways to include a more non-local measure of flexibility that may induce larger or smaller fluctuations across a biomolecule, without dramatically changing the conditional amino acid distribution?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
the paper address an interesting problem where one wan to generate novel proteins while accounting for protein flexibility—a feature that is critical for protein function but has proven challenging to model and integrate due to data scarcity, heterogeneity, and the high costs of data generation.  The paper provide benchmark dataset and ways to quantify flexibility. They  adapt the inverse folding model for flexibility, guiding protein sequences towards targeted flexibility in specific regions.

### Strengths
1. The study provides a robust review of various protein flexibility quantification methods and assesses the utility of diverse data sources. 
2. The development of Flexpert-Design to adjust flexibility in specified regions is significant, especially for applications in enzyme engineering, drug design, and biocatalysis.

### Weaknesses
1. A major limitation is the lack of extensive experimental validation.

### Questions
1. Given the relatively small ATLAS dataset, how does the predictor generalize to other protein datasets? Has the model been tested on proteins outside the ATLAS dataset to evaluate robustness?
2. I am wondering if the proteins generated  by this model (due to the flexibility-aware nature of the model) show evidence of functional improvements (e.g., improved stability, interaction specificity), or is the flexibility change primarily theoretical?
3. The Flexpert loss function combines flexibility and sequence objectives with a mixing parameter, θ. How sensitive is the final model performance to different values of θ, and how was the optimal value chosen?

### Soundness
3

### Presentation
3

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
This paper proposes a method called Flexpert-Design to enable guidance of inverse folding models toward increased flexibility. The authors first quantify protein flexibility and identify data relevant to learning, then train flexibility predictors with protein language models. Finally, they fine-tune inverse folding models to steer it toward desired flexibility.

### Strengths
- **New field scenario**: Protein flexibility is rarely considered by current protein engineering methods.

- **Reasonable baseline settings**: The authors arange a reasonable set of flexibility-related baselines.

- **Quantified metric evaluation**: The authors apply several methods to quantify protein flexibility comprehensively.

### Weaknesses
- The technical novelty is somewhat limited. While the problem is important, the overall framework is simple and trivial.

- Flexpert-Design uses Gumbel-Softmax trick to enable back propagation. However, this estimation is not extremely accurate.

- Undiscussed choice consideration for the backbone. 

- The experimental results are not comprehensive. Only flexibility-related metrics are shown. Besides, only ProteinMPNN is considered as the inverse folding model.

- Lack of evaluations on the overall protein quality (reliability, plausibility, etc.) for the sequences generated by Flexpert-Design. This can be an important proof of its practical application ability

- No clear explanation for the engagement of the proportion of flexibility-increasing mutations (What is its meaning? What properties and biological representations does it reflect?

### Questions
Here are some questions **from weaknesses**.

(1) Since estimation with Gumbel-Softmax trick is not extremely accurate, how does this method compared to other RL-based methods with the L\_flex as the reward?

(2) Why you choose ProtTrans as the backbone, do you try ESM series?

(3) Can the proposed method be applied to any inverse folding models?

And here are questions **about experiment details and considerations**.

(4) In Table 1, how do you match the RMSF of MD with other methods’ metrics (like plddt of AF2 and ESMFold2)? And in Sec. 3.1, you mention that MD is considered an accurate physics-based method. Is that a common sense? If so, why include the whole experiment in Table 1?

(5) How do you get the average PCC upper bound of 0.88? Is that for MD? Does it make sense to set an average value as the upper bound?

(6) How do you prove that the protein sequences designed by Flexpert-Design are natural (or useable) other than with good flexibility only? Since you mentioned that the loss function makes a balance between flexibility and sequence recovery, is it worth sacrificing the overall quality of proteins to enhance their flexibility?

### Soundness
3

### Presentation
3

### Contribution
2
