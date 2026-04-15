# Gradual Optimization Learning for Conformational Energy Minimization

- Decision: Accept (poster)
- Scores: 8, 6, 6, 6

## Abstract
Molecular conformation optimization is crucial to computer-aided drug discovery and materials design.
Traditional energy minimization techniques rely on iterative optimization methods that use molecular forces calculated by a physical simulator (oracle) as anti-gradients.
However, this is a computationally expensive approach that requires many interactions with a physical simulator.
One way to accelerate this procedure is to replace the physical simulator with a neural network.
Despite recent progress in neural networks for molecular conformation energy prediction, such models are prone to errors due to distribution shift, leading to inaccurate energy minimization.
We find that the quality of energy minimization with neural networks can be improved by providing optimization trajectories as additional training data.
Still, obtaining complete optimization trajectories demands a lot of additional computations.
To reduce the required additional data, we present the Gradual Optimization Learning Framework (GOLF) for energy minimization with neural networks.
The framework consists of an efficient data-collecting scheme and an external optimizer.
The external optimizer utilizes gradients from the energy prediction model to generate optimization trajectories, and the data-collecting scheme selects additional training data to be processed by the physical simulator. 
Our results demonstrate that the neural network trained with GOLF performs \textit{on par} with the oracle on a benchmark of diverse drug-like molecules using significantly less additional data.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an active learning approach to train neural network potentials by employing a cheap surrogate oracle before querying the much more expensive genuine oracle.

### Strengths
The paper is very well written with clear insight and motivation. The method is very neat and the results are promising. I am excited to see more work that follows these mixed genuine and surrogate oracle approach.

### Weaknesses
Additional baselines and metrics could make the result stronger.

In terms of the baseline, the true innovation of the framework is the "active learning" component, and not "conformer generation" itself. Therefore, the baseline conversation probably should focus on other active learning approaches such as Kulichenko et al (2023) or Chem et al. (2019). While these methods require OG, the author can still compare to these methods by contrasting the OG query budget to the same amount, but use random selection instead of SG estimates as proposed by GOLF. The comparison to TD/ConfOpt, while interesting, the problem setup and training data requirement are very different from what the authors are trying to demonstrate here.

In terms of the metrics, it would be very helpful if the authors can provide more context about why "percentage of minimized energy" is a meaningful metric, and why >98% is considered solving the optimization. If >98% is broadly considered as solving the optimization, can the authors report what percentage of targets in the test set is "solved" under different experiment setup?

### Questions
I would consider raising my score if the authors can address my concerns around baselines and metrics as mentioned in the Weaknesses section.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the paper, the authors present Gradual Optimization Learning Framework (GOLF), a framework for improving the efficiency of generating low-energy molecular conformation prediction models, a crucial technology used in computer-aided drug discovery and materials design.

Overall, the paper is well written and presents some valuable insights into applying active learning efficiently for the discovery of energy minimized conformations. Traditional approaches, such as Density-functional theory (DFT) models, use high-fidelity physics-based numerical quantum chemistry simulators whose computational costs are exponential with respect to the complexity of the molecule under study. Unfortunately, this limits their applicability to simple molecules with few atoms or electrons. To address computational complexity and to scale conformal optimization to more complex molecules, researchers have explored various alternatives based on lower fidelity linear models and, more recently, neural network models that leverage the availability of computed quantum property molecular databases. Sadly, these alternate approaches lead to inaccurate predictions and suffer from distribution shift. To scale conformal energy prediction to larger molecules while addressing computational cost, the authors propose GOLF, an automated data augmentation scheme and a hybrid computational approach that combines the use of both high and low-fidelity simulators as well as neural networks.

In Section 1, the authors present the concept of low-energy molecular conformations and their uses. This section is well written and provides an adequate background of the approach and the insights that motivate the solution. The authors indicate that the fundamental problem with traditional approaches, such as DFT, to obtain optimal conformations is their high computational cost. These approaches, which are based on numerical quantum chemistry simulations that calculate anti-gradients representing molecular forces, are iterative by nature and, when given a sufficiently complex molecule or physical system, may fail to complete even a single iteration. For this reason, the authors claim that “reducing the number of interactions with the physical simulator” is crucial for efficiency. The authors then go on to describe current methods that apply Neural Network Potentials (NNP), a form of deep neural networks, to the problem. NNP-based techniques significantly reduce computational complexity by using the gradients inherent to neural networks to model the molecular forces, thereby obviating the need for expensive simulations. Unfortunately, the NNP approach suffers from distribution shift resulting in inaccurate predictions. The authors then introduce GOLF, which employs a data augmentation active learning scheme to improve the diversity of the training dataset, thereby alleviating the distribution shift. By doing so, GOLF achieves energy minimized conformation prediction accuracy comparable with that of high-fidelity simulations while retaining the intrinsic efficiencies gained by using neural networks. A novel framework for data-efficient training of NNPs, GOLF comprises three components: 1) a computationally expensive high-fidelity simulation that is a genuine oracle (GO) used to calculate the ground truth energies and forces; 2) an optimizer that uses the NNP gradients to produce optimization trajectories that constitute the additional training dataset; and 3) a computationally inexpensive low-fidelity simulation that is a surrogate oracle (SO) used to augment the training dataset. Finally, the authors then conclude this section by summarizing their contributions.

However, there are various weaknesses in this section as well that can be addressed to improve the quality of the paper. 1) The statement that “reducing the number of interactions with the physical simulator” is unclear and the reviewer assumes the efficiency objective is attained by reducing the number of iterations required to produce optimal low-energy conformations. 2) The authors state that they augment the dataset with “optimization trajectories” without explaining what such a trajectory is and how the trajectories address distribution shift. At a minimum, providing a reference to the discussion of augmented data in Section 5 would be useful. 3) Moreover, as GOLF requires running the high-fidelity simulation, GO, to produce the anti-gradients, it is unclear whether GOLF can be successful when GO fails to complete a single iteration. This is a serious flaw in the paper and needs to be addressed. 4) In addition, the “Ab initio property” phrase is used without a definition or description and seems rather superfluous to the narrative. 5) The adequacy of the requirement for 5 X 105  “additional oracle interactions”, which presumably means optimization trajectories that augment the training dataset, is likely anecdotal based on the molecules selected for the experiments. If that is not the case, an explanation of why it is generally applicable should be articulated.

In the “Related Works” section (Section 2), the authors describe a variety of contemporary approaches to conformation generation. The benefits and drawbacks of these methods are discussed. However, it is not clear to the reviewer how GOLF addresses the drawbacks of these approaches. Successfully addressing the drawbacks could demonstrate GOLF’s superiority. Also, form the exposition in this section, it is not clear how significantly different the GOLF approach is to the active learning technique presented by Kulichecnko et al. (2023). Finally, the phrase “we believe it is necessary to explore further the ability …” is confusing. Are the authors proposing future work or teeing up the discussion in the remainder of the paper?

In the “Notations and Preliminaries” section (Section 3), the authors summarize the theoretical foundation of their approach. Although informative, the notation is somewhat cryptic and can benefit from slightly greater verbosity or additional graphics. Also, mentioning GOLF models in this section, without any discussion as to what they are or how they differ from ftraj, seems premature and confusing. At the very least, there should be a forward reference to Section 5 that articulates how GOLF intelligently identifies the datasets that promote diversity, which enhances prediction performance. Moreover, a small discussion of the NNP architecture used in the experimentation would be useful for the sake of completeness.

Section 4 presents “Conformation Optimization”. This section is well written and the both the graphic, Figure 1, and the table, Table 1, provide valuable insight. The Figure 1 graphic clearly depicts the distribution shift, in terns of Mean Square Error (MSE), increasing as the optimization progresses. The graphic also depicts that the prediction accuracy improves – MSE decreases – when augmenting the training dataset with GO produced optimization trajectories. This is an important result but without highlighting it, the reader can easily miss that it is one of the contributions of the paper. Table 1 seems to be highlighting precision of the approach, but the word is not used in the discussion. It is unclear to the reviewer as to the innovativeness of the approach, which may be construed as a weakness. Moreover, using the GO to provide the baseline training dataset may limit the scalability of the approach.

The authors present a sound argument in Section 5 where they present the GOLF algorithm and discuss using a high-performance low-cost surrogate oracle to make the data generation computationally tractable. The “Experiments” section, Section 6 is reasonably complete though much of the discussion seems anecdotal. The reviewer is unable to determine how many of the experimental results were achieved through a fortuitous selection of the molecules under study. Also, the authors report that the GOLF technique can produce “a high percentage of diverged conformations”. This is not surprising as the training dataset is likely to be much noisier as a result of the choice to use a low-cost simulation. It would be nice to get better characterization of the noise and its effects, including the loss in efficiency resulting from these unusable conformations.

Sections 7, 8, and Appendices conclude the paper. An explicit tie back to the goals and contributions identified in the Introduction would be beneficial.

### Strengths
Overall, the paper is well written and informative. It seems relevant to improving the computational tractability of conformational energy minimization. The insight to use active learning to address the distribution shift and improve accuracy is valuable. Also, the approach to active learning by using low-cost simulation to augment the training data set without impacting the quality of the subsequent model is somewhat innovative.

### Weaknesses
The various weaknesses are already detailed above. Here I summarize the most important ones. Using GO to generate the baseline dataset may limit the scalability of the approach. It is unclear why the authors do not use the molecular databases they mention in the “Introduction” section to extract the baseline dataset. Some of the discussion is somewhat cryptic and can benefit from some additional discussion or graphics. The results seem anecdotal, tied to the selected dataset and molecules, and thus may not generalize particularly well.

### Questions
Suggestions:
1) Clean up the use of "interactions with the physical simulator" and the "number of iterations" in several locations in the paper. They seem to imply the same concept. If they are, just use a single phrase for both.
2) Additional references early on in the paper to the results later on in the paper will help the reader question many of the seemingly unsupported statements.
3) A better tie-in to how GOLF addresses the drawbacks of the "Related Works" section will improve the quality of  the paper.
4) Reduce the amount of mathematical notation in the “Notations and Preliminaries” section (Section 3) to simplify the narrative and improve understandability.
5) The main result in Section 4, the decrease in MSE by augmenting the dataset, should be highlighted and tied-in to the wording of the contributions listed in the introduction section of the paper.

Questions:
1) GOLF requires running the high-fidelity simulation, GO, to produce the anti-gradients initial training data. In a different section of the paper, the claim is that for a sufficiently complex system, the physical simulation may not succeed in completing even a single iteration in a reasonable amount of time. Taken together, these two statements seem to suggest that GOLF is atomic complexity scale limited. How do the authors claim to address this apparent limitation?
2) In the Experiments section (Section 6), how much of the results are related to the selection of the molecules under study? Stated differently, how do the authors plan to address the generalizability of the approach?
3) The additional data generated for Active Learning are selected based only on errors. Without some sort of approach to balance the introduction of new data, does the approach bias the dataset distribution causing the learned distribution to fail to generalize to other molecules?
4) How different is the GOLF approach from the active learning technique presented by Kulichecnko et al. (2023).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a gradual optimization learning framework (GOLF) for molecular conformation minimization. The framework is designed to improve the training of Neural Network Potentials (NNP). The authors first claim that NNPs trained on existing datasets suboptimal in energy minimization due to the distribution shift and perform experiments to show (a large amount of) additional data from the optimization trajectories can help improve the NNP's performance.  The GOLF framework uses a surrogate oracle (MMFF) to evaluate the conformation energy and expand the training data by selecting the incorrect prediction and re-evaluating with the genuine oracle (DFT), which reduces the required additional data. The experiments on the nablaDFT dataset demonstrate the effectiveness of the proposed method.

### Strengths
The proposed method sounds reasonable and the experiments shows its effectiveness.  The method also looks easy to implement, which can improve the conformation energy minimization performance at a small cost.

### Weaknesses
- The writing is not very clear, especially in the introduction section. It takes me a while to understand simply enriching the training dataset is actually a preliminary baseline method the authors want to compare with. Also, lots of experiment details are mixed with the method, which makes the paper not very easy to read. 
- The calculation of COV and MAT looks problematic. It seems the authors optimize **one conformation per molecule** and take them of the entire test set as the generation set. However, in the conformation generation setting, models generate **multiple conformations per molecule** to construct the generation set, and then COV and MAT are calculated per molecule,  and finally the average / median of them on the entire test set is reported.

### Questions
- Are the conformations in nablaDFT dataset equilibrium ones or the intermediate state sampled from the optimization process? How large is the training set D0? More description about this dataset is needed. 
- ConfOpt and TorsionDiff are designed to generate equilibrium low-energy conformers, and not guaranteed to achieve a lower energy by repeatedly applied. Thus, I think it's unfair to compare these models with GOLF in terms of pct. 
- The statement of "We hypothesize that in the case of ConfOpt, the main problems are the choice of the architecture and the fact that the model generates optimal conformations from SMILES and does not use initial geometries. " doesn't make sense to me. ConfOpt takes the 2D molecular graph as input and also utilizes initial 3D conformations. 
- Does $f^{traj-10k /100k }$ keep the total number of updates equal to $5 \times 10^5$? If so, please provide more training details, otherwise, the comparison between them and $f^{GOLF}$ is unfair.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces GOLF, a framework for improving molecular conformation optimization with neural networks. GOLF addresses distribution shift issues, enhancing energy prediction and optimization. It outperforms traditional methods and reduces the need for physical simulator interactions by 50 times.

### Strengths
1. The development of the proposed GOLF is clear.
2. Demonstrated outstanding performance in conformation optimization tasks.
3. Generalization to larger molecules.

### Weaknesses
1. Dataset Limitation: The paper may be limited by the availability and diversity of datasets used for testing, potentially impacting the generalizability of the results.
2. Complexity: It seems that the complexity of GOLF is not clearly discussed in the paper.
3. Practical Implementation: Though the algorithm is not very complicated, this paper does not release code, which leaves me cautious about the practical implementation and complexity of the algorithm.

Overall, while the paper presents valuable contributions, addressing these weaknesses could enhance its overall impact and relevance in the field of molecular conformation optimization.

### Questions
Same with the 'weaknesses' part:
1. Why not use more datasets besides nablaDFT?
2. What's the complexity of GOLF? It seems that you only show experiment results on subsets of nablaDFT. Is it because the computational complexity of the method is very high?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
