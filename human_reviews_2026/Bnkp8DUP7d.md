# OpenPhase: Condition-Aware Exploration of Multicomponent Biosystem Phase-Separating Behavior

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Liquid-liquid phase separation (LLPS) is a fundamental biophysical process
      in which biomolecules, such as proteins, DNA and RNA, demix from solution
      to form distinct liquid phases. Crucially, experimental conditions, such as
      salt, temperature, pH and concentration, profoundly influence LLPS
      properties and dynamics, often determining whether phase separation occurs
      and modulating the propensity of the resulting biomolecular condensates.
      While numerous machine learning methods have been developed for protein
      phase behavior prediction, their capabilities are frequently constrained by
      the inherent complexity of biosystems and the vast variability of environmental
      conditions. Here, we introduce \textbf{OpenPhase}, the first condition-aware
      platform for system-level exploration of biomolecular phase behavior.
      OpenPhase uniquely provides well-structured and ready-to-use datasets of
      experimentally verified phase outcomes coupled with corresponding
      conditions. We also formalize three canonical tasks (1) condition-aware phase
      outcome prediction, (2) condition inference of LLPS, and (3) conditional phase-separating
      system design. These tasks well articulate the interplay between system
      components, environmental conditions, and emergent phase properties. For each
      task, we propose novel solutions and benchmark them against strong
      baseline models. OpenPhase also includes a user-friendly API to facilitate
      \textit{in-silico} development and evaluation of novel machine learning
      methods for complex biosystem phase behavior modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces OpenPhase, a framework for modeling liquid–liquid phase separation (LLPS) in multicomponent biomolecular systems.  The authors compile datasets combining proteins, RNAs, and DNAs with detailed experimental conditions and define three tasks: predicting phase outcomes, inferring conditions, and designing phase-separating systems.  They propose specialized models for each task, including a transformer (Condformer), a conditional VAE, and a diffusion-based generator.  OpenPhase achieves strong results across benchmarks and provides the first condition-aware platform for predictive and generative modeling of LLPS.

### Strengths
This paper addresses the problem of liquid–liquid phase separation in biomolecular systems.  
The authors provide well-curated, publicly available datasets that integrate experimental conditions, which is a valuable contribution to the community.  
While the proposed models are relatively simple, they serve as a solid baseline and demonstrate practical utility for condition-aware phase behavior modeling and protein design.

### Weaknesses
1. The proposed gated transformer architecture is not particularly novel and closely follows existing conditional transformer designs. The gating mechanism is a straightforward modification commonly used in multimodal and conditional attention frameworks.

2. All the trained models rely on standard, well-established architectures rather than introducing new methodological ideas. The Condformer is a basic transformer variant, the branched cVAE is a conventional conditional autoencoder with a classification branch, and the diffusion model directly builds upon prior protein generative models like PRO-LDM and MapDiff with minimal adaptation. Their main contribution lies in applying these existing methods to the new dataset rather than advancing model design or theory.

### Questions
1. How reliable are the experimental annotations of LLPS, and is there a consistent standard for defining phase separation across datasets?  
2. How well does success on these datasets correlate with experimental validation in the lab?  
3. Given the variability of LLPS assays, how noisy are the datasets, and how accurate are the underlying experimental measurements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new dataset for liquid-liquid phase separation (LLPS) and architecture for conditional generation of phase outcome. The dataset may be beneficial to practitioners in biological machine learning space and the architecture shows promise in performance, but serious presentation and motivation issues make me doubt this promise at this point. Should the authors be able to help clarify these issues, then this paper may be beneficial to the field.

### Strengths
The paper builds a new dataset that can be used for biocondensate discovery. Additionally, the paper proposes a novel conditional VAE architecture that uses a transformer, as well as latent diffusion, to guide phase outcome prediction. The results seem to perform better than alternative architectures on a benchmark of conditon-aware phase outcome predictions. The authors demonstrate the ability to predict experimental conditions in section 5.2. Finally, they briefly show the benefit of protein structure design backtracked from the LLPS architecture and what I'm assuming is the protein embedding, which is quite interesting but left for a small section.

### Weaknesses
**Overview**

The paper is replete with grammatical, spelling, and exposition issues, indicating a rushed submission. The methodology section first introduces the new architecture to perform inference. It's hard to follow what aspects of the system are being explained where n the first part. For example, gating is introduced but it's not clear where that's used in the architecture in Figure 1. Also, it's not clear why the section is split up into three tasks in the first place. A motivating sentence or paragraph would greatly help orient the reader.

**Methods**

The new dataset construction is confusing. I'm assuming it's an aggregate of db1, db2, and db3? The authors state that db2 comes from manually collected LLPS records, but what are those and how can we trust their veracity? It seems that their new dataset is mainly the new 'db2' dataset whereas the other two are from previous studies, so the naming convention is awkward that the newest one is db2.

Task 2 in section 4 is straightforward but there is no explanation of which hyperparameters were used for their final loss function, which seems important to recreate their work.

Task 3 is the most opaque and confusing section. The purpose was briefly mentioned in the introduction and summary of the paper. I don't exactly see the benefit of training latent diffusion on the model, nor the purpose of training both a conditional and unconditional model.

**Experiments**

Again, a guiding paragraph would immensely help. Also, all experimental results would instill more confidence should they have standard error calculations. I see they show the mean value.

Why are the metrics for first experiment based on 50/50 train/test split? This seems arbitrary to me. 

The second section evaluates MSE and $r^2$ metrics within a condition with the model. The $r^2$ seems poor for all of the conditions and I'm not sure how well Table 5 helps us in determining the accuracy of this method. Figure 3 states "accepted" is in green whereas I see blue. 

For section 5.3, I think more analysis should be done. The authors show droplet-forming regions, but how do they quantify that? What chemical/physical features support those predictions? these look like interesting proteins but how do we know they are relevant and if the predictions are accurate?

Finally, the authors address accuracy of their method but I highly encourage they also evaluate the calibration of their model. Having a measure of uncertainty in predictions would be very beneficial for this task.

### Questions
Many questions are posed in the weaknesses section. Here are remaining questions that I have:
- What is the difference in the few-shot and zerio-shot split in section 5.1?
- Can they show other phase diagrams that are unique? They should be able to do this using the cVAE and more evaluation of the phase plots would help show the benefit of the system.
- What is "FuzDrop"?

### Soundness
1

### Presentation
1

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
This paper proposes an integrated computational framework for analyzing and designing biomolecular phase separation phenomena from multiple perspectives. The authors define three distinct tasks related to understanding and controlling phase separation and develop new machine learning models for each. In Task 1, they construct a model that predicts whether phase separation occurs based on molecular features and experimental conditions, achieving higher predictive accuracy than existing approaches. In Task 2, they formulate the inverse problem of inferring experimental conditions from molecular features and observed phase separation results, proposing an inference model effective for condition optimization and mechanistic understanding. Furthermore, in Task 3, they present a model that generates protein sequences capable of realizing a desired phase separation behavior under specified experimental conditions, demonstrating its potential as a novel molecular design approach. These three tasks are interrelated and collectively constitute a comprehensive framework that supports the prediction, analysis, and design of biomolecular phase separation.

### Strengths
- This study presents an integrated machine learning framework for understanding biomolecular phase separation, a fundamental phenomenon underlying cellular function. By providing a data-driven approach to analyze, predict, and design complex biological processes, the framework offers a novel computational approach with significant implications for the life sciences field.

- The authors integrated three existing databases to construct a data infrastructure that enables the application of machine learning to diverse problems related to phase separation. This dataset has the potential to serve as a standard benchmark for future researchers, fostering the development of new methods in molecular design and condition optimization.

- A distinctive feature of this paper is the systematic formulation of biomolecular phase separation into three tasks—prediction, inverse inference, and generation. This framework enables a quantitative and computational treatment of phenomena that were previously discussed only qualitatively, thereby providing a foundation for future model development and theoretical studies.

### Weaknesses
- This paper is heavily oriented toward the life sciences, focusing on biomolecular phase separation, and is therefore not well suited for a machine learning conference such as ICLR. The primary contributions lie in dataset construction and problem formulation, while the novelty and theoretical advancement in machine learning methodology are limited. Hence, this work would be more appropriately evaluated in a life science–focused journal.

- Tasks 1 (phase separation prediction) and 3 (protein generation) address general problems that have already been extensively studied, yet the paper compares its approach with only a subset of existing methods. The lack of systematic benchmarking against the latest state-of-the-art techniques makes it difficult to assess whether the proposed models truly outperform existing approaches.

- The formulation of Task 3 appears highly dependent on the specific dataset used in this study, raising concerns about its general applicability. The assumption that proteins can be generated solely from phase separation phenomena and experimental conditions seems biologically unrealistic, and the paper does not clearly demonstrate whether the proposed framework can generalize to other molecular systems or experimental contexts.

### Questions
- The number of proteins included in the dataset constructed in this paper seems considerably smaller compared to those in recent protein foundation model databases. Would it not be possible to build a larger database? Also, is there a risk that this database is specialized for certain types of proteins and therefore lacks generality?

- Task 1 is essentially a binary classification problem based on biomolecular features, for which various approaches have already been proposed. Why is it necessary to develop a new method here? In particular, does the proposed approach incorporate any mechanisms that are specifically designed for modeling phase separation phenomena?

- In Task 3, the claim that proteins can be generated from phase diagrams and experimental conditions seems to lack generality. Could it be that the method is applicable only to specific types of proteins or experimental setups, and therefore not broadly generalizable? It would be valuable to include validation experiments using entirely different types of proteins. Would such experiments be feasible?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors consider a complex study of Liquid–Liquid Phase Separation (LLPS) in biophysical processes. During such a process, molecules such as proteins, DNA, and RNA start forming little clusters together and separate from their solutions. While remaining a liquid, the systems have distinct properties and dynamics compared to individual systems in solution. Notably, LLPS properties and dynamics are also significantly impacted by experimental conditions such as temperature and pH. All of this together makes LLPS a challenging area of study that has so far been largely overlooked.

Within the context of LLPS, the authors consider three major tasks: 1) Condition-aware phase outcome prediction, 2) Condition inference for phase systems, and 3) Phase system design. For all three problems, the authors discuss a concrete architecture and training objective to train the models to solve the tasks.

In the case of condition-aware phase outcome prediction, the authors train a large transformer-based embedding model which, combined with a simple classification or regression head, can predict the phase.

For condition inference, the authors propose training a VAE that encodes both the system and the experimental conditions into a latent space. Inference is then done by thresholding a reconstruction loss and a branched phase loss. This is a rather unusual construction in my opinion, and I have commented on it below in the weaknesses section.

Lastly, for phase-system design, the authors propose using a diffusion process, with both conditional and unconditional variants. This section, however, requires some further clarification, as the different approaches considered are not entirely clear. There is mention of two different design pipelines in Figure 2, but only one seems to be discussed in the text.

To train their models, the authors combine three small datasets, referred to as db1, db2, and db3, to form the “OpenPhase” dataset. The dataset is accompanied by a larger development framework that also includes access to methods for component and condition embeddings, predefined dataset splits, and a user-friendly interface.

Lastly, the authors evaluate their proposed approaches for the different tasks. Across all three tasks, the authors highlight that their proposed methods perform well but ultimately have limited comparisons against other methods.

### Strengths
The presented work covers an interesting and somewhat overlooked application domain in the form of Liquid–Liquid Phase Separation, and within this context the work proposes possible solutions for a wide range of tasks. The additional contribution of the combined dataset also provides a service to the community by simplifying future research. Furthermore, the paper is well written and clearly motivated.

### Weaknesses
While interesting and well executed, I unfortunately find the paper to lack sufficient contribution and novelty for it to be accepted at the conference at this time. As stated, while I believe that the OpenPhase framework provides an important service to the community, due to primarily combining existing datasets the novelty is limited. Furthermore, while extensive in considering three different tasks within the LLPS domain, the proposed solutions have limited novelty. They primarily adjust existing methods from other domains to the specific problems at hand. While this is in itself interesting, I personally believe this to be more appropriate for a domain-specific journal/conference as opposed to ICLR.

It is for this reason that I believe the paper is not ready for acceptance to ICLR. However, I am open to hearing from the authors if they believe that I have missed or misunderstood important parts of their contribution. Outside of the significance and novelty issues, I would be happy to accept the paper, as the work is otherwise well executed.

I have listed a few more points of weakness below, but I believe these can most likely all be addressed in small updates during the rebuttal and/or in the process of producing the camera-ready paper:
- While the Liquid–Liquid Phase Separation domain that the work tackles is very interesting, the discussion of how it is fundamentally different from more conventionally studied domains needs further clarification. For a non-expert audience not intimately familiar with the biochemistry domain (i.e., the general ICLR audience), this is currently hard to follow. A clear illustration would benefit the paper a lot.
- The paper has a few minor mistakes that are mostly the result of last-minute changes. These need to be flushed out in future versions. E.g., a small error on line 208, a weird overrun from the inline graphics on line 445, and, in general, some large overrun into the margins in Table 5.
- Am I correct in understanding that the second task, condition inference, requires an exhaustive search over all possible c? If this is the case, I suspect this to be quite computationally expensive, and this should be clearly discussed in the paper.
- While Conformer seems to provide a significant improvement over the two embedding methods it is compared to, it is unclear what computational cost this entails. It would be interesting if the authors could include a short study on this.
- It is unclear how the different dataset splits are used. My assumption was that the different categories of splits would be used to make sure specific components are equally represented, but given the stated 50–50 train–test split, this does not seem to be the case.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1
