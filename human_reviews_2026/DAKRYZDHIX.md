# Rethinking Federated Aggregation Under Heterogeneity: Scalabe Ensembles With Open-Set Recognition

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 4, 6, 2

## Abstract
Federated learning (FL) has gained widespread adoption as a privacy-preserving framework for distributed model training. However, it continues to face persistent challenges, most notably statistical heterogeneity and high communication cost. The current dominant paradigm in FL is consensus-driven averaging of model parameters across clients. Most recent methods, despite their innovations, remain anchored in repeated round averaging as the backbone of their design. The substantial communication overhead from repeated rounds is an obvious drawback, but another matter of debate is whether this approach can succeed under heterogeneous data, which forms the central focus of this paper. We argue that this prevailing approach fails to address heterogeneity. Using extreme label skew as a lens to expose its limitations, we demonstrate that even the most recent methods that ultimately rely on parameter averaging remain fundamentally limited in such settings. We instead advocate for an emerging alternative: ensemble-based FL with open-set recognition (OSR), which, by preserving client-specific models and selectively leveraging their strengths, directly mitigates the information loss and distortion caused by parameter averaging in heterogeneous settings. We consider this approach a principled path forward for addressing heterogeneity, substantiating our view through both theoretical analysis and extensive experiments. However, we acknowledge its primary limitation: the linear growth of ensemble size with client count, which hinders scalability. As a step forward in this direction, we introduce FedEOV, which incorporates improved negative sample generation to prevent shortcut cues, and FedEOV-pruned, which explores pruning as a solution to the scalability problem, rather than relying on distillation, thus avoiding the need for server-side data or additional training at the server. Our experiments across multiple datasets and heterogeneity settings confirm the superiority of our method, achieving an average improvement of 16.76% over the state-of-the-art ensemble baseline, FedOV, under extreme label skew and up to 102% over FedGF, the top-performing parameter averaging method. Furthermore, we show that pruned federated ensembles achieve performance on par with distilled ensembles, without any server-side data or training requirements, even when the latter is distilled using data from the same datasets. Code is available at: https://github.com/Anonymous6868-hue/FedEOV

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper aims at improving the federated learning with open-set recognition algorithm FedOV (Diao et al., 2023). The problem tackled here is to produce a good classification prediction in an FL context with extreme label heterogeneity, eg disjoint and 1 class per client.
The authors introduce FedEOV, a method similar to FedOV with an extra step introducing "shuffled-patch augmentations with smoothed transitions" to avoid learning spurious features of unknown classes.
They also provide theoretical justification of the open-set approach to tackle heterogeneity (in labels) in the FL context and test their method experimentally on standard classification datasets.
The authors also explore dimension reduction techniques like distillation and pruning to scale their method with the number of clients.

### Strengths
**Strengths**
- Good related work section and definition of open set voting in FL
- Reasonable experiments (standard datasets, several heterogeneity levels, extensions to tackle scalability issues)

### Weaknesses
**Weaknesses**
- Theorem 1 is correct, but not really informative: OSR is optimal because of assumption B4.
- Wrong theorem 2. Vector $x$ is not fixed in the problem (3) thus it cannot appear in the solution (4). 
- No proof of theorem 3 in the appendix...
- There is a lack of clarity how the two paradigms FedAvg (and thus any standard FL algorithm) and FedOV (and thus FedEOV too) differ at training and at inference. This makes the discussion a bit confusing: in FL once the model is trained and aggregated one last time on the server, it's fixed. Then no other communication cost arises. As far as I understood, communication is still required for all clients to the server for every single inference, which typically makes it impractical in case of clients dropping the pool.

### Questions
Please find rapid comments and questions below. 

**Comments**
- C1) the abstract and first 4 pages are too long. It would be better to focus on the differences between FedOV and FedEOV
- C2) (line 177 page 4) other papers tried to combine FL with pruning, especially in the case of model heterogeneity, which make sense in the context of ensemble learning as one can train locally "weak learner" and combine them on the server in a "strong learner". References: 
    - Horvath et al, FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout (2021)
    - Alam et al, FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (2022)
    - Yi et al, FedP3: Federated Personalized and Privacy-friendly Network Pruning under Model Heterogeneity (2024)
I let the authors check if the above references are relevant or not.
- C3) page 5, theorem 2, $f_c (x)_\bot$ not defined
- C4) a deeper explanation of the difference between FedOV and FedEOV is required here, especially experiments to check if FedEOV consistently brings an improvement. Or highlighting the spurious correlation learned by FedOV and not by FedEOV, visual examples are missing in this submission.
- C5) Figure 2 is not very clear, it's not sequential and present variants, which is confusing. What is the right-hand side part about? Maybe clearer to put the algorithm in the main text. Also, concerning the algorithm present in the appendix, I recommend a rewriting after drawing inspiration from FedOV algorithm that explicitly tells what is run on the clients and on the server.
- C6) Line 457-460 page 9, where is 16.76% visible? I disagree with the conclusion as it is written here. If FedEOV and FedOV perform the same in homogeneous setting, it implies the finale (3rd) stage of FedEOV dos not bring any benefit over FedOV. This does not prove that Open-Set Recognition is not effective. To conclude that, one need to compare to the baseline without OSR, eg FedAvg.

**Questions**
- Q1) in (4) there might be a mismatch in dimension as $f_c (x)$ do not necessarily belong to the same vector space for all $c$
- Q2) In the discussion after Theorem 2, it seems there is a comparison between FedOV with the assumption each $f_c (x)$ is optimal, ie trained, vs FedAvg training steps. Could you precise the discussion here to avoid confusion between training and inference stages. Also, aggregation in FL has never been a descent step. 
- Q3) Is the experimental setting following the one of FedOV ? Why limiting to 20 clients and not 80 as authors do in the later? Are FedOV results reproduced?
- Q4) Why is the number of parameters increasing with the number of clients for open-set methods?
- Q5) What is "*" symbol for in the entire paper. It's always written "FedEOV*" not "FedEOV". Is there a footnote missing?
- Q6) Is $lr=0.001$ optimal for all methods?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an algorithm to address extreme label heterogeneous data in FL setting (where the clients have a disjoint set of classes). Each client learns a classifier using its own local data and prunes the model as it does. Then shares the model with the server. Each of these models is trained to differentiate between the samples from classes that are not there in the local data. This is done by training with an extra 'unknown' class. The method augments the local samples to produce negative samples to train the 'unknown' class. The server uses the ensemble of (pruned) client models for inference.

### Strengths
1. Communication efficiency in FL is an important problem.
2. The paper explores an interesting solution using ensembles and open set recognition (if the category of a test sample was seen during training)

### Weaknesses
1. Authors should improve readability by breaking long paragraphs at logical places.
2. I'm unclear about the setting. Is each model stored on every device and used for inference? Or does the server store all the models and make the inference?
3. I think the assumptions in the appendix should be in the main document. And the theorems need to be formulated taking them into account. Otherwise, the current state of the theorems is confusing. For example, in Theorem 1 statement, it is unclear what is Z_nosr and Z_osr and a natural question is how the OSR loss function is defined.
4. The novelty of the paper is a bit concerning. The authors mentioned three contributions: i) theoretical analysis, ii) augmentation to generate negative samples, and iii) pruning models. 
	a. I feel the theoretical part needs to be more formal and concrete by bringing in material from the appendix. 
	b. The second contribution is explained as a single line 331 by only saying "shuffled patch augmentations with smoothed transitions". It is too short to be a main contribution. For example, why does this augmentation help the model learn to differentiate between seen and unseen classes?
	c. The pruning module uses existing methods in pruning.
5. The paper needs to cite previous works that it got inspiration from. For example, pruning methods.
2. Explain the pruning method used in line 342. The details are unclear - how to decide the threshold for pruning. Also, I'm unsure about: "Surviving weights are reset to their original pre-training values before training resumes" (line 344). Can you elaborate more on why this will work? The main challenge in FL + pruning comes from parameter averaging of pruned models, which is not relevant in this context. 
6. It is unclear how the ensemble predictions are combined to produce the final prediction, taking into account the OSR confidence values.
7. I'm not sure why the authors have made the case using on-shot FL, whereas the method seems to be one-shot.


Detailed comments:
1. fix citation to remove redundant string: Guha et al. Guha et al. (2019) in line 165
2. The main paper should include the details for the experiment settings regarding datasets, models, and FL setting.
3. One of the major claims of the paper is that the resulting method is efficient, but it only shows the parameter count. It will be more convincing to talk in terms of CPU cycles, peak memory usage, and GPU memory.
4. None of the results have confidence bounds.

### Questions
As above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper critiques the dominant parameter-averaging paradigm in FL, arguing that it is fundamentally flawed for settings with high statistical heterogeneity, particularly extreme label skew. Averaging-based methods may suffer from information loss and local drift, which cannot be adequately corrected. As an alternative, the paper champions an ensemble-based approach using OSR, which aggregates models in the function space, thereby preserving client-specific knowledge. The proposed methods include FedEOV, an enhanced version of the SOTA ensemble method, FedOV, using a more robust three-stage negative sample generation, and FedEOV-Pruned, which addresses the scalability limitation of ensemble methods.

### Strengths
- The core argument that parameter averaging is the wrong approach for heterogeneity and ensemble-based OSR is clear, direct, and well-motivated. It seems reasonable to me.

- Theoretical grounding is provided for the claims made.

- The empirical results look strong, demonstrating significant improvement.

### Weaknesses
- The paper dismisses the poor performance of the hybrid method FedConcat, attributing it to hyperparameter sensitivity. While this may be true, it slightly weakens the comparative analysis. A more in-depth exploration of why its clustering mechanism fails under extreme skew would be more conclusive than a note on hyperparameters.

- The 3-stage OSR training for FedEOV is a key contribution. However, there is no ablation study to show the marginal benefit of each stage. Is the performance gain primarily from Stage 3?

- In the main experiments, all models use a simple CNN with two convolutional layers and one fully connected layer. While the appendix mentions experiments with larger models, the core claims in the main paper would be strengthened if some of them could be included in the main body.

### Questions
- FedEOV-Pruned is described as using iterative pruning with reinitialization. How crucial are these specific (and more complex) "lottery ticket" style elements? How does it compare to a simpler, one-shot pruning of weights based on magnitude at the end of local training? Is the iterative process essential for maintaining accuracy at high sparsity?

- The paper focuses on the extreme skew case, so how does the performance gap between FedEOV and parameter-averaging methods (like FedGF) change as heterogeneity decreases?

- Also see weaknesses please.

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
4

### Summary
This paper presents FedEOV, an ensemble-based algorithm for one-shot FL designed for scenarios of extreme label skew where the clients own disjoint label sets. The authors argue that when the clients own and train on completely different labels, parameter averaging (e.g. FedAvg) is bound to fail due to information collapse and misalignment. Instead they propose training with open-set recognition (OSR) like in FedOV, and further introduce model pruning to compress the resulting ensemble. The paper includes some theoretical analysis on mutual information, functional aggregation, and the bounds of averaging vs ensembling error (but see W2 below). Experiments cover MNIST, FashionMNIST, SVHN, CIFAR10/100, and TinyImageNet and multiple baselines in 3 heterogeneity settings, with the proposed method coming out on top (but see W4).

### Strengths
The paper offers clear problem framing, despite dealing with an extreme edge case and requiring very strong assumptions. The experimental coverage is comprehensive encompassing multiple datasets and baselines. The limitations of previous work it attempts to address, namely the non scalability of ensembles, are very relevant and the proposed approach via pruning seems intuitive.

### Weaknesses
## 1. Limited applicability of the motivating scenario

One-shot FL can address important challenges in resource-constrained environments and large-scale model deployments. However, this paper's focus is on image classification (on toy datasets), with a 3 layer CNN, on completely disjoint label distributions, representing an extreme edge case. To strengthen the motivation, I encourage the authors to: (a) provide concrete real-world examples where image classification tasks would exhibit disjoint label sets across clients, and (b) discuss whether the theory generalizes to more common scenarios with partial label overlap. The information collapse argument would also benefit from extending beyond the single-class-per-client example to demonstrate robustness across varying degrees of label skew.

## 2. Theoretical framework requires rehauling

* In Theorem 1, two of the assumptions are mutually inconsistent A3: Deterministic Prediction on In-Distribution Labels means the network is perfect but A.4 Uniform Prediction on Out-of-Distribution Labels (No OSR) requires complete randomness in misclassification of unseen labels. For instance, if a model perfectly classifies ships and dogs (A3), it seems unlikely to uniformly misclassify airplanes across all known classes (A4), rather, airplanes would likely be consistently classified as ships due to visual similarity.
* Theorem 2: the optimality claim would be more compelling if the authors justified why the objective in Eq. (3) is the appropriate choice for federated aggregation. Currently, the theorem shows that ensembles minimize a specific objective, but doesn't establish why this objective is superior to alternatives. Connecting this to a global federated learning objective or comparing different functional aggregation objectives would strengthen this result.
* Theorem 3 is not currently contributing to the paper's argument, since it does not compare the averaging error and the ensembling error. The inequality $\mathcal{E_avg}$ > $\mathcal{E}_{ens}$ is qualitative, not predictive. Consider either deriving bounds under some specific conditions or repositioning this as an empirical observation rather than a theoretical result.

## 3. Critical evaluation methodology issue

The results presented in Tables 2, 3, 4 caught my attention: Tables 3 (Dirichlet 0.1) and 4 (IID partitions) are easy to cross reference with other literature, whereby it becomes evident the proposed method has extremely high results, e.g. FedEOV - 20 clients - CIFAR100 achieves **98.23%** with a 3-layer CNN on CIFAR100.

After reviewing the provided code, I have a significant concern with the experimental protocol. The authors do not form disjoint label sets only for the training sets, **but also the test sets**. The ```compute_ensemble_accuracy``` function then iterates over client-specific test sets (test_loaders), where each test set appears to be partitioned similarly to the training data. For extreme label skew, this means each client's model may be evaluated primarily on classes it was trained on, rather than on a global test set representing the full label distribution. This could substantially inflate reported accuracies and explain results that appear inconsistent with other benchmarks.

Currently the partitioning strategy means in many of the experimental scenarios every client has a separate test set with only 1 class, and the task for the ensemble training is binary classification (class image vs abstention token), which of course can be done nearly perfectly, while the models trained via averaging receive conflicting gradient updates that do not allow them to make meaningful progress in this task.

## 4. Incremental novelty and clarification of technical contributions needed 

The paper's main idea is the same as FedOV, so it would benefit from more clearly delineating its contributions relative to FedOV. Moving the core algorithm from the appendix to the main text with a clear walkthrough would help to this effect, along with expanding the technical description of the enhanced OSR mechanism and pruning strategy which is currently relegated to two paragraphs on page 7/9. Additionally, while the enhanced negative sample generation and pruning strategies are interesting, additional discussion would help readers understand: why harder negative samples are necessary given the relatively simple datasets used, and how the pruning insights on this 3 layer CNN would transfer to modern architectures (ResNets, Vision Transformers).

### Questions
1. Please clarify the evaluation protocol. Specifically: (i) Are test sets partitioned identically to training sets? (ii) Is the final accuracy computed over a single global test set or averaged across client-specific test sets? (iii) Can you provide results using a standard, unpartitioned test set for comparison?

### Soundness
1

### Presentation
2

### Contribution
1
