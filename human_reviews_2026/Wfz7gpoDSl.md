# RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Federated Learning (FL) has gained prominence in machine learning applications across critical domains, offering collaborative model training without centralized data aggregation. However, FL frameworks that protect privacy often sacrifice fairness and reliability; differential privacy reduces data leakage but hides sensitive attributes needed for bias correction, worsening performance gaps across demographic groups. This work explores the trade-off between privacy and fairness in FL-based object detection and introduces RESFL, an integrated solution optimizing both. RESFL incorporates adversarial privacy disentanglement and uncertainty-guided fairness-aware aggregation. The adversarial component uses a gradient reversal layer to remove sensitive attributes, reducing privacy risks while maintaining fairness. The uncertainty-aware aggregation employs an evidential neural network to weight client updates adaptively, prioritizing contributions with lower fairness disparities and higher confidence. This ensures robust and equitable FL model updates. We demonstrate the effectiveness of RESFL in high-stakes autonomous vehicle scenarios, where it achieves high mAP on FACET and CARLA, reduces membership-inference attack success by 37%, reduces equality-of-opportunity gap by 17% relative to the FedAvg baseline, and maintains superior adversarial robustness. However, RESFL is inherently domain-agnostic and thus applicable to a broad range of application domains beyond autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RESFL, a method for privacy-preserving and fair object detection in federated learning (FL). The main idea is to optimize a composite loss on client-side, with separate terms for utility, privacy (essentially minimizing mutual information between the feature extractor output and sensitive features), and fairness (minimizing group-wise disparity in uncertainty). On the server-side, the proposed method uses averaging using weights derived from the client-specific uncertainty values. The proposed method is compared to several baselines (mostly) on FACET and CARLA datasets on several metrics: utility (mean average precision), privacy (membership inference & attribute inference attacks, MIAs, AIAs, respectively), fairness (disparate impact, equality of opportunity), and robustness (poisoning).

### Strengths
* Empirical results seem quite nice (although see more detailed comments on this also under Weaknesses & Questions)
* Focus on object detection seems interesting,: as far as I know, this direction has not been too much explored under privacy and fairness constraints.
* Using uncertainty gap for fairness mitigation is, to my knowledge, a novel contribution.

### Weaknesses
1) The writing could be clearly improved; eg, the main learning problem as well as several acronyms are only introduced in the appendix (or not introduced at all), some notations are used in more than one sense, there is no explanation why the performance metrics include robustness to poisoning when the stated objective is to guarantee privacy and fairness, etc.

2) Much of the experimental details needed to make sense of the results are missing (see Questions for details).

3) There might be some implementation problems at least for some of the baselines in the provided code, although it is hard to be certain as the scripts used for actually generating the results are missing (see Questions for details).

4) There are plenty of inaccurate or unsubstantiated claims in the text (see Questions for details).

### Questions
## Questions and suggested changes, in decreasing order of importance:

1) Please provide sufficient experimental details to evaluate the results: clarify how all hypers are tuned (eg all grids, do you do separate tuning for each method including the baselines), add some error bars to the results, clarify why recall is the one chosen metric for calculating DI/EOP or does this make a difference, clarify what does it mean in practice to use per-client training time of 8 hours (lines 1137-38), does this imply that some methods are terminated before hitting the same number of total training rounds as others; clarify how model finetuning is done (eg, is this all params, last layer only or something else), clarify how much heterogeneity the chosen Dirichlet allocation produces in the non-IID setup (and why do all clients have the same number of samples even in non-IID setting?).

2) Clarify how exactly MIA and AIA are done (eg, do you assume the adversary has access to some data from the same distribution or how do you train a model to predict sensitive attribute values from the model updates; following Carlini et al. 2022: MIA from first principles arguments, MIA is typically reported as TPR vs FPR, can you show these in addition to just the success rate; if the adversary can train shadow models, why not use eg standard LiRA attack from the Carlini et al. paper with loss values; how many shadow models do you actually use)

3) Which $(\varepsilon,\delta)$ values are used for each of the DP baselines, why are these specific values chosen, and how is the privacy accounting done for each of the methods (at least some parts of the code seems to use the classical Gaussian mechanism noise scaling that can be very loose and has limitations on possible $\epsilon$ values)?

4) For several of the baselines (eg, PUFFLE, FairFed) you need to choose a specific fairness metric to optimize. Which one do you use?

5) Looking at the gradient_dp_custom_trainer.py code, it looks like you might not be clipping per-example grads but the aggregate. Can you clarify if this is so for the DP baselines, and state the corresponding threat model explicitly (also in the paper: state the DP adjacency and granularity you use for the baselines).

6) Some of the provided code (CHFL_train.py) seem to do eg quantization before FedAvg, and client clustering functions, which are not mentioned anywhere in the paper. Can you clarify if these are used somewhere in training, and add all the relevant training details at least in the Appendix.

7) Considering the jump from Thm B.1 & Cor B.2 (which do not explicitly consider FL setting) to the claims on lines 165-171: can you write this step to FL out more formally to make it clear what do you actually claim about the connection of locally optimizing UFM to reducing global DI/EOP gap (ie, move from local optimization to global fairness)?

8) On finetuning with local data: do the DP baselines do DP also when finetuning with local data?

9) Eq.(6) and lines 185-86: is there something that prevents giving high weight to a client with equally low confidence for all groups?

10) Why is robustness one of the performance metrics (the paper says next to nothing about robustness on any level before the experiments)?

11) Lines 889-92: "We compute $UFM_i$ per client on a held-out local validation split..." please clarify how this splitting is done (do you assume a separate dataset, split the local training data for the proposed method or something else).

12) Lines 897-98: "Model updates $\Delta \theta_i$ continue to use secure aggregation..." do you actually use secure aggregation for something?

13) Do you apply same augmentation for non-DP and DP methods, and how do you do clipping with augmentations (see eg De et al. 2022: Unlocking high accuracy... for a good discussion on how augmentations affect DPSGD, and how it should be implemented)?
### Smaller issues (no pressing need to comment on these, ok to just clarify in the paper when appropriate)

14) Eq.(2): does $\sim$ mean approximately here? Is the lhs otherwise exact epistemic variance or is that also an approximation?
15) Lines 83-90: we know from Dinur & Nissim 2003 that there is an inescapable privacy-utility trade-off; any method (not just DP) that guarantees non-trivial privacy needs to compromise on the utility. DP and secure computation are orthogonal, not alternative approaches as they have very different goals.
16) In several places, eg, lines 647-48 it is claimed in so many words that FL maintains privacy. If this were true, we would not need something like the proposed method but could just use vanilla FedAvg.
17) Lines 48-49: "Quantifying both epistemic and aleatoric uncertainty is therefore essential to ensure equitable reliability across demographic cohorts". Do you actually try to quantify aleatoric uncertainty?
18) Lines 140-141: does the regularization provably lead to calibrated uncertainty estimates, or is this more empirical observation?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RESFL, an FL training framework designed to simultaneously improve three model properties: utility (defined as downstream performance), fairness (defined as group fairness across sensitive attributes e.g. skin tone) and privacy against inference attacks. To mitigate sensitive attribute leakage, the authors propose adding an auxiliary attribute classifier to the client's representation. The classifier is trained to predict the sensitive attribute and the reverse of its gradient is added to the feature extractor, with the net effect of the extractor learning to hide the sensitive attribute so that attribute-inference attacks see a random signal in the representation. The guarantees stated are not in terms of DP but stems from reducing the MI between the representation and the sensitive attribute. The strength of this effect is controlled by a hyper-parameter of the corresponding loss component $\lambda_{priv}$. Simultaneously, to address sensitive confounding factors such as skin tone leading to higher false-negatives rates in difficult cases (e.g. driving at night), the authors employ evidential learning to boost contributions from clients with lower inter-group disparities. This is done via a Dirichlet head, the outputs of which are shaped into per-client “Uncertainty Fairness Metric” (UFMs) that then modify the aggregation weights via a temperature-scaled exponential.

The proposed method is benchmarked on two datasets, FACET which tests contains real-world images with instance annotations for people (and skin tone labels) and is meant to test algorithmic fairness, and a simulated dataset (produced using CARLA) containing self-driving ego-camera frames, which contain at least one pedestrian. Metrics reported include mAP, two fairness metrics, and inference attacks success rate. The presented results show improvement across the board, and the authors claim domain-agnosticity.

### Strengths
The paper addresses an important problem in FL, jointly managing performance, fairness, and privacy. The proposed mechanisms are cohesive, practical, well-motivated, and appear empirically effective, thus forming a principled technical approach. The experimental design is thorough, including multiple baselines, attack scenarios (MIA, AIA, Byzantine, data poisoning), robustness tests under weather variations, and ablation studies. There is great experimental rigour and breadth (albeit a lot of it buried in the appendix) that the authors have the right to be proud of. The paper is generally well-written, easy to read and with good pacing.

### Weaknesses
1. Requirement for sensitive attribute annotations. The method presented requires sensitive attribute labels which are hard to procure. I can't see how either of the two components would function without them. The authors seem aware of this limitation and hint towards potential avenues to ablate this requirement, but currently this is a significant limitation. See also Q1.

2. There is a lot of very relevant detail in various parts of the appendix. Were I to not look at it, my score would be lower. For example the main paper results are limited to only 4 IID clients. On the other hand, it is my opinion figures 2 and 3 do little to advance the paper. Perhaps consider moving parts of appendix F to the main paper.

2. **$\beta$ sweep**:  Could you add an ablation where $\beta$ is swept (including $\beta=0$) to quantify the marginal contribution of UFM-guided aggregation independent of the local uncertainty loss? There is brief discussion a value of 2 was chosen after a grid search but no results to gauge sensitivity to this parameter.

3. Computational overhead is listed as a defficiency of relevant work, yet there is no mention of the overhead RESFL occurs.

Misc

- DI/EOP are not introduced in line 168.
- Epistemic uncertainty in the context of FL has been previously proposed in other works (e.g. **FedEvi** https://papers.miccai.org/miccai-2024/paper/2717_paper.pdf). I believe it would be relevant to include these works in the related work discussion.

I will defer to other reviewers with respect to details about the attack models and related claims as my expertise on this specific topic is limited.

### Questions
Q1. Attribute availability: In which application domains do you expect client-side sensitive labels to be available? Could you elaborate on the future directions (attribute-free proxies, vacuity/dissonance) and how partial label scenarios (noisy/missing S) would degrade performance? See also W1

1. Can the authors justify the evaluation set construction for the CARLA-produced dataset? It seems counter-intuitive to include as many clear weather frames as with full adverse weather intensities.

2. Have the authors conducted a sensitivity analysis over the arbitrary choices of T=100 communication rounds and E=1 local epochs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work targets the challenge of achieving privacy, group fairness, and utility simultaneously in federated object detection. Existing approaches often rely on differential privacy noise that degrades accuracy or fairness, or require the server to access sensitive attributes for fairness optimization, and rarely exploit uncertainty as a signal for aggregation. To overcome these limitations, this work propose RESFL: a framework that (i) suppresses sensitive attribute information in client representations via adversarial privacy disentanglement with gradient reversal, and (ii) estimates epistemic uncertainty using an evidential (Dirichlet) head, from which a Uncertainty Fairness Metric (UFM) measures cross-group uncertainty disparity. The server then performs fairness-aware aggregation by giving larger weights to updates with lower UFM, without accessing any sensitive labels centrally. This design reduces privacy leakage at the representation level and aligns cross-group uncertainty at the aggregation level, jointly improving privacy, fairness, and utility. Experiments on FACET and CARLA show improved equality-of-opportunity fairness and reduced membership/attribute inference risk, while maintaining competitive detection accuracy; ablations and robustness tests further support the method.

### Strengths
1. This work tackles a fundamental and highly challenging issue in Federated Learning (FL): the trilemma among privacy protection, model fairness, and model utility. In standard FL, enhancing privacy (e.g., via Differential Privacy) often degrades both fairness and accuracy. This work aims to jointly optimize all three, addressing a key open problem in the emerging field of Responsible AI.
    
2. The proposed RESFL framework demonstrates strong innovation. It integrates two advanced components: (1) adversarial privacy disentanglement, using a Gradient Reversal Layer (GRL) to remove sensitive attribute information from local representations; and (2) uncertainty-guided, fairness-aware aggregation, which employs an Evidential Neural Network (ENN) to quantify model uncertainty and introduces a novel Uncertainty Fairness Metric (UFM). The server uses UFM to weight client updates, prioritizing those that are both “fairer” (i.e., have smaller inter-group uncertainty gaps) and “more confident.” This design allows the server to promote global fairness without ever accessing sensitive data.
    
3. The experimental design is comprehensive and well-motivated. The authors evaluate RESFL in the high-stakes autonomous driving (AV) domain using two challenging datasets: FACET (real-world demographic data) and CARLA (a simulator with environmental variations). The evaluation includes not only standard utility metrics (mAP) but also fairness measures (DI, EOP gap) and privacy metrics (MIA, AIA attack success rates). Importantly, robustness is tested under adverse weather conditions (cloud, rain, fog)—a crucial aspect for AV systems. Additional ablation studies and cross-domain evaluations (Adult, TweetEval) in the appendix further support the method’s effectiveness and generalization ability.

### Weaknesses
1. Computational overhead at clients is considerably higher than standard FL. Each client must train not only a backbone detector (YOLOv8) but also an adversarial classifier (GRL) and an evidential head (ENN). This “three-in-one” setup is far more complex than FedAvg. Although Appendix F reports training times (~8 hours on RTX 3070), such overhead may be impractical for resource-constrained edge devices (e.g., vehicles or mobile units). The paper lacks discussion on scalability—how the framework behaves when the number of clients scales to hundreds or when hardware heterogeneity increases.
    
2. The framework assumes access to sensitive attribute labels (e.g., skin tone) at each client. The GRL-based privacy module and UFM computation both rely on these labels for grouping local data. While the authors stress that these labels remain local and are never shared—technically compliant with FL privacy principles—many real-world applications (e.g., healthcare, finance) legally or ethically prohibit collecting or using sensitive attributes. The paper lists this as future work, but it remains a major limitation for real-world deployment.
    
3. The server relies on a single scalar (UFM) reported by each client to determine its fairness contribution and aggregation weight. Compressing complex fairness characteristics (possibly across multiple sensitive groups) into a single number inevitably loses nuance. Moreover, the approach assumes honest reporting: a malicious client could falsify a very low UFM to gain disproportionate aggregation weight. The paper does not discuss any mechanism for verifying UFM authenticity, leaving a potential security vulnerability in untrusted FL environments.

### Questions
Can UFM be computed without sensitive labels? Please (i) clarify feasibility, (ii) add a robustness study for no/partial/noisy labels, and (iii) discuss proxy/unsupervised alternatives and their privacy implications. If not feasible by design, state this limitation explicitly.

### Soundness
2

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
2

### Summary
This paper proposes RESFL, a composite approach in federated learning to balance privacy, fairness, and performance (utility). The proposed approach replaces softmax with an evidential head that yields calibrated epistemic uncertainty and uses a scale-invariant Uncertainty Fairness Metric (UFM) for aggregation. Evaluations, on two benchmarks, FACET and Carla, use mAP, |1-DI|, delta EOP, and membership/attribute inference success rate to show utility fairness and privacy. The results show that RESFL can maintain or gracefully degrade utility while preserving privacy and fairness.

### Strengths
+ Evidential uncertainty modeling neatly fits in the framework
+ Joint optimization over fairness and privacy
+ Convincing results show tunable knobs and the intended tradeoff

### Weaknesses
- Figure layout could use some polishing. Both Figure 1 and Figure 2 are placed far away from where they were mentioned.

### Questions
In the ablation study, when fixing the privacy weight and adjusting the fairness weight, a lower fairness weight results in a higher mAP, which is a bit counterintuitive. It would be helpful to give some insight into this. 

Figure 2 has a lot of information, which may warrant a bit more detail in text descriptions. For example, why does the privacy attack success rate and fairness score dramatically increase in the fog situation as the intensity increases?

Minor, it would be nice to add an arrow after each metric indicating the lower the better or the higher the better.

### Soundness
3

### Presentation
2

### Contribution
3
