# ReactionLLM: Data-Centric  Learning for a Compact Single-step Chemical reaction prediction model

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Chemical reaction prediction faces three fundamental challenges that limit practical deployment: (1) ineffective molecular representations that fail to capture essential chemical context, (2) unfair comparison with test-time augmentation and without mention the usage of AAM(atom-atom mapping, and (3) unsatisfactory performance of large-scale pretrained models. To address these limitations, we present a unified framework that enables a compact 0.5B parameter model to outperform significantly larger counterparts (7B/13B parameters) through three strategic innovations: the AAM-0 molecular representation that bridges mapped and unmapped data via implicit contrastive learning; bidirectional multi-task learning that creates a unified chemical representation space across retrosynthesis and forward prediction tasks; and structured plan-based reasoning that ensures chemically plausible step-by-step rationalizations. Extensive evaluation with rigorous separate assessment of mapped and unmapped performance demonstrates +14\% accuracy improvement over strong baselines, establishing that carefully designed compact models with built-in chemical intelligence can surpass larger, less specialized alternatives while maintaining computational efficiency.The implementation is available at: \url{https://anonymous.4open.science/r/ReactionLLM-DF4C}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ReactionLLM, a fine-tuned language model (based off the Qwen2.5-0.5B base model), for reaction prediction and one-step retrosynthesis. ReactionLLM is trained on multiple reaction-related tasks jointly: (1) forward reaction prediction, (2) one-step retrosynthesis, and (3) one-step retrosynthesis with the reaction type known. This training is done in multiple stages; first, a general set of weights is trained, before specific weights are found for each task using LoRA (Hu et al., 2022). The authors additionally train their method on plan-based reasoning traces to improve performance as well as introducing an “AAM-0” representation of the same reaction (where the integers corresponding to atom mapped integers in the SMILES strings have all been set to zero). 

The authors evaluate their model on the standard USPTO forward and reverse reaction tasks. ReactionLLM shows very strong performance against previous baselines (69.5% top-1 accuracy on USPTO-50k without atom maps, compared to ~55–58% for the strongest relevant baselines). One-step backwards prediction performance is broken down across reaction classes (Figure 2) and ablations into aspects of the model are presented (table 3)

> Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR (2022)

### Strengths
### S1 Multi-task (bidirectional) training interesting for creating a general purpose model 
ReactionLLM is able to do both forward and two kinds of backward (with and without reaction class) reaction prediction. This enables the model to both share information between these tasks (which seems to result in better overall performance) but also simplifies the deployment of such a model. Although this has been done in an encoder-decoder framework before (see below), this seems a useful avenue to explore in decoder-only models.

> Lu, Jieyu, and Yingkai Zhang. "Unified deep learning model for multitask reaction predictions with explanation." Journal of chemical information and modeling 62.6 (2022): 1376-1387.

Likewise, adding atom-mapped reaction data into the training pipeline seems an interesting way to add additional supervision to the models during training (for instance, in forward prediction this can tell the model not only what products occurred, but also where the reaction happened, and potentially even some sense of how it happened).


### S2 Very strong performance
The results suggest much better performance than previous approaches (e.g., 14% improvement in top-1 accuracy on USPTO-50k). Putting aside some concerns I have with the evaluation for now (see W1 below), this seems to be a significant and large improvement over previous approaches (1.2x), which seemed to be reaching somewhat of a plateau.

### Weaknesses
### W1 Possible data leakage?
ReactionLLM is evaluated on the USPTO dataset, which is public. Given that this model is based off of Qwen2.5-0.5B, it is possible that the original base model might have seen this data in training, making a comparison to the existing baseline methods (which are trained from scratch) unfair? I’m particularly worried about this as it seems ReactionLLM performs substantially better than the previous methods even in the ablation study without the multi-task or other modifications. (See also Q3).

This is the main reason I have gone with a lower soundness score for now (and contribution score -- as see this also dependent on soundness to an extent). Happy to change if addressed in rebuttal.

### W2 Paper could be clearer in places
Although I felt the authors did a good job compactly summarizing previous work and outlining at a very high level their main contributions, I found parts of the paper hard to understand, particularly around how these method developments were implemented in practice. Specific examples include:    
i. I was not actually sure how the AAM-0 method discussed in Section 3.3 ensures the contrastive objective in Proposition 3.1 is met? My understanding is that the training on S, f(S) pairs is done by creating independent, augmented examples, rather than using an explicit contrastive loss? Is it then not also similar to other augmentation schemes? (see also Q1 + Q2 below).    
ii. How are the ground truth planning mechanisms obtained for training (i.e., the $z^\ast$ under the expectation in Equation 4)?    
iii. How is the top-k sampling from the model done? (Line 474, discusses how the top-1 prediction is sampled greedily from the model with no temperature, but how are the other top-k members sampled/picked?). 


Aside from these more general issues there were also sentences/parts which I found confusing, such as:
* Line 194, $\Sigma^\ast \to \Sigma^\ast$ what does this notation mean?
* Line 231& 236: seems to be repeated sentences but for different parts? (i.e., T1 and T3 vs T1 and T2).
* Line 238: “The multi-task objective learns a latent representation $\phi(x)$ whose topology reflects the true reaction manifold.” What does a topology reflecting the “true reaction manifold" mean?
* What was the parameter count of the final model (with the LoRA adapters) and how does this compare to the baselines? (Ideally it would be nice to have this as an extra column in Table 1). 
* What does “contrasting with AAM” mean in Table 3?
* Line 431: “Also is datasets is unbalanced across different class.” Was unclear what was meant here?

### Questions
Q1. What is the difference between the last two lines in table 1? The checkmarks indicate that the second from last line did not use any atom maps, but Section 4.3 suggests that this is only during inference? Does this mean maps were still used during training? 

Q2. What is the intuition behind “AAM-0” doing so much better than “Raw SMILES” for the ablation study in table 3? To make sure I understand correctly, this means training and evaluating on molecule encoded similar to `[CH3:0][CH2:0][OH:0]` rather than `[CH3][CH2][OH]`? 

Q3. When relevant, how do you calculate the correct atom maps? Are these taken from the original USPTO data and shuffled or computed anew? (The ones in the original dataset have been shown to contain information about the reaction center and so I just wanted to double check that this information had been removed here – see BEST PRACTICE S5 of Maziarz, Krzysztof, et al. "Re-evaluating retrosynthesis algorithms with syntheseus." (2025))

Q4. Have you evaluated the planning/reasoning traces? Do these look reasonable?

Q5. Did you try augmentation of SMILES too, to see if it further helped your model? (Or alternatively, removed this augmentation from the relevant baselines, to see how they performed without this additional improvement).

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates strategies for adapting large language models (LLMs) to chemical reaction prediction, covering both forward and backward (retrosynthetic) directions. The proposed approach involves the following key steps:

(1) Leveraging atom-mapped SMILES for contrastive learning to enhance molecular representations;

(2) Jointly training the model on both forward and backward reaction prediction tasks to encourage bidirectional understanding;

(3) Applying parameter-efficient fine-tuning techniques (such as LoRA) to adapt foundational models across different reaction tasks; 

(4) Designing prompt templates to promote multi-step reasoning during reaction generation and prediction.

### Strengths
The presentation provides a clear explanation of the motivation and the proposed idea.

### Weaknesses
(1) **The paper lacks sufficient detail in articulating its motivations.** Many arguments are presented at an abstract level without concrete evidence or examples. For instance, the introduction repeatedly mentions “ineffective molecular representations,” yet provides neither specific examples nor references to support this claim, nor an explanation of why atom mapping is considered central to addressing it. Additionally, the three challenges and three limitations are discussed separately, even though they are closely related. This section would benefit from clearer organization that explicitly links each challenge to its corresponding limitation. Overall, the motivation in the introduction currently appears scattered and lacks strong, conclusive arguments to guide the reader;

(2) **The proposed workflow shows limited novelty from an algorithmic perspective.** Joint task training has already been explored in several prior works, such as “Towards Understanding Retrosynthesis by Energy-Based Models” (NeurIPS 2021), while multi-task learning has been applied in Reaction-T5 and related models. Overall, this work primarily integrates existing techniques with a different base model, which, while technically sound, does not offer substantial methodological innovation.

(3) **Missing related work and fair comparison.** The empirical and literature coverage is incomplete. As this paper focuses on applying LLMs to chemical reaction prediction, it should engage more comprehensively with existing studies that have systematically explored this topic. Only a few relevant works [3,4,5] are cited, but there appear to be many other important references missing from both the discussion and the experimental comparisons. Furthermore, LLM-based reaction prediction models should be compared primarily with their LLM counterparts rather than traditional models trained from scratch. This is particularly important because LLMs may have already encountered both training and test reactions during pretraining, as such datasets are often derived from patent literature—a known source of potential data leakage. Additionally, LLMs inherently possess much larger parameter counts and model capacities, which further complicates fair comparison.

In conclusion, this reviewer finds that the paper is still incomplete and lacks sufficient consistency for publication at this stage.

References:

[1] Towards understanding retrosynthesis by energy-based models [NeurIPS 2021]

[2] ReactionT5: a large-scale pre-trained model towards application of limited reaction data 

[3] Augmenting large language models with chemistry tools [Nature Machine Intelligence]

[4] What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks [NeurIPS 2023]

[5] A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists [Nature Chemistry]

### Questions
Questions:

(1) Have you used atom-mapping information during inference stage of retrosynthesis?

(2) Have you fine-tuned the 7B model on chemical reactions? May I ask why there are many unreported numbers for the 7B model?

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
4

### Summary
The paper describes a unified framework for reaction prediction. It is based on three technical components: contrastive learning on mapped and unmapped data, bi-directional multi-task learning for forward reaction prediction and retrosynthetic analysis, and structured plan-based reasoning. The resulting model is compact and demonstrates 14% accuracy improvement over baselines.

### Strengths
AAM-0 contrastive learning appears useful, so does bi-directional training. After all, both forward reaction prediction and retrosynthetic analysis come down to determining a set of reactants, a set of products, and a set of condition (in the broad sense of the word).

### Weaknesses
Peculiar citation logic: retrosynthetic analysis citation points at 1969 paper, yet forward reaction prediction started only in 2019 according to the selected citation. Please do better.

It seems from the language of the paper, that the authors don't quite realize a fundamental difference between forward- and retrosynthetic tasks. The forward task starts with completely specified set of the reactants (along with conditions) and aims to predict the set of products along with yields. The retrosynthetic task starts with a single desirable product - this doesn't mean that it has to be the only product of the predicted reactions. This makes retrosynthesis a fundamentally underspecified task that is supposed to recover multiple sets of reactants producing different sets of products all containing target product with different yields. Retrosynthesis is fundamentally a prediction of ensembles of forward reactions, such that the intersection of their product sets has at least one common product (the target). 

The prompt appears to be explicitly referring to known reaction types - it cannot be better in any sense than the approaches criticized in the introduction for combinatorial explosion. The only reason why the authors do not have combinatorial explosion of the length of their prompt is because they did not properly enumerate the reaction types.

The argument about "inefficient representation" is confusing. SMILES can be canonicalized (just like IUPAC names) for uniqueness, ambition to use SMILES to describe reactions led to the development of Reaction SMILES. Throw in SMARTs for the completeness of the discourse. These are chemoinformatic representations - the paper does nothing to improve them. Internal (latent and such) representations obtained by deep neural models are as numerous as deep neural models - the paper doesn't really describe any novelty. Using contrastive learning protocol does not change the nature of the representation, this contrastive training protocol appears to improve its quality which is often the case with contrastive learning. 

There's only one task in one benchmark (out of eight tasks, two options re: reaction types,  and three benchmark) where 14% accuracy improvement is observed. In majority of the comparisons the gains are smaller. This is important considering that some baselines are already in +90% range.

### Questions
Please fix the references.

Please tighten the narrative and just address what you realistically accomplished (contrastive training protocol on mapped an unmapped SMILES is not a new representation; prompt referring to reaction types is an uncontrolled sample of a combinatorial set of reaction types, etc).

Please discuss the sensitivity of the outcomes to LLM prompting. This is the biggest source of the uncertainty in the framework.

As far as uncertainty goes, please at least discuss uncertainty propagation in this framework.

### Soundness
3

### Presentation
3

### Contribution
3
