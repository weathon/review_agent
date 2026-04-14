# Concept Bottleneck Language Models For Protein Design

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 6, 6

## Abstract
We introduce Concept Bottleneck Protein Language Models (CB-pLM), a generative masked language model with a layer where each neuron corresponds to an interpretable concept. Our architecture offers three key benefits: i) Control: We can intervene on concept values to precisely control the properties of generated proteins, achieving a 3$\times$ larger change in desired concept values compared to baselines. ii) Interpretability: A linear mapping between concept values and predicted tokens allows transparent analysis of the model's decision-making process. iii) Debugging: This transparency facilitates easy debugging of trained models. Our models achieve pre-training perplexity and downstream task performance comparable to traditional masked protein language models, demonstrating that interpretability does not compromise performance. While adaptable to any language model, we focus on masked protein language models due to their importance in drug discovery and the ability to validate our model's capabilities through real-world experiments and expert knowledge. We scale our CB-pLM from 24 million to 3 billion parameters, making them the largest Concept Bottleneck Models trained and the first capable of generative language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper takes the approach called concept bottleneck language modeling and it applies it in the protein language model context.  This provides creative control in terms of human-interpretable biological concepts, as well as the ability to debug models.  The proposed approach appears to work better than other generic approaches to controllable generation such as ProGen/CTRL, as demonstrated in experimental comparison with other models trained by the researchers themselves.  A case study with redesign of siltuximab shows the ability to quite precisely control certain properties, though no wet lab experiments are shown to verify.

### Strengths
Human-interpretable creative control is a super key in generative AI broadly, but especially in protein design.  As such, the proposed approach based on the concept bottleneck is quite compelling.  The computational experiments provide evidence that the proposed architecture is effective.  The paper is written in a clear way.

### Weaknesses
No wet lab experiments were done to verify results.  This would strengthen the paper considerably.

Comparisons to other architectural approaches are based on the authors' own implementations, rather than by external research groups that may be motivated to optimize the other architectures more.  Notwithstanding, the performance differences are quite considerable.

No discussion on biosecurity risks is given.

### Questions
If even a small wet lab experiment is possible, I would be super curious to see the result.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work applies concept bottleneck models to protein language models. The approach uses fairly standard ideas in concept bottleneck models, but they are well-executed and lead to novel applications in e.g. controllability.

### Strengths
The controllability experiments are fairly convincing, though the stated application on debugging is fairly cursory and could be improved (the intervention to test debugging is fairly blunt, and also the results are just hand inspection and thus not reproducible). However, for me the controllability experiments are enough to be a solid contribution.

The perplexity results were moderately convincing, though more discussion of comparisons to LBSTER and ESM2 would be useful -- it was not clear to me for instance if your method was given extra information (the tags) that LBSTER / ESM2 did not have. It was also strange that the largest model had worse perplexity.

The writing is overall good and easy to follow, but I felt that the level of detail in some experiment descriptions was not enough to be fully reproducible from the text.

### Weaknesses
See above (strengths and weaknesses are merged)

### Questions
There are some minor typos e.g. "00:01 01:15" in Section 3.1, as well as a missing line break before \textbf{Scaling} in the same subsection.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper introduces a novel protein language model designed to provide predictions that can be interpreted by human experts without compromising performance. The proposed model, dubbed Concept Bottleneck Protein Language Model (CB-pLM), empirically proves to be easier to control and interpret over previously proposed State of the Art Protein Language Models.

### Strengths
Developing and testing new models for protein modeling is an important problem. In particular, having methods that can be interpretable and that can allow better control for drug design is a high impact project.

### Weaknesses
The paper contains many typos, it is meandering and unfocused, and generally is very hard to read (with many repetitions and vaguely defined concepts).  

For example, what does “debuggable model“ mean? This is stated as one of the main contributions of the proposed architecture, hence the reader expects it to be clearly defined and thoroughly assessed through empirical validation. In the current version of the manuscript this is not properly defined and hence not convincingly validated with experiments. 

What are the novel contributions specific to the Protein Language domain? Can the authors separate the specific novelty introduced for their protein language model? What are the main differences with the previously proposed concept Bottleneck Generative Models? If the main contribution of the paper is to test whether CB-LMs work on the Protein domain then it would be good to devote more space in the paper to the empirical validation with a thorough comparison with other State of the Art models at different scales.

The discussion section briefly mentions some properties of the model that have not been tested in the main empirical section of the paper. It would be good either move the strong empirical evidence supporting concept composition and generalization to novel concepts (added with fine-tuning) in the main or, if results are not convincing enough, remove the claim. Similarly, in the conclusions (and abstract) it is mentioned that the current model has been tested up to the 3B scale and compared with other public models. However I did not manage to find evidence of experiments on scales larger than 150M (see for example Table 1 in which the proposed method does not perform comparably with others).

### Questions
1. Line 76, what are “human understandable concepts”? This is not very well defined in the paper and it is left vague. Similarly, what does debuggable model mean? Even if the proposed model leverages a bottleneck to force some representations to align to a set of pre-specified “concepts” the whole model before the bottleneck is a complete black-box. Does debuggable mean that the user can inspect the black box before the bottleneck too? 
2. In the abstract and intro it is mentioned that the proposed model has been scaled to 3B parameters, however there are no empirical results showcasing models larger than 150M (most of the results are reported on the small 24M model). Furthermore, in Table 1 the comparison with other State of The Art models suggests that the proposed model regresses in perplexity over similarly sized non-interpretable models. Can the authors provide an apples to apples comparison with other public models at scales larger than 150M?
3. Is the current method supposed to scale to a large number of protein sequences? Will it generalize in the case of mislabelled or unlabelled samples? Given annotation cost, a model capable of leveraging partially annotated/unlabeled samples is more likely to scale.
4. What is the maximum context length (length of the protein sequence) that the model supports?
5. It is not clear how the concepts in the bottleneck layer should be defined and why the model needs to learn to associate specific concepts to “disentangled” representations in one-to-one relation with the concepts bank. From the paper it appears that the concepts are annotated and hence the “factor of variations” are known at training time. This could become problematic since scaling on annotated data is not always possible. Does the proposed method allow one to not directly observe the factors of variations and still learn some minimal and sufficient concept bank, similarly to the method proposed in [1]? Furthermore, when annotations are used, are the learned concepts compositional? 
6. Experiments in section 4.1.1 and 4.1.2 have been conducted on ablations of the proposed method. Why is it not compared with other State of The Art methods of similar sizes?


[1] M. Fumero et al., “Leveraging sparse and shared feature activations for disentangled representation learning”, NeurIPs 2023.

Minor:
- Line 121, typo.
- Line 125, labels in match with Figure 2 even if they are not related.
- Line 226 repeated.
- Line 245, typo

### Soundness
2

### Presentation
1

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
This paper proposes a Transformer Masked Language Model (MLM) variant of concept-bottleneck generative models (CBGM) with three goals in mind: controllable generation, interpretability, and debuggability. This paper adapts the training objectives of CBGM to the masked language model architecture, and proposes a control method specifically for their CB MLM. Evaluation was conducted on the domain of protein sequence modeling, for controllability, interpretability and debuggability. The proposed model leads to improvements over non concept-bottlenecked baselines on controllability. The proposed model demonstrates interpretability and debuggability via inspection of the weights of the linear decoder head. 

Comment: I have experience with language modeling but not with protein domain, so my review will focus more on the method part of the paper (as recommended by AC). Please take this into account when considering my review and recommendation.

### Strengths
* Adapting concept-bottleneck generative models to masked language models is an interesting and novel idea.
* Using integrated gradients to find tokens to resample to is an interesting and novel idea.
* Writing is generally easy to follow.
* This work focuses on improving controllability, interpretability, and debuggability of language models (and protein sequence in particular as an example) which is an important topic with large potential impact.

### Weaknesses
Method

(considering a broader application of the proposed method to general language/sequence modeling, as authors mentioned in the abstract) 

M.1 The model proposed is only a conditional generative model given a partial sequence rather than a full generative model of an entire sequence. Classical masked language models (like BERT) do not model the joint distribution over sequences [1]. There are ways to derive joints from MLMs by making their conditionals consistent with one another (see, e.g. [1] and the related works cited there), or by modifying the training objective so that the conditionals are encouraged to be consistent during training (e.g. [2] [3]). Other protein models based on MLM (e.g. [3], which the authors cited) have done the latter kind of training modifications to enable sampling full sequences. I'm not sure how this limitation matters for protein modeling, could authors give more discussions about this? 

M.2 Naturalness of the controllably generated sequence is not considered during intervention, but only used as a measurement afterwards. See point below for consequence.

M.3 The proposed controllability technique only proposes changes local to a particular known sequence x, by changing n% (5 in experiments) of x at a time. Again, I cannot comment on the significance of this limitation for protein modeling. For other modalities like images and language, it can be difficult to change sequence-level concepts via chaining a sequence of such local changes. Even if it is possible, since naturalness is not considered as a factor during intervention but only afterwards, applying a sequence of interventions could lead to going outside of the training distribution (e.g. the interventions produced in Figure 20 goes outside of the color-MNIST training distribution, where all digits are uniformly colored). Empirically though, (Fig 4b 15 16) the proposed method seems to outperform baselines so maybe this is less of an issue in practice, for the concepts in this paper and/or for the protein domain.

Experiments

E.1 The model is trained with 25% masking, but is used to generate with 5% masking in the experiments. This could make the model worse because it’s a different input distribution. Is there a particular justification for masking 5%?

E.2 When evaluating controllability against baselines (and more specifically CC-pLM, since it's closer to CB-pLM), it’s not clear whether the gain comes from intervening at better positions (due to applying integrated gradients to a different architecture), better sensitivity of the model to interventions (due to the proposed concept bottleneck), or a combination, making it hard to evaluate the limitations / benefits of the architecture versus the intervention technique.

[1] Hennigen and Kim, 2023. Deriving Language Models from Masked Language Models.

[2] Germain et al., 2015. MADE: Masked Autoencoder for Distribution Estimation.

[3] Hayes et al., 2024. Simulating 500 million years of evolution with a language model.

### Questions
Q1 How does compute / training vary across C-pLM, CC-pLM, and CB-pLM. Is total compute/wallclock time controlled (e.g. 24 hr, as the cited cramming paper) or perhaps early stopping on validation is used?

Q2 It is interesting that there is negative scaling happening in figures 10 and 11, with sometimes large decreases in performance as the model becomes larger. Out of curiosity, do authors have any speculations about why this happens?

Q3 In lines 142-144, “This enables us to answer [… q1…], or [q2] counterfactual questions like if we want to decrease hydrophobicity, which amino acid should we replace “E” with?”. I understand q1 is an interpretability question and linear decoder does help with that. For q2 however, I am not sure I fully understood. Is it accurate to rephrase it as “if we intervened on the concept strength of hydrophobicity, which amino acid would the decoder output?” My confusion is on the verb “enables” – I believe the linear layer gives you a more interpretable answer to q2 (say compared to a nonlinear decoder), but you would still be able to answer q2 with a nonlinear decoder,  by intervening on the concept value $\hat{c}_i$ and then running a forward pass of the decoder.

Q4 What’s the spread along the x-axis on figure 4b? Is it naturalness? It might be helpful to also plot the reference lines for no-change (for both naturalness and intervened property) as you do in figure 4a.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors apply concept bottleneck language models to protein design. They propose this model has three benefits; control, interpretability, and facilitated debugging. To support this, they demonstrate evidence that the model can optimize for single or double property control, better than existing conditioning approaches. They showcase this by redesigning the hydrophobic patch of an antibody, showing that it responds to the in-silico metrics as expected. Finally, they correlate model weights with trained concepts, and propose these correlations can help model debugging efforts

### Strengths
I believe this CB-pLM is the first model which applies concept bottleneck models to protein design, the authors show with strong evidence that CB-pLM effectively shifts concept distributions better than existing approaches such as C-pLMs and CC-pLMs. The paper is well written, and does a good job at showcasing the in-silico results for conditioning.

### Weaknesses
A limitation of this approach is that biophysical concepts need to be explicitly defined for controlling design. The argument that a CB-pLM is more interpretable because of the training approach is less convincing, as the results appear to be evidence that the model learns the provided concepts correctly, and doesn’t expand to a new interpretation of known biological properties. Concepts used in the model are also straightforward to calculate and interpret the outputs of with bioinformatics tools, alone. The model does not provide additional interpretability on proteins that otherwise exists.  Similarly, while the redesign example appears to currently be tested in a lab, it is not possible to asses if the model improves the solubility and function of  Siltuximab, based on the data provided.

### Questions
* Line 245: it appears the time is denoted as a typo 
* Table 1: how was the validation set chosen, is there overlap between the validation set and LBSTER/ESM2 train sets? 
* Section 4.1 
    * Why was the C-pLM baseline tested with random substitutions instead of feature attribution. 
    * Model size does not appear to improve intervention accuracy or change, significantly. 
* Section 4.2
    * What is the motivation for using WSJ here. Why not use ESM2 or an inverse-design approach like ProteinMPNN or ESM-IF, for sequence redesign in the hydrophobic region? 
    * It seems useful to include a non deep-learning baseline here that at minimum randomly re-samples residues in the hydrophobic patch with hydrophilic residues. 
    * Is the hydrophobic patch critical for the function of the antibody? How do you know if the redesigned region has improved or reduced function. 
* Section 5: 
     * Is it not expected that the model should correctly learn or not learn the concepts provided during training? 
     * It is also clear in what application the model debugging of a CB-pLM might be relevant, based on the given example.

### Soundness
3

### Presentation
3

### Contribution
3
