# Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection

- Decision: Reject
- Scores: 4, 2, 2, 10

## Abstract
LLMs are increasingly embedded in everyday decision-making, yet their outputs can encode subtle, unintended behaviours that shape user beliefs and actions. We refer to these covert, goal-directed behaviours as hidden intentions, which may arise from training and optimisation artefacts, or be deliberately induced by an adversarial developer, yet remain difficult to detect in practice. We introduce a taxonomy of ten categories of hidden intentions, organised by intent, mechanism, context, and impact, shifting attention from surface-level behaviours to design-level strategies of influence. We show how hidden intentions can be easily induced in controlled models, providing both testbeds for evaluation and demonstrations of potential misuse. We systematically assess detection methods, including reasoning and non-reasoning LLM judges, and find that detection collapses in realistic open-world settings, particularly under low-prevalence conditions, where false positives overwhelm precision and false negatives conceal true risks. Stress tests on precision–prevalence and precision–FNR trade-offs reveal why auditing fails without vanishingly small false positive rates or strong priors on manipulation types. Finally, a qualitative case study shows that all ten categories manifest in deployed, state-of-the-art LLMs, emphasising the urgent need for robust frameworks. Our work provides the first systematic analysis of detectability failures of hidden intentions in LLMs under open-world settings, offering a foundation for understanding, inducing, and stress-testing such behaviours, and establishing a flexible taxonomy for anticipating evolving threats and informing governance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
After an argument about the risks of embedded "hidden intentions" in LLMs discourse, this paper proposes to operationalize them through a ten-category taxonomy. The authors propose a testbed to observe them in single-turn outputs.
For this, the authors propose an LLM-based question generation scheme, and LLM judges to decide upon the categories. Ground truth labels verification is performed on a sub-sample by human annotators. Experiments for the accuracy of the proposed framework are performed on open-source models in a more favorable setup, before the authors perform experiments "in the wild" on online LLMs. The main experimental conclusion is that for realistic audit setups in the wild, the proposed framework does not function, ie, it is insufficient for a robust audit of the ten proposed categories.

### Strengths
* enjoyable paper to read
* important problem considering the impact of LLMs on the industry and their position facing users
* interesting set of categories

I think this is an important topic, and the authors made a significant effort to bring it up clearly and place related works in the discussion. The paper is good at raising a problem, but does not provide any solid means to address it, even partially well.

### Weaknesses
* use of LLMs as judges to judge upon categories, which might introduce bias
* arbitrary categories, without leveraging other experts from other fields (sociologists, political scientists, etc), while working on arguably part of their field for a precise definition
* The solution proposed fails, and thus categories are not operationalizable

I am in the end disappointed by the provided results: if the proposal works so poorly, it is either because 1) the categories are not relevant for the problem as they cannot be operationalized to achieve the defined detection goals (which is not my primary guess) or 2) the tools and setups brought in the experimental parts are not satisfying. For 2), as no systematic human labeling nor question crafting was performed but the work was mainly left to LLMs, it lacks an experimental section demonstrating an absence of bias due to the use of such automated tools. If so, the single-turn questioning would be insufficient for the target resolution, and a better scheme would have to be proposed to go beyond the poor results displayed e.g., on Figure 3. 
I cannot agree more with the "Limitations." section at the end, that in my opinion, must be implemented in order to give ground to this interesting research track.

### Questions
* are there existing works by sociologists demonstrating that "LLMs... [are] capable of steering beliefs, preferences and behaviors" (p.1)? More generally, are the categories built according to a consensus on concepts accepted in social sciences, where this whole set of considerations belongs? I believe the paper to be unconvincing on these, as I see this pertains to their field better than to computer science.

### Soundness
2

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
3

### Summary
The paper aims to study “hidden intentions”, defined as “covert, goal-directed behavioural patterns”. It proposes a list of 10 of these hidden intentions.

It then runs experiments using this list: for each intention, ChatGPT generates prompts to trigger these intentions, as well as innocuous prompts as a baseline. Mistral-7B and Llama3.2-3B (“lab models”) respond to these prompts, to represent models acting with hidden intentions.

Next, 10 LLMs (5 reasoning and 5 non-reasoning) are evaluated as judges to classify whether the lab models were acting with hidden intentions. The paper finds high performance in the “category-specific” setting, where models are aware of exactly which hidden intention they’re looking for; but lower performance in the “category-agnostic” setting, where models are told to look for hidden intentions but are not given specifics of what to look for.

Finally, the paper designs prompts to elicit each hidden intention, “guided by our taxonomy but without relying on adversarial prompt engineering”. It evaluates 5 prompts and 3 LLMs per category, finding examples of models showing each hidden intention.

### Strengths
1. The introduction is well written and provides a strong motivation for this research direction. In particular, the idea of decoupling *intentions* from *surface-level tendencies* is interesting to explore: the same behavior (e.g. agreeing with the user) could be benign or undesirable, depending on the context.

2. The topic is important: as models become increasingly capable and coherent, identifying goal-directed behavioral patterns will be an important topic to understand.

3. The qualitative examples of models demonstrating these intentions (Table 10) are interesting.

4. The paper evaluates a good variety of judge models for the detection task: 5 reasoning and 5 non-reasoning models, including strong models like Claude 4 Opus and GPT 4.1.

### Weaknesses
1. The paper claims to introduce a "taxonomy" of hidden intentions. But typically a taxonomy refers to the categorization of some existing set of things in the world - what exactly is being taxonomized here, and how was the taxonomy generated? If this taxonomy of 10 intentions is intended to be a contribution, what makes *these specific 10* intentions interesting to study?

1a. The paper says "Building on existing literature and conceptual analysis, we propose ten broad categories of hidden intentions". Is it a taxonomy of intentions behind behaviors discussed in the literature? But the connection to prior work feels very sparse - if the paper wants to justify this taxonomy as a review/synthesis of prior work, it should at the very least given a more detailed explanation of how each class relates to prior work.

1b. Is it a taxonomy of intentions which LLMs exhibits in natural settings? But the connection to "in the wild" LLM behavior also feels very sparse - once these intentions are identified, the authors demonstrate that it's possible to construct prompts which elicit each of these intentions (or at least, behaviors which seem consistent with these intentions - unclear exactly how you'd identify *intentions* in the wild!). But this experiment feels post-hoc; I imagine that with the right prompting, we could elicit a wide range of undesirable behaviors. It would be more convincing to start with a natural, non-targeted distribution of prompts, and to identify these behaviors emerging naturally; but the paper doesn't appear to attempt this kind of study.

2. IIUC, after providing the taxonomy, most of the dataset is AI-generated by LLMs (Figure 5.) - how can we be confident that evaluations on this dataset are actually supporting the claims made by the paper, i.e. that “We show that hidden intentions, covert, goal-directed behaviours in LLM outputs, are both easily inducible and difficult to detect”? I wasn’t able to find qualitative examples from the paper’s dataset, other than the single example in Figure 1 - what do ChatGPT’s generated prompts look like, and what do the lab model’s responses look like? The paper should provide randomly sampled examples, in order to assess how “natural” these behaviors are.

3. The paper’s intro claims that “In isolation, such surface-level statements cannot reveal function or intent, as they may be supportive, manipulative, or simply contextually adaptive.” However, the paper’s experiments on both human annotation (Appendix B) and real world manifestation of hidden intentions (Section 6, Appendix F) appear to rest on the premise that humans are indeed able to *evaluate whether a given response matches a given intention*, presumably based on the surface-level content of the response - on what basis can you claim to identify intentions, separately from surface-level behaviors?

3b. Also, the very high human agreement in Appendix B based on just one turn seems to empirically contradict the claim that “surface-level statements cannot reveal function or intent”?

4. The “lab models” studied are quite weak - Mistral-7B and Llama3.2-3B. Why so much weaker than the judge models?

### Questions
1. To what extent is an LLM prompted to display an intention a good representation of an LLM actually displaying that intention in the wild? How could we tell?

2. How is “Unsafe Coding Practices” an intention? Seems more like a description of a surface-level behavior that could arise for a number of different reasons?

3. In figures and tables referring to specific hidden intentions, the paper uses shorthand, e.g. “C01”. This means that the reader always needs to refer base to page 4 to interpret any of the results. It would improve readability to use more evocative shorthand - maybe abbreviated names, e.g. “Vague” for “Strategic Vagueness”. Even acronyms (“SV”) would be easier to remember than just numbers.

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
3

### Summary
This paper focuses on LLM auditing with the objective of detecting a large range of inappropriate LLM behaviour referred to as "hidden intentions". To that end, the authors propose a taxonomy of hidden intentions, and construct a dataset of both appropriate and inappropriate answers that spans the taxonomy. They then rely on different LLMs as judges, tasked to detect the hidden intentions. In other words, the LLM judges have to label the question/answer pair as appropriate or not. The (in)accuracy of the labeling leads the authors to conclude that LLMs-as-judges perform poorly.

### Strengths
-Important topic, devising methods to improve and control the safety of LLM answers is crucial.
-Difficult approach: trying to construct a taxonomy of non-desirable behaviors is a difficult task always prone to debate.
-Stressing that LLMs are poor judges of LLMs is important (but arguably new see below)
-I appreciated the authors reflexivity on the limits of their approach, and the idea of exhibiting the limits of detection in best case scenarios to advocate for a larger risk in the wild.

### Weaknesses
-In my opinion, there is a strong mismatch between the high level goals of the paper (answer to "why hidden intentions evade detection") and the actual contents (analyzing the precision/recall on a benchmark of ok/not ok interactions). In other words, the paper showcases some intentions evading some detection, but does not explain why.
-Similarly I am disturbed by notion of "functional but not anthropomorphic" use of "hidden intentions". Intention without agency sounds like "dehydrated water". Why then call it that way and not undesired/toxic/problematic/unsafe or whatever ? In my view intent is a central building block of our society, for instance drawing a legal line between accidental homicide and murder. This creates a frustrating experience where what is discussed eg line 48 ("latent agendas") or figure 1 (adversarial developer) is discretely dismissed in a footnote line 45. This frustration is reinforced by recurring critiques of the paper against the ambiguous (l.83) or imprecise (l.89) existing terminologies concerning AI risks.
-to a lesser extent, I missed a deeper discussion on the difference between "surface level behavior" (also called "isolated quirks") and the single turn interaction benchmark based detection exploited in the paper.
-Related works section adequately cites some references, but I missed a precise positioning (and arguments regarding novelty/contributions) of the paper with respect to those cited references. I believe this reference is missing "Carroll, Micah, et al. "Characterizing manipulation from AI systems."" ACM EAAMO'23
In addition, I missed a positioning wrt a broad body of works on LLM-as-judges (eg Dorner et al, ICLR'25, and connected bibliographic references like "Large language models are not fair evaluators") that already delve into this paradigm and highlight its limits.
-I believe  a software testing and/or facct like conference would represent a better fit for the paper. Its ICLR-wise technical depth is rather shallow, namely the analysis of one large experience, and although the objectives are of great relevance to the community, I do not feel qualified for discussing the soundness of the proposed taxonomy.

### Questions
-can you clarify the notion of surface-level and how you address deeper aspects?
-can you clarify the papers contribution wrt llm-as-judges literature ? 
-in C05 definition: do you consider "omitting proper licensing information" a hidden intention 
-I missed a comment on Gemma3-12B surprising 0.01 FNR
-Can you clarify if figure 2,6,7 are simply plots of equation l.371 instantiated with table1 parameters ?
-fig3 is interesting, but the conflicting "better" direction of x (higher) and y (lower) axis impede its readability. Perhaps y=TNR would be better ?
-minor: typo l149: withing

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper identifies a new class of behaviors of language models that may be malicious for the users, and coins the term hidden intentions for them. This larger group is further classified into ten categories through a detailed design-based and extendable taxonomy. In order to evaluate these behaviors the authors craft a dataset containing examples of hidden intentions introduced in the paper whose labels are further verified through human annotators. A number of language models is then put to a task of judging whether responses contain hidden intentions on this dataset in two scenarios: one where a specific intention is described in the judge's prompt, and another, more realistic, where the judge looks for any hidden intention. The results using confusion matrix based metrics reveal that models perform significantly better when looking for specific category compared to looking for just any. Moreover, surprisingly, the results also show that reasoning models underperform compared to their non-reasoning counterparts in some cases. Finally, the paper identifies some of these behaviors in real world deployed LLM services.

### Strengths
The paper introduces a very interesting and important concern for the future use of LLMs in a way that is not only accessible to scientific community, but general public as well. Examples given throughout the paper are good and thought provoking. The design of experiments is sound: appropriate metrics are used to evaluate the results, extensive set of models is evaluated. The appendices include more detailed results which one reading a paper may want to look into for more fine-grained overview. The paper introduces a dataset which the authors commit to releasing which would add a great value to the community willing to further explore this problem. Moreover, the authors take prevalence of these hidden intentions in account and visualize the results as a function of them in Section 5.3, together with category-agnostic judging this gives a very realistic performance evaluation of LLM judges.

Overall, I see this paper having a major value not only for machine learning community, but also the general public and legislative bodies looking to regulate the field. While complete, this paper has many potential connections with human psychology and opens directions that can be explored by further research.

### Weaknesses
Plots in Figures 2, 3 and 7 are hard to discern even for non color blind readers due to repetition and similarity of colors. Using different line shapes or geometric shapes across the length of the lines would greatly improve the readability of the plots.

### Questions
The introduction of the paper sets up a reader for active deception of sorts, yet in C10 you classify disinformation and bias as a part of your taxonomy. How do you consider this an active deception, especially in line with the arguments of the paragraph between lines 75 to 88? Furthermore, can you expand on C05 making into the taxonomy, similarly in line with C10, as this can easily be attributed to incompetence rather than malice?

While the evaluation spans a number of models, there isn't any relatively small ones which can be run locally on a mobile device for instance. I can imagine a future where communication with LLM services is done over a unified API, and having a small locally run language model actively monitoring responses of the more knowledgeable LLM service for any sign of a manipulation in order to protect the user (think of it as a form of a guardian angel). Based on the results of Table 1, Gemma appears to underperform compared to the other models, do you have any insights whether this is due to its size (implying smaller models being inherently bad at judging) or is there something else at play here?

In paragraph 337-346 you note that different models make different mistakes, and this is something to be expected if we draw analogy with decision threshold, and considering all models are calibrated differently through training it is expected they behave accordingly. An interesting idea to see in this case would be to modify sampling procedure of judges when they are making YES/NO decision using a form of structured decoding to mimic different thresholds and visualize ROC curves in such cases.

In Table 2, the example of C08 gives significant sycophany feeling, therefore can you expand how C08 in your taxonomy differs from sycophany?

### Soundness
3

### Presentation
4

### Contribution
3
