# ParaScopes: What do Language Models Activations Encode About Future Text?

- Decision: Reject
- Scores: 4, 4, 0, 4

## Abstract
Interpretability studies in language models often investigate forward-looking representations of activations. However, as language models become capable of doing ever longer time horizon tasks, methods for understanding activations often remain limited to testing specific concepts or tokens. We develop a framework of Residual Stream Decoders as a method of probing model activations for paragraph-scale and document-scale plans. We test several methods and find information can be decoded equivalent to 5+ tokens of future context in small models. These results lay the groundwork for better monitoring of language models and better understanding how they might encode longer-term planning information.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigate the extent to which transformer residual stream activations at a new-paragraph token ("\n\n") contain enough information to reconstruct the transformer's next output paragraph (sampled autoregressively), which they refer to as "planning" for the next paragraph. They employ two different probes:
- "continuation ParaScope", which involves patching the residual stream activations at "\n\n" into a blank prompt and running the transformer forward autoregressively
- "TAE ParaScope", which involves fitting a linear mapping from residual stream activations to the SONAR embedding of the next paragraph.
The authors find that the performance of these probes beats random baselines and is roughly comparable to providing ~5 tokens of the ground-truth upcoming paragraph. They also extend these methods to document-outline-level information and find weaker results.

In addition, the authors study where planning takes place along the layer and position dimensions. They interpret their experimental results as showing that, for the model they study, 1) layers 60-80% into the model are most planning-relevant; 2) the model is relatively myopic and mostly focuses on the immediate next paragraph.

### Strengths
- The writing and especially the figures are clear. It is easy to understand what questions the paper investigates and its methodology for doing so.
- The performance of the two ParaScope probes are compared to various meaningful baselines.
- The layer-wise and temporal investigations in section 6 are interesting.

### Weaknesses
- In my view, this is a fairly incremental follow-up to an existing line of work. Previous works [1, 2] find that residual stream activations contain enough information to predict ~5 future tokens. Hence it is not so surprising that these activations can be used to reconstruct the next paragraph about as well as by providing ~5 ground truth tokens.
- The methods (activation patching, linear probes, text embeddings) are not especially novel, being straightforward adaptations of existing techniques to the specific next-paragraph setting.
- The notion of "planning" here is a fairly weak information-theoretic one, and it is possibly misleading to refer to this as "planning" at all. While the authors are careful to note this in the introduction, the writing elsewhere in the paper is sometimes more imprecise/anthropomorphic. E.g. 448: "myopic behavior suggests a form of just-in-time updating creation of plans"; 
- I'm not convinced that the results of the token-position experiments (6.2) show that the model is "relatively myopic". E.g. one cartoon picture consistent with the Continuation ParaScope results is that, before the "\n\n" token, the activations encode the plan "I will talk about blockchain until I reach a \n\n token, then switch to Mayan architecture."; then, after the token, simply "I will talk about Mayan architecture." This does not seem "myopic" to me.

[1] Pal et al. "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State". 2023

[2] Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads". 2024

### Questions
- Are the results in Section 4 statistically significant? It's worth computing p-values for all the comparisons being made (e.g. "When looking at subject match, we find that TAE ParaScopes generally seem to outperform Continuation ParaScopes"; "when it comes to achieving at least Moderate Depth details [...], Continuation ParaScope (15%) is significantly better than the TAE ParaScope (3%)".)
- Error bars for Fig 8 (right) would also be appreciated.
- Any guesses as to how the qualitative differences between continuation and TAE ParaScope arise? Do they tell us anything about how models do planning?
- What is the significance of the 0.3 temperature? This introduces some irreducible noise into the ground truth next paragraph; is there a reason why 0 was not used instead?
- Can you reproduce the results of 6.1 and 6.2 using TAE instead of Continuation ParaScope?
- I would appreciate a bit more discussion on applications to monitoring and interpretability (though I realize this is mostly out-of-scope for this paper). When would ParaScope be more useful than simply monitoring the ground-truth next-paragraph output before it is used?

Nit:
- 030: Run-on sentence
- 044: malformed citation (Team)
- 068: malformed citation (team et al.)
- 121: "For our dataset, we require data that represents the outputs of the LLM so we can understand its thinking process." Somewhat confusingly and imprecisely phrased. You require data sampled autoregressively from the LLM because the ground truth is by definition the next paragraph sampled autoregressively.
- 162: "residuals diffs" -> "residual diffs"? I don't know what this term refers to. Are these the MLP/attention outputs?
- 185: K\in \{1, 5, 10\}
- 219: "ruberic" -> "rubric"
- 353: "generating outputs in both directions". Not clear what this means. I guess this means you tried both A below B and B below A?
- 560: malformed citation (Anonymous or Various)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ParaScopes, a framework designed to investigate the extent to which language models encode information about future text within their internal activations. The authors propose the Planning Decodability Hypothesis, which operationalizes the concept of model planning into a testable claim: that information about upcoming content is decodable from activations before that content is generated. The study introduces two Residual Stream Decoder methods, Continuation ParaScope and TAE ParaScope, to probe for paragraph-level and outline-level plans. Experiments on a Llama 3.2 3B model show that limited information, comparable to a 5-10 token lookahead, can be decoded, with planning signals being most prominent in the model's middle layers and appearing "just-in-time" at paragraph transition tokens.

### Strengths
- The paper introduces two distinct and complementary probing methods, the intervention-based Continuation ParaScope and the mapping-based TAE ParaScope, which strengthens the validity of its conclusions.
- The study goes beyond simply finding evidence of planning by localizing the relevant signals, identifying the middle layers of the network as the primary location for paragraph-level planning information.
- The paper provides a specific temporal account of planning, presenting evidence for a "just-in-time" mechanism where plans are sharply formed or refined at the moment of paragraph transitions.

### Weaknesses
- The findings are based almost entirely on a single, relatively small (3B parameter) model, making it unclear if they apply to larger, more capable architectures.
- The main TAE ParaScope method uses a linear map, which may be too simple to extract more complex, non-linearly encoded plans from the model's activations.
- The work primarily establishes that future-looking information is present in activations, but provides limited evidence that this information is causally used by the model to guide generation.

### Questions
1. Is the observed just-in-time planning an artifact of the 3B model's scale, or do you expect it to persist in larger models?
2. Have you tried a steering experiment by injecting a synthetic plan embedding to test for causality?
3. What is your hypothesis for why the Continuation ParaScope is sometimes better at fine-grained details than the TAE ParaScope?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper proposes two new methods for probing the "forward looking" or "planning" information present in language model hidden states. The common setup in the paper is given some text, bisect it, feed the first half through a target model you wish to study, take the hidden states of the last token in the first half and then try to generate the second half of the text from just those hidden states. The first method they propose, Continuation ParaScope, uses the target model to generate the second half of the text. The second method they propose, TAE Parascope, first trains a TAE on hidden states and generations from the target model and then uses the trained TAE to generate the second half of the text.

### Strengths
* The layer-wise and temporal dynamics studies in sections 6.1 and 6.2 provide some interesting analysis of language model hidden states
* Method diagram are a helpful aid to the writing

### Weaknesses
* Experimental setup for the main experimental results (in Figure 4) is problematic. They use data generated by the target model to then evaluate the target model. Obviously text sampled from the target model is likely under the target model. This introduces a major source of confounding. I expect a language model would generate the same or similar next couple tokens only given its last hidden state. Not necessarily because it has plan them out but possibly because its just the likely next thing to say given what is captured in that current hidden state. Given this it is unclear to me that the experiment actual measures the amount of "planning" information present in the hidden state.
* There is a second issue in the main experimental results (in Figure 4). In the description of the Cheat-K baseline they state "For fairness, we filter examples where more than 50% of the text would be exposed" this is actually completely unfair. For a fair comparison all methods should be evaluated on the same evaluation set. It is improper to remove these examples at all in my opinion, the whole point of that baseline is it cheats, but at the very least if you are going to remove them you need to remove the examples for all methods. You cannot compare two methods if they are evaluated on different subsets of the data.
* The results displayed in the violin plot would be much more convincing if demonstrated to be significant with a statistically test.
* The paper presents parascope as a general method for probing information, but only provides experiments with one model.
* There are a number of figures which are never referenced in the text. If the figure is important enough to be in the main paper it should be discussed and referenced.
* The presentation also could be improved. The text is redundant in places (e.g. splitting the paragraphs by "\n\n" is described on lines 035-036, 113-115, 125-128). The text in figures is often very small, and sometimes overlaps itself (e.g. Figure 4, Figure 5, Figure 7, Figure 8, Figure 10). There is a lot of whitespace around some diagrams (e.g. Figure 1, Figure 8).
* Some very minor details. The citation (team et al., 2024) on line 068 I believe should be capitalized "Team", the white grid lines in Figure 5 make the plot harder for me to read (I think because the bars are translucent), and Figure 10 x axis is labeled Token Index but the values on the x axis are decimal values 2.5, 7.5, etc.

### Questions
* What is the training/testing setup for the TAE Parascope? Where does the training data come from? Assuming the dataset constructed for evaluating Parascope variants is held out for unbiased evaluation.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates whether large language models exhibit evidence of planning within their internal representations. The authors operationalize planning as the presence of information about future tokens within the model's residual stream. To test this “Decodability Hypothesis,” they introduce two decoding methods, termed ParaScopes, which attempt to reconstruct upcoming paragraphs from residual stream activations. By comparing these decoders against several baselines, the study finds some but limited evidence that model activations contain information about upcoming paragraph.

### Strengths
- The paper is well-written with a clear structure. Both the methodology and experimental setup are presented in a straightforward, easy-to-follow manner.
- The authors offer a valuable methodological advance by operationalizing planning as the decodability of upcoming text from residual stream activations.

### Weaknesses
- The paper equates the ability to decode future tokens or paragraphs from residual stream activations with evidence of "planning." This interpretation is not convincing to me. The residual stream contains contextual information that makes certain continuations more likely, even if no explicit future information is stored. For example, if the context describes the first half of a soccer match, the model predicting the second half next is just a result of coherence, not evidence of a stored "plan".
- The improvements over random or baseline methods are marginal. The results suggest shallow correlations rather than robustly decodable "plans."
- The core experiments are performed on a single mode, Llama 3.2 3B. Expanding to other family models, and especially to larger models, would be valuable.
- The "subject match" in outline-level experiments is insufficient to claim planning, and the other measures perform poorly.

### Questions
Why do you use temperature>0? By doing this, each continuation you sample, both in the dataset as for Continuation ParaScope can lead to different paragraphs. Would it make sense to resample each continuation multiple times and consider that when computing the metrics? Or using temperature=0 instead?

Your operationalization defines planning as decodable information about future text within the residual stream. How do you distinguish this from contextual predictability, i.e., the residual stream encoding contextual information that makes certain future tokens plausible rather than future planning?

Comments:
- line 219: ruberic-based -> rubric-based
- line 308 "we show the results in 6." -> "we show the results in Figure 7"

### Soundness
3

### Presentation
3

### Contribution
2
