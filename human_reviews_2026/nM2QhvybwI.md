# Cognitive models can reveal interpretable value trade-offs in language models

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 8

## Abstract
Value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in language models are limited. In cognitive science, so-called "cognitive models" provide formal accounts of such trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. Here, we show that a leading cognitive model of polite speech can be used to systematically evaluate alignment-relevant trade-offs in language models via two encompassing settings: degrees of reasoning "effort" and system prompt manipulations in closed-source frontier models, and RL post-training dynamics of open-source models. Our results show that LLMs' behavioral profiles under the cognitive model a) shift predictably when they are prompted to prioritize certain goals, b) are amplified by a small reasoning budget, and c) can be used to diagnose other social behaviors such as sycophancy. Our findings from LLMs' post-training dynamics reveal large shifts in values early on in training and persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. Our framework offers a flexible tool for probing behavioral profiles across diverse model types and gaining insights for shaping training regimes that better control trade-offs between values during model development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a cognitive model of polite speech as an interpretability tool to quantify value trade-offs in LLMs, formalizing the balance between informational, social, and presentational utilities. The authors apply this method to analyze both frontier black-box models under different reasoning budgets and open-source models during RL alignment. The results demonstrate that models' utility preferences shift predictably in response to goal-based prompts, revealing a signature for sycophancy, and show that base model choice has a more persistent impact on these trade-offs than the specific alignment method, with utility values converging early in training.

### Strengths
The research method in this paper is very interesting, by estimating the parameters of cognitive models, it enables interpretable analyses of the internal value trade-offs within LLMs.

The paper also compares the analysis results of human cognitive models, providing valuable reference points.

### Weaknesses
As the authors discussed in the conclusion, I have some concerns about the methodology of using human cognitive models to analyze LLMs’ value trade-offs. This approach can at best offer correlational explanations, but it essentially does not move beyond treating LLMs as black-box models. If the proposed cognitive model could be mapped to some kind of circuit within the LLM, I believe it would be more convincing.

There are numerous real-world scenarios involving value trade-offs. While I agree that politeness is indeed an important one, focusing only on this type of scenario feels somewhat limited.

The experimental analysis is rather rough. First, the figures are not clear enough to allow readers to intuitively understand the results. Second, there is a lack of deeper analysis, the paper seems to merely present evaluation results without offering further insight.

There are also some minor typos. For instance, a period should follow paragraph names, and punctuation should be added after equations. The use of quotation marks in LaTeX is incorrect (e.g., Line 262). In addition, the $\phi$ symbol in the footnote on Line 377 did not render properly.

### Questions
See the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this work, the authors use the cognitive model of polite speech, a newer version of the influential rational speech act (RSA), to understand value tradeoff in LLMs. The RSA model provides a normative framework to model the weights an agent, human or LLM, attributes for different utility functions in speech, such as information utility, social utility, presentational utility. The authors began by conducting a vignette study on  LLMs -- drawing on experiments run on humans -- and fit RSA model to the LLM generated responses. They found that by default reasoning LLMs, particularly the closed-source ones, tended to weigh informational utility higher than social utility, with the utility values being amenable to prompting methods that prescribe the goals more explicitly. They additionally analyzed how these utilities change over the course of fine-tuning on two different open-source models and found that utility values shift quite early on in the training process with the effect of base model and pre-training data out weighing those of feedback dataset and alignment method.

### Strengths
Although the idea of using cognitive models to study the behavior of LLM models is not new, the setting the authors have chosen to study and the model of choice serves as a great example of how one can use cognitive modeling to draw insights about LLMs.  I find both the setting and model to be ecological for the study of value trade-off in LLMs. The findings are intuitive and corroborate many existing findings.I also liked the fact the they considered both open and closed models, and used them appropriately. For example, by using them in cases where certain class of models can be helpful to draw relevant conclusions. The writing was clear, figures legible, and analyses well motivated.

### Weaknesses
The RSA model explanation is concurrent a bit too dense, especially for ML audience. It would be extremely helpful if the authors can dig into the model a bit more, with examples to provide readers with more intuition and also interpretation of what different values of the fitted parameters mean. This is especially important as the rest of the papers build on the fact that the readers understand the model and its parameters well. 
The authors could have also motivated the datasets used for fine-tuning a bit better, focusing on how they are useful for the answering the questions of interest.

### Questions
How well do these models fit human data?
Are there other competing models that can used to compare against? 
What happens in these measures if you have more than one back and forth? Can the cognitive model update its parameter with additional observations? It would interesting to see how these measures change over the course of a short, constrained conversation.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigated LLM preference balance between informativeness and social focus in response generation using a pre-established cognitive model of politeness in humans. They analyzed the default behavior of multiple black-box models, as well as shifts across various reasoning budgets and prompt-induced goal variations. Finally, they analyzed the change in preference balance of white-box models during fine-tuning on established RLHF datasets.

### Strengths
- Very well articulated and formulated discussion of weaknesses (section 6).

- Important early step in LLM behavior analysis influenced by pre-existing cognitive models of behavior.

### Weaknesses
- The experiment design as described arguably only probes the LLM's model of how it would expect others to behave in this scenario (i.e. this is a Theory of Mind task). It is unclear whether this directly predicts model overt behavior. Given appropriate analysis, this might be addressed by comparisons between LLM-as-judge, LLM-as-agent, and LLM-as-assistant perspectives, but analysis along this dimension seems to be absent.

- Section 5.2 provides p-values but does not specify the test being used.

- There is no mention of precautions to control for positional or label biases when prompting the models (i.e. via utterance choice orderings randomization/etc). This is likely non-problematic if all experiments used a shared fixed option ordering.

- It's unclear why you chose to discard the assumption of greater cognitive cost for negated expressions (Appendix C.2) as used by Yoon et al. The explicit need for additional tokens and generation steps would suggest that greater cost is certain for LLMs and thus more important to include than in human cognitive modeling where that increased cost is fully assumed. This is only further exacerbated by the possibility that some utterances may require more than one token. My best guess is that the assumption was made here because of the label-based cloze test methodology, but this may not be a safe assumption, as the cost of internally associating labels to utterances may still be affected by utterance complexity.

- Minor: The intuitive meaning of V(s) should be defined alongside the existing description of equation (2), which is where it is first used, albeit indirectly (via in-text definition of U_soc). In a similar vein, it is difficult and inconvenient to parse the consequences of Figure 1 when it is so far separated from its use in text (section 5.2).

### Questions
- Are the changes to utility value from alignment training robust to obliterative fine-tuning? Relatedly, can models that are sensitive to obliterative finetuning be trained to value align and task align simultaneously?

- Did you perform any analysis on variation across framings (LLM-as-[judge/agent/assistant]) for inferred parameter behavior shifts? If not, and if the results in figures 1 and 4 are based on fitting to all three framings in unison, could this have compromised the result? e.g. could the models show strong sycophancy when acting as an assitant but weak anti-sycophancy when acting as a judge or agent, but the assistant framing effectively dominates the other two framings.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- The paper proposes using formal "cognitive models" to interpret how LLMs handle value trade-offs. Specifically, it applies a model of polite speech that formalizes communication as a trade-off between three competing utilities.
- The authors collect behavioral data by giving LLMs "experimental vignettes" where they must choose a polite (or impolite) utterance in a social context (e.g., judging a friend's bad cake) . By fitting the LLMs' responses to the cognitive model, they infer the underlying utility weights.
- In a study of Anthropic, Google, and OpenAI models, the paper finds that models with higher "reasoning budgets" (e.g., using more tokens or an explicit reasoning mode) demonstrate a higher default preference for informational utility over social utility. These utility weights also shift in predictable ways when the models are explicitly prompted with "informative" or "social" goals.
- The choice of the base model and its pretraining data has a more "outsized" and persistent impact on the final utility trade-offs than the specific feedback dataset.

### Strengths
- The authors propose create a behavioral "signature" for sycophancy (hypothesized as high presentational utility but low informational and social utility). They find that when models are prompted with a purely "social" goal, their inferred parameters converge to this "sycophantic" signature---this has some practical use, given increasing worry around syncophancy.
- The experiments are quite thorough, using both open- and closed-source models, running a lot of ablations, and putting detailed results in the Appendix.
- The findings around the centrality of the base model and pretraining data re: sycophancy has important implications for how models will be post-trained.

### Weaknesses
- Perhaps it's because I lack a proper cogsci background, but the specific framework used by Yoon et al. that was borrowed by the authors wasn't fully clear to me. The overall approach made sense, however.
- The paper's conclusions about general LLM "value trade-offs" are based entirely on polite speech (i.e., judging a friend's cake or poem) . The authors concede that these cognitive models "are often bespoke to the target domain" and "do not easily generalize to the open-ended nature of natural language use".
- The authors admit that their method has technical challenges. They state that fitting such a complex model "could potentially pose a challenge for making robust inferences".

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
