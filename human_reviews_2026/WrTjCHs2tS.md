# Decomposing Prompts: Discovering Reusable Scaffolds and Task-Specific Residuals

- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Traditional automatic prompt optimization methods treat prompts as monolithic text blocks, necessitating costly, from-scratch optimization for each new task and precluding knowledge reuse across related tasks. We revisit this monolithic paradigm by proposing an approach where prompts are decomposed into two components: a reusable, domain-invariant **Instructional Scaffold** that captures high-level task structure, and a concise **Task-specific Residual** for fine-grained adaptation. We introduce **DiSR** (**D**iscovering **i**nstructional **S**caffolds and **R**esiduals), a two-stage algorithm inspired by the Minimum Description Length (MDL) principle, which motivates decomposing prompts into reusable components. This approach systematically discovers the components, allowing for efficient knowledge reuse and task-specific adaptation. In contrast to existing methods like OPRO and AutoPrompt, DiSR enables task-level knowledge reuse, improving performance and reliability across different tasks. Extensive experiments demonstrate that DiSR not only achieves competitive accuracy but also generates better-calibrated confidence estimates, especially on large models and in professional domains. This compositional hypothesis is validated through semantic analysis, revealing that optimized prompts form distinct, scaffold-centered clusters in their embedding space. Our findings establish a compositional view of prompt engineering, facilitating more robust, interpretable, and reusable optimization strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Most existing prompt optimization framework optimizes a single string to corresponding tasks. This paper argues for a more structured perspective that we shall both optimize for a domain-general string and also adapt by optimizing additional task modules. This lens is intriguing and is really necessary to industrial applications. They thus adapt the framework across several NLP tasks, by using ORPO as optimizer, showing some performance improvements over ORPO, which is slightly expected.

### Strengths
Most of all, like I said, the lens of "modular prompts" is important, especially considering industrial applications where we always have to update the prompts for LLM agents whenever we have any new task requests.

They present some experiments and show their effectiveness.

### Weaknesses
1. In my opinion, a few sentences are a bit hard to follow. 

For example, what do you mean by "better-calibrated, more trustworthy prompts"? I am rather confused. 

And from the intro, I'd rather use some examples to illustrate this. Otherwise, it could be too abstract. For instance, when you talk about the idea of compositional structure, I think it is good, and it is really common for practitioners to expect this. Then, the overall prompt refinement could just work as how engineers specify new requirements in yaml. Expanding your introduction with an example like this could be better from my lens.

2. The idea is relatively too simple... It is just a direct implementation of ORPO in a variety of NLP tasks. The only difference is they use ORPO under their decomposition framework. This is really limited. This framework is quite easy to come up with, and the formulation is so simple that any prompt optimization framework can be adapted. So I am not convinced enough regarding the contribution of this work. Yea, some derivations may look interesting, but it is limited in a top AI conference, at least to me. It is quite hard for us to get convinced that the performance improvement could be any sense meaningful as scientific insights..

3. BTW, could you provide more baselines other than ORPO. I am confused about the setup. I also think the related work section misses a lot of relevant work, e.g., [1].

[1] Decomposed Prompting: A Modular Approach for Solving Complex Tasks

And I am not sure whether it is proper to use "challenge this monolithic paradigm" in the abstract.

### Questions
Please answer my critique on the weaknesses. Let me know if I miss anything.

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
This paper introduces DiSR (Discovering instructional Scaffolds and Residuals), a novel framework for automatic prompt optimization that challenges the conventional monolithic view of prompts. The authors propose decomposing prompts into two components: (1) an Instructional Scaffold—a reusable, domain-invariant component capturing high-level task structure; and (2) a Task-specific Residual—a concise adaptation for fine-grained task requirements. Inspired by the Minimum Description Length (MDL) principle, DiSR operates in two stages: first discovering a shared scaffold across multiple source tasks, then learning minimal residuals for adapting to specific target tasks. The paper provides theoretical grounding, a practical algorithm, and extensive empirical validation across commonsense reasoning and professional domains (medicine, law, finance). Results demonstrate that DiSR achieves competitive accuracy while generating better-calibrated prompts, with particularly strong performance on larger models.

### Strengths
- The paper makes a significant conceptual contribution by reframing prompts as decomposable structures rather than monolithic units. This represents a paradigm shift in prompt engineering that could influence future research directions.

- The connection to Minimum Description Length (MDL) principle provides a solid theoretical basis for the decomposition approach. The integration of cognitive science concepts (Schema Theory) further strengthens the conceptual framework.

- The authors provide multiple lines of evidence supporting their hypothesis, including performance comparisons, scaffold generalization tests, semantic visualizations (Figure 4b), and ablation studies on residual portability and efficiency.

### Weaknesses
- While the paper demonstrates that scaffolds generalize well, it lacks a systematic analysis of what specific properties make certain scaffolds more effective than others. A deeper linguistic or structural analysis of successful scaffolds would strengthen the theoretical contribution.

- The operational definition of a "domain" as "a set of tasks that share a common goal, format, and reasoning process" is somewhat vague. The paper would benefit from clearer guidance on how to define domains in practice and analysis of sensitivity to domain boundaries.

- The paper focuses primarily on successful cases without adequately exploring when and why the approach fails. A dedicated analysis of failure modes would provide valuable insights for practical implementation.

- All experiments use multiple-choice question answering tasks. Testing the approach on more diverse task types (e.g., text generation, classification, summarization) would strengthen claims about general applicability.

- While the paper cites the MDL principle as theoretical motivation, the concrete connection between the DiSR algorithm and MDL minimization is not explicitly formulated. A mathematical demonstration of how DiSR minimizes description length would strengthen the theoretical foundation.

### Questions
- One concern about the experiments is the claim that DiSR "generates more trustworthy prompts," which extends beyond the presented evidence. Trustworthiness involves multiple dimensions beyond calibration (ECE), and the paper doesn't address aspects like factual accuracy, bias mitigation, or robustness to adversarial inputs. 


- Some sections could be more concise (e.g., parts of the introduction), and the novelty compared to related work could be highlighted. Moreover, the paper would benefit from a clearer statement of the specific research questions being addressed.


- The paper appropriately positions itself against existing APO methods. One gap is the limited discussion of work on modular prompting or structured prompt engineering beyond the mentioned PromptSource library. 


- Technical details are a little unclear. For example, how are the  prompts in stage 1 intialized, and how are the prompts optimized in two stages?

- Experimental settings are not clear by saying "For each domain, the source task set for Scaffold Discovery was constructed from a small set of distinct sub-tasks (e.g., different legal subjects for law, or diverse reasoning types for commonsense), chosen to represent a reasonable diversity of challenges within that domain. All remaining sub-tasks were held out as target tasks for evaluation."

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
5

### Summary
This paper introduces a novel framework, DiSR, for automatic prompt optimization that challenges the conventional treatment of prompts as monolithic text blocks. The authors propose that effective prompts can be decomposed into two components: a reusable, domain-invariant "Instructional Scaffold" that captures the general structure of a task domain, and a concise "Task-specific Residual" that fine-tunes the prompt for a specific task. Inspired by the Minimum Description Length (MDL) principle, their two-stage approach first discovers a general scaffold by optimizing a prompt across a diverse set of source tasks within a domain, and then efficiently learns the residual to adapt this scaffold to a new target task. The work provides extensive empirical validation, demonstrating that this compositional approach not only achieves competitive accuracy but also improves prompt reliability and computational efficiency compared to methods like OPRO that optimize prompts from scratch for every task.

### Strengths
The primary strength of this paper lies in its conceptual shift towards a compositional and reusable view of prompt engineering. This is a well-motivated and timely contribution, as the cost and inefficiency of re-optimizing prompts for every new task is a growing bottleneck. By grounding their framework in the MDL principle and drawing parallels to cognitive science, the authors provide a solid theoretical underpinning for their approach. The proposed DiSR method is elegant and practical, cleverly leveraging an existing optimization paradigm (LLM-as-optimizer from OPRO) to focus on the novel contributions of the decompositional framework itself. The empirical results are convincing, particularly the t-SNE visualization which offers strong, intuitive evidence for the existence of the claimed scaffold-centered semantic structure. Furthermore, the consistent improvements in Expected Calibration Error (ECE) are a significant result, highlighting that the method produces not just accurate, but more reliable and trustworthy prompts, which is a critical aspect for real-world deployment. The demonstration of amortized efficiency, where the initial cost of scaffold discovery is quickly offset by rapid adaptation for subsequent tasks, presents a compelling practical advantage.

### Weaknesses
Despite the promising results, the framework's effectiveness appears to be highly dependent on a well-curated set of source tasks to define a "task domain" for the initial scaffold discovery. The paper acknowledges that the definition of a domain is heuristic, which introduces a degree of subjectivity and potential fragility; if the source tasks are not sufficiently diverse or representative, the resulting scaffold may not generalize well, undermining the entire premise. The paper also notes that the performance benefits of DiSR are most pronounced on large-scale models like GPT-4, while its advantage diminishes on smaller models where monolithic baselines can perform comparably. This suggests the approach may not be a universal solution but rather a technique that specifically capitalizes on the advanced abstract reasoning capabilities of the most powerful models, potentially limiting its immediate applicability for users of smaller, more accessible models. Lastly, the significant upfront computational investment required for the Stage 1 scaffold discovery, while justified by long-term amortized gains, could still be a barrier in domains where few related tasks are available or where computational resources are constrained.

### Questions
1.	The quality of the discovered scaffold seems critical. How sensitive is the framework to the selection and number of source tasks in Stage 1? For example, how does the scaffold's performance on held-out tasks degrade if it is generated from only two source tasks versus the fuller set used in the experiments, or if an outlier task is included in the source set?
2.	The paper describes the "Task-specific Residual" as the semantic change from the scaffold. Operationally, however, the final prompt is generated by re-running the optimization initialized from the scaffold. Does the meta-prompt in Stage 2 explicitly instruct the LLM to adapt the scaffold, or does it simply treat the scaffold as a high-quality starting point for a standard optimization? Could you elaborate on whether it's possible to explicitly learn or isolate the "residual" itself, for example as a set of interpretable edit operations?
3.	Given that the benefits are most significant on larger models, do you hypothesize that this is purely due to the model's ability to better utilize the abstract scaffold, or could it be that the monolithic optimization process (like OPRO) is somehow less effective on these larger models, creating a wider gap for DiSR to outperform it?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes decomposing prompts into a reusable, domain-invariant "Instructional Scaffold" and a concise "Task-specific Residual." This challenges the standard, inefficient "monolithic" approach to prompt optimization, where each new task requires optimization from scratch. The authors introduce DiSR, a two-stage algorithm inspired by the Minimum Description Length (MDL) principle , which first discovers the shared scaffold from a set of source tasks and then rapidly adapts it for a new target task. Experiments demonstrate that this compositional approach achieves competitive accuracy while being significantly more efficient and generating more reliable and trustworthy (better-calibrated) prompts than monolithic baselines like OPRO, particularly on large-scale models and in specialized professional domains.

### Strengths
1. Novel Problem Formulation. I think the domain specific scaffold and task residual concept inspiring in the space of prompt engineering. We see positive results of the paradigm in many other cases, e.g., multi-task adapter in soft prompt tuning. The methods also document positive performance consistently. 

2. The empirical results are thorough. The paper documents consistent and superior performance on main experiments. It also present semantics analysis to visualize the scaffold difference in different domains.

### Weaknesses
My main concern is on the effectiveness of domain specific scaffold:

- By looking at the examples you give in the appendix, domain scaffold is very generic. How much can it be replaced with a manual reasonably good prompt as a starting point? The marginal contribution and residual and scaffold is not properly analyzed. 

- The definition of a domain is a bit vague. It remains unclear how sensitive the quality of the discovered scaffold is to the quantity and diversity of these source tasks. What if the the task it self is very diverse within one domain? For example, a task could be multi-choice numerical reasoning while other is financial summary. Are they considered one domain?

### Questions
1. In your analysis of residual portability, did you try add residual directly rather than pairing with scaffold of other domains? Also It could be informative to run ablation study of marginal contribution of residual and scaffold on task performance by keeping one while removing another. I look at the sample scaffold you give in the appendix, they seem pretty general. I am not sure how much they contribute to the task performance.

2. What is the starting prompt of ORPO?

### Soundness
3

### Presentation
4

### Contribution
3
