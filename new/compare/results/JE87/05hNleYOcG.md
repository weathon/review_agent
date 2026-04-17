# Review

## Summary
This paper proposes a framework for multi-turn jailbreaks on LLMs. The framework consists of three phases: planner, primer, and finisher. The planner phase generates a multi-step plan for the attack. The primer phase escalates the context with the first n-1 steps of the plan. The finisher phase delivers the final blow. The framework also includes lifelong learning, where successful strategies are stored in a memory bank and retrieved for future attacks. The paper evaluates the framework on several models and shows that it outperforms existing attacks.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
1. The paper proposes a new framework for multi-turn jailbreaks on LLMs, which consists of three phases: planner, primer, and finisher. The framework also includes lifelong learning, where successful strategies are stored in a memory bank and retrieved for future attacks. 
2. The paper evaluates the framework on several models and shows that it outperforms existing attacks.

## Weaknesses
1. The paper does not provide enough details on how the framework works. The paper should provide more information on how the planner, primer, and finisher phases are implemented, and how lifelong learning is integrated into the framework. 
2. The paper does not provide enough analysis on why the framework is effective. The paper should provide more insights into how the three phases work together to jailbreak the LLMs and why the framework is more effective than existing attacks.

## Questions
1. How does the framework ensure that the planner, primer, and finisher phases work together effectively? Is there any mechanism to coordinate between the three phases?
2. How does the lifelong learning component help the framework? Can you provide some examples of successful strategies retrieved from the memory bank and how they help the attack?
3. How does the framework handle the evolution of LLMs? Can the framework adapt to new models or updated versions of existing models?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4