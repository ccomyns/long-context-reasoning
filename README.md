**Overview**

Investigating the impact of a long-context data mixture on the performance of LLMs on long-context tasks. The data mixture consists of codebases, sorted into buckets based on how interconnected (ie. dependency rich) the code is. 

A notable limitation of past studies is that a data mixture of entirely long-context data degrades the LLMs performance on short-context tasks and vice versa with short-context data and long-context tasks

In addition to the effects of the given data mixture, this project seeks to investigates these limitaitons by performing targeted weight updates to specific layers. Experiment idea: update only the earlier layers on long-context training data to try and teach the earlier layers to think at a higher-level of abstraction than the later layers. 


**Interesting LLM Behaviors**
1. Degradation in reasoning as context window grows
2. Dependency invocation rate (ie. tendency of model to rewrite, rather than reuse code)


**Motivations**
1. Agentic deployments and commercially valuable tasks require long-context reasoning
    - Autonomous research (AI-Scientist-v2 by Sakana AI)
    - Making a pitch deck 
    - Searching through the internet
    - etc...
2. Longer context windows as an alternative to new continual learning techniques
    - Memory is already the fundamental bottleneck from a GPU perspective
    - Continual learning techniques and individualized weight updates would exacerbate this 
    - In-context learning could effectively "teach" the model the relevant context


**Related Publications**
1. Gistify: Codebase-Level Understanding via Runtime Execution (2025)
2. How to Train Long-Context Language Models (Effectively) (2024)
3. Lost in the Middle: How Language Models Use Long Contexts (2023)
4. RULER: What’s the Real Context Size of Your Long-Context Language Models? (2024)
5. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (2024)
6. Extending Context Window of LLMs Via Position Interpolation (2023)

