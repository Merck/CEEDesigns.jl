# CEEDesigns.jl: Overview

A decision-making framework for the cost-efficient design of experiments, balancing the value of acquired experimental evidence and incurred costs. We have considered two different experimental setups, which are outlined below.

```@raw html
<figure style="float: right; width: 400px;">
  <img src="assets/front_static.png" alt="Pareto front of static experimental designs" width="400">
  <figcaption>Pareto front of experimental designs: each point is a set of experiments lying on the optimal tradeoff between information gain and incurred costs.</figcaption>
</figure>
```

## Static experimental designs
Here we assume that the same experimental design will be used for a population of examined entities, hence the word "static".

For each subset of experiments, we consider an estimate of the value of acquired information. To give an example, if a set of experiments is used to predict the value of a specific target variable, our framework can leverage a built-in integration with [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) to estimate predictive accuracies of machine learning models fitted over subset of experimental features.

In the cost-sensitive setting of CEEDesigns.jl, a user provides the monetary cost and execution time of each experiment. Given the constraint on the maximum number of parallel experiments along with a fixed tradeoff between monetary cost and execution time, we devise an arrangement of each subset of experiments such that the expected combined cost is minimized.

Assuming the information values and optimized experimental costs for each subset of experiments, we then generate a set of cost-efficient experimental designs.

```@raw html
<figure style="float: right; width: 400px;">
  <img src="assets/front_generative.png" alt="Personalized Pareto fronts of generative experimental designs" width="400">
  <figcaption>Pareto front for generative designs: the difference here is that a personalized front is generated for each experimental entity.</figcaption>
</figure>
```

## Generative experimental designs

We consider 'personalized' experimental designs that dynamically adjust based on the evidence gathered from the experiments. This approach is motivated by the fact that the value of information collected from an experiment generally differs across subpopulations of the entities involved in the triage process.

At the beginning of the triage process, an entity's prior data is used to project a range of cost-efficient experimental designs. Internally, while constructing these designs, we incorporate multiple-step-ahead lookups to model likely experimental outcomes and consider the subsequent decisions for each outcome. Then after choosing a specific decision policy from this set and acquiring additional experimental readouts (sampled from a generative model, hence the word "generative"), we adjust the continuation based on this evidence.

```@raw html
<figure style="float: left; width: 400px;">
  <img src="assets/search_tree.png" alt="Search tree of the Markov decision process over experimental designs" width="400">
  <figcaption>Explainability: a search tree produced by solving the Markov decision process, showing the information gain and accumulated cost of each path through the space of possible designs.</figcaption>
</figure>
```

We conceptualized the triage as a Markov decision process, in which we iteratively choose to conduct a subset of experiments and then, based on the experimental evidence, update our belief about the distribution of outcomes for the experiments that have not yet been conducted. The information value associated with the state, derived from experimental evidence, can be modeled through any statistical or information-theoretic measure such as the variance or uncertainty associated with the target variable posterior.

We implemented the following two variants of the decision-making process: Firstly, assuming that the decision-making process only terminates when the uncertainty drops below a given threshold, we minimize the expected resource spend. Secondly, we can optimize the value of experimental evidence, adjusted for the incurred experimental costs.

```@raw html
<div style="clear: both;"></div>
```