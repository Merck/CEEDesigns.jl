### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fc17a594-02f3-46b5-a012-136ab5d0ba38
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate("..")

    using PlutoUI
    md"""
   Case [_TCGA-HT-8564_](https://portal.gdc.cancer.gov/cases/f625e522-226b-450f-af94-dd2f5adb605e?filters=%7B%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LGG%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%2C%22op%22%3A%22and%22%7D), Diagnosis _Astrocytoma, anaplastic_:
   	
   $(LocalResource("glioma_slide.jpeg", :width => 500, :style => "display: block; margin-left: auto; margin-right: auto;"))
       """
end

# ╔═╡ d66ad52c-4ac9-4a04-884b-bfa013f9acd9
begin
    using CSV, DataFrames
    data = CSV.File("../data/glioma_grading.csv") |> DataFrame
end

# ╔═╡ b7b62e65-9bd3-4bd8-9a71-2bea4ba109c4
begin
    using MLJ
    import BetaML, MLJModels
    using Random: seed!

    md"We fix the scientific types of features."
end

# ╔═╡ 6ebd71e0-b49a-4d12-b441-4805efc69520
begin
    using CEEDesigns, CEEDesigns.StaticDesigns

    md"""
    ### Cost-Efficient Feature Selection

    We use `evaluate_experiments` from `CEEDesigns.StaticDesigns` to evaluate the predictive accuracy over subsets of experiments. We use `LogLoss` as a measure of accuracy. It is possible to pass additional keyword arguments, which will be passed to `MLJ.evaluate` (such as `measure`, shown below).
    """
end

# ╔═╡ 7f2e19e0-4cf3-11ee-0e10-dba25ffccc94
md"""
# Cost-Efficient Experimental Design Towards Glioma Grading

Gliomas are the most common primary tumors of the brain. They can be graded as LGG (Lower-Grade Glioma) or GBM (Glioblastoma Multiforme), depending on the histological and imaging criteria. Clinical and molecular or genetic factors are also very crucial for the grading process. The ultimate aim is to identify the optimal subset of clinical, molecular or genetic, and histological features for the glioma grading process to improve diagnostic accuracy and reduce costs.

## Theoretical Framework

Let us consider a set of $n$ experiments $E = \{ e_1, \ldots, e_n\}$.

For each subset $S \subseteq E$ of experiments, we denote by $v_S$ the value of information acquired from conducting experiments in $S$.

In the cost-sensitive setting of CEEDesigns, conducting an experiment $e$ incurs a cost $(m_e, t_e)$. Generally, this cost is specified in terms of monetary cost and execution time of the experiment.

To compute the cost associated with carrying out a set of experiments $S$, we first need to introduce the notion of an arrangement $o$ of the experiments $S$. An arrangement is modeled as a sequence of mutually disjoint subsets of $S$. In other words, $o = (o_1, \ldots, o_l)$ for a given $l\in\mathbb N$, where $\bigcup_{i=1}^l o_i = S$ and $o_i \cap o_j = \emptyset$ for each $1\leq i < j \leq l$.

Given a subset $S$ of experiments and their arrangement $o$, the total monetary cost and execution time of the experimental design is given as $m_o = \sum_{e\in S} m_e$ and $t_o = \sum_{i=1}^l \max \{ t_e : e\in o_i\}$, respectively.

For instance, consider the experiments $e_1,\, e_2,\, e_3$, and $e_4$ with associated costs $(1, 1)$, $(1, 3)$, $(1, 2)$, and $(1, 4)$. If we conduct experiments $e_1$ through $e_4$ in sequence, this would correspond to an arrangement $o = (\{ e_1 \}, \{ e_2 \}, \{ e_3 \}, \{ e_4 \})$ with a total cost of $m_o = 4$ and $t_o = 10$.

However, if we decide to conduct $e_1$ in parallel with $e_3$, and $e_2$ with $e_4$, we would obtain an arrangement $o = (\{ e_1, e_3 \}, \{ e_2, e_4 \})$ with a total cost of $m_o = 4$, and $t_o = 3 + 4 = 7$.

Given the constraint on the maximum number of parallel experiments, we devise an arrangement $o$ of experiments $S$ such that, for a fixed tradeoff between monetary cost and execution time, the expected combined cost $c_{(o, \lambda)} = \lambda m_o + (1-\lambda) t_o$ is minimized (i.e., the execution time is minimized).

In fact, it can be readily demonstrated that the optimal arrangement can be found by ordering the experiments in set $S$ in descending order according to their execution times. Consequently, the experiments are grouped sequentially into sets whose size equals to the maximum number of parallel experiments, except possibly for the final set.

Continuing our example and assuming a maximum of two parallel experiments, the optimal arrangement is to conduct $e_1$ in parallel with $e_2$, and $e_3$ with $e_4$. This results in an arrangement $o = (\{ e_1, e_2 \}, \{ e_3, e_4 \})$ with a total cost of $m_o = 4$ and $t_o = 2 + 4 = 6$.

Assuming the information values $v_S$ and optimized experimental costs $c_S$ for each subset $S \subseteq E$ of experiments, we then generate a set of cost-efficient experimental designs.

### Application to Predictive Modeling 

Consider a dataset of historical readouts over $m$ features $X = \{x_1, \ldots, x_m\}$, and let $y$ denote the target variable that we want to predict.

We assume that each experiment $e \in E$ yields readouts over a subset $X_e \subseteq X$ of features.

Then, for each subset $S \subseteq E$ of experiments, we may model the value of information acquired by conducting the experiments in $S$ as the accuracy of a predictive model that predicts the value of $y$ based on readouts over features in $X_S = \bigcup_{e\in S} X_e$.
"""

# ╔═╡ 2edaa133-6907-44f9-b39c-da06dae3eead
md"""
## Glioma Grading Clinical and Mutation Dataset

In this dataset, the instances represent patient records of those diagnosed with brain glioma. The dataset is publicly available at [Glioma Grading Clinical and Mutation Features](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset). It is constructed based on the TCGA-LGG and TCGA-GBM brain glioma projects available at the [NIH GDC Data Portal](https://portal.gdc.cancer.gov).

Each record is characterized by 

- 3 clinical features (age, gender, race),
- 5 mutation factors (IDH1, TP53, ATRX, PTEN, EGFR; each of which can be 'mutated' or 'not_mutated').

We list somatic mutations with the highest number of affected cases in cohort:
"""

# ╔═╡ 6737a98e-20b9-4016-85f6-c8f4a07d04f8
md"""$(LocalResource("mutations.png", :width => 500, :style => "display: block; margin-left: auto; margin-right: auto;"))"""

# ╔═╡ 1364c91a-4331-4a1b-8c3d-fb40743c1df7
md"We load the dataset:"

# ╔═╡ 87f9826f-865f-4d1f-b122-e1ccae51bfdf
md"## Assessing the Predictive Accuracy

We specify the clinical features and mutation factors."

# ╔═╡ 9835f38b-b720-429d-8a8b-eab742fe7e05
features_clinical = ["Age_at_diagnosis", "Gender", "Race"]

# ╔═╡ 0bb794a2-d0b4-493d-8a10-d19a515271ec
features_mutation = ["IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16"]

# ╔═╡ 1c72a85d-19cb-44f3-b330-a312b9ef9b7c
md"Classification target is just the glioma grade"

# ╔═╡ 7e2bb98c-1175-4964-8c41-8e3ac8e6eb9f
target = "Grade"

# ╔═╡ d74684f6-7fd4-41c8-9917-d99dfc1f5f64
md"In the cost-sensitive setting of CEEDesigns, obtaining additional experimental evidence comes with a cost. We assume that each gene mutation factor is obtained through a separate experiment."

# ╔═╡ f176f8cf-8941-4257-ba41-60fff864aa56
# We assume that each feature is measured separately and the measurement incurs a monetary cost.
experiments = Dict(
    ## experiment => features
    "TP53" => 3.0 => ["TP53"],
    "EGFR" => 2.0 => ["EGFR"],
    "PTEN" => 4.0 => ["PTEN"],
    "ATRX" => 2.0 => ["ATRX"],
    "IDH1" => 3.0 => ["IDH1"],
    "CIC" => 1.0 => ["CIC"],
    "MUC16" => 2.0 => ["MUC16"],
)

# ╔═╡ 9547fc15-7dff-49a3-b5d2-43b54a6dc443
md"""
### Classifier

We use a package called [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) to evaluate the predictive accuracy over subsets of experimental features.
"""

# ╔═╡ 61219167-805c-4624-932e-d050a13ada07
begin
    types = Dict(
        name => Multiclass for name in [Symbol.(features_mutation); :Grade; :Gender; :Race]
    )

    data_typefix = coerce(data, types)
    schema(data_typefix)
end

# ╔═╡ 63b7b1f3-660d-4a42-8b2c-3c0851d2b859
md"Next, we choose a particular predictive model that will evaluated in the sequel. We can list all models that are compatible with the dataset:"

# ╔═╡ d0973a67-f24e-4ffd-8343-7fe7be400d07
models(matching(data_typefix, data_typefix[:, target]))

# ╔═╡ f797ccd0-e98e-4617-a966-455469d16096
md"Eventually, we fix `RandomForestClassifier` from [BetaML](https://github.com/sylvaticus/BetaML.jl)"

# ╔═╡ b987567b-f500-4837-8f23-412a2eecec51
classifier = @load RandomForestClassifier pkg = BetaML verbosity = -1

# ╔═╡ 88a20956-e665-4b0f-81f2-0e7ec8a1e28f
model = classifier(; n_trees = 8, max_depth = 5)

# ╔═╡ c1860fca-f304-4bb4-9266-5a6d71467e27
# ╠═╡ show_logs = false
begin
    seed!(1) # evaluation process generally is not deterministic
    perf_eval = evaluate_experiments(
        experiments,
        model,
        data_typefix[!, Not(target)],
        data_typefix[!, target];
        zero_cost_features = features_clinical,
        measure = LogLoss(),
        resampling = CV(; nfolds = 10),
    )
end

# ╔═╡ ccdef55c-0570-404a-bb3d-ea0b9487c321
# ╠═╡ show_logs = false
begin
    using Plots
    plotly()
    designs = efficient_designs(experiments, perf_eval)

    function plot_front_invert(
        designs;
        grad = cgrad(:Paired_12),
        xlabel = "combined cost",
        ylabel = "information measure",
        labels = make_labels(designs),
    )
        xs = map(x -> x[1][1], designs)
        ys = map(x -> x[1][2], designs)

        p = scatter(
            [xs[1]],
            [1 - ys[1]];
            xlabel,
            ylabel,
            label = labels[1],
            c = grad[1],
            mscolor = nothing,
            fontsize = 16,
            legendposition = :bottomright,
            title = "cost-efficient experimental designs",
        )
        for i = 2:length(designs)
            if xs[i] < 10_000
                scatter!(
                    p,
                    [xs[i]],
                    [1 - ys[i]];
                    label = labels[i],
                    c = grad[i],
                    mscolor = nothing,
                )
            end
        end

        return p
    end

    plot_front_invert(
        designs;
        labels = make_labels(designs),
        ylabel = "accuracy (1-logloss)",
    ) # fill("", (1, length(designs)))
end

# ╔═╡ 01646835-55a9-42af-8eb3-29816c9780b7
md"We proceed to construct the set of cost-efficient experimental designs. In doing so, our goal is to identify the optimal sets of mutation factors for the glioma grading task, balancing the conflicting objectives of enhancing prediction accuracy and reducing incurred costs."

# ╔═╡ dcd790bd-36a5-4536-9e21-4f54fe2cf4e4
begin
    function predict(
        X;
        positive_label = 1,
        negative_label = 0,
        sensitivity::Float64,
        specificity::Float64,
    )
        y_pred = similar(X, Union{typeof(positive_label),typeof(negative_label)})

        # simulate a predictor with given sensitivity and specificity
        for i in eachindex(X)
            if X[i] == positive_label
                y_pred[i] = rand() < sensitivity ? positive_label : negative_label
            else
                y_pred[i] = rand() < (1 - specificity) ? positive_label : negative_label
            end
        end

        return y_pred
    end

    sensitivity_slider =
        @bind sensitivity Slider(0:0.01:1, default = 0.73, show_value = true)
    specificity_slider =
        @bind specificity Slider(0:0.01:1, default = 0.74, show_value = true)
    cost_slider = @bind cost Slider(1:1:10, default = 1, show_value = true)

    md"""
 ## Assessing the Impact of Histoathology Image Analysis on the Experimental Cost-Efficiency

 Building on the previous example, we will consider the introduction of a new feature in the task of glioma grading, where this feature will essentially function as a predictor of the glioma grade. 

 In [Glioma Grading via Analysis of Digital Pathology Images Using Machine Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7139732/), the authors proposed a computational method that exploits pattern analysis methods for grade prediction in gliomas using digital pathology images. 

 From the abstract, _according to the remarkable performance of computational approaches in the digital pathology domain, we hypothesized that machine learning can help to distinguish low-grade gliomas (LGG) from high-grade gliomas (HGG) by exploiting the rich phenotypic information that reflects the microvascular proliferation level, mitotic activity, presence of necrosis, and nuclear atypia present in digital pathology images. A set of 735 whole-slide digital pathology images of glioma patients (median age: 49.65 years, male: 427, female: 308, median survival: 761.26 days) were obtained from TCGA. Sub-images that contained a viable tumor area, showing sufficient histologic characteristics, and that did not have any staining artifact were extracted. Several clinical measures and imaging features, including conventional (intensity, morphology) and advanced textures features (gray-level co-occurrence matrix and gray-level run-length matrix), extracted from the sub-images were further used for training the support vector machine model with linear configuration._

 $(LocalResource("architecture.png", :width => 600, :style => "display: block; margin-left: auto; margin-right: auto;"))

 The authors aimed to evaluate the combined effect of conventional imaging, clinical, and texture features by assessing the predictive value of each feature type and their combinations through a predictive classifier.

 For our specific intent, we will focus on the predictive accuracy of a classifier that utilizes only imaging features.
 	
 $(LocalResource("accuracy.png", :width => 600, :style => "display: block; margin-left: auto; margin-right: auto;"))

 We will artificially produce the grade predictions, modelling them as predictor outputs with the defined sensitivity and specificity. Note that we will not incorporate any further correlations with other features such as clinical factors, mutation factors, or histology.

 In addition, we consider the cost of the predictive classifier development.

 | parameter | value picker |
 | :-----: | :------: |
 | sensitivity | $sensitivity_slider |
 | specificity | $specificity_slider |
 | cost | $cost_slider |
 """
end

# ╔═╡ c2121983-7135-4833-a33a-2dbc331272f3
digital_pathology = map(
    x -> x == "LGG" ? "lower grade" : "glioblastoma",
    predict(
        data_typefix[!, target];
        positive_label = "GBM",
        negative_label = "LGG",
        sensitivity,
        specificity,
    ),
)

# ╔═╡ a271f163-046e-40cc-8134-c7619bf6a63a
begin
    experiments_new_feature =
        push!(copy(experiments), "digital_pathology" => cost => ["digital_pathology"])
end

# ╔═╡ 9ee21001-8404-4d5d-875b-6ea7952f65b8
md"We add the new feature to the dataset:"

# ╔═╡ bc31504a-9374-4da3-9d68-030994ba8fcf
begin
    data_new_feature = data_new_feature = copy(data_typefix)
    data_new_feature.digital_pathology = digital_pathology

    data_new_feature_typefix = coerce(
        data_new_feature,
        Dict(
            (
                name => Multiclass for
                name in [Symbol.(features_mutation); :Grade; :Gender; :Race]
            )...,
            :new_feature => Multiclass,
        ),
    )

    data_new_feature_typefix[
        !,
        [
            ["IDH1", "digital_pathology"]
            setdiff(names(data_new_feature_typefix), ("digital_pathology", "IDH1"))
        ],
    ]
end

# ╔═╡ 0c58ddff-ad6f-4e80-8f1a-3c0c8d5245d2
# ╠═╡ show_logs = false
begin
    seed!(1) # evaluation process generally is not deterministic
    perf_eval_new_feature = evaluate_experiments(
        experiments_new_feature,
        model,
        data_new_feature_typefix[!, Not(target)],
        data_new_feature_typefix[!, target];
        zero_cost_features = features_clinical,
        measure = LogLoss(),
        resampling = CV(; nfolds = 10),
    )
end

# ╔═╡ 3d710dad-9b8c-47a3-966b-9718cb76fad8
md"We evaluate performance measures across different experimental subsets, comparing those that include the histology feature and those that do not."

# ╔═╡ 6ed9b5e5-754c-47f6-b1e4-2514d23039d2
begin
    evals1_new_feature = filter(x -> "digital_pathology" ∉ x[1], perf_eval_new_feature)
    xs1_new_feature =
        sort!(collect(keys(evals1_new_feature)); by = x -> -perf_eval_new_feature[x])
    #xs1_new_feature = xs1_new_feature[begin:2:end]
    ys1_new_feature = map(xs1_new_feature) do x
        return if perf_eval_new_feature[x] isa Number
            perf_eval_new_feature[x]
        else
            perf_eval_new_feature[x].loss
        end
    end

    evals2_new_feature = filter(x -> "digital_pathology" ∈ x[1], perf_eval_new_feature)
    xs2_new_feature_new_feature = sort!(
        collect(keys(evals2_new_feature));
        by = x -> -perf_eval_new_feature[setdiff(x, ["histology"])],
    )
    #xs2_new_feature_new_feature = xs2_new_feature_new_feature[begin:2:end]
    ys2_new_feature = map(xs2_new_feature_new_feature) do x
        return if perf_eval_new_feature[x] isa Number
            perf_eval_new_feature[x]
        else
            perf_eval_new_feature[x].loss
        end
    end

    p_evals_new_feature = Plots.sticks(
        1:length(xs2_new_feature_new_feature),
        1 .- min.(ys1_new_feature, ys2_new_feature);
        ticks = 1:4:length(evals2_new_feature),
        #xformatter=i -> isempty(xs2_new_feature_new_feature[Int(i)]) ? "∅" : join(xs1_new_feature[Int(i)], ", "),
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel = "accuracy",
        c = CEEDesigns.colorant"rgb(110,206,178)",
        label = "w/ histology feature",
        xrotation = 50,
    )

    Plots.sticks!(
        p_evals_new_feature,
        1:length(xs1_new_feature),
        1 .- ys1_new_feature;
        ticks = 1:4:length(evals1_new_feature),
        xformatter = i ->
            isempty(xs1_new_feature[Int(i)]) ? "∅" : join(xs1_new_feature[Int(i)], ", "),
        guidefontsize = 8,
        tickfontsize = 8,
        ylabel = "accuracy",
        c = CEEDesigns.colorant"rgb(104,140,232)",
        label = "w/o histology feature",
        width = 2,
        xrotation = 50,
    )
end

# ╔═╡ 765ed41a-50e5-4989-98a1-1d2abfa3ba28
md"We proceed to construct the set of cost-efficient experimental designs, comparing the frontier that incorporates the histology feature with the one that does not."

# ╔═╡ 95567d4d-6594-4a25-8449-b7b95738c56c
# ╠═╡ show_logs = false
begin
    experiments_new_feature_no_feature = copy(experiments_new_feature)
    delete!(experiments_new_feature_no_feature, "digital_pathology")

    seed!(1)
    perf_eval_new_feature_no_feature = evaluate_experiments(
        experiments_new_feature_no_feature,
        model,
        data_new_feature_typefix[!, Not(target)],
        data_new_feature_typefix[!, target];
        zero_cost_features = features_clinical,
        measure = LogLoss(),
        resampling = CV(; nfolds = 10),
    )

    for (k, v) in perf_eval_new_feature_no_feature
        perf_eval_new_feature[k] = v
    end

    design_new_feature = efficient_designs(experiments_new_feature, perf_eval_new_feature)

    design_new_feature_no_feature = efficient_designs(
        experiments_new_feature_no_feature,
        perf_eval_new_feature_no_feature,
    )

    p_new_feature = scatter(
        map(x -> x[1][1], design_new_feature),
        map(x -> 1 - x[1][2], design_new_feature);
        xlabel = "combined cost",
        ylabel = "accuracy",
        label = "w/ histology feature",
        c = CEEDesigns.colorant"rgb(110,206,178)",
        mscolor = nothing,
        fontsize = 16,
        #fill = (0, CEEDesigns.colorant"rgb(110,206,178)"),
        fillalpha = 0.2,
        legend = :bottomright,
    )

    scatter!(
        p_new_feature,
        map(x -> x[1][1], design_new_feature_no_feature),
        map(x -> 1 - x[1][2], design_new_feature_no_feature);
        label = "w/o histology feature",
        c = CEEDesigns.colorant"rgb(104,140,232)",
        mscolor = nothing,
        fontsize = 16,
        #fill = (0, CEEDesigns.colorant"rgb(104,140,232)"),
        fillalpha = 0.15,
        title = "sensitivity = $sensitivity, specificity = $specificity, cost = $cost",
    )
end

# ╔═╡ 53023ce3-2c74-46b7-82eb-589a78ca89c0
md"This is a dynamic illustration demonstrating the efficient frontiers generated for a range of predictive model parameters."

# ╔═╡ 6900000d-8bfd-4f8b-bf5b-46cefce652e4
LocalResource("anim.gif")

# ╔═╡ 471eaf63-ca16-486e-b63a-ea3852768a0f
html"""<style>
main {
    text-align: justify;
  	text-justify: inter-word;
}
"""

# ╔═╡ Cell order:
# ╟─7f2e19e0-4cf3-11ee-0e10-dba25ffccc94
# ╟─2edaa133-6907-44f9-b39c-da06dae3eead
# ╟─6737a98e-20b9-4016-85f6-c8f4a07d04f8
# ╟─fc17a594-02f3-46b5-a012-136ab5d0ba38
# ╟─1364c91a-4331-4a1b-8c3d-fb40743c1df7
# ╟─d66ad52c-4ac9-4a04-884b-bfa013f9acd9
# ╟─87f9826f-865f-4d1f-b122-e1ccae51bfdf
# ╟─9835f38b-b720-429d-8a8b-eab742fe7e05
# ╟─0bb794a2-d0b4-493d-8a10-d19a515271ec
# ╟─1c72a85d-19cb-44f3-b330-a312b9ef9b7c
# ╟─7e2bb98c-1175-4964-8c41-8e3ac8e6eb9f
# ╟─d74684f6-7fd4-41c8-9917-d99dfc1f5f64
# ╟─f176f8cf-8941-4257-ba41-60fff864aa56
# ╟─9547fc15-7dff-49a3-b5d2-43b54a6dc443
# ╟─b7b62e65-9bd3-4bd8-9a71-2bea4ba109c4
# ╟─61219167-805c-4624-932e-d050a13ada07
# ╟─63b7b1f3-660d-4a42-8b2c-3c0851d2b859
# ╟─d0973a67-f24e-4ffd-8343-7fe7be400d07
# ╟─f797ccd0-e98e-4617-a966-455469d16096
# ╟─b987567b-f500-4837-8f23-412a2eecec51
# ╟─88a20956-e665-4b0f-81f2-0e7ec8a1e28f
# ╟─6ebd71e0-b49a-4d12-b441-4805efc69520
# ╟─c1860fca-f304-4bb4-9266-5a6d71467e27
# ╟─01646835-55a9-42af-8eb3-29816c9780b7
# ╟─ccdef55c-0570-404a-bb3d-ea0b9487c321
# ╟─dcd790bd-36a5-4536-9e21-4f54fe2cf4e4
# ╟─c2121983-7135-4833-a33a-2dbc331272f3
# ╟─a271f163-046e-40cc-8134-c7619bf6a63a
# ╟─9ee21001-8404-4d5d-875b-6ea7952f65b8
# ╟─bc31504a-9374-4da3-9d68-030994ba8fcf
# ╟─0c58ddff-ad6f-4e80-8f1a-3c0c8d5245d2
# ╟─3d710dad-9b8c-47a3-966b-9718cb76fad8
# ╟─6ed9b5e5-754c-47f6-b1e4-2514d23039d2
# ╟─765ed41a-50e5-4989-98a1-1d2abfa3ba28
# ╟─95567d4d-6594-4a25-8449-b7b95738c56c
# ╟─53023ce3-2c74-46b7-82eb-589a78ca89c0
# ╟─6900000d-8bfd-4f8b-bf5b-46cefce652e4
# ╟─471eaf63-ca16-486e-b63a-ea3852768a0f
