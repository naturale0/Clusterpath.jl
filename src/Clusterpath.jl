module Clusterpath

using LinearAlgebra
using Random
using DataFrames
using Interpolations
using Distributed
using MultivariateStats
using Distributions
import Gaston

include("AVLTree.jl")

export clusterpath, plot_cluster, plot_path, assign_cluster
export clusterpath_pop, cond_mean_on_LR, find_truncation, find_split
export generate_mixture_normal

"""
    clusterpath!(y; α=0.1, jitter=1e-6, return_split=false, silence_warning=false)
Perform convex clustering via α-thresholded clusterpath algorithm to `y`.
To avoid ties, add a small perturbation (`jitter`) to each data, and then sort it.
Returns a dictionary of
* (1) idx: cluster indices assigned to each elements in `y`
* (2) lambda: λ values in each merge
* (3) alpha: solution corresponding to each observation where merge happened
* (4) n_cluster: number of clusters wrt merging algorithm
* (5) n_splits: number of total splits, in corresponding top-down algorithm
* (6) splits: sample split point (only get returned when `return_split=true`)
"""
function clusterpath(y; α=0.1, jitter=1e-6, return_split=false, silence_warning=false)
    n = length(y)

    # remove ties with small perturbation `jitter`.
    x = copy(y)
    Random.seed!(0)
    x += randn(n) * jitter

    ii = sortperm(sortperm(x))
    sort!(x)  # sort `x` in ascending order

    # Initialize variables
    n_cluster = n  # (n) number of clusters
    probability = 0

    ## linkage between clusters
    next = Array{Union{Missing,Int64},1}(2:n+1)
    next[n] = missing
    prev = Array{Union{Missing,Int64},1}(0:n-1)
    prev[1] = missing

    ## status of clusters
    cardinals = ones(Int64, n)  # (freq) cardinality(freq) of each clusters
    centroids = copy(x)
    distance = diff(centroids) ./ (cardinals[1:n_cluster-1] + cardinals[2:end])
    λ_old = 0
    idx_cluster = Array(1:n) # (ind) cluster indices of observations

    ## big merge tracker
    was_big = false  # true if last merge was big
    is_big = false   # true if current merge is big

    # store returns along the solution path
    alphas = Array{Array{Float64,1},1}([copy(centroids)[ii]])
    λs = Float64[λ_old]  # (lambda) lambdas
    idx_along_path = Array{Array{Int64,1},1}()#[Array(1:n)[ii]])
    n_cluster_along_path = Int[]#n]
    n_splits = Int64(0)
    splits = Float64[]
#    println(idx_cluster[ii])
#    println(round.(distance,digits=5))

    # tree initailization
    ## (tree of pairwise distance btw successive indices)
    tree = AVLTree{Float64}()
    for i in 1:n-1
        push!(tree, distance[i], i)
    end


    """
    function for status update at each lambda, except for alpha
    """
    function update_return!(; update_lambda=true)
        if update_lambda
            append!(λs, λ_old)
        end
        append!(idx_along_path, [idx_cluster[ii]])
        append!(n_cluster_along_path, n_cluster)
    end
    
    """
    function for status update just for alpha
    """
    function update_alpha!()
        walk = 1
        alpha = Array{Float64}(undef, n)
        while !ismissing(walk)
            if !ismissing(next[walk])
                #n_repeat = next[walk] - walk
                alpha[walk:next[walk]-1] .= centroids[walk] + λ_old *((n-next[walk]+1)-(walk-1))
            else
                alpha[walk:end] .= centroids[walk] + λ_old *(-(walk-1))
            end
            walk = next[walk]
        end
        append!(alphas, [alpha[ii]])
    end

    ## main loop
    while n_cluster > 1

        # find minimum distance
        pivot = minimum_node(tree.root)
        λ_old, i = pivot.data, pivot.idx  # λ of next merge (minimal distance)

        #println(cardinals[i], " ", cardinals[next[i]])
#        println((i, next[i]))

        # detect and handle big merge
        if min(cardinals[i], cardinals[next[i]]) > ceil(α * n)  # But the paper says ⌈n * α⌉
            #println(cardinals[i], " ", cardinals[next[i]], " ", i, " ", λ_old, " ", n)

            is_big = true
            probability = (cardinals[i] + cardinals[next[i]]) / n

            n_splits += 1
            if return_split
                a_1 = max((x[idx_cluster .== idx_cluster[i]])...)
                a_2 = min((x[idx_cluster .== idx_cluster[next[i]]])...)
                split = (cardinals[next[i]] * a_2 + cardinals[i] * a_1) / (cardinals[next[i]] + cardinals[i])
                append!(splits, split)
            end
        end

        if is_big || was_big
            if is_big
                was_big = true
                is_big = false
            else
                was_big = false
            end
            # update λ path
            update_return!()
        end

        # update status
        centroids[i] = (centroids[i] * cardinals[i] + centroids[next[i]] * cardinals[next[i]]) / (cardinals[i] + cardinals[next[i]])
        cardinals[i] += cardinals[next[i]]
        idx_cluster[idx_cluster .== next[i]] .= i
        #n_right[i] = n_right[next[i]]

        # update distance and tree
        delete!(tree, distance[i])

        if !ismissing(prev[i])
            delete!(tree, distance[prev[i]])
            d_prev = (centroids[i] - centroids[prev[i]]) / (cardinals[prev[i]] + cardinals[i])
            distance[prev[i]] = d_prev
            push!(tree, d_prev, prev[i])
        end
        if !ismissing(next[next[i]])
            delete!(tree, distance[next[i]])
            d_next = (centroids[next[next[i]]] - centroids[i]) / (cardinals[i] + cardinals[next[next[i]]])
            distance[i] = d_next
            push!(tree, d_next, i)
        end

        # update linkage
        next[i] = next[next[i]]
        if !ismissing(next[i])
            prev[next[i]] = i
        end

        n_cluster -= 1

        if is_big || was_big
            if is_big
                was_big = true
                is_big = false
            else
                was_big = false
            end
            # update λ path
            #update_return!()
            update_alpha!()
        end
    end

    # last update for status at n_cluster = 1
    update_return!(update_lambda=false)

    # if the last Big Merge has sum of probabilities < 50%,
    # consider the case as if no splits have ever occured
    if probability < 0.5
        if !silence_warning
            @warn "No clusters were found: the last big merge has probability $(probability) < 0.5"
        end
        idx_along_path = Array(1:n)
        n_cluster_along_path = n
        n_splits = 0
    end

    return Dict("lambda" => λs,
                "idx" => idx_along_path,
                "alpha" => alphas,
                "n_cluster" => n_cluster_along_path,
                "n_splits" => n_splits,
                "splits" => splits)

end


"""
    rep(x, times)
Julia equivalent of R's `rep`.
"""
function rep(x, times)
    vcat(fill.(x, times)...)
end


"""
    merge_dfs(df1, df2)
merge two solution path dataframes with `outerjoin()`
"""
function merge_dfs(df1, df2)
    outerjoin(df1, df2; on=[:obs, :lambda], makeunique=true)
end

"""
    cast_solution(x; α=0.1, silence_warning=false)

A generator function that casts a dataframe of solution path using `clusterpath()`. The function yields the solution path of each observation one at an iteration. This return will be used in `plot_path()` and `plot_cluster()`.

* `x`: the data for which we want to cluster.
* `α`: threshold for Big Merge Tracker procedure. `α` shoulbe be ≥ 0. The smaller the value of `α`, the more detailed the solution path will be. The larger the `α`, the faster the algorithm will run. (default: `0.1`)
"""
function cast_solution(x; α=0.1, silence_warning=false)

    n, p = try
             n, p = size(x)
             (n, p)
           catch e
             if isa(e, BoundsError)
               (length(x), 1)
             else
               throw(e)
             end
           end
#     println(n, " ", p)

    #lambdas = Array{Union{Nothing,Array{Float64,1}},1}(repeat([nothing], p))
    ccs = Array{Dict,1}(undef, p)
    lambdas = Array{Array{Float64,1},1}(undef, p)
    Threads.@threads for k=1:p
        cc = clusterpath(x[:, k]; α=α, silence_warning=silence_warning)
        ccs[k] = cc
        lambdas[k] = cc["lambda"]
    end

    lambdas = vcat(lambdas...)
    unique!(lambdas)
    sort!(lambdas)

    Channel{DataFrame}() do channel
        # report one obs at a time (for memory management)
        for i=1:n
            # initialize solution path matrix for ith obs
            solution = Array{Union{Float64, Missing}, 2}(missing, length(lambdas), p+2)
            solution[:, 1] .= i
            solution[:, 2] = lambdas

            # loop over dimensions
            for k=1:p
                cc = ccs[k]

                idx = [l in cc["lambda"] for l in lambdas]

                alpha_idx = 1
                for (lam_idx, is_in_lambda) in enumerate(idx)
                    if is_in_lambda
                        solution[lam_idx, k+2] = cc["alpha"][alpha_idx][i]
                        alpha_idx += 1
                    end
                end

#                 alpha = Float64[a[i] for a in cc["alpha"]]

#                 df = DataFrame(obs = repeat([i], n_path),
#                                lambda = lambdas[k],
#                                dim = alpha)#,
#                                #col = repeat(["dim$k"], n * n_path))

#                 if isnothing(solution)
#                     solution = df
#                 else
#                     solution = merge_dfs(solution, df)
#                 end

                # calculate cluster number based on n_splits
            end

            solution = DataFrame(solution)
            rename!(solution, vcat(["obs", "lambda"], ["dim$k" for k=1:p]))
#             sort!(solution, [:obs, :lambda])

            put!(channel, solution)
        end
    end
end


"""
    _prepare(solution_i; for_path::Bool)

prepare the return of `cast_solution()` for `plot_path()`, or `plot_cluster()`.
* `solution_i`: single solution path element of `cast_solution()`.
* `for_path`: if `true` then prepare for `plot_path()`, `false` then prepare for `plot_cluster()`.
"""
function _prepare!(solution_i; for_path::Bool)
    p = size(solution_i, 2) - 2
    n = length(unique(solution_i.obs))

    #unique!(solution_i, vcat(["obs"], ["dim$k" for k=1:p]))
    #Impute.interp!(solution_i)

    if !for_path
        nothing
        #solution_i[1:end-n_node-1, :] .= missing
        #dropmissing!(solution_i)
    end

    for j=1:p
        idx_missing = ismissing.(solution_i[!, "dim$j"])
        #trunc = findlast(.~idx_missing)

        lambda_obs = solution_i[.~idx_missing, "lambda"]
        point_obs = Float64.(solution_i[.~idx_missing, "dim$j"])
        itp = interpolate((lambda_obs,), point_obs, Gridded(Linear()))
        etp = extrapolate(itp, Flat())

        solution_i[idx_missing, "dim$j"] = etp.(solution_i.lambda[idx_missing])
    end

    #Impute.locf!(solution_i)
end


"""
    plot_path(x; α=0.1, gt=ones(Int64, size(x,1)), force2dim=true, show::Bool, savefig=false, fname="path_plot", verbose=false)

plot clusterpath with the data(`x`).
If the dimension of `x` is greater than 2, only plot a combination of first two dimensions.

***`Gaston.jl` and `gnuplot` should be installed and on the PATH of your system. Install gnuplot [here](https://sourceforge.net/projects/gnuplot/files/gnuplot/).***

* `x`: the data to cluster.
* `α`: threshold for BMT-clusterpath
* `gt`: the ground truth labels.
* `force2dim`: force dim>2 data to 2-dim data (default: `true`)
* `show`: whether to show the plot in the notebook. **Highly recommended not to show if the number of samples is large.**
* `savefig`: whether to save the figure as a PNG file. (default: `false`)
* `fname`: image file name to be used when `savefig` is `true`. (default: `"path_plot"`)
"""
function plot_path(x; α=0.1, gt=ones(Int64, size(x,1)), force2dim=true, show::Bool, savefig=false, fname="path_plot", verbose=false)
    n_dim = length(size(x)) == 1 ? 1 : size(x, 2)
    if n_dim > 2 && !force2dim
        x = x[:, 1:3]
    elseif (n_dim > 2 && force2dim) || n_dim == 2
        x = x[:, 1:2]
    end
    solution = cast_solution(x, α=α, silence_warning=true)
#     n_plot = Int64(n_dim * (n_dim-1) / 2)
#     if n_plot > 6
#         n_plot = 6
#         n_dim = 4
#     end

    # initialize plot
#     if n_dim > 2
#         p = Plots.plot(layout=(1,n_plot), size=(400*n_plot, 400), legend=false)
#         counter = 0
#             for i=1:n_dim
#                 for j=(i+1):n_dim
#                     counter += 1
#                     Plots.plot!(p[counter], view(x, :, i), view(x, :, j), seriestype=:scatter, color=gt, alpha=.2)
#                 end
#             end
    if n_dim > 2 && !force2dim
        #p = Plots.plot(view(x, :, 1), view(x, :, 2),
        #               #series_annotations = text.(1:size(x, 1), :bottom),
        #               seriestype=:scatter, color=gt, alpha=.2, legend=false)
        clst = unique(gt)
        p = Gaston.scatter3(
                x[:, 1], x[:, 2], x[:, 3], supp=gt .% 8, variable="using 1:2:3:(4)",
                pointtype="ecircle", lw=0.75, lc="palette",
                Gaston.Axes(key="noautotitle",
                            grid=:on,
                            style="fill transparent solid 0.005;
                                   set style circle radius screen 0.005;
                                   set border 21;
                                   unset colorbox;
                                   set auto fix;
                                   set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                                   set cbrange [0:7];
                                   set size square 1,1;
                                   set format y \"\"; set format x \"\"; set format z \"\"")
            )
#         p = Gaston.plot(
#                 x[:, 1], x[:, 2], z=x[:, 3], supp=gt.%8, variable="using 1:2:3:4", w=:circles, lw=0.75, lc="palette",
#                 Gaston.Axes(key="noautotitle",
#                             style="fill transparent solid 0.005;
#                                    set style circle radius screen 0.005;
#                                    set border 3;
#                                    unset colorbox;
#                                    set auto fix;
#                                    set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
#                                    set cbrange [0:7];
#                                    set size square 1,1;
#                                    set format y \"\"; set format x \"\"")
#             )
    elseif (n_dim > 2 && force2dim) || n_dim == 2
        #p = Plots.plot(view(x, :, 1), view(x, :, 2),
        #               #series_annotations = text.(1:size(x, 1), :bottom),
        #               seriestype=:scatter, color=gt, alpha=.2, legend=false)
        clst = unique(gt)
        p = Gaston.plot(
                x[:, 1], x[:, 2], supp=gt.%8, variable="using 1:2:3", w=:circles, lw=0.75, lc="palette",
                Gaston.Axes(key="noautotitle",
                            grid=:on,
                            style="fill transparent solid 0.005;
                                   set style circle radius screen 0.005;
                                   set border 3;
                                   unset colorbox;
                                   set auto fix;
                                   set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                                   set cbrange [0:7];
                                   set size square 1,1;
                                   set format y \"\"; set format x \"\"")
            )
        #if length(clst) > 1
        #    for cl in clst[2:end]
        #        Gaston.plot!(x[gt .== cl, 1], x[gt .== cl, 2], w=:circles, lw=0.75)
        #    end
        #end
    else
        n = size(x, 1)
        #p = Plots.plot(zeros(n), x, seriestype=:scatter, color=gt, alpha=.2, legend=false)

        ## USE GASTON (GNUPLOT) FOR FASTER PLOTTING
        clst = unique(gt)
        p = Gaston.plot(
                zeros(n), x, supp=gt.%8, variable="using 1:2:3", w=:circles, lw=0.75, lc="palette",
                Gaston.Axes(key="noautotitle",
                            grid=:on,
                            style="fill transparent solid 0.005;
                                   set style circle radius screen 0.01;
                                   set border 3;
                                   unset colorbox;
                                   set auto fix;
                                   set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                                   set cbrange [0:7];
                                   set size square 1,1;
                                   set format y \"\"; set format x \"\"")
            )
        #if length(clst) > 1
        #    for cl in clst[2:end]
        #        Gaston.plot!(zeros(n)[gt .== cl], x[gt .== cl], w=:circles, lw=0.75)
        #    end
        #end
    end

    # draw paths, one at a time
    i = 1
    for solution_i in solution
        _prepare!(solution_i; for_path=true)
        #println(solution_i)

#         if n_dim > 2
#             counter = 0
#             for i=1:n_dim
#                 for j=(i+1):n_dim
#                     counter += 1
#                     Plots.plot!(p[counter],
#                                 view(solution_i, "dim$i"), view(solution_i, "dim$j"),
#                                 seriestype=:path, color="black", alpha=0.25, legend=false)
#                 end
#             end
        if n_dim > 2 && !force2dim
            verbose && (i % 500 == 0) && print("$(i) ")

            #Plots.plot!(p, view(solution_i, "dim1"), view(solution_i, "dim2"),
            #            seriestype=:path, color="black", alpha=0.25, legend=false)
            Gaston.scatter3!(view(solution_i, "dim1"), view(solution_i, "dim2"), view(solution_i, "dim3"), 
                            w=:lines, lw=0.05, lc="black")
#             Gaston.plot!(view(solution_i, "dim1"), view(solution_i, "dim2"), z=view(solution_i, "dim3"),
#                          w=:lines, lw=0.05, lc="black")
        elseif (n_dim > 2 && force2dim) || n_dim == 2
            verbose && (i % 500 == 0) && print("$(i) ")

            #Plots.plot!(p, view(solution_i, "dim1"), view(solution_i, "dim2"),
            #            seriestype=:path, color="black", alpha=0.25, legend=false)
            Gaston.plot!(view(solution_i, "dim1"), view(solution_i, "dim2"),
                         w=:lines, lw=0.05, lc="black")
        else
            verbose && (i % 500 == 0) && print("$(i) ")

            #Plots.plot!(p, view(solution_i, "lambda"), view(solution_i, "dim1"),
            #            seriestype=:path, color="black", alpha=0.25, legend=false)
            Gaston.plot!(view(solution_i, "lambda"), view(solution_i, "dim1"),
                         w=:lines, lw=0.1, lc="black")
        end

        i += 1
    end

    if savefig
        #png(p, fname*".png")
        Gaston.save(term = "png",
                    output= "$(fname).png",
                    size = "500,500",
                    background = "white")
    end

    if show
        return(p)
    end
end


"""
    assign_cluster(x; α=0.1, verbose=false)

assign cluster to each of the observations in `x`.
returns an array of length=size(x, 1) of cluster indices.

* `x`: data
* `α`: threshold for BMT-clusterpath
"""
function assign_cluster(x; α=0.1, verbose=false)

    n = size(x, 1)
    p = length(size(x)) == 1 ? 1 : size(x, 2)

    final = Array{Float64,2}(undef, n, p)

    clst = repeat([""], n)
    for i=1:p
        splits = clusterpath(x[:, i], α=α, return_split=true, silence_warning=true)["splits"]
        clst_dim = zeros(Int64, n)
        for s in splits
            clst_dim .+= Int64.(x[:, i]  .> s)
        end
        clst .*= string.(clst_dim)
    end

    DataFrames.levelcode.(DataFrames.CategoricalArray(clst))
end
# function assign_cluster!(x, solution; n_node=1, verbose=false)

#     n = size(x, 1)
#     p = length(size(x)) == 1 ? 1 : size(x, 2)

#     final = Array{Float64,2}(undef, n, p)
#     lambda = nothing

#     i = 1
#     for solution_i in solution
#         if verbose && (i % 500 == 0)
#             print("$(i) ")
#         end
#         _prepare!(solution_i; for_path=false, n_node=n_node)

#         #println(solution_i[end, 3:end])
#         final[i, :] = collect(solution_i[end-n_node, 3:end])
#         isnothing(lambda) && (lambda = solution_i.lambda[end-n_node])
#         i += 1
#     end

#     centroids = unique(final, dims=1)
#     n_centroids = size(centroids, 1)

#     idx_x = Array{Int64}(undef, n)

#     Threads.@threads for i=1:n_centroids
#         filter = vec(reduce(&, final .== centroids[i, :]', dims=2))
#         idx_x[filter] .= i
#     end

#     idx_x, lambda
# end


"""
    plot_cluster(x; α=0.1, n_node=1, show::Bool, savefig=false, fname="plot_clst", verbose=false)

Plots the scatter plot of the data `x` colored according to the cluster assigned by clusterpath algorithm.
If the dimension of `x` is greater than 2, perform PCA and plot two PCs.

***`Gaston.jl` and `gnuplot` should be installed and on the PATH of your system. Install gnuplot [here](https://sourceforge.net/projects/gnuplot/files/gnuplot/). ***

* `x`: data
* `α`: threshold for BMT-clusterpath
* `n_node`: if greater than `1`, will assign clusters from previous merge status. (default: `1`)
* `show`: whether to show the figure.
* `savefig`: whether to save the figure as a png file. (default: `false`)
* `fname`: file name to save if `savefig` is true. (default: `"plot_clst"`)
* `verbose`: print out current iteration. (default: `false`)
"""
function plot_cluster(x; α=0.1, force2dim=true, show::Bool, savefig=false, fname="plot_clst", verbose=false)
    clst = assign_cluster(x; α=α, verbose=verbose)

    #solution = cast_solution(x, α=α)
    #solution_1 = take!(solution)

    k = levels(clst)[end]

    p = size(x, 2)
    
    if p > 3
        force2dim && (pca = fit(PCA, Array(x'); maxoutdim=2))
        !force2dim && (pca = fit(PCA, Array(x'); maxoutdim=3))
        x = MultivariateStats.transform(pca, x')'
    elseif p == 3
        force2dim && (pca = fit(PCA, Array(x'); maxoutdim=2); x = MultivariateStats.transform(pca, x')')
    end

    # use Gaston for plotting
    #uniq_clusters = unique(clst)
    #cl = uniq_clusters[1]
    #idx = clst .== cl
    if p > 2 && !force2dim
        plt = Gaston.scatter3(
                x[:, 1], x[:, 2], x[:, 3], supp=clst .% 8, variable="using 1:2:3:(4)",
                pointtype="ecircle", lw=0.75, lc="palette",
                Gaston.Axes(key="noautotitle",
                            grid=:on,
                            style="fill transparent solid 0.005;
                                   set style circle radius screen 0.005;
                                   set border 21;
                                   unset colorbox;
                                   set auto fix;
                                   set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                                   set cbrange [0:7];
                                   set size square 1,1;
                                   set format y \"\"; set format x \"\"; set format z \"\"")
            )
    elseif (p > 2 && force2dim) || p == 2
        plt = Gaston.plot(
            x[:, 1], x[:, 2], supp=clst .% 8, variable="using 1:2:3", 
            w=:circles, lw=0.75, lc="palette",
            Gaston.Axes(key="noautotitle",
                        grid=:on,
                        xlabel="\"k=$(k)\"",
                        style="fill transparent solid 0.005;
                               set style circle radius screen 0.005;
                               set border 3;
                               unset colorbox;
                               set auto fix;
                               set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                               set cbrange [0:7];
                               set size square 1,1;
                               set format y \"\"; set format x \"\"")
        )
    else
        plt = Gaston.plot(
            x, randn(length(x)) * 0.02, supp=clst .% 8, variable="using 1:2:3", 
            w=:circles, lw=0.75, lc="palette",
            Gaston.Axes(key="noautotitle",
                        grid=:on,
                        xlabel="\"k=$(k)\"",
                        style="fill transparent solid 0.005;
                               set style circle radius screen 0.005;
                               set border 3;
                               set yrange [-1:1];
                               unset colorbox;
                               set auto fix;
                               set palette defined ( 0 '#E41A1C', 1 '#377EB8', 2 '#4DAF4A', 3 '#984EA3', 4 '#FF7F00', 5 '#d2d200', 6 '#A65628', 7 '#F781BF' );
                               set cbrange [0:7];
                               set size square 1,1;
                               set format y \"\"; set format x \"\"")
        )
    end

#     if length(uniq_clusters) > 1
#         for cl in uniq_clusters[2:end]
#             idx = clst .== cl
# #             println(sum(idx))
#             Gaston.plot!(x[idx, 1], x[idx, 2], w=:circles, lw=0.75)
#         end
#     end

    if savefig
        Gaston.save(term = "png",
                    output= "$(fname).png",
                    size = "500,500",
                    background = "white")
    end

#     plot(x[:, 1], x[:, 2], color=clst,
#          seriestype=:scatter, legend=false, alpha=.2,
#          title="lambda=$(round(λ, digits=3)), k=$(k)")
    if show
        return(plt)
    end
end


function count_final_cluster(x; α=0.1)
    levels(assign_cluster(x, α=α))[end]
end


# ---- population procedure ---- #


"""
    pnorm(x)

Julia equivalent of R's `pnorm`
"""
function pnorm(x)
    cdf(Normal(), x)
end


"""
    dnorm(x)

Julia equivalent of R's `dnorm`
"""
function dnorm(x)
    pdf(Normal(), x)
end


# """
#     cond_mean_on_LR(L::T, R::T, a1::T, a2::T, a3::T, μ1::T, μ2::T, μ3::T) where T <: Real

# Conditional mean on (L, R), defined as μ_{L,R} = (∫_L^R f(x) dx)^(-1) / (∫_L^R x f(x) dx)
# """
# function cond_mean_on_LR(L::T, R::T, a1::T, a2::T, a3::T, μ1::T, μ2::T, μ3::T) where T <: Real
#     pL = a1 * pnorm(L-μ1) + a2 * pnorm(L-μ2) + a3 * pnorm(L-μ3)
#     pR = a1 * pnorm(R-μ1) + a2 * pnorm(R-μ2) + a3 * pnorm(R-μ3)

#     ∫_L = -(a1 * dnorm(L-μ1) + a2 * dnorm(L-μ2) + a3 * dnorm(L-μ3)) +
#           a1 * μ1 * pnorm(L-μ1) + a2 * μ2 * pnorm(L-μ2) + a3 * μ3 * pnorm(L-μ3)
#     ∫_R = -(a1 * dnorm(R-μ1) + a2 * dnorm(R-μ2) + a3 * dnorm(R-μ3)) +
#           a1 * μ1 * pnorm(R-μ1) + a2 * μ2 * pnorm(R-μ2) + a3 * μ3 * pnorm(R-μ3)

#     μ_LR = (∫_R - ∫_L)/(pR - pL)

#     (pL, pR, ∫_L, ∫_R, μ_LR)
# end


"""
    cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

Conditional mean on (L, R), defined as μ_{L,R} = (∫_L^R f(x) dx)^(-1) / (∫_L^R x f(x) dx)
"""
function cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)
    # (∫_l^r f(x) dx) = pR - pL
    pL = a1 * pnorm(L.-μ1) + a2 * pnorm(L.-μ2) #+ a3 * pnorm(L.-μ3)
    pR = a1 * pnorm(R.-μ1) + a2 * pnorm(R.-μ2) #+ a3 * pnorm(R.-μ3)

    # (∫_l^r x f(x) dx) = ∫_R - ∫_L
    ∫_L = -(a1 * dnorm(L.-μ1) + a2 * dnorm(L.-μ2)) + # + a3 * dnorm(L.-μ3)) +
          a1 * μ1 * pnorm(L.-μ1) + a2 * μ2 * pnorm(L.-μ2) #+ a3 * μ3 * pnorm(L.-μ3)
    ∫_R = -(a1 * dnorm(R-μ1) + a2 * dnorm(R-μ2)) + # + a3 * dnorm(R-μ3)) +
          a1 * μ1 * pnorm(R-μ1) + a2 * μ2 * pnorm(R-μ2) #+ a3 * μ3 * pnorm(R-μ3)
    ## adding a3/μ3 terms yields inaccurate result.

    # population conditional mean on (L, R)
    μ_LR = (∫_R .- ∫_L) ./ (pR .- pL)

    (pL, pR, ∫_L, ∫_R, μ_LR)
end


"""
    find_split(d, a1, a2, μ1, μ2, L, R;
               a3=0., μ3=0., find_deltas=false, find_split=!find_deltas)

Split the cluster (L, R) into subclusters that maximizes G_{L,R}.
Returns centered conditional mean (deltas) of the left and right subcluster
if `find_deltas` is true. Otherwise return the split point.
"""
function find_split(d, a1, a2, μ1, μ2, L, R;
                    a3=0., μ3=0., find_deltas=false, find_split=!find_deltas)
    # p, int, int_en evaluated at `d`
    ∫_d = -(a1 * dnorm(d.-μ1) + a2 * dnorm(d.-μ2)) + # + a3 * dnorm(d.-μ3)) +
          a1 * μ1 * pnorm(d.-μ1) + a2 * μ2 * pnorm(d.-μ2) #+ a3 * μ3 * pnorm(d.-μ3)
    p_d = a1 * pnorm(d.-μ1) + a2 * pnorm(d.-μ2) #+ a3 * pnorm(d.-μ3)

    pL, pR, ∫_L, ∫_R, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

    # find split point with normlized G'_{L,R} (=h)(supp. p.32)
    h = μ_LR .* (p_d .- pL).^2 + (pL + pR .- 2p_d) .* (∫_d .- ∫_L) - d .* (p_d .- pL) .* (pR .- p_d)
    split = d[2:end][h[2:end] .* h[1:end-1] .< 0]

    if find_deltas && (sum(μ1 .< split .< μ2) == 0)
        return(1., 1.)
    end

    # split point
    s = max(split[μ1 .< split .< μ2]...)

    if find_split
        return(s)
    else
        _, _, _, _, δ_1 = cond_mean_on_LR(L, s, a1, a2, a3, μ1, μ2, μ3)
        δ_1 -= (L + s)/2  # centering

        _, _, _, _, δ_2 = cond_mean_on_LR(s, R, a1, a2, a3, μ1, μ2, μ3)
        δ_2 -= (s + R)/2  # centering

        return(δ_1, δ_2)
    end

end


"""
    find_truncation(a1, a2, μ1, μ2,
                    R_ub=(μ2 + 8), L_lb=(μ1 - 8),
                    tuning_par1=0.01, tuning_par2=0.005,
                    a3=0, μ3=0, σ=1)

Find population right-truncation points for bimodal Normal mixtures.
Inputs:
    a1, a2, a3: weights for Normal distributions. a1 + a2 + a3 should be == 1
    μ1, μ2, μ3: means for normals μ1 = -μ2

Possible returns:
    0: negligible result (not a symmetric problem)
    -1: no truncation
"""
function find_truncation(a2, μ2;
                         a1=1-a2, μ1=-μ2, R_ub=μ2+8., L_lb=μ1-8.,
                         tuning_par1=0.01, tuning_par2=0.005,
                         a3=0., μ3=0., σ=1.)

    r = Array(R_ub:(-tuning_par1):μ1)

    # if the bimodal distn is symmetric, and a1 == a2,
    # then truncate both ends equally
    if a1 == a2 && μ2 == -μ1
        for i=1:length(r)

            # truncate both left and right with the same amount
            R = r[i]
            L = -R

            d = Array(L:0.01:R)
            δ_1, δ_2 = find_split(d, a1, a2, μ1, μ2, L, R; find_deltas=true)

            # L^* is the smallest L that satisfies δ_1 < 0 < δ_2. (supp. p.30)
            # Thus returns right-trunctation point
            if δ_1 < 0 < δ_2
                return(R)
            end
        end
    end

    # possible cases:
    #   a1 != a2 && μ2 != -μ1
    #   a1 != a2 && μ2 == -μ1
    if a1 != a2
        L_old = -1000
        for i=1:length(r)
            R = r[i]
            L = Array{Float64,1}(L_old:tuning_par2:(R-0.05))

            pL, pR, ∫_L, ∫_R, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

            # locate R_L using the equation μ_{L,R_L} - (L + R_L)/2.  (supp. p.30)
            temp = sign.(μ_LR .- (L .+ R)./2)

            if sum(temp[2:end] .* temp[1:end-1] .< 0) ≠ 0
                # this is L^* from (supp. p.28)
                L = L[2:end][temp[2:end] .* temp[1:end-1] .< 0][1]

                d = Array(L:0.01:R)
                δ_1, δ_2 = find_split(d, a1, a2, μ1, μ2, L, R; find_deltas=true)

                pL = a1 * pnorm(L-μ1) + a2 * pnorm(L-μ2) + a3 * pnorm(L-μ3)
                pR = a1 * pnorm(R-μ1) + a2 * pnorm(R-μ2) + a3 * pnorm(R-μ3)

                if sign((R-L)/(pR-pL) * (a1 * dnorm(L-μ1) + a2 * dnorm(L-μ2) + a3 * dnorm(L-μ3)) - 1) ≠ -1
                    println("🤔🤔 input is not good")
                end

                # L^* is the smallest L that satisfies δ_1 < 0 < δ_2. (supp. p.30)
                # Thus returns right-trunctation point
                if δ_1 < 0 < δ_2 && L_old ≤ L

                    # if L^* is achieved at i==1, it means there is no truncation.
                    if i == 1
                        return(-1)
                    end

                    return(R)
                end

                # increment L
                if L_old ≤ L
                    L_old = L
                end

                # if L > 0, it means roughly that L > m_ (m_≤0)  (supp. p.29 table and p.28)
                if L > 0
                    return(-1)
                end

            end
        end
    end

    # If a1 == a2 and μ1 != -μ2, then return 0
    # : do not need to assess in this case, since it can WLOG be reduced to a symmetric case)
    return(0)
end


"""
    clusterpath_pop(a2::T, μ2::T; a1=1-a2, μ1=-μ1, a3=0., μ3=0, σ=1) where T <: Real

Inputs:
    a2: weight on 2nd normal distribution
    μ2: mean of 2nd normal distribution
Returns:
    A dictionary of weights (`a1`, `a2`), means (`μ1`, `μ2`) and
    description of truncation (`L*`, `R*`) and split point (`s`).
"""
function clusterpath_pop(a2::T1, μ2::T2; a1=1-a2, μ1=-μ2, a3=0., μ3=0, σ=1.) where T1 <: Real where T2 <: Real
    # search right truncation point
    R = find_truncation(a2, μ2; a1=a1, μ1=μ1, a3=a3, μ3=μ3, σ=σ)

    # R == 0 or R == -1
    if R == 0 || R == -1
        d = Array(μ1:1e-04:μ2)
        # d[argmin(abs.(a1 * dnorm(d.-μ1) .- a2 * dnorm(d.-μ2)))]  ## ???????
        # d[argmin(a1 * dnorm(d.-μ1) .+ a2 * dnorm(d.-μ2))]

        return(Dict("a1" => a1, "a2" => a2,
                    "mu1" => μ1, "mu2" => μ2,
                    "s" => NaN, "L*" => NaN, "R*" => NaN))
    end

    # If R is a proper right-truncation point, then find split point
    L_lb = -3 * max(R, 3)  # lower bound of searching grid of L
    L_ub = R - 0.01  # upper bound
    L = Array(L_lb:5e-04:L_ub)

    _, _, _, _, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)
    δs = sign.(μ_LR .- (L .+ R)/2)

    # L^* is the smallest L that satisfies δ_1 < 0 < δ_2
    L = L[2:end][δs[2:end] .* δs[1:end-1] .< 0][1]

    d = Array(L:0.001:R)
    split = find_split(d, a1, a2, μ1, μ2, L, R)

    return(Dict("a1" => a1, "a2" => a2,
                "mu1" => μ1, "mu2" => μ2,
                "s" => split, "L*" => L, "R*" => R))
end


# ---- simulation data generation ---- #


"""
    generate_mixture_normal(n, m, p)

generate `n` observations from mixture of univariate normals 
each with standard deviation 1 and mean parameters `m` and proportion `p`.
"""
function generate_mixture_normal(n::Int64, m::Array{T1,1}=[-20., 0., 20.], p::Array{T2,1}=ones(length(m))/length(m)) where T1 <: Real where T2 <: Real
    @assert sum(p) ≈ 1
    mixed_normal = Array{Float64}(undef, 0)  # placeholder for the resulting mixture
    n_mixtures = floor.(Int64, n * p)  # sample sizes for each univariate normals
    
    # leftover n's is assigned to the first distribution > sum(n_mixtures)
    n_mixtures[1] += (n - sum(n_mixtures))
    
    # sample from Normal(0, 1)
    for i=1:length(m)
        mixed_normal = [mixed_normal; randn(n_mixtures[i]) .+ m[i]]
    end
    
    mixed_normal 
end


end  # end module
