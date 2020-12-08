"""
    generate_mixture_normal(n, m, p)

generate `n` observations from mixture of univariate normals 
each with standard deviation 1 and mean parameters `m` and proportion `p`.
"""
function generate_mixture_normal(n::Int64, m::Array{T,1}=[-20., 0., 20.], p::Array{T,1}=ones(length(m))/length(m)) where T <: AbstractFloat
    mixed_normal = Array{Float64}(undef, 0)  # placeholder for the resulting mixture
    n_mixtures = Int64.(floor.(n * p))  # sample sizes for each univariate normals
    
    # leftover n's is assigned to the first distribution > sum(n_mixtures)
    n_mixtures[1] += (n - sum(n_mixtures))
    
    # sample from Normal(0, 1)
    for i=1:length(m)
        mixed_normal = [mixed_normal; randn(n_mixtures[i]) .+ m[i]]
    end
    
    mixed_normal 
end