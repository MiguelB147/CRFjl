
module CRF

export SimData, LogLik, spline, Profile, AdaptiveProfile

using Distributions
using Random
using Splines2
using Distributed

addprocs(6)

IndGreater = function(x::Array)
    n = length(x)
    elem = Matrix{Int}(undef, n, n)
    for i in 1:n
        for j in 1:n
            if x[j] >= x[i]
                elem[j,i] = 1
            else
                elem[j,i] = 0
            end
        end
    end
    return elem
end

IndLess = function(x::Array)
    n = length(x)
    elem = Matrix{Int}(undef, n, n)
    for i in 1:n
        for j in 1:n
            if x[j] <= x[i]
                elem[j,i] = 1
            else
                elem[j,i] = 0
            end
        end
    end
    return elem
end

IndEqual = function (x::Array)
    n = length(x)
    elem = Matrix{Int}(undef, n, n)
    for i in 1:n
        for j in 1:n
            if x[j] == x[i]
                elem[j,i] = 1
            else
                elem[j,i] = 0
            end
        end
    end
    return elem
end

Ind2 = function (x::Array, y::Array, a::Float64, b::Float64)
    n = length(x)
    sum = 0
    for i in 1:n
        if x[i] >= a && y[i] >= b
            sum += 1
        end
    end
    return sum
end

Riskset = function (x::Array, y::Array)
    n = length(x)
    riskset = Matrix{Int}(undef, n, n)
    for i in 1:n
        for j in 1:n
            riskset[j,i] = Ind2(x, y, x[j], y[i])
        end
    end
    return riskset
end

SimData = function(;K::Int, df::Int, degree::Int, ub = 0; alpha = 0.0023)

    Random.seed!(123)

    u1 = rand(Uniform(0,1), K)
    u2 = rand(Uniform(0,1), K)
    
    a = alpha.^u1 + (alpha .- alpha.^u1).*u2
    
    # Fan 2000
    T1 = -log.(u1)
    T2 = -log.(log.(alpha, a ./ (a .+ (1-alpha) .* u2)))

    if ub == 0
        X1 = copy(T1)
        X2 = copy(T2)

        delta1 = fill(1,K)
        delta2 = copy(delta1)

    else
        C1 = rand(Uniform(0, ub), K)
        C2 = rand(Uniform(0, ub), K)

        X1 = min(T1,C1)
        X2 = min(T2,C2)

        delta1 = 1 .* (T1 .<= C1)
        delta2 = 1 .* (T2 .<= C2)
    end

    X = hcat(X1,X2)

    qq1 = quantile(X1[delta1 .== 1], collect(range(0,1,length=df-degree+2)))
    knots1 = vcat(minimum(X1)-1, qq1[2:length(qq1)])

    qq2 = quantile(X2[delta2 .== 1], collect(range(0,1,length=df-degree+2)))
    knots2 = vcat(minimum(X2)-1, qq1[2:length(qq2)])  

    if (ub > 0 && ub < 5)
        knots1[length(knots1)] = maximum(X1)
        knots2[length(knots2)] = maximum(X2)
    end

    I1 = IndGreater(X1)
    I2 = IndLess(X2)
    I5 = IndEqual(X1)
    I6 = IndEqual(X2)

    deltaprod = delta1 .* delta2'

    N = Riskset(X1, X2)

return (X = X, knots = hcat(knots1, knots2), delta = deltaprod, riskset = N, I1 = I1, I2 = I2, I5 = I5, I6 = I6)
end

LogLik = function (; riskset::Matrix, logtheta::Matrix, delta::Matrix, I1::Matrix, I2::Matrix, I5::Matrix, I6::Matrix)

    n = size(riskset, 1)
    sum1::Float64 = 0
    sum2::Float64 = 0

    for i in 1:n
        for j in 1:n
            if riskset[j,i] > 0
                sum1 += delta[j,i]*I1[i,j]*(logtheta[j,i]*I5[i,j] - log(riskset[j,i] + I2[i,j]*(exp(logtheta[j,i]) - 1)))
            else
                 sum1 += 0
            end

            if riskset[i,j] > 0
                sum2 += delta[i,j]*I2[j,i]*(logtheta[i,j]*I6[i,j] - log(riskset[i,j] + I1[j,i]*(exp(logtheta[i,j]) - 1)))
            else
                sum2 += 0
            end


        end
    end
    return -sum1-sum2;
end

spline = function(t1::Vector, t2::Vector, coef::Vector; degree::Int, knots::Matrix)

    df = size(knots,1)
    coefmat = reshape(coef, df, df)

    B1 = bs(t1, order = degree+1, knots = knots[:,1])
    B2 = bs(t2, order = degree+1, knots = knots[:,2])

    tensor = B1 * coefmat * B2'

    return tensor
end

polynomial = function(t1::Vector, t2::Vector, coef::Vector)

    X = hcat(fill(1,length(t1)), t1, t2, (t1).^2, (t2).^2, t1 .* t2, ((t1).^2).* t2, ((t2).^2).* t1, t1 .^3, t2 .^3)
    
    return X * coef
end

Profile = function (datalist::NamedTuple, degree::Int; start::Vector, lower::Int, upper::Int, step::Float64, iter = 10)

    nparam = length(start)

    result = Array{Float64}(undef, iter, nparam)
    result[1,:] = start

    grid = collect(lower:step:upper)
    gridsize = length(grid)

    println("Profiling...")

    for i in 1:iter
        printstyled("Iteration ", i, "\n"; color = :blue)
        for j in 1:nparam
            ll = Array{Float64}(undef, gridsize)
            for k in 1:gridsize
                result[i,j] = grid[k]
                logtheta = spline(datalist.X[:,1], datalist.X[:,2], result[i,:], degree = degree, knots = datalist.knots)
                ll[k] = LogLik(riskset = datalist.riskset, logtheta = logtheta, delta = datalist.delta, I1 = datalist.I1, I2 = datalist.I2, I5 = datalist.I5, I6 = datalist.I6)
            end
            result[i,j] = grid[argmin(ll)]
            println("Iteration ", i, ", Parameter ", j, ": ", result[i,j])

        end

        if i < iter
            result[i+1, :] = result[i, :]
        end

    end

    return result
end

AdaptiveProfile = function (datalist::NamedTuple, degree::Int; start::Vector, step::Float64, iter = 10)

    nparam = length(start)

    result = Array{Float64}(undef, iter, nparam)
    result[1,:] = start

    println("Profiling...")

for i in 1:iter

        printstyled("Iteration ", i, "\n"; color = :blue)
        
        for j in 1:nparam
            grid = collect((result[i,j]-10):step:(result[i,j]+10))
            gridsize = length(grid)
            ll = Array{Float64}(undef, gridsize)

            for k in 1:gridsize
                result[i,j] = grid[k]
                logtheta = spline(datalist.X[:,1], datalist.X[:,2], result[i,:], degree = degree, knots = datalist.knots)
                ll[k] = LogLik(riskset = datalist.riskset, logtheta = logtheta, delta = datalist.delta, I1 = datalist.I1, I2 = datalist.I2, I5 = datalist.I5, I6 = datalist.I6)
            end

            result[i,j] = grid[argmin(ll)]
            println("Iteration ", i, ", Parameter ", j, ": ", result[i,j])

        end

        if i < iter
            result[i+1, :] = result[i, :]
        end

    end

    return result
end

end