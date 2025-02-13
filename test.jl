
include("CRF.jl")

using .CRF
using Plots

testdata = SimData(K=1000, df = 5, degree = 2, ub = 0)

logtheta = spline(testdata.X[:,1],testdata.X[:,2], fill(1,25), degree = 2, knots = testdata.knots)
LogLik(riskset = testdata.riskset, logtheta = logtheta, delta = testdata.delta, I1 = testdata.I1, I2 = testdata.I2, I5 = testdata.I5, I6 = testdata.I6)

solution = Profile(testdata, 2, start = ones(25), lower = -20, upper = 20, step = 0.1, iter = 5)

solution = AdaptiveProfile(testdata, 2, start = ones(25), step = 0.1, iter = 10)


grid = collect(15:0.1:50)
gridsize = length(grid)
ll = Array{Float64}(undef, gridsize)
# coef = ones(25)
coef = solution[5,:]
for i in 1:gridsize
    coef[5] = grid[i]
    logtheta = spline(testdata.X[:,1], testdata.X[:,2], coef, degree = 2, knots = testdata.knots)
    ll[i] = LogLik(riskset = testdata.riskset, logtheta = logtheta, delta = testdata.delta, I1 = testdata.I1, I2 = testdata.I2, I5 = testdata.I5, I6 = testdata.I6)
end