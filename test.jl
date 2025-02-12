
include("CRF.jl")

using .CRF

testdata = SimData(K=1000, df = 5, degree = 2, ub = 0)

logtheta = spline(testdata.X[:,1], testdata.X[:,2], fill(1,25), degree = 2, knots = testdata.knots)
LogLik(riskset = testdata.riskset, logtheta = logtheta, delta = testdata.delta, I1 = testdata.I1, I2 = testdata.I2, I3 = testdata.I5)

solution = Profile(testdata, 2, start = fill(1,25), lower = -20, upper = 20, step = 0.1, iter = 20)

solution = AdaptiveProfile(testdata, 2, start = fill(1,25), step = 0.1, iter = 50)


grid = collect(-20:0.1:20)
gridsize = length(grid)
ll = Array{Float64}(undef, gridsize)
coef = fill(1.000000,25)
for i in 1:gridsize
    coef[25] = grid[i]
    logtheta = spline(testdata.X[:,1], testdata.X[:,2], coef, degree = 2, knots = testdata.knots)
    ll[i] = LogLik(riskset = testdata.riskset, logtheta = logtheta, delta = testdata.delta, I1 = testdata.I1, I2 = testdata.I2, I3 = testdata.I5)
end