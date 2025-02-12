
include("CRF.jl")

using .CRF

testdata = SimData(K=1000, df = 5, degree = 2, ub = 0)

logtheta = spline(testdata.X[:,1], testdata.X[:,2], fill(1,25), degree = 2, knots = testdata.knots)
LogLik(riskset = testdata.riskset, logtheta = logtheta, delta = testdata.delta, I1 = testdata.I1, I2 = testdata.I2, I3 = testdata.I5)

solution = Profile(testdata, 2, start = fill(1,25), lower = -50, upper = 50, step = 0.1, iter = 5)

solution = AdaptiveProfile(testdata, 2, start = fill(1,25), step = 0.1, iter = 5)