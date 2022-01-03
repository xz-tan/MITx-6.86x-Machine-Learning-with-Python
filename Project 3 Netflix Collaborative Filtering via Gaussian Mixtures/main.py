import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")
# TODO: Your code here

# for K in range(1, 5):
#     list_log_likelihood = []
#     list_cost = []
#     for seed in range(5):
#         mixture_EM, post, log_likelihood = naive_em.run(X, *common.init(X, K, seed))
#         list_log_likelihood.append(log_likelihood)
#         common.plot(X, mixture_EM, post, title=f"K={K}, seed={seed}")
#         mixture_kmeans, post, cost = kmeans.run(X, *common.init(X, K, seed))
#         list_cost.append(cost)
#         common.plot(X, mixture_kmeans, post, title=f"K={K}, seed={seed}")
#
#     print(f"{K}: Max log-likelihood = {max(list_log_likelihood)}")
#     print(f"{K}: Min cost = {min(list_cost)}")


# for K in range(1, 5):
#     mixture_EM, post, log_likelihood = naive_em.run(X, *common.init(X, K, 0))
#     BIC_value = common.bic(X, mixture_EM, log_likelihood)
#     print(f"{K}: BIC = {BIC_value}")

X = np.loadtxt("netflix_incomplete.txt")
list_log_likelihood = []
for seed in range(5):
    mixture_EM, post, log_likelihood = em.run(X, *common.init(X, 1, seed))
    list_log_likelihood.append(log_likelihood)
print(f"K1:{max(list_log_likelihood)}")

# list_log_likelihood = []
# for seed in range(1):
#     mixture_EM, post, log_likelihood = em.run(X, *common.init(X, 12, seed))
#     list_log_likelihood.append(log_likelihood)
# print(f"K12:{max(list_log_likelihood)}")



