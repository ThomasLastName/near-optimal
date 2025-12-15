
import torch

k = 15
m = 2*k
a = torch.randn( k, m )
b = torch.randn( k, m )
target = torch.zeros(m,m)
for j in range(k):
    target += torch.outer(a[j],a[j])
    target -= torch.outer(b[j],b[j])


C = torch.row_stack([a,b]).T
D = torch.cat([ torch.ones(k), -torch.ones(k) ]).diag()
(target - C@D@C.T).abs().max()



U,s,Vt = torch.linalg.svd(C)
Lambda = torch.randn_like(s).diag()
vals, vecs = torch.linalg.eigh(C@Lambda@C.T)

assert ( C - U@s.diag()@Vt ).abs().mean() < 1e-5
mu,Q = torch.linalg.eigh(C@D@C.T)
assert ( C@D@C.T - Q@mu.diag()@Q.T ).abs().mean() < 1e-5