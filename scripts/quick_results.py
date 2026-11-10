"""
Quick benchmark: trains PINN, MC Dropout, ReFANN, Deep GP, BNN-VI on 3 PDEs.
- PINN:    trained on BC + PDE collocation (no labeled interior)
- Others:  trained on 500 labeled interior points (noisy exact solution)
- BNN-VI:  Variational Bayesian NN (Bayes by Backprop, mean-field Gaussian)
Run from project root: python3 scripts/quick_results.py
"""
import sys, os, json, warnings, importlib.util
warnings.filterwarnings('ignore')

def load_src(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
gen  = load_src('generators', os.path.join(SRC, 'data',       'generators.py'))
metr = load_src('metrics',    os.path.join(SRC, 'evaluation', 'metrics.py'))

import torch, torch.nn as nn, numpy as np
torch.manual_seed(42); np.random.seed(42)

EPOCHS = 5000

# ── Loss ───────────────────────────────────────────────────────────
def gnll(mu, ls, y):
    return 0.5*(ls + 0.5*torch.log(torch.tensor(6.2832)) + (y-mu)**2/torch.exp(2*ls)).mean()

# ── Early stop ─────────────────────────────────────────────────────
class ES:
    def __init__(self, p=1500): self.p=p; self.b=float('inf'); self.c=0
    def step(self, v):
        if v < self.b-1e-6: self.b=v; self.c=0
        else: self.c+=1
        return self.c >= self.p

# ── Models ─────────────────────────────────────────────────────────
class Sin(nn.Module):
    def __init__(self,w=30.): super().__init__(); self.w=w
    def forward(self,x): return torch.sin(self.w*x)

class PINNModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Linear(2,64),Sin()]
        for _ in range(3): layers+=[nn.Linear(64,64),Sin()]
        self.bb=nn.Sequential(*layers)
        self.mu=nn.Linear(64,1); self.ls=nn.Linear(64,1)
    def forward(self,x): h=self.bb(x); return self.mu(h), self.ls(h).clamp(-6,2)

class MCDModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Linear(2,128),nn.Tanh(),nn.Dropout(0.1)]
        for _ in range(3): layers+=[nn.Linear(128,128),nn.Tanh(),nn.Dropout(0.1)]
        self.bb=nn.Sequential(*layers)
        self.mu=nn.Linear(128,1); self.ls=nn.Linear(128,1)
    def forward(self,x): h=self.bb(x); return self.mu(h), self.ls(h).clamp(-6,2)

class RB(nn.Module):
    def __init__(self,d=128): super().__init__(); self.net=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d)); self.act=nn.Tanh()
    def forward(self,x): return self.act(x+self.net(x))

class ReFANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(2,128),nn.Tanh(),RB(),RB())
        self.mu0=nn.Linear(128,1); self.ls0=nn.Linear(128,1)
        self.ref=nn.Sequential(nn.Linear(130,128),nn.Tanh(),RB())
        self.mu1=nn.Linear(128,1); self.ls1=nn.Linear(128,1)
    def forward(self,x):
        h=self.enc(x); m=self.mu0(h); s=self.ls0(h).clamp(-6,2)
        h2=self.ref(torch.cat([h,m.detach(),s.detach()],-1))
        return m+self.mu1(h2), (s+self.ls1(h2)).clamp(-6,2)

# ── Variational BNN (Bayes by Backprop) ────────────────────────────
import torch.nn.functional as F

class VBLayer(nn.Module):
    """Mean-field Gaussian variational linear layer."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -4.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -4.0))
    def forward(self, x):
        ws = F.softplus(self.w_rho)
        bs = F.softplus(self.b_rho)
        w  = self.w_mu + ws * torch.randn_like(self.w_mu)
        b  = self.b_mu + bs * torch.randn_like(self.b_mu)
        return F.linear(x, w, b)
    def kl(self):
        ws = F.softplus(self.w_rho); bs = F.softplus(self.b_rho)
        kl_w = -0.5*(1 + 2*ws.log() - self.w_mu.pow(2) - ws.pow(2))
        kl_b = -0.5*(1 + 2*bs.log() - self.b_mu.pow(2) - bs.pow(2))
        return kl_w.sum() + kl_b.sum()

class VBNNModel(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.l1=VBLayer(2,h); self.l2=VBLayer(h,h); self.l3=VBLayer(h,h)
        self.mu_h=VBLayer(h,1); self.ls_h=VBLayer(h,1)
        self.act=nn.Tanh()
    def forward(self, x):
        h=self.act(self.l1(x)); h=self.act(self.l2(h)); h=self.act(self.l3(h))
        return self.mu_h(h), self.ls_h(h).clamp(-6,2)
    def kl(self):
        return self.l1.kl()+self.l2.kl()+self.l3.kl()+self.mu_h.kl()+self.ls_h.kl()

# ── Train standard NN ──────────────────────────────────────────────
def train_nn(model, x_tr, y_tr, x_val, y_val, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=lr*0.01)
    es  = ES()
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        mu, ls = model(x_tr)
        gnll(mu, ls, y_tr).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if ep % 200 == 0:
            model.eval()
            with torch.no_grad(): mv,lv=model(x_val); vl=gnll(mv,lv,y_val).item()
            if es.step(vl): break

# ── Train VBN ──────────────────────────────────────────────────────
def train_vbnn(model, x_tr, y_tr, x_val, y_val, lr=1e-3):
    n   = len(x_tr)
    opt = torch.optim.Adam(model.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=lr*0.01)
    es  = ES()
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        mu, ls = model(x_tr)
        loss = gnll(mu, ls, y_tr) + 1e-4 * model.kl() / n
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if ep % 200 == 0:
            # validation: average over 10 stochastic passes
            model.train()
            with torch.no_grad():
                preds = [model(x_val) for _ in range(10)]
                mv = torch.stack([p[0] for p in preds]).mean(0)
                lv = torch.stack([p[1] for p in preds]).mean(0)
                vl = gnll(mv, lv, y_val).item()
            if es.step(vl): break

# ── Train PINN ─────────────────────────────────────────────────────
def train_pinn(model, data, pde_type, pde_params):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    es  = ES()
    x_bc=data.x_bc; u_bc=data.u_bc; x_val=data.x_test; u_val=data.u_test_noisy
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        # BC loss
        mu_bc, ls_bc = model(x_bc)
        loss = 10.0*nn.MSELoss()(mu_bc, u_bc)
        # PDE residual (heat only for speed)
        if pde_type == 'heat':
            xc = data.x_col[:2000].clone().requires_grad_(True)
            mu_c, _ = model(xc)
            g = torch.autograd.grad(mu_c.sum(), xc, create_graph=True)[0]
            ux, ut = g[:,0:1], g[:,1:2]
            uxx = torch.autograd.grad(ux.sum(), xc, create_graph=True)[0][:,0:1]
            loss = loss + (ut - pde_params.get('alpha',0.01)*uxx).pow(2).mean()
        elif pde_type == 'wave':
            xc = data.x_col[:2000].clone().requires_grad_(True)
            mu_c, _ = model(xc)
            g = torch.autograd.grad(mu_c.sum(), xc, create_graph=True)[0]
            ux, ut = g[:,0:1], g[:,1:2]
            uxx = torch.autograd.grad(ux.sum(), xc, create_graph=True)[0][:,0:1]
            utt = torch.autograd.grad(ut.sum(), xc, create_graph=True)[0][:,1:2]
            c   = pde_params.get('c', 1.0)
            loss = loss + (utt - c**2*uxx).pow(2).mean()
        elif pde_type == 'poisson':
            xc = data.x_col[:2000].clone().requires_grad_(True)
            mu_c, _ = model(xc)
            g = torch.autograd.grad(mu_c.sum(), xc, create_graph=True)[0]
            ux, uy = g[:,0:1], g[:,1:2]
            uxx = torch.autograd.grad(ux.sum(), xc, create_graph=True)[0][:,0:1]
            uyy = torch.autograd.grad(uy.sum(), xc, create_graph=True)[0][:,1:2]
            f   = 2*np.pi**2*torch.sin(np.pi*xc[:,0:1])*torch.sin(np.pi*xc[:,1:2])
            loss = loss + (-uxx-uyy-f).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if ep % 200 == 0:
            model.eval()
            with torch.no_grad(): mv,lv=model(x_val); vl=gnll(mv,lv,u_val).item()
            if es.step(vl): break

# ── Deep GP (Deep Kernel Learning: DNN feature extractor + SVGP) ───
def run_deep_gp(x_tr, y_tr, x_test):
    import gpytorch

    class DKLModel(gpytorch.models.ApproximateGP):
        def __init__(self, ip):
            vd = gpytorch.variational.CholeskyVariationalDistribution(ip.shape[0])
            vs = gpytorch.variational.VariationalStrategy(self, ip, vd, learn_inducing_locations=True)
            super().__init__(vs)
            self.feat = nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,8))
            self.mean = gpytorch.means.ConstantMean()
            self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))
        def forward(self, x):
            z = self.feat(x)
            return gpytorch.distributions.MultivariateNormal(self.mean(z), self.covar(z))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    ip = x_tr[torch.randperm(len(x_tr))[:128]].clone()
    model = DKLModel(ip)
    model.train(); likelihood.train()
    opt = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-4)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_tr))
    y_flat = y_tr.squeeze(-1)
    es = ES()
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = -mll(model(x_tr), y_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if ep % 200 == 0:
            if es.step(loss.item()): break
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(x_test))
    return preds.mean.numpy(), preds.variance.sqrt().numpy()

# ── Datasets ───────────────────────────────────────────────────────
print('\nLoading datasets ...')
CFGS = {
    'Heat Eq.': dict(cls=gen.HeatEquationDataset,
                     cfg=dict(alpha=0.01,n_collocation=5000,n_boundary=200,n_initial=200,n_test=2500,noise_std=0.01),
                     pde='heat', params={'alpha':0.01}),
    'Wave Eq.': dict(cls=gen.WaveEquationDataset,
                     cfg=dict(c=1.0,n_collocation=5000,n_boundary=200,n_initial=400,n_test=2500,noise_std=0.01),
                     pde='wave', params={'c':1.0}),
    'Poisson':  dict(cls=gen.PoissonDataset,
                     cfg=dict(n_collocation=5000,n_boundary=400,n_test=2500,noise_std=0.01),
                     pde='poisson', params={}),
}

results = {}
rng = np.random.default_rng(42)

for task, info in CFGS.items():
    data = info['cls'](info['cfg']).get()
    print(f'\n{"="*55}\n  Task: {task}\n{"="*55}')
    results[task] = {}

    # Interior training set for non-PINN models (500 labeled points)
    idx = rng.choice(len(data.x_test), 500, replace=False)
    x_in = data.x_test[idx];  y_in = data.u_test_noisy[idx]
    # Validation: remaining test points
    mask = np.ones(len(data.x_test), dtype=bool); mask[idx] = False
    x_val = data.x_test[mask];  y_val = data.u_test_noisy[mask]
    # Full test set for evaluation
    x_tst = data.x_test;  y_tst = data.u_test_noisy

    # --- PINN ---
    print('  PINN ...', end=' ', flush=True)
    m = PINNModel(); train_pinn(m, data, info['pde'], info['params'])
    m.eval()
    with torch.no_grad(): mu, ls = m(x_tst)
    r = metr.evaluate_model(y_tst.numpy().ravel(), mu.numpy().ravel(), torch.exp(ls).numpy().ravel())
    results[task]['PINN'] = r
    print(f'NLL={r.gaussian_nll:.3f}  RMSE={r.rmse:.4f}  Cov90={r.coverage_90:.3f}')

    # --- MC Dropout ---
    print('  MC Dropout ...', end=' ', flush=True)
    m = MCDModel(); train_nn(m, x_in, y_in, x_val, y_val)
    m.train()
    samps, sigs = [], []
    with torch.no_grad():
        for _ in range(100):
            mu_s, ls_s = m(x_tst); samps.append(mu_s); sigs.append(torch.exp(ls_s))
    samps=torch.stack(samps); sigs=torch.stack(sigs)
    mean=(samps.mean(0)).numpy().ravel()
    std =((samps.var(0)+sigs.pow(2).mean(0)).sqrt()).numpy().ravel()
    r = metr.evaluate_model(y_tst.numpy().ravel(), mean, std)
    results[task]['MC Dropout'] = r
    print(f'NLL={r.gaussian_nll:.3f}  RMSE={r.rmse:.4f}  Cov90={r.coverage_90:.3f}')

    # --- ReFANN ---
    print('  ReFANN ...', end=' ', flush=True)
    m = ReFANNModel(); train_nn(m, x_in, y_in, x_val, y_val)
    m.eval()
    with torch.no_grad(): mu, ls = m(x_tst)
    r = metr.evaluate_model(y_tst.numpy().ravel(), mu.numpy().ravel(), torch.exp(ls).numpy().ravel())
    results[task]['ReFANN'] = r
    print(f'NLL={r.gaussian_nll:.3f}  RMSE={r.rmse:.4f}  Cov90={r.coverage_90:.3f}')

    # --- Deep GP ---
    print('  Deep GP ...', end=' ', flush=True)
    mean_gp, std_gp = run_deep_gp(x_in, y_in, x_tst)
    r = metr.evaluate_model(y_tst.numpy().ravel(), mean_gp, std_gp)
    results[task]['Deep GP'] = r
    print(f'NLL={r.gaussian_nll:.3f}  RMSE={r.rmse:.4f}  Cov90={r.coverage_90:.3f}')

    # --- BNN (VI) ---
    print('  BNN (VI) ...', end=' ', flush=True)
    m = VBNNModel(h=64); train_vbnn(m, x_in, y_in, x_val, y_val)
    m.train()
    samps_mu, samps_sig = [], []
    with torch.no_grad():
        for _ in range(200):
            mu_s, ls_s = m(x_tst)
            samps_mu.append(mu_s); samps_sig.append(torch.exp(ls_s))
    mu_all  = torch.stack(samps_mu)   # (200, N, 1)
    sig_all = torch.stack(samps_sig)
    pred_mean = mu_all.mean(0).squeeze(-1).numpy()
    pred_std  = (mu_all.var(0) + sig_all.pow(2).mean(0)).sqrt().squeeze(-1).numpy()
    r = metr.evaluate_model(y_tst.numpy().ravel(), pred_mean, pred_std)
    results[task]['BNN (VI)'] = r
    print(f'NLL={r.gaussian_nll:.3f}  RMSE={r.rmse:.4f}  Cov90={r.coverage_90:.3f}')

# ── Summary table ──────────────────────────────────────────────────
print('\n\n' + '='*80)
MODEL_ORDER = ['PINN','MC Dropout','ReFANN','Deep GP','BNN (VI)']
for task, tres in results.items():
    rows = sorted([(m,tres[m]) for m in MODEL_ORDER if m in tres], key=lambda x: x[1].gaussian_nll)
    print(f'\n{task}')
    print(f"{'Rank':<5}{'Model':<14}{'NLL':>8}{'RMSE':>9}{'CRPS':>9}{'ECE':>9}{'Cov90':>8}{'IS90':>9}")
    for i,(model,r) in enumerate(rows,1):
        d=r.to_dict()
        print(f"{i:<5}{model:<14}{d['gaussian_nll']:>8.3f}{d['rmse']:>9.4f}{d['crps']:>9.4f}{d['ece']:>9.4f}{d['coverage_90']:>8.3f}{d['interval_score_90']:>9.3f}")

os.makedirs('results', exist_ok=True)
with open('results/quick_results.json','w') as f:
    json.dump({t:{m:v.to_dict() for m,v in tr.items()} for t,tr in results.items()}, f, indent=2)
print('\nSaved: results/quick_results.json')
