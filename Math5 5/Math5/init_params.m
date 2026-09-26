function params = init_params(cfg, L)
%% INIT_PARAMS  Initialize learnable parameters
%  L=1: Kedge=2.0 (paper Table 2: K=2 for L=1 boat)
%  L=10: Kedge=1.0 (paper Table 2: K=1 for L=10)
if nargin < 2, L = 1; end
T=cfg.T; Nk=cfg.Nk; fs=cfg.fsize;
params.T=T; params.Nk=Nk; params.fsize=fs;
params.phi_type='students_t';
dct_basis=compute_dct_basis(fs);
n_dct=size(dct_basis,2);
for t=1:T
    K_init=zeros(fs,fs,Nk);
    for i=1:Nk
        idx=mod(i-1,n_dct-1)+2;
        f_vec=dct_basis(:,idx); f_vec=f_vec-mean(f_vec);
        n_=norm(f_vec); if n_>1e-8, f_vec=f_vec/n_; end
        K_init(:,:,i)=reshape(f_vec,fs,fs);
    end
    params.K_t{t}=K_init;
    params.nu_t(t)=1.0;
    if L==1
        params.lambda_t(t)=0.5;
        params.Kedge_t(t)=2.0;
    else
        params.lambda_t(t)=0.5;
        params.Kedge_t(t)=1.0;
    end
end
end
function dct_basis=compute_dct_basis(fs)
D1=zeros(fs,fs);
for p=0:fs-1
    for x=0:fs-1
        if p==0, D1(x+1,p+1)=sqrt(1/fs);
        else,    D1(x+1,p+1)=sqrt(2/fs)*cos(pi*p*(2*x+1)/(2*fs));
        end
    end
end
nb=fs*fs; dcb=zeros(nb,nb); fo=zeros(1,nb); col=1;
for q=1:fs
    for p=1:fs
        b2=D1(:,p)*D1(:,q)'; dcb(:,col)=b2(:);
        fo(col)=(p-1)^2+(q-1)^2; col=col+1;
    end
end
[~,idx]=sort(fo); dct_basis=dcb(:,idx);
end