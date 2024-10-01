import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import torch
    return mo, np, torch


@app.cell
def __():
    from torch.profiler import profile, ProfilerActivity
    return ProfilerActivity, profile


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Create Data""")
    return


@app.cell
def __(np):
    def create_data(N, k):
        node_list = np.array([np.random.rand(N), np.random.rand(N)]).reshape((N, 2))
        link_list = []
        for i in range(N):
            distance = np.array(
                [np.linalg.norm(node_list[i] - node_list[j]) for j in range(N)]
            )
            neighbors = np.argsort(distance)[1 : (k + 1)]
            link = np.zeros((N, k))
            link[i, :] = -1
            link[neighbors, np.array(range(k))] = 1
            link_list.append(link)
        A = np.hstack(link_list)
        p = np.random.permutation(A.shape[1])
        return A[:, p]
    return create_data,


@app.cell
def __():
    device = "cuda:0"
    return device,


@app.cell
def __():
    N = 2000
    k = 50
    seed = 0
    return N, k, seed


@app.cell
def __(N, create_data, device, k, np, seed, torch):
    torch.cuda.empty_cache()

    # Create data
    np.random.seed(seed)

    A = create_data(N, k)
    A = torch.tensor(A, device=device, dtype=torch.float32)

    F = torch.rand((N, N * k), device=device, dtype=torch.float32)  # F is same size as A
    return A, F


@app.cell
def __(A, F):
    print(F.shape)
    print(A.shape)
    return


@app.cell
def __(A, N, torch):
    pos_ind = torch.where(torch.Tensor(A).T == 1)[1]
    neg_ind = torch.where(torch.Tensor(A).T == -1)[1]

    # Add dimension and expand along axis
    # Note: this step creates a view and does not allocate memory
    pos_ind = pos_ind.unsqueeze(1).expand((-1, N))
    neg_ind = neg_ind.unsqueeze(1).expand((-1, N))
    return neg_ind, pos_ind


@app.cell
def __(pos_ind):
    print(pos_ind.shape)
    return


@app.cell
def __(pos_ind):
    pos_ind[:5, :5]
    return


@app.cell
def __(mo):
    mo.md(r"""## Dense Profile""")
    return


@app.cell
def __(A, F, ProfilerActivity, profile, torch):
    torch.cuda.empty_cache()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof_dense:
        B_dense = F @ A.T

    print(prof_dense.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return B_dense, prof_dense


@app.cell
def __(mo):
    mo.md(r"""## Sparse Profile""")
    return


@app.cell
def __(F, N, ProfilerActivity, device, neg_ind, pos_ind, profile, torch):
    torch.cuda.empty_cache()
    # Sparse calculation
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof_sparse:
        B_sparse_pos = torch.zeros((N, N), device=device)
        B_sparse_pos = B_sparse_pos.scatter_add(0, pos_ind, F.T)

        B_sparse_neg = torch.zeros((N, N), device=device)
        B_sparse_neg = B_sparse_neg.scatter_add(0, neg_ind, F.T)

        B_sparse = B_sparse_pos - B_sparse_neg
        B_sparse = B_sparse.T

    print(prof_sparse.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return B_sparse, B_sparse_neg, B_sparse_pos, prof_sparse


@app.cell
def __(mo):
    mo.md(r"""## Check Correctness""")
    return


@app.cell
def __(B_dense, B_sparse):
    assert B_dense.shape == B_sparse.shape
    return


@app.cell
def __(B_dense, B_sparse, torch):
    torch.sum(torch.abs(B_dense - B_sparse)) / torch.sum(torch.abs(B_dense))
    return


@app.cell
def __(B_dense):
    B_dense
    return


@app.cell
def __(B_sparse):
    B_sparse
    return


if __name__ == "__main__":
    app.run()
