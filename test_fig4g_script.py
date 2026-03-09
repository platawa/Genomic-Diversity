import numpy as np, sys
sys.path.insert(0, "/orcd/data/zhang_f/001/platawa/jan31_files")
from run_sae_on_chromosome_drops import plot_region_figure4g, KNOWN_BIO_FEATURES

np.random.seed(42)
seq_len = 1200
feature_ts = np.zeros((seq_len, 32768), dtype=np.float32)
feature_ts[300:500, 15680] = np.random.exponential(1.5, 200)
feature_ts[650:900, 15680] = np.random.exponential(1.5, 250)
feature_ts[500:650, 28339] = np.random.exponential(2.0, 150)
feature_ts[300:305, 1050] = np.random.exponential(4.0, 5)
feature_ts[650:655, 1050] = np.random.exponential(4.0, 5)
feature_ts[495:500, 25666] = np.random.exponential(4.0, 5)
feature_ts[895:900, 25666] = np.random.exponential(4.0, 5)
feature_ts[420:425, 24278] = np.random.exponential(3.0, 5)
for fid in [32710, 13657, 17323]:
    feature_ts[:, fid] = np.random.exponential(0.5, seq_len)

result = {
    "region": {
        "genomic_start": 21710630, "genomic_end": 21710849,
        "padded_start": 21710130, "padded_end": 21711349,
        "drop_local_pos": 500, "rise_local_pos": 719,
        "method": "mad", "start_confidence": 17.80, "strand": "+",
    },
    "feature_ts": feature_ts,
    "top_feature_idx": [32710, 13657, 17323, 15680, 28339],
}

entropy = np.random.exponential(1.5, 21712000)
entropy[21710630:21710849] *= 0.3

annotations = [[300, 500, "CDS", {}], [600, 900, "CDS", {}], [200, 1000, "gene", {}]]

gtf_features = [
    {"feature_type": "gene", "start": 21710000, "end_exclusive": 21711500, "name": "YPEL1", "strand": "-"},
    {"feature_type": "exon", "start": 21710200, "end_exclusive": 21710450, "name": "", "strand": "-"},
    {"feature_type": "exon", "start": 21710700, "end_exclusive": 21710900, "name": "", "strand": "-"},
    {"feature_type": "CDS", "start": 21710250, "end_exclusive": 21710450, "name": "", "strand": "-"},
    {"feature_type": "CDS", "start": 21710700, "end_exclusive": 21710880, "name": "", "strand": "-"},
]

plot_region_figure4g(result, 0, "/tmp/test_fig4g_v3.png",
    annotations=annotations, entropy=entropy, gtf_features=gtf_features,
    n_plot_features=5, chrom="chr22")
print("SUCCESS")
