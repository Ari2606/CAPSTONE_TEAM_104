"""make_segments.py -- run on Colab (or anywhere with wfdb + the MIT-BIH files). Cuts one 30 s raw MLII segment per DS2 record
and saves segments.npz for pi_pipeline_bench.py.   python make_segments.py  [DATA_DIR]"""
import sys, numpy as np, wfdb
DATA = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/physioguard_space/data/mitdb"
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
FS, SEC, START = 360, 30, 90          # 30 s segment starting 90 s into each record (past filter warm-up)
sigs, recs = [], []
for r in DS2:
    rec = wfdb.rdrecord(f"{DATA}/{r}")
    names = [n.strip() for n in rec.sig_name]
    s = rec.p_signal[:, names.index("MLII")].astype(np.float64)
    sigs.append(s[START * FS:(START + SEC) * FS]); recs.append(r)
np.savez("segments.npz", sig=np.stack(sigs), rec=np.array(recs), fs=FS)
print("saved segments.npz", np.stack(sigs).shape)
