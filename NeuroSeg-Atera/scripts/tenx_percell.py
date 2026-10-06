import pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow as pa, pandas as pd
d=ds.dataset("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/Cerebellum_sample/transcripts.parquet"); acc=[]
for i,b in enumerate(d.to_batches(columns=["cell_id"],filter=(pc.field("qv")>=20)&pc.field("is_gene"),batch_size=20_000_000)):
    t=pa.table({"c":b.column(0)}).group_by("c").aggregate([("c","count")]); acc.append(t.to_pandas())
s=pd.concat(acc).groupby("c").c_count.sum(); s=s[~s.index.isin(["UNASSIGNED",""])]
s.rename("qc_transcripts").to_frame().to_parquet("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/tenx_qc_counts_per_cell.parquet"); print(len(s), s.median())
