import pyarrow.dataset as ds, pyarrow.compute as pc, json
d=ds.dataset("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/Cerebellum_sample/transcripts.parquet")
n=a=0
for b in d.to_batches(columns=["cell_id"],filter=(pc.field("qv")>=20)&pc.field("is_gene"),batch_size=2_000_000):
    c=b.column(0); n+=len(c); a+=len(c)-(pc.sum(pc.equal(c,"UNASSIGNED")).as_py() or 0)-(pc.sum(pc.equal(c,"")).as_py() or 0)
json.dump({"n_qc_tx":n,"tenx_assigned":a,"pct":round(100*a/n,2)},open("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/tenx_summary.json","w")); print(n,a,round(100*a/n,2))
