import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.feature import peak_local_max
UM=0.2125; SSD=Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); RUNS=Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
setA=set(json.load(open(SSD/"compare/purkinje/gene_sets.json"))["purkinje_A"]); rows=[]
for win in sys.argv[1:]:
    x0,y0,x1,y1=json.load(open(RUNS/"crops_PF_500"/win/"stats.json"))["window"]; H,W=y1-y0,x1-x0
    b=SSD/"work"/("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    tx=ds.dataset(b/"transcripts.parquet").to_table(columns=["x_location","y_location","feature_name"],filter=(pc.field("x_location")>=x0*UM)&(pc.field("x_location")<x1*UM)&(pc.field("y_location")>=y0*UM)&(pc.field("y_location")<y1*UM)&(pc.field("qv")>=20)&pc.field("is_gene")).to_pandas(strings_to_categorical=True)
    r=np.clip((tx.y_location.to_numpy()/UM).astype(int)-y0,0,H-1); c=np.clip((tx.x_location.to_numpy()/UM).astype(int)-x0,0,W-1); isA=tx.feature_name.astype(str).isin(setA).to_numpy()
    dA=ndi.gaussian_filter(np.bincount(r[isA]*W+c[isA],minlength=H*W).reshape(H,W).astype(np.float32),1.5/UM)/UM**2
    dall=ndi.gaussian_filter(np.bincount(r*W+c,minlength=H*W).reshape(H,W).astype(np.float32),1.0/UM)/UM**2
    z=zarr.open(tifffile.TiffFile(b/"morphology_focus"/"ch0000_dapi.ome.tif").aszarr(level=0,series=0),mode="r")
    s18=(ndi.gaussian_filter(np.asarray(z[2,y0:y1,x0:x1]).astype(np.float32),0.5/UM)-2)/606
    core=ndi.binary_opening((dA>=0.534)&(dall>20),iterations=int(1/UM)); cl,n=ndi.label(core)
    for p in regionprops(cl):
        a=p.area*UM**2
        if a<60: continue
        sl=p.slice; m=cl[sl]==p.label; d=np.where(m,dA[sl],0)
        pk=peak_local_max(ndi.gaussian_filter(d,2/UM),min_distance=int(12/UM),threshold_abs=2*0.534,labels=m.astype(int))
        rows.append(dict(window=win,label=p.label,area=a,solidity=p.solidity,elong=p.major_axis_length/max(p.minor_axis_length,1),med18=float(np.median(s18[sl][m])),peaks=len(pk),maxA=float(d.max())))
d=pd.DataFrame(rows); d.to_csv(sys.stdout.buffer if False else "/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad/soma_shape.csv",index=False)
pd.set_option("display.width",200); print(d.describe().round(2).to_string()); print(d.sort_values("area",ascending=False).head(12).round(2).to_string()); print(d.sort_values("med18").head(12).round(2).to_string())
