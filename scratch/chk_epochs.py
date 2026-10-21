import glob,os,re,numpy as np
files=sorted(glob.glob("S-band-fits/*2D_Fitting*.txt"))
ra0=(19+15/60+11.558/3600)*15; de0=10+56/60+44.905/3600; cd=np.cos(np.radians(de0))
for f in files:
    ts=int(re.search(r"img_(\d+)",os.path.basename(f)).group(1)); mjd=ts/86400+40587
    comps=[];c=None
    for l in open(f):
        if l.startswith("Component #"): c={}; comps.append(c)
        elif "Center X" in l and c is not None:
            m=re.search(r"=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)",l)
            if m: h,mi,s,e=map(float,m.groups()); c['ra']=(h+mi/60+s/3600)*15; c['ex']=e*15*cd
        elif "Center Y" in l and c is not None:
            m=re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)",l)
            if m:
                d,a,s,e=map(float,m.groups()); sg=-1 if '-' in m.group(1) else 1
                c['dec']=sg*(abs(d)+a/60+s/3600); c['ey']=e
        elif "Integrated flux" in l and c is not None:
            m=re.search(r"=\s*([\d\.eE+-]+)\s*±",l)
            if m: c['flux']=float(m.group(1))
    out=[]
    for c in comps:
        if 'ra' not in c or 'dec' not in c: continue
        dx=(c['ra']-ra0)*3600*cd; dy=(c['dec']-de0)*3600
        kind='core' if 288.7979<c['ra']<288.7982 else ('N' if dy>0 else 'S')
        out.append(f"{kind}(dx={dx:.2f},dy={dy:.2f},err={c['ex']:.3f},{c['ey']:.3f},F={c.get('flux',0):.2e})")
    print(f"{mjd:.2f} ts={ts}", out)
