import common.star_api.design_manager as dm

version = dm.STARCCMInstall(r"E:\Siemens\STARCCM\starpy\STAR-CCM+19.04.007-2-ga404231\star\bin\starccm+.bat")
path = r"E:\OneDrive - Siemens AG\mdx\hackathon\2024\starPy\eCubded\proj_taw2.dmprj"
port = 47827
dmprj = dm.DesignManagerProject.get_live_proj(path, port, version)

study = dmprj.get_study("Opt2")
angle_params = [p for p in study.parameters if "angle" in p.name]
weight_params = [p for p in study.parameters if "P_W_" in p.name]
print(f"There are {len(angle_params)} angle parameters.")
print(f"There are {len(weight_params)} weight parameters.")
for p in angle_params:
    print(f"{p.name}: [{p.min}, {p.max}]")

for p in angle_params:
    mode = "abs"
    p.set_ranges(mode=mode, u_bnd=3.0, l_bnd=-3.0)

for p in weight_params:
    p.set_ranges(u_bnd=25.0, l_bnd=-25.0, sig_fig=4)

dmprj.sync(push_to_star=True)
