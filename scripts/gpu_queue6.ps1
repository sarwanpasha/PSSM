# GPU queue 6: ESM-2 650M wild-type masked marginals for the ProteinGym multi-mutant assays.
$env:SYCL_CACHE_PERSISTENT = "1"
$py = "$env:USERPROFILE\miniforge3\envs\pssm\python.exe"; $code = Split-Path -Parent $MyInvocation.MyCommand.Path; $root = "$env:USERPROFILE\pssm_work"
$log = "$root\logs\gpu_queue6.log"
& $py -u "$code\extract_wt_marginals.py" esm2_t33_650M_UR50D $root GFP_AEQVI_Sarkisyan_2016 GRB2_HUMAN_Faure_2021 F7YBW8_MESOW_Aakre_2015 DLG4_HUMAN_Faure_2021 YAP1_HUMAN_Araya_2012 RASK_HUMAN_Weng_2022_abundance GCN4_YEAST_Staller_2018 PABP_YEAST_Melamed_2013 CAPSD_AAV2S_Sinai_2021 2>&1 | Out-File -Append -Encoding utf8 $log
New-Item -ItemType Directory -Force "$code\..\stage\pgw" | Out-Null
Copy-Item "$root\emb\*__pg_*__wtlogp.npy" "$code\..\stage\pgw\" -Force
"GPU QUEUE6 DONE $(Get-Date)" | Out-File -Append -Encoding utf8 $log
