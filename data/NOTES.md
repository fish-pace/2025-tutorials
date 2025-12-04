# The data

## CHLA

* `CHLA_argo_profiles`: All BGC-Argo profiles Mar 2024 to Nov 2025. Includes whole profile (depths) by platform and cycle.
* In `data/matchups` directory, `CHLA_argo_Rrs_chlor_a_Kd_all` and `CHLA_argo_Rrs_chlor_a_Kd`: The matchups with PACE Rrs, chlor_a and Kd. `all` includes NaNs in Rrs and duplicates (cases where 2 PACE files cover same Argo Time, while the other file has NaN Rrs rows and duplicates removed. Duplicated are removed as follows: if a delayed and NRT file is present for both profiles, chose delayed. If 2 files are present because of the timestamp, chose the file where Argo time is closest to middle of the file time mid-point (t_start to t_end). NaNs are places where clouds or glint or something prevented a Rrs value. The `pace_Rrs_lat`/`pace_Rrs_lon` columns are the matched (nearest) values from the daily PACE Level 3 4km data while the `LATITUDE`, `LONGITUDE`, `TIME` columns are what is in the Bio-Argo data.
* `CHLA_argo_profile_plus_PACE`: This is `CHLA_argo_Rrs_chlor_a_Kd` with the CHLA profile data added. Profile is average CHLA for each depth bin: 0-10dbar, 10-20dbar, etc to 190-200dbar using PRES for dbar (a proxy for depth).

## BBP700 older before I changed approach a bit

* `argo_bgc_global_surface_BBP700`: All the mean surface measurements from Bio-Argo for March 2024 to Nov 23, 2025. BBP700 is depth < 10m. OLDER version
* `bbp700_argo_rrs_all` and `bbp700_argo_rrs`: The matchups with PACE Rrs. all includes NaNs in Rrs while the other file has NaN rows removed. NaNs are places where clouds or glint or something prevented a Rrs value. The `pace_Rrs_lat`/`pace_Rrs_lon` columns are the matched (nearest) values from the PACE Rrs (level 3) data while the `LATITUDE`/`LONGITUDE` columns are what is in the Bio-Argo data.

# Notebooks

* `argopy.ipynb`: created `CHLA_argo_profiles`.
* `argopy-matchups.ipynb`: created `CHLA_argo_profiles_plus_PACE`, `CHLA_argo_Rrs_chlor_a_Kd` and `CHLA_argo_profile_Rrs_chlor_a_Kd`.