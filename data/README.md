# Official UCI data and provenance

Source: [Glass Identification, UCI](https://archive.ics.uci.edu/dataset/42/glass+identification).  
Citation: German, B. (1987). *Glass Identification* [Dataset]. DOI [10.24432/C5WW2P](https://doi.org/10.24432/C5WW2P).  
Licence: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by UCI.

The successful acquisition used `ucimlrepo.fetch_ucirepo(id=42)` on 2026-10-22. Actual API data URL: https://archive.ics.uci.edu/static/public/42/data.csv. No fallback was used. The runner contains an official-CSV fallback with a bounded 1 MB read, reports the primary failure if invoked, and never fabricates replacement samples.

- `glass.csv`: 214 original rows returned by the API, serialized locally, 11,415 bytes. Nine features plus sample ID and target.
- `dictionary.csv`: official variable roles, numeric/categorical types, descriptions and units.
- `uci_metadata.json`: original API metadata snapshot.
- `manifest.json`: acquisition method/time, citation, licence, file sizes and SHA-256 values.

The cached table is not claimed to be byte-identical to the HTTP payload. SHA-256:
`c4677c97a86cc9f63c5756f13671a7a9ffb3f7dc3031524023deebcae75d14ad`.

Feature allowlist: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe. RI is refractive index; the other features are weight percentages in their corresponding oxides. ID is excluded from predictors. Present target labels are 1,2,3,5,6,7; class 4 has no observations.

No missing cells or nonfinite predictors were found. IDs 39 and 40 have identical feature vectors and class 1 labels. Modeling keeps ID 39 and excludes ID 40 before splitting; this original table is not edited. No conflicting labels were found for identical feature vectors. The resulting 213-row experiment and train/test counts are recorded in results/data_quality.json.

Transformations and evaluation are not endorsed by UCI or the dataset creator. Labels are benchmark annotations, not new laboratory determinations. Broad categories cannot establish individual source identity. No personally identifying case or patient records are introduced.

