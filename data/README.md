# Input data

- `simulated_data.csv` contains the measurements used to reconstruct the state
  and input scalers required by the serialized model.
- `evenements-pointe.csv` defines demand-response intervals with `dateDebut` and
  `dateFin` columns. Timestamps include UTC offsets and are converted to
  `America/Toronto` by the controller.
- `initial_zone_temperatures_0000_all_dr_days_wide.csv` contains one midnight
  initial-condition row per demand-response day for the ten controlled zones.

Column names and zone order are part of the model interface. Do not reorder or
rename them without updating the data-loading and zone-mapping code.
