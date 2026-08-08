# Trained model

`NOCDE_MEEB_Sim.eqx` contains the serialized Equinox leaves for the ten-zone
NODE/NCDE model. `MPC_Script.py` constructs the matching model skeleton before
deserialization.

The model also depends on scaling metadata reconstructed from
`data/simulated_data.csv`.
