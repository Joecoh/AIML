from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pickle

model = BayesianNetwork([('Guest', 'Host'), ('Price', 'Host')])

cpd_guest = TabularCPD('Guest', 3, [[0.33], [0.33], [0.33]])
cpd_price = TabularCPD('Price', 3, [[0.33], [0.33], [0.33]])
cpd_host = TabularCPD('Host', 3, 
    [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
     [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
     [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
    evidence=['Guest', 'Price'],
    evidence_card=[3, 3])

model.add_cpds(cpd_guest, cpd_price, cpd_host)
assert model.check_model()  # Check if the model is valid

with open("bayesian_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model has been defined and saved.")
