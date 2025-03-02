from pgmpy.inference import VariableElimination
import pickle

# Load the model
with open(r"d:\Codes\AIML\Baysian Network\bayesian_model.pkl", "rb") as f:
    model = pickle.load(f)


# Performing inference
infer = VariableElimination(model)
posterior_p = infer.query(['Host'], evidence={'Guest': 2, 'Price': 2})
print("Posterior Probability:\n", posterior_p)
