import joblib

model_file = "models/rf_best_model_rs_selected.joblib"
obj = joblib.load(model_file, mmap_mode='r')
print(type(obj))
print(obj.__dict__.keys())