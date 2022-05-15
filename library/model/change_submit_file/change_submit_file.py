import pandas as ps

def change_submit_file(id_name, predict_id_name, test, predict):
  id_name = str(id_name)
  predict_id_name = str(predict_id_name)
  
  submit = pd.DataFrame({
     id_name: test[id_name], 
     predict_id_name: predict})
  submit.to_csv("submit.csv", header=True, index=False)
  print("Your submission was successfully saved!")
