classfier_chatgpt api calls gpt-40-mini to act as an classifier
classfier_detoxify uses Detoxify and prints metric scores along with labels
classifier_hybrird uses Detoxify+ Rule based override to further augument quality
distilbert trains a distilbert on dataset and returns confusion matrix and other graphs
load_openchat, load_prompts, define_query, query are pipline for local instance of openchat 
response_openchat sends red-teaming prompts to a hosted openchat/openchat-3.5-1210 on Hugging Face and collects the model's responses
mitigation.py is the mitigation script

id like to add that my dumbass somehow questioned how will we get resposnes to apply the mitigation principles and vaguely decided to use gemini, instead of using the dataset of prompts i already have. due to this, i was unable compare the effectiveness of the mitigation techniques i have applied and the corresponding comparsion visualization is missing. however i have added run examples in the documentaion as a depiction of the effectiveness of applied mitigation principles. 
since it is as honest mistake which can we easily rectified if not time constraints, ill appreciate your oversight on this.