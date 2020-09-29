# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List, Union

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
import numpy as np
import pandas as pd
import joblib
from joblib import load
from rasa_sdk.events import SlotSet
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml



class DiabetiesForm(FormAction):
    """Collects sales information and adds it to the spreadsheet"""

    def name(self) -> Text:
        return "diabetic_form1"
        
        
    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        return ["details","gender","famh","smoke","drink"]
 

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:

        dispatcher.utter_message("Thanks for getting in touch, we’ll contact you soon")
        return []
       

    
    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        return{
             "details": [self.from_entity(entity="details", intent="age_details"),self.from_text()],
             "height1": [self.from_entity(entity="height1",intent="BMIcalculation"),self.from_text()],
             "famh": [self.from_entity(entity="famh", intent="family_history"),self.from_text()],
             "gender": [self.from_entity(entity="gender", intent="Gender_type"),self.from_text()],
             "weight": [self.from_entity(entity="weight", intent="BMIcalculation"),self.from_text()],
             "smoke": [self.from_entity(entity="smoke", intent="smoker_status"),self.from_text()],
             "drink": [self.from_entity(entity="drink", intent="drink_status"),self.from_text()]
             }
        gender1 = tracker.get_slot('gender')
        gender1[gender1=='Male'] = 1
        gender1[gender1=='Female'] = 0
        
     
    
    

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Define what the form has to do
            after all required slots are filled"""

        # utter submit template
        dispatcher.utter_message(template="utter_submit")

        return []
        
class ActionshowBMI(Action):

     def name(self) -> Text:
        return "action_BMI"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         meter = 100
         weightt = tracker.get_slot("weight")
         weight = float(weightt)
         heightt = tracker.get_slot("height1")
         height1 = float(heightt)
         height2 = height1/meter
         BMII = round(weight/(height2 * height2),2)
         BMIX ="your {} and {} BMI is: {}".format(weight,height2, BMII)
         dispatcher.utter_message(text=BMIX)


         return [SlotSet("BMI", BMII)]
         BMI = tracker.get_slot('BMI')
         print('BMI')

    

         
class DiabeticResult(FormAction):
    """Collects sales information and adds it to the spreadsheet"""

    def name(self) -> Text:
        return "diabetic_result"
        
    @staticmethod
    def slot_key_db() -> Dict[str, List]:
        """Database of slot values & 
        corresponding questions"""
        

        return {'age': 'details',
            'gender': 'gender',
            'BMI': 'BMI',
            'smoking': 'smoke',
            'drinking': 'drink',
            'family': 'famh'
            }
        
         
            
    def format_user_input(self, dispatcher, tracker, domain):
        """ Format user input as a pd series with the question
        key as the row name, should match format of test_case
        before encoding. 
        """
        
        user_input = ""

        return(user_input)


    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(template= "utter_working_on_it")
        
        # get information from the form & format it
        # for encoding
        slot_question_key = self.slot_key_db()
        formatted_responses = pd.Series(index = slot_question_key.keys())
        
        for index, value in formatted_responses.items():
            formatted_responses[index] = tracker.get_slot(slot_question_key[index])
            print(formatted_responses[index])
      
        loaded_model2, test_case = ClassifierPipeline.load_data()
        answers = ClassifierPipeline.predict_diabetes(formatted_responses,loaded_model2)
        print(answers)
        return[SlotSet("dialect", answers)]
            
class ClassifierPipeline():
    """Load in calssifier & encoders"""

    def name(self) -> Text:
        """Unique identifier of the classfier """

        return "SVM_output"
                   
    
    def load_data():
        d = load("labelencoder1_X.joblib1.dat")
        yaml_file = open('model.yaml', 'r')
        loaded_model_yaml1 = yaml_file.read()
        yaml_file.close()
        loaded_model2 = model_from_yaml(loaded_model_yaml1)
        # load weights into new model
        loaded_model2.load_weights("model.h5")
        print("Loaded model from disk")
        test_case = load("test_case1.joblib1.dat")
        test_case = array(test_case).reshape(1, 1, 6)
        
        
       
        return loaded_model2, test_case

        
        
    def predict_diabetes(test_case,loaded_model2):
        test_case[test_case=='Yes'] = 1
        test_case[test_case=='No'] = 0
        test_case[test_case=='Male'] = 1
        test_case[test_case=='Female'] = 2
        test_case[test_case=='Everyday'] = 1
        test_case[test_case=='Regularly'] = 2
        test_case[test_case=='Never'] = 3
        print(test_case)
        

        test_case = np.asarray(test_case).astype(np.float32)
        print(test_case)
        
        test_case = test_case.reshape(1, -1)
        print(test_case)
        
        test_case = array(test_case).reshape(1, 1, 6)
        print(test_case)
        predict = loaded_model2.predict_classes(test_case)
        print(predict)
        if predict == 0:
            return 'low'
        elif predict == 1:
            return 'high'
            

        
        
    
            