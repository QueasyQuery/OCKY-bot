import time
import numpy as np
from datetime import datetime

class ResponseSystem():
    '''response system: decide if it should respond'''
    def __init__(self, config, data_manager, sentence_model):
        self.config = config
        self.data_manager = data_manager
        self.sentence_model = sentence_model
        self.response_classifier = data_manager.load_response_classifier()
        self.feature_scaler = data_manager.load_feature_scaler()    

    def should_respond(self, message, bot_user):
        ''' decide if it should respond based on given message
            Returns float between 0 and 1.
        '''
        # get features from message
        features = self._extract_message_features(message, bot_user)
        
        # register as training point
        self.data_manager.record_user_message(message, features)

        # model not trained yet: never respond to messages
        if (not hasattr(self.response_classifier, 'coef_')): return 0.0
        
        # use trained model
        basic_scaled = self.feature_scaler.transform([features['basic_features']])
        probability = self.response_classifier.predict_proba(basic_scaled)[0][1]

        # is training? then don't respond but print the respond result
        if (self.config.get('training', 0) == 1):
            print(f"respond? {probability}")
            return 0.0

        return probability
    
    def _extract_message_features(self, message, bot_user):
        '''extract features for the response system'''
        channel_id = message.channel.id
        q_words = ['?''what', 'how', 'why', 'when', 'where', 'wat', 'hoe', 'waarom', 'wanneer', 'waar']

        # append features (customizable)
        features = []
        features.append(len(message.content))                               # message length
        features.append(len(message.content.split()))                       # amount of words
        features.append(time.time() - self.data_manager.prev_response.get(channel_id, 0))  # time since last bot response
        features.append(len(self.data_manager.msg_activity[channel_id]))     # messages per hour
        features.append(1 if any(word in message.content.lower() for word in q_words) else 0) # question
        features.append(1 if bot_user in message.mentions else 0)           # has mention to bot
        features.append(1 if len(message.content) < 10 else 0)              # really short
        features.append(1 if datetime.now().weekday() >= 5 else 0)          # is it weekend

        # message embedding
        embedding = self.sentence_model.encode(message.content)

        return {'basic_features': np.array(features), 'embedding': embedding, 'message': message}
