import pickle
import time
import json
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class DataManager():
    '''Manages the training data and model/embed loading.'''
    def __init__(self, config):
        self.config = config
        self.training_data = []  # training data
        self.prev_response = {}  # channel_id -> timestamp of last bot response
        self.msg_activity = defaultdict(list)  # channel_id -> list of message timestamps

    async def process_feedback(self, reaction, user, is_add, bot_user):
        '''Process user feedback reactions adding it to training data'''
        message = reaction.message
        
        # case 1: reaction on user message: response emoji
        if (message.author != bot_user) and (reaction.emoji == "🗣️") and is_add:
            for data in self.training_data:
                if data['type'] != 'respond_request': continue
                if (data.get('message_id') == message.id):
                    data['should_respond'] = 1
                    break
            return

        # case 2: reaction on a bot message: feedback 
        feedback_value = self._get_feedback_value(reaction.emoji)
        if feedback_value is None: return
        
        for data in self.training_data:
            if (data['type'] == 'bot_response' and data['bot_message_id'] == message.id):
                if is_add:
                    # weirdass averaging system
                    score = 0.5*(data['feedback_score'] + feedback_value)
                else:
                    # this undoes it but only once ^_^ after multiple scores this breaks. But does something
                    # in the right direction. TODO: make a decent system.
                    score = 2 * data['feedback_score'] - feedback_value
                data['feedback_score'] = score
                return score 

    def _get_feedback_value(self, emoji):
        '''Convert emoji reactions to feedback scores'''
        emoji_scores = {"👍": 1.0,"👎": -1.0,"🟩": 1.5,"🟥": -1.5}
        score = emoji_scores.get(str(emoji), None)
        return score

    def record_bot_response(self, original_message, bot_message, response_id, sentence_model):
        '''Record a bot response for learning'''
        self.training_data.append({
            'type': 'bot_response',
            'original_message': original_message.content,
            'original_embedding': sentence_model.encode(original_message.content),
            'response_id': response_id,
            'bot_message_id': bot_message.id,
            'channel_id': bot_message.channel.id,
            'timestamp': time.time(),
            'feedback_score': 1
        })

    def record_user_message(self, message, features):
        '''record this message as a training point for ocky responses. called in ResponseManager'''
        self.training_data.append({
            'type': 'respond_request',
            'message': message.content,
            'message_id': message.id,
            'features': features['basic_features'],
            'should_respond': 0,  # Will be set to 1 if user reacts with 🗣️
            'timestamp': time.time()
        })

    def track_channel_activity(self, message):
        ''' track message activity for feature extraction during response training.
            also add onto the training data
        '''
        channel_id = message.channel.id
        current_time = time.time()

        # add current message
        self.msg_activity[channel_id].append(current_time)
        
        # keep only messages from last hour
        hour_ago = current_time - 3600
        self.msg_activity[channel_id] = [t for t in self.msg_activity[channel_id] if t > hour_ago]

    def load_model(self, path, default, name):
        '''loads a model if it exists'''
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                print(f'Loaded {name}')
                return obj
        except:
            print(f"Starting {name} fresh.")
            return default

    def load_response_classifier(self):
        return self.load_model('response_classifier.pkl', LogisticRegression(), 'response classifier')

    def load_feature_scaler(self):
        return self.load_model('source/models/feature_scaler.pkl', StandardScaler(), 'feature scaler')

    def load_response_embeddings(self):
        return self.load_model('source/models/response_embeddings.pkl', {}, 'response embeddings')

    def load_response_training_data(self):
        return self.load_model('source/models/response_data.pkl', [], 'response training data')

    def save_models(self, response_classifier, feature_scaler, response_embeddings, response_training_data):
        '''Save trained models to disk'''
        models = [
            (response_classifier, 'response_classifier.pkl'),
            (feature_scaler, 'feature_scaler.pkl'),
            (response_embeddings, 'response_embeddings.pkl'),
            (response_training_data, 'response_data.pkl')
        ]
        
        for model_object, filename in models:
            self.save_model(model_object, filename)

    def save_model(self, model_object, filename):
        try:
            with open(f'source/models/{filename}', 'wb') as f:
                pickle.dump(model_object, f)
            print(f"Saved {filename}")
            return True
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return False

    def save_responses(self, data):
        '''Save responses to JSON file'''
        with open(self.config.get('response_file', 'responses.json'), 'w', encoding='utf-8') as f:
            json.dump(data, indent=2, ensure_ascii=False, fp=f)