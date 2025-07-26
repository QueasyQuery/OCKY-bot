import time
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class ChoiceSystem():
    def __init__(self, config, data_manager, sentence_model, responses):
        self.config = config
        self.data_manager = data_manager
        self.sentence_model = sentence_model
        self.responses = responses
        
        self.response_dict = {}  # response_id -> response
        self.response_embeddings = {}  # response_id -> target embedding
        
        self._load_embeddings()

    def get_response(self, message):
        ''' Choice System: selects best response using embedding similarity matching'''
        msg_embed = self.sentence_model.encode(message.content)
        best_score = -1
        best_response = None
        
        # find most similar response embedding
        for response_id, target_embed in self.response_embeddings.items():
            similarity = np.dot(msg_embed, target_embed) / (
                np.linalg.norm(msg_embed) * np.linalg.norm(target_embed)
            )
            # randomness
            noise = np.random.normal(0, self.config['randomness'])
            similarity += noise
            # check if best
            if similarity > best_score:
                best_score = similarity
                best_response = response_id
 
        # get the actual response text
        if best_response:
            print(f'CHOICE: {best_response}, score {best_score}')
            response_text = self.response_dict[best_response]
            return {'id': best_response, 'text': response_text, 'score': best_score}
        
        return None

    def _load_embeddings(self):
        ''' create initial embeddings for each response, using example_input if available.
            these are overridden then by datamanager loading the models.
        '''
        # create initial embeddings
        for category, category_data in self.responses.items():
            # parse per catgory
            responses_list = category_data.get('responses', [])
            example_input = category_data.get('example_input')
            
            for response in responses_list:
                response_hash = hashlib.md5(response.encode('utf-8')).hexdigest()[:10]
                response_id = (category, response_hash)
                
                # Use example_input for embedding if available, otherwise fall back to response text
                if example_input != 'None': self.response_embeddings[response_id] = self.sentence_model.encode(example_input)
                else: self.response_embeddings[response_id] = self.sentence_model.encode(response)
                    
                # add a smidgeon of randomness
                self.response_dict[response_id] = response

        # override embeddings with saved embeds. Filter out responses no longer in the response file
        saved_embeddings = self.data_manager.load_response_embeddings()
        filtered_saved = {k: v for k, v in saved_embeddings.items() if k in self.response_dict}
        self.response_embeddings.update(filtered_saved)