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
        ''' create initial embeddings for each response, starts as basic encoded texts. 
            These are overridden then by data loading the models.'''
        # create initial embeddings
        for category, responses in self.responses.items():
            for response in responses:
                response_hash = hashlib.md5(response.encode('utf-8')).hexdigest()[:10]
                response_id = (category, response_hash)
                self.response_embeddings[response_id] = self.sentence_model.encode(response)
                self.response_dict[response_id] = response
        
        # override embeddings with saved embeds
        saved_embeddings = self.data_manager.load_response_embeddings()
        self.response_embeddings.update(saved_embeddings)