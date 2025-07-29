from sentence_transformers import SentenceTransformer
import gc

class RAMTransformer:
    ''' stand-in for the SentenceTransformer class that loads/unloads 
        SentenceTransformer so it's only in RAM when needed
    '''
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        '''load model if not already loaded'''
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
    
    def encode(self, sentences, **kwargs):
        '''load model if needed and encode'''
        self._load_model()
        return self._model.encode(sentences, **kwargs)
    
    def unload(self):
        '''unload the model to free memory'''
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()