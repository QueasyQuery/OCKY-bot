import numpy as np

class TrainingManager():
    def __init__(self, config, data_manager):
        self.config = config
        self.data_manager = data_manager

    async def training_loop(self, response_system, choice_system, message=None):
        '''Periodic training of both systems'''
        training_data = self.data_manager.training_data
        
        # separate data types
        response_data = [d for d in training_data if d['type'] == 'respond_request']
        choice_data = [d for d in training_data if d['type'] == 'bot_response']

        # check if we have too little data to train
        if (len(response_data) < 10) or (len(choice_data) < 10):
            print(f"not enough data! response:{len(response_data)}, choice:{len(choice_data)} < 10 each")
            return

        # start training message
        print("Training started!")
        if message: 
            status = await message.channel.send(f"Aan het trainen op {len(response_data)} feedback points...")
        
        # train systems
        self._train_response_system(response_system, response_data)
        self._train_choice_system(choice_system, choice_data)
        
        # save models
        self.data_manager.save_models(
            response_system.response_classifier,
            response_system.feature_scaler,
            choice_system.response_embeddings
        )
        
        self._train_stats(choice_system)

        # end training message
        print("Training update complete!")
        if message: 
            await status.edit(content=f"Training afgerond!")
    
    def _train_response_system(self, response_system, datapoints):
        '''Train the response classifier'''
        X, y = [], []
        for data in datapoints:
            if 'features' in data and 'should_respond' in data:
                X.append(data['features'])
                y.append(data['should_respond'])
        
        if len(X) < 2:
            print("Not enough response training data")
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # scale and train
        X_scaled = response_system.feature_scaler.fit_transform(X)
        response_system.response_classifier.fit(X_scaled, y)

    def _train_choice_system(self, choice_system, datapoints):
        '''update response embeddings based on feedback'''
        learning_rate = self.config.get('learning_rate', 0.1)
        
        for data in datapoints:
            if data['feedback_score'] == 0: continue

            response_id = data['response_id']
            feedback = data['feedback_score']
            input_embedding = data['original_embedding']
            
            if response_id not in choice_system.response_embeddings: continue

            # update specific response
            current_embedding = choice_system.response_embeddings[response_id]
            mu = learning_rate * (input_embedding - current_embedding)

            # give feedback by adding/subtracting mu from embedding
            if feedback > 0: choice_system.response_embeddings[response_id] = current_embedding + mu
            else: choice_system.response_embeddings[response_id] = current_embedding - mu

            # update others in same category
            category_learning_rate = learning_rate * 0.5
            target_category = response_id[0]

            for other_response_id in choice_system.response_embeddings:
                if (other_response_id[0] != target_category): continue # other category
                if (other_response_id == response_id): continue # original response

                other_embedding = choice_system.response_embeddings[other_response_id]
                mu = category_learning_rate * (input_embedding - other_embedding)

                # give feedback by adding/subtracting mu from embedding
                if feedback > 0: choice_system.response_embeddings[other_response_id] = other_embedding + mu
                else: choice_system.response_embeddings[other_response_id] = other_embedding - mu

    def _train_stats(self, choice_system):
        # training data breakdown
        respond_data = [d for d in self.data_manager.training_data if d['type'] == 'respond_request']
        positive = sum(1 for d in respond_data if d['should_respond'] == 1)
        negative = sum(1 for d in respond_data if d['should_respond'] == 0)
        
        if negative > 0:
            ratio = positive / negative
            print(f"{len(respond_data)} Training Feedback Points: ({ratio:.4f} POS/NEG ratio)")
        else:
            print(f"{len(respond_data)} Training Feedback Points: ({positive} positive, {negative} negative)")

        # embedding drift (how much embeddings changed from original)
        total_drift = 0
        drift_count = 0
        
        for resp_id, curr_embed in choice_system.response_embeddings.items():
            if resp_id not in choice_system.response_dict: 
                continue # skip if not in dict
                
            original_text = choice_system.response_dict[resp_id]
            original_embed = choice_system.sentence_model.encode(original_text)

            # calculate cosine distance between embeddings
            similarity = np.dot(curr_embed, original_embed) / (
                np.linalg.norm(curr_embed) * np.linalg.norm(original_embed)
            )
            drift = 1 - similarity
            total_drift += drift
            drift_count += 1
        
        if drift_count > 0:
            avg_drift = total_drift / drift_count
            print(f"Embedding Drift: {avg_drift:.4f}")