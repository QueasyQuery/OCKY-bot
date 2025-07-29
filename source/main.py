import json
import importlib
from transformer_ram import RAMTransformer
import asyncio
import bot_logic
import response_sys
import choice_sys
import training
import data

class OCKYBot:
    def __init__(self, config_file="config.json"):
        # load config, responses and transformer
        self.config = self.load_json(config_file)
        self.responses = self.load_json(self.config.get('response_file', 'responses.json'))
        self.sentence_model = RAMTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # initialize systems
        self.data_manager     = data.DataManager(self.config)
        self.training_manager = training.TrainingManager(self.config, self.data_manager)
        self.response_system  = response_sys.ResponseSystem(self.config, self.data_manager, self.sentence_model)
        self.choice_system    = choice_sys.ChoiceSystem(self.config, self.data_manager, self.sentence_model, self.responses)
        self.discord_handler  = bot_logic.DiscordHandler(self.config, self.data_manager, self)
        self.sentence_model.unload() # remove from RAM

        # misc
        self.bot = self.discord_handler.bot
        self.channel_id = self.config['channel']

    def load_json(self, filename):
        '''load dict from JSON file'''
        with open(filename, 'r',encoding="utf-8") as f:
            return json.load(f)
    
    def run(self, token):
        '''Start the bot'''
        self.bot.run(token)

    async def shutdown(self):
        print("shutting down...")
        if self.config['training'] == 1:
            channel = self.bot.get_channel(self.channel_id)
            await channel.edit(topic='**STATUS: OFFLINE**')
        await self.bot.close()

    def reload(self, module):
        match module:
            case "response": 
                importlib.reload(response_sys)
                self.response_system  = response_sys.ResponseSystem(self.config, self.data_manager, self.sentence_model)
            case "choice": 
                importlib.reload(choice_sys)
                self.choice_system = choice_sys.ChoiceSystem(self.config, self.data_manager, self.sentence_model, self.responses)
            case "training": 
                importlib.reload(training)
                self.training_manager = training.TrainingManager(self.config, self.data_manager)
            case "data": 
                importlib.reload(data)
                self.data_manager = data.DataManager(self.config)

if __name__ == "__main__":
    # read token
    with open('TOKEN.txt', 'r') as f:
        token = f.readline().strip()

    # run
    print("initializing...")
    bot = OCKYBot()
    try:
        bot.run(token)
    except Exception as e:
        print(f"Error: {e}")
        asyncio.run(bot.shutdown())